# Copyright 2026 EngineAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish metric and colorized range images from organized lidar clouds."""

import struct

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField


def xyz_offsets(message: PointCloud2):
    """Return FLOAT32 x/y/z field offsets from an organized PointCloud2."""
    offsets = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(
                    "PointCloud2 x, y and z must be scalar FLOAT32 fields"
                )
            offsets[field.name] = field.offset
    if set(offsets) != {"x", "y", "z"}:
        raise ValueError("PointCloud2 must contain x, y and z fields")
    if any(offset + 4 > message.point_step for offset in offsets.values()):
        raise ValueError("PointCloud2 XYZ field exceeds point_step")
    return offsets


def pointcloud_to_xyz(message: PointCloud2) -> np.ndarray:
    """Decode an organized XYZ cloud into an HxWx3 float32 array."""
    if message.height < 1 or message.width < 1:
        raise ValueError("PointCloud2 must be organized with height and width")
    if (
        message.point_step <= 0
        or message.row_step < message.width * message.point_step
        or len(message.data) < message.row_step * message.height
    ):
        raise ValueError("PointCloud2 data dimensions are inconsistent")
    offsets = xyz_offsets(message)
    byte_order = ">" if message.is_bigendian else "<"
    points = np.empty((message.height, message.width, 3), dtype=np.float32)
    for row in range(message.height):
        row_start = row * message.row_step
        for column in range(message.width):
            start = row_start + column * message.point_step
            points[row, column, 0] = struct.unpack_from(
                byte_order + "f", message.data, start + offsets["x"]
            )[0]
            points[row, column, 1] = struct.unpack_from(
                byte_order + "f", message.data, start + offsets["y"]
            )[0]
            points[row, column, 2] = struct.unpack_from(
                byte_order + "f", message.data, start + offsets["z"]
            )[0]
    return points


def ranges_from_xyz(points: np.ndarray, ray_origin: np.ndarray) -> np.ndarray:
    """Return per-pixel range from the virtual lidar origin."""
    origin = np.asarray(ray_origin, dtype=np.float32).reshape(3)
    valid = np.all(np.isfinite(points), axis=2)
    valid &= np.linalg.norm(points, axis=2) > 1.0e-12
    ranges = np.full(points.shape[:2], np.nan, dtype=np.float32)
    ranges[valid] = np.linalg.norm(points[valid] - origin, axis=1)
    return ranges


def colorize_range(
    ranges: np.ndarray,
    min_distance: float,
    max_distance: float,
) -> np.ndarray:
    """Colorize a range image with OpenCV JET.

    Near values map toward blue and far values toward red, matching
    ``cv2.COLORMAP_JET``. Missing or out-of-range pixels are black.
    """
    span = max_distance - min_distance
    valid = np.isfinite(ranges)
    gray = np.zeros(ranges.shape, dtype=np.uint8)
    if span > 0.0:
        normalized = np.clip((ranges - min_distance) / span, 0.0, 1.0)
        gray[valid] = np.rint(normalized[valid] * 255.0).astype(np.uint8)
        valid &= (ranges >= min_distance) & (ranges <= max_distance)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


def numpy_to_image(header, array: np.ndarray, encoding: str) -> Image:
    """Pack a numpy image into a sensor_msgs/Image."""
    message = Image()
    message.header = header
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = False
    if encoding == "32FC1":
        pixels = np.ascontiguousarray(array, dtype=np.float32)
        message.step = message.width * 4
    elif encoding == "bgr8":
        pixels = np.ascontiguousarray(array, dtype=np.uint8)
        message.step = message.width * 3
    elif encoding == "mono8":
        pixels = np.ascontiguousarray(array, dtype=np.uint8)
        message.step = message.width
    else:
        raise ValueError(f"unsupported image encoding '{encoding}'")
    message.data = pixels.tobytes()
    return message


class LidarRangeImageNode(Node):
    """Convert organized navigation lidar into range-image topics."""

    def __init__(self):
        super().__init__("lidar_range_image")
        self.cloud_topic = self.declare_parameter(
            "cloud_topic", "/navigation/lidar/points"
        ).value
        self.range_image_topic = self.declare_parameter(
            "range_image_topic", "/navigation/lidar/range_image"
        ).value
        self.range_image_viz_topic = self.declare_parameter(
            "range_image_viz_topic", "/navigation/lidar/range_image_viz"
        ).value
        self.ray_origin = np.asarray(
            self.declare_parameter(
                "ray_origin", [0.105, 0.0, 0.185]
            ).value,
            dtype=np.float32,
        )
        self.min_distance = float(
            self.declare_parameter("min_distance", 0.1).value
        )
        self.max_distance = float(
            self.declare_parameter("max_distance", 5.0).value
        )
        self.viz_scale = int(self.declare_parameter("viz_scale", 10).value)
        self.flip_vertical = bool(
            self.declare_parameter("flip_vertical", True).value
        )
        self.flip_horizontal = bool(
            self.declare_parameter("flip_horizontal", True).value
        )
        if (
            self.ray_origin.shape != (3,)
            or not np.all(np.isfinite(self.ray_origin))
            or self.min_distance < 0.0
            or self.min_distance >= self.max_distance
            or self.viz_scale < 1
        ):
            raise ValueError("invalid lidar_range_image parameters")

        self.range_publisher = self.create_publisher(
            Image, self.range_image_topic, qos_profile_sensor_data
        )
        self.viz_publisher = self.create_publisher(
            Image, self.range_image_viz_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Publishing range images from {self.cloud_topic} to "
            f"{self.range_image_topic} and {self.range_image_viz_topic}"
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            points = pointcloud_to_xyz(message)
        except (ValueError, struct.error) as error:
            self.get_logger().warning(
                f"Ignoring incompatible point cloud: {error}",
                throttle_duration_sec=2.0,
            )
            return
        ranges = ranges_from_xyz(points, self.ray_origin)
        self.range_publisher.publish(
            numpy_to_image(message.header, ranges, "32FC1")
        )
        self.viz_publisher.publish(
            numpy_to_image(
                message.header,
                self._prepare_color(
                    colorize_range(
                        ranges, self.min_distance, self.max_distance
                    )
                ),
                "bgr8",
            )
        )

    def _prepare_color(self, color: np.ndarray) -> np.ndarray:
        """Flip to camera view and nearest-neighbor scale a BGR range image."""
        if self.flip_vertical:
            color = np.flipud(color)
        if self.flip_horizontal:
            color = np.fliplr(color)
        if self.viz_scale > 1:
            color = cv2.resize(
                color,
                (
                    color.shape[1] * self.viz_scale,
                    color.shape[0] * self.viz_scale,
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        return color


def main(args=None):
    rclpy.init(args=args)
    node = LidarRangeImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
