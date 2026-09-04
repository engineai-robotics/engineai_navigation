"""ROS 2 node for PM01 reinforcement-learning lidar navigation."""

import struct
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy, PointCloud2, PointField
from std_srvs.srv import SetBool

from .core import (
    NUM_AZIMUTHS,
    NUM_CHANNELS,
    NUM_POINTS,
    PoseVelocityEstimator,
    build_observation,
    goal_observation,
    lidar_observation,
    projected_gravity,
)
from .policy import NavigationPolicy


class RlNavigationNode(Node):
    """Run the recurrent navigation actor from ROS lidar and LIO odometry."""

    def __init__(self):
        super().__init__("rl_nav_policy")
        default_policy = (
            get_package_share_directory("rl_nav_policy") + "/policy/817.onnx"
        )
        policy_path = self.declare_parameter(
            "policy_path", default_policy
        ).value
        if not policy_path:
            policy_path = default_policy
        self.cloud_topic = self.declare_parameter(
            "cloud_topic", "/navigation/lidar/points"
        ).value
        self.odom_topic = self.declare_parameter(
            "odom_topic", "/lio/robo/odom"
        ).value
        self.goal_topic = self.declare_parameter(
            "goal_topic", "/goal_pose"
        ).value
        self.cmd_vel_topic = self.declare_parameter(
            "cmd_vel_topic", "/cmd_vel"
        ).value
        self.joystick_cmd_vel_topic = self.declare_parameter(
            "joystick_cmd_vel_topic", "/cmd_vel_joystick"
        ).value
        self.joy_topic = self.declare_parameter(
            "joy_topic", "/joy"
        ).value
        self.mode_button = int(
            self.declare_parameter("mode_button", 6).value
        )
        self.disable_button = int(
            self.declare_parameter("disable_button", 2).value
        )
        self.enable_button = int(
            self.declare_parameter("enable_button", 3).value
        )
        self.policy_enabled = bool(
            self.declare_parameter("initial_policy_enabled", False).value
        )
        self.policy_hz = float(
            self.declare_parameter("policy_hz", 5.0).value
        )
        self.input_timeout = float(
            self.declare_parameter("input_timeout", 0.5).value
        )
        self.goal_tolerance = float(
            self.declare_parameter("goal_tolerance", 0.5).value
        )
        self.max_distance = float(
            self.declare_parameter("max_distance", 5.0).value
        )
        self.ray_origin = np.asarray(
            self.declare_parameter(
                "ray_origin", [0.105, 0.0, 0.185]
            ).value,
            dtype=np.float64,
        )
        self.action_filter_alpha = float(
            self.declare_parameter(
                "action_filter_alpha", 0.8819113783
            ).value
        )
        velocity_filter_alpha = float(
            self.declare_parameter("velocity_filter_alpha", 0.8).value
        )
        velocity_max_dt = float(
            self.declare_parameter("velocity_max_dt", 0.5).value
        )
        initial_goal_enabled = bool(
            self.declare_parameter("initial_goal_enabled", False).value
        )
        initial_goal = np.asarray(
            (
                float(self.declare_parameter("initial_goal_x", 0.0).value),
                float(self.declare_parameter("initial_goal_y", 0.0).value),
                float(self.declare_parameter("initial_goal_z", 0.5).value),
            ),
            dtype=np.float64,
        )

        if (
            self.policy_hz <= 0.0
            or self.input_timeout <= 0.0
            or self.goal_tolerance < 0.0
            or self.max_distance <= 0.0
            or self.ray_origin.shape != (3,)
            or initial_goal.shape != (3,)
            or min(
                self.mode_button,
                self.disable_button,
                self.enable_button,
            ) < 0
            or self.joystick_cmd_vel_topic == self.cmd_vel_topic
            or not 0.0 <= self.action_filter_alpha < 1.0
        ):
            raise ValueError("invalid rl_nav_policy parameters")

        self.policy = NavigationPolicy(str(policy_path))
        self.velocity_estimator = PoseVelocityEstimator(
            velocity_filter_alpha, velocity_max_dt
        )
        self.cloud_points: Optional[np.ndarray] = None
        self.cloud_received_at = 0.0
        self.odom_received_at = 0.0
        self.position = np.zeros(3, dtype=np.float64)
        self.quaternion = np.asarray((0.0, 0.0, 0.0, 1.0))
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.odom_frame = ""
        self.goal: Optional[np.ndarray] = (
            initial_goal if initial_goal_enabled else None
        )
        self.pending_goal: Optional[tuple[np.ndarray, str]] = None
        self.last_raw_action = np.zeros(3, dtype=np.float32)
        self.filtered_action = np.zeros(3, dtype=np.float32)
        self.last_status_at = 0.0
        self.last_hit_count = 0
        self.last_stop_reason = ""
        self.disable_combo_pressed = False
        self.enable_combo_pressed = False

        self.cmd_publisher = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.odom_subscription = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, 20
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped, self.goal_topic, self._on_goal, 10
        )
        self.joy_subscription = self.create_subscription(
            Joy,
            self.joy_topic,
            self._on_joy,
            qos_profile_sensor_data,
        )
        self.joystick_cmd_subscription = self.create_subscription(
            Twist,
            self.joystick_cmd_vel_topic,
            self._on_joystick_cmd,
            10,
        )
        self.set_policy_enabled_service = self.create_service(
            SetBool,
            "~/set_policy_enabled",
            self._on_set_policy_enabled,
        )
        self.timer = self.create_timer(1.0 / self.policy_hz, self._step)
        self.get_logger().info(
            f"RL navigation ready: policy={policy_path}, "
            f"cloud={self.cloud_topic}, odom={self.odom_topic}, "
            f"goal={self.goal_topic}, output={self.cmd_vel_topic}, "
            f"mode={'navigation' if self.policy_enabled else 'joystick'}"
        )
        self.get_logger().info(
            f"Joy gate on {self.joy_topic}: MODE({self.mode_button}) + "
            f"X({self.disable_button}) disables, MODE({self.mode_button}) + "
            f"Y({self.enable_button}) enables"
        )
        self.get_logger().info(
            "ROS gate: "
            f"{self.get_fully_qualified_name()}/set_policy_enabled "
            "(std_srvs/SetBool, data=true enables RL navigation)"
        )
        if not initial_goal_enabled:
            self.get_logger().info(
                "Waiting for a world-frame goal on /goal_pose"
            )

    @staticmethod
    def _xyz_offsets(message: PointCloud2):
        offsets = {}
        for field in message.fields:
            if field.name in ("x", "y", "z"):
                if (
                    field.datatype != PointField.FLOAT32
                    or field.count != 1
                ):
                    raise ValueError(
                        "PointCloud2 x, y and z must be scalar FLOAT32 fields"
                    )
                offsets[field.name] = field.offset
        if set(offsets) != {"x", "y", "z"}:
            raise ValueError("PointCloud2 must contain x, y and z fields")
        if any(offset + 4 > message.point_step for offset in offsets.values()):
            raise ValueError("PointCloud2 XYZ field exceeds point_step")
        return offsets

    @classmethod
    def pointcloud_to_numpy(cls, message: PointCloud2) -> np.ndarray:
        """Decode the strict organized XYZ contract into a 656x3 array."""
        if (
            message.height != NUM_CHANNELS
            or message.width != NUM_AZIMUTHS
        ):
            raise ValueError(
                f"expected PointCloud2 {NUM_CHANNELS}x{NUM_AZIMUTHS}, "
                f"got {message.height}x{message.width}"
            )
        if (
            message.point_step <= 0
            or message.row_step < message.width * message.point_step
            or len(message.data) < message.row_step * message.height
        ):
            raise ValueError("PointCloud2 data dimensions are inconsistent")
        offsets = cls._xyz_offsets(message)
        byte_order = ">" if message.is_bigendian else "<"
        points = np.empty((NUM_POINTS, 3), dtype=np.float32)
        index = 0
        for row in range(message.height):
            row_start = row * message.row_step
            for column in range(message.width):
                start = row_start + column * message.point_step
                points[index, 0] = struct.unpack_from(
                    byte_order + "f", message.data, start + offsets["x"]
                )[0]
                points[index, 1] = struct.unpack_from(
                    byte_order + "f", message.data, start + offsets["y"]
                )[0]
                points[index, 2] = struct.unpack_from(
                    byte_order + "f", message.data, start + offsets["z"]
                )[0]
                index += 1
        return points

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            self.cloud_points = self.pointcloud_to_numpy(message)
            self.cloud_received_at = time.monotonic()
        except (ValueError, struct.error) as error:
            self.get_logger().warning(
                f"Ignoring incompatible processed point cloud: {error}",
                throttle_duration_sec=2.0,
            )

    def _on_odom(self, message: Odometry) -> None:
        position = np.asarray(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ),
            dtype=np.float64,
        )
        quaternion = np.asarray(
            (
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ),
            dtype=np.float64,
        )
        timestamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        try:
            linear, angular, _ = self.velocity_estimator.update(
                timestamp, position, quaternion
            )
            self.position = position
            self.quaternion = quaternion
            self.linear_velocity = linear
            self.angular_velocity = angular
            self.odom_frame = message.header.frame_id
            self.odom_received_at = time.monotonic()
            if self.pending_goal is not None:
                pending_goal, pending_frame = self.pending_goal
                if pending_frame == self.odom_frame:
                    self.pending_goal = None
                    self._accept_goal(pending_goal)
                else:
                    self.get_logger().warning(
                        f"Discarding pending goal in frame "
                        f"'{pending_frame}'; odometry uses "
                        f"'{self.odom_frame}'"
                    )
                    self.pending_goal = None
        except ValueError as error:
            self.get_logger().warning(
                f"Ignoring invalid odometry: {error}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, message: PoseStamped) -> None:
        goal = np.asarray(
            (
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(goal)):
            self.get_logger().warning("Ignoring non-finite navigation goal")
            return
        if not message.header.frame_id:
            self.get_logger().warning("Ignoring goal with an empty frame")
            return
        if not self.odom_frame:
            self.pending_goal = (goal, message.header.frame_id)
            self.get_logger().info(
                f"Goal in '{message.header.frame_id}' queued until "
                "odometry establishes its world frame"
            )
            return
        if message.header.frame_id != self.odom_frame:
            self.get_logger().warning(
                f"Ignoring goal in frame '{message.header.frame_id}'; "
                f"expected '{self.odom_frame}'"
            )
            return
        self._accept_goal(goal)

    @staticmethod
    def _button_pressed(message: Joy, index: int) -> bool:
        return index < len(message.buttons) and message.buttons[index] != 0

    def _set_policy_enabled(self, enabled: bool, source: str) -> bool:
        """Enable or disable RL velocity output. Return True if mode changed."""
        enabled = bool(enabled)
        if enabled == self.policy_enabled:
            return False
        self.policy_enabled = enabled
        self._reset_policy_state()
        self._publish(np.zeros(3))
        if enabled:
            self.last_stop_reason = ""
            self.get_logger().info(
                f"RL navigation velocity output enabled by {source}"
            )
        else:
            self.last_stop_reason = f"disabled by {source}"
            self.get_logger().info(
                f"Joystick velocity control enabled by {source}"
            )
        return True

    def _on_set_policy_enabled(self, request, response):
        """Switch RL navigation from a ROS SetBool service call."""
        changed = self._set_policy_enabled(
            request.data, "set_policy_enabled"
        )
        response.success = True
        if request.data:
            response.message = (
                "RL navigation enabled"
                if changed
                else "RL navigation already enabled"
            )
        else:
            response.message = (
                "joystick control enabled"
                if changed
                else "joystick control already enabled"
            )
        return response

    def _on_joy(self, message: Joy) -> None:
        """Gate policy velocity output using MODE+X and MODE+Y."""
        mode_pressed = self._button_pressed(message, self.mode_button)
        disable_pressed = mode_pressed and self._button_pressed(
            message, self.disable_button
        )
        enable_pressed = mode_pressed and self._button_pressed(
            message, self.enable_button
        )

        if disable_pressed and not self.disable_combo_pressed:
            self._set_policy_enabled(False, "MODE+X")
        elif enable_pressed and not self.enable_combo_pressed:
            self._set_policy_enabled(True, "MODE+Y")

        self.disable_combo_pressed = disable_pressed
        self.enable_combo_pressed = enable_pressed

    def _on_joystick_cmd(self, message: Twist) -> None:
        """Forward joystick velocity only while navigation is disabled."""
        if not self.policy_enabled:
            self.cmd_publisher.publish(message)

    def _accept_goal(self, goal: np.ndarray) -> None:
        """Activate a validated goal in the odometry world frame."""
        self.goal = goal
        self._reset_policy_state()
        self.get_logger().info(
            f"Navigation goal set to "
            f"({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) "
            f"in {self.odom_frame}"
        )

    def _reset_policy_state(self) -> None:
        self.policy.reset()
        self.last_raw_action.fill(0.0)
        self.filtered_action.fill(0.0)

    def _publish(self, action: np.ndarray) -> None:
        command = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        message = Twist()
        message.linear.x = float(command[0])
        message.linear.y = float(command[1])
        message.angular.z = float(command[2])
        self.cmd_publisher.publish(message)

    def _stop(self, reason: str) -> None:
        self._reset_policy_state()
        self._publish(np.zeros(3))
        if reason != self.last_stop_reason:
            self.get_logger().info(f"Navigation stopped: {reason}")
            self.last_stop_reason = reason

    def _step(self) -> None:
        if not self.policy_enabled:
            return

        now = time.monotonic()
        if self.goal is None:
            reason = (
                "waiting for odometry to accept queued goal"
                if self.pending_goal is not None
                else "waiting for goal"
            )
            self._stop(reason)
            return
        if (
            self.cloud_points is None
            or self.cloud_received_at == 0.0
            or now - self.cloud_received_at > self.input_timeout
        ):
            self._stop("processed point cloud unavailable or stale")
            return
        if (
            self.odom_received_at == 0.0
            or now - self.odom_received_at > self.input_timeout
        ):
            self._stop("odometry unavailable or stale")
            return

        try:
            goal, planar_distance = goal_observation(
                self.position, self.quaternion, self.goal
            )
            if planar_distance <= self.goal_tolerance:
                self._stop("goal reached")
                return
            lidar, hit_count = lidar_observation(
                self.cloud_points,
                self.quaternion,
                self.ray_origin,
                self.max_distance,
            )
            observation = build_observation(
                self.linear_velocity,
                self.angular_velocity,
                projected_gravity(self.quaternion),
                self.last_raw_action,
                goal,
                lidar,
            )
            raw_action = self.policy.infer(observation)
            target_action = np.tanh(raw_action)
            blend = 1.0 - self.action_filter_alpha
            self.filtered_action = (
                self.action_filter_alpha * self.filtered_action
                + blend * target_action
            ).astype(np.float32)
            self.last_raw_action = raw_action
            self.last_hit_count = hit_count
            self._publish(self.filtered_action)
            self.last_stop_reason = ""
        except Exception as error:  # Safety boundary around native inference.
            self.get_logger().error(
                f"Navigation inference failed: {error}",
                throttle_duration_sec=2.0,
            )
            self._stop("inference error")
            return

        if now - self.last_status_at >= 1.0:
            self.get_logger().info(
                f"goal_dist={planar_distance:.2f}m "
                f"hits={self.last_hit_count}/{NUM_POINTS} "
                f"body_v=[{self.linear_velocity[0]:+.2f}, "
                f"{self.linear_velocity[1]:+.2f}, "
                f"{self.linear_velocity[2]:+.2f}] "
                f"raw=[{raw_action[0]:+.2f}, {raw_action[1]:+.2f}, "
                f"{raw_action[2]:+.2f}] "
                f"cmd=[{self.filtered_action[0]:+.2f}, "
                f"{self.filtered_action[1]:+.2f}, "
                f"{self.filtered_action[2]:+.2f}]"
            )
            self.last_status_at = now

    def stop(self) -> None:
        """Publish a final zero command."""
        if rclpy.ok():
            self._publish(np.zeros(3))


def main(args=None):
    """Run the ROS 2 navigation policy node."""
    rclpy.init(args=args)
    node = None
    try:
        node = RlNavigationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
