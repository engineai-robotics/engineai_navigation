#!/usr/bin/env python3
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

"""Convert sensor_msgs/Joy axes into PM01 velocity commands."""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty


class JoyVelocityNode(Node):
    """Publish smoothed cmd_vel commands from a joystick."""

    def __init__(self):
        super().__init__('pm01_joy_velocity')
        self.linear_axis = int(
            self.declare_parameter('linear_axis', 7).value)
        self.lateral_axis = int(
            self.declare_parameter('lateral_axis', 0).value)
        self.angular_axis = int(
            self.declare_parameter('angular_axis', 3).value)
        self.linear_scale = float(
            self.declare_parameter('linear_scale', 0.7).value)
        self.lateral_scale = float(
            self.declare_parameter('lateral_scale', 1.0).value)
        self.angular_scale = float(
            self.declare_parameter('angular_scale', 1.0).value)
        self.linear_direction = float(
            self.declare_parameter('linear_direction', 1.0).value)
        self.lateral_direction = float(
            self.declare_parameter('lateral_direction', 1.0).value)
        self.angular_direction = float(
            self.declare_parameter('angular_direction', 1.0).value)
        self.deadzone = min(max(float(
            self.declare_parameter('deadzone', 0.2).value
        ), 0.0), 0.99)
        self.timeout = max(float(
            self.declare_parameter('timeout', 0.5).value
        ), 0.0)
        self.publish_rate = max(float(
            self.declare_parameter('publish_rate', 50.0).value
        ), 1.0)
        self.smoothing_weight = min(max(float(
            self.declare_parameter('smoothing_weight', 0.25).value
        ), 0.0), 1.0)
        self.reset_button = int(
            self.declare_parameter('reset_button', 7).value)
        joy_topic = self.declare_parameter('joy_topic', '/joy').value
        cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', '/cmd_vel').value
        reset_service = self.declare_parameter(
            'reset_service', '/pm01/reset_stand').value

        self.target_linear = 0.0
        self.target_lateral = 0.0
        self.target_angular = 0.0
        self.filtered_linear = 0.0
        self.filtered_lateral = 0.0
        self.filtered_angular = 0.0
        self.last_joy_stamp = None
        self.reset_button_pressed = False
        self.reset_future = None
        self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.reset_client = self.create_client(Empty, reset_service)
        self.subscription = self.create_subscription(
            Joy, joy_topic, self.on_joy, qos_profile_sensor_data
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_command)
        self.get_logger().info(
            f'Joy mapping: axis {self.linear_axis} -> forward/backward, '
            f'axis {self.lateral_axis} -> left/right, '
            f'axis {self.angular_axis} -> yaw, button {self.reset_button} -> '
            f'{reset_service}; publishing {cmd_vel_topic}')

    def read_axis(self, axes, index):
        """Return a finite, deadzone-adjusted axis value."""
        if index < 0 or index >= len(axes):
            self.get_logger().warning(
                f'Joy message has {len(axes)} axes; axis {index} is missing',
                throttle_duration_sec=2.0,
            )
            return 0.0
        value = float(axes[index])
        if not math.isfinite(value) or abs(value) <= self.deadzone:
            return 0.0
        magnitude = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return math.copysign(min(magnitude, 1.0), value)

    def on_joy(self, message):
        reset_pressed = (
            0 <= self.reset_button < len(message.buttons)
            and message.buttons[self.reset_button] != 0
        )
        if reset_pressed and not self.reset_button_pressed:
            self.request_reset()
        self.reset_button_pressed = reset_pressed

        linear = self.read_axis(message.axes, self.linear_axis)
        lateral = self.read_axis(message.axes, self.lateral_axis)
        angular = self.read_axis(message.axes, self.angular_axis)
        self.target_linear = max(min(
            linear * self.linear_scale * self.linear_direction, 1.0
        ), -1.0)
        self.target_lateral = max(min(
            lateral * self.lateral_scale * self.lateral_direction, 1.0
        ), -1.0)
        self.target_angular = max(min(
            angular * self.angular_scale * self.angular_direction, 1.0
        ), -1.0)
        self.last_joy_stamp = self.get_clock().now()

    def request_reset(self):
        """Request one Gazebo stand reset on the button rising edge."""
        if self.reset_future is not None and not self.reset_future.done():
            self.get_logger().warning('A PM01 reset request is already active')
            return
        if not self.reset_client.service_is_ready():
            self.get_logger().warning(
                'PM01 reset service is unavailable',
                throttle_duration_sec=2.0,
            )
            return

        self.target_linear = 0.0
        self.target_lateral = 0.0
        self.target_angular = 0.0
        self.filtered_linear = 0.0
        self.filtered_lateral = 0.0
        self.filtered_angular = 0.0
        self.publisher.publish(Twist())
        self.reset_future = self.reset_client.call_async(Empty.Request())
        self.reset_future.add_done_callback(self.on_reset_complete)
        self.get_logger().info('Start button pressed; requesting PM01 reset')

    def on_reset_complete(self, future):
        """Report completion of the asynchronous reset request."""
        try:
            future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(f'PM01 reset failed: {error}')
        else:
            self.get_logger().info('PM01 Gazebo reset completed')
        self.reset_future = None

    def publish_command(self):
        stale = self.last_joy_stamp is None
        if self.last_joy_stamp is not None:
            age = (
                self.get_clock().now() - self.last_joy_stamp
            ).nanoseconds * 1.0e-9
            stale = age > self.timeout
        if stale:
            self.target_linear = 0.0
            self.target_lateral = 0.0
            self.target_angular = 0.0
            self.filtered_linear = 0.0
            self.filtered_lateral = 0.0
            self.filtered_angular = 0.0
        else:
            self.filtered_linear += self.smoothing_weight * (
                self.target_linear - self.filtered_linear
            )
            self.filtered_lateral += self.smoothing_weight * (
                self.target_lateral - self.filtered_lateral
            )
            self.filtered_angular += self.smoothing_weight * (
                self.target_angular - self.filtered_angular
            )

        message = Twist()
        message.linear.x = self.filtered_linear
        message.linear.y = self.filtered_lateral
        message.angular.z = self.filtered_angular
        self.publisher.publish(message)

    def stop(self):
        """Publish an explicit zero command before shutting down."""
        self.publisher.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = JoyVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
