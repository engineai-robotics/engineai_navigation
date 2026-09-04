#!/usr/bin/env python3

"""Real-robot test launch for MID360 preprocessing and LIO TF.

Assumes the Livox driver and Super-LIO are already running:

    ros2 launch livox_ros_driver2 msg_MID360_launch.py
    ros2 launch super_lio Livox_mid360.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start CustomMsg resampling and world -> torso TF."""
    preprocessor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mid360_preprocessor"),
                "launch",
                "custom_msg_hw.launch.py",
            )
        ),
        launch_arguments={
            "input_topic": LaunchConfiguration("input_topic"),
            "output_topic": LaunchConfiguration("output_topic"),
            "source_frame_override": LaunchConfiguration(
                "source_frame_override"
            ),
        }.items(),
    )

    lio_robo_tf = Node(
        package="rl_nav_policy",
        executable="lio_robo_tf",
        name="lio_robo_tf",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "odom_topic": LaunchConfiguration("odom_topic"),
            "parent_frame": LaunchConfiguration("parent_frame"),
            "child_frame": LaunchConfiguration("child_frame"),
            "odom_body_frame": LaunchConfiguration("odom_body_frame"),
        }],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic",
                default_value="/livox/lidar",
                description="Livox CustomMsg topic from the real MID360",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/navigation/lidar/points",
                description="Resampled 16x41 PointCloud2 for navigation",
            ),
            DeclareLaunchArgument(
                "source_frame_override",
                default_value="",
                description="Optional lidar frame override, e.g. livox_frame",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/lio/robo/odom",
            ),
            DeclareLaunchArgument(
                "parent_frame",
                default_value="world",
            ),
            DeclareLaunchArgument(
                "child_frame",
                default_value="LINK_TORSO_YAW",
                description=(
                    "Frame published from Super-LIO odometry. "
                    "Use LINK_TORSO_YAW when the robot URDF TF is not running."
                ),
            ),
            DeclareLaunchArgument(
                "odom_body_frame",
                default_value="LINK_TORSO_YAW",
                description="Body frame of /lio/robo/odom",
            ),
            preprocessor,
            lio_robo_tf,
        ]
    )
