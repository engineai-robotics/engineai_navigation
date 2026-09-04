#!/usr/bin/env python3

"""Real-robot RL navigation launch.

Assumes the Livox driver and Super-LIO are already running:

    ros2 launch livox_ros_driver2 msg_MID360_launch.py
    ros2 launch super_lio Livox_mid360.py

Stop pm01_cloud_tf_test.launch.py first; this launch starts the same
preprocessor and world -> LINK_TORSO_YAW TF, then the navigation policy.
RViz loads rviz/myconfig.rviz unless rviz:=false.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(package, launch_file, arguments):
    """Create an include action for a launch file in a package."""
    package_share = get_package_share_directory(package)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", launch_file)
        ),
        launch_arguments=arguments.items(),
    )


def generate_launch_description():
    """Start CustomMsg resampling, LIO TF, and the navigation policy."""
    cloud_tf = include(
        "rl_nav_policy",
        "pm01_cloud_tf_test.launch.py",
        {
            "input_topic": LaunchConfiguration("input_topic"),
            "output_topic": LaunchConfiguration("cloud_topic"),
            "source_frame_override": LaunchConfiguration(
                "source_frame_override"
            ),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "parent_frame": LaunchConfiguration("parent_frame"),
            "child_frame": LaunchConfiguration("child_frame"),
            "odom_body_frame": LaunchConfiguration("odom_body_frame"),
        },
    )

    range_image = Node(
        package="rl_nav_policy",
        executable="lidar_range_image",
        name="lidar_range_image",
        output="screen",
        parameters=[{
            "cloud_topic": LaunchConfiguration("cloud_topic"),
            "range_image_topic": LaunchConfiguration("range_image_topic"),
            "range_image_viz_topic": LaunchConfiguration(
                "range_image_viz_topic"
            ),
        }],
    )

    navigation = include(
        "rl_nav_policy",
        "rl_nav_policy.launch.py",
        {
            "use_sim_time": "false",
            "policy_path": LaunchConfiguration("policy_path"),
            "cloud_topic": LaunchConfiguration("cloud_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "goal_topic": LaunchConfiguration("goal_topic"),
            "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
            "joystick_cmd_vel_topic": LaunchConfiguration(
                "joystick_cmd_vel_topic"
            ),
            "gamepad_type": "gamepad_keys",
            "gamepad_topic": LaunchConfiguration("gamepad_topic"),
            "mode_button": LaunchConfiguration("navigation_mode_button"),
            "disable_button": LaunchConfiguration(
                "navigation_disable_button"
            ),
            "enable_button": LaunchConfiguration(
                "navigation_enable_button"
            ),
            "initial_policy_enabled": LaunchConfiguration(
                "initial_policy_enabled"
            ),
            "initial_goal_enabled": LaunchConfiguration(
                "initial_goal_enabled"
            ),
            "initial_goal_x": LaunchConfiguration("initial_goal_x"),
            "initial_goal_y": LaunchConfiguration("initial_goal_y"),
            "initial_goal_z": LaunchConfiguration("initial_goal_z"),
        },
    )

    rviz_config = os.path.join(
        get_package_share_directory("rl_nav_policy"),
        "rviz",
        "myconfig.rviz",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rl_nav_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": False}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic",
                default_value="/livox/lidar",
                description="Livox CustomMsg topic from the real MID360",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
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
            DeclareLaunchArgument(
                "goal_topic",
                default_value="/goal_pose",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
            ),
            DeclareLaunchArgument(
                "joystick_cmd_vel_topic",
                default_value="/cmd_vel_joystick",
                description="Joystick input forwarded in MODE+X mode",
            ),
            DeclareLaunchArgument(
                "gamepad_topic",
                default_value="/hardware/gamepad_keys",
                description="Hardware GamepadKeys topic",
            ),
            DeclareLaunchArgument(
                "navigation_mode_button",
                default_value="6",
                description="GamepadKeys BACK used as MODE",
            ),
            DeclareLaunchArgument(
                "navigation_disable_button",
                default_value="4",
                description="GamepadKeys X used with BACK to disable navigation",
            ),
            DeclareLaunchArgument(
                "navigation_enable_button",
                default_value="5",
                description="GamepadKeys Y used with BACK to enable navigation",
            ),
            DeclareLaunchArgument(
                "initial_policy_enabled",
                default_value="false",
                description="Publish policy velocity before MODE+Y is pressed",
            ),
            DeclareLaunchArgument("policy_path", default_value=""),
            DeclareLaunchArgument(
                "initial_goal_enabled", default_value="false"
            ),
            DeclareLaunchArgument("initial_goal_x", default_value="0.0"),
            DeclareLaunchArgument("initial_goal_y", default_value="0.0"),
            DeclareLaunchArgument("initial_goal_z", default_value="0.5"),
            DeclareLaunchArgument(
                "range_image_topic",
                default_value="/navigation/lidar/range_image",
                description="32FC1 metric range image in meters",
            ),
            DeclareLaunchArgument(
                "range_image_viz_topic",
                default_value="/navigation/lidar/range_image_viz",
                description="BGR8 JET-colored range image for RViz",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz with rviz/myconfig.rviz",
            ),
            cloud_tf,
            range_image,
            navigation,
            rviz,
        ]
    )
