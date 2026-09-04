#!/usr/bin/env python3

"""Launch the PM01 reinforcement-learning navigation policy."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    """Create the configurable RL navigation launch description."""
    share = get_package_share_directory("rl_nav_policy")
    config = os.path.join(share, "config", "rl_nav_policy.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_path", default_value=""),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/navigation/lidar/points",
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/lio/robo/odom"
            ),
            DeclareLaunchArgument(
                "goal_topic", default_value="/goal_pose"
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic", default_value="/cmd_vel"
            ),
            DeclareLaunchArgument(
                "joystick_cmd_vel_topic",
                default_value="/cmd_vel_joystick",
            ),
            DeclareLaunchArgument(
                "gamepad_type", default_value="gamepad_keys"
            ),
            DeclareLaunchArgument(
                "gamepad_topic",
                default_value="/hardware/gamepad_keys",
            ),
            DeclareLaunchArgument("mode_button", default_value="6"),
            DeclareLaunchArgument("disable_button", default_value="4"),
            DeclareLaunchArgument("enable_button", default_value="5"),
            DeclareLaunchArgument(
                "initial_policy_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "initial_goal_enabled", default_value="false"
            ),
            DeclareLaunchArgument("initial_goal_x", default_value="0.0"),
            DeclareLaunchArgument("initial_goal_y", default_value="0.0"),
            DeclareLaunchArgument("initial_goal_z", default_value="0.5"),
            Node(
                package="rl_nav_policy",
                executable="rl_nav_policy_node",
                name="rl_nav_policy",
                output="screen",
                parameters=[
                    config,
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"),
                            value_type=bool,
                        ),
                        "policy_path": LaunchConfiguration("policy_path"),
                        "cloud_topic": LaunchConfiguration("cloud_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "goal_topic": LaunchConfiguration("goal_topic"),
                        "cmd_vel_topic": LaunchConfiguration(
                            "cmd_vel_topic"
                        ),
                        "joystick_cmd_vel_topic": LaunchConfiguration(
                            "joystick_cmd_vel_topic"
                        ),
                        "gamepad_type": LaunchConfiguration("gamepad_type"),
                        "gamepad_topic": LaunchConfiguration(
                            "gamepad_topic"
                        ),
                        "mode_button": ParameterValue(
                            LaunchConfiguration("mode_button"),
                            value_type=int,
                        ),
                        "disable_button": ParameterValue(
                            LaunchConfiguration("disable_button"),
                            value_type=int,
                        ),
                        "enable_button": ParameterValue(
                            LaunchConfiguration("enable_button"),
                            value_type=int,
                        ),
                        "initial_policy_enabled": ParameterValue(
                            LaunchConfiguration("initial_policy_enabled"),
                            value_type=bool,
                        ),
                        "initial_goal_enabled": ParameterValue(
                            LaunchConfiguration("initial_goal_enabled"),
                            value_type=bool,
                        ),
                        "initial_goal_x": ParameterValue(
                            LaunchConfiguration("initial_goal_x"),
                            value_type=float,
                        ),
                        "initial_goal_y": ParameterValue(
                            LaunchConfiguration("initial_goal_y"),
                            value_type=float,
                        ),
                        "initial_goal_z": ParameterValue(
                            LaunchConfiguration("initial_goal_z"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
