#!/usr/bin/env python3

"""Launch the complete PM01 Gazebo RL navigation stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
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
    """Start Gazebo, walking, lidar preprocessing, LIO and navigation."""
    walking = include(
        "rl_walking",
        "pm01_walking.launch.py",
        {
            "gui": LaunchConfiguration("gui"),
            "world": LaunchConfiguration("world"),
            "gazebo_master_uri": LaunchConfiguration("gazebo_master_uri"),
            "x": LaunchConfiguration("x"),
            "y": LaunchConfiguration("y"),
            "z": LaunchConfiguration("z"),
            "yaw": LaunchConfiguration("yaw"),
            "enable_lidar": "true",
            "lidar_update_rate": LaunchConfiguration("lidar_update_rate"),
            "lidar_horizontal_samples": LaunchConfiguration(
                "lidar_horizontal_samples"
            ),
            "lidar_vertical_samples": LaunchConfiguration(
                "lidar_vertical_samples"
            ),
            "lidar_visualize": LaunchConfiguration("lidar_visualize"),
            "enable_joystick": LaunchConfiguration("enable_joystick"),
            "joy_cmd_vel_topic": LaunchConfiguration(
                "joystick_cmd_vel_topic"
            ),
            "start_delay": LaunchConfiguration("walking_start_delay"),
        },
    )

    pointcloud_preprocessor = include(
        "mid360_preprocessor",
        "pointcloud2.launch.py",
        {
            "input_topic": "/livox/lidar/pointcloud",
            "output_topic": "/navigation/lidar/points",
        },
    )

    # Super-LIO is currently started separately:
    # ros2 launch super_lio Livox_mid360.py rviz:=true
    # localization = include(
    #     "super_lio",
    #     "Livox_mid360.py",
    #     {
    #         "rviz": LaunchConfiguration("rviz"),
    #     },
    # )

    lio_robo_tf = Node(
        package="rl_nav_policy",
        executable="lio_robo_tf",
        name="lio_robo_tf",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "odom_topic": "/lio/robo/odom",
            "parent_frame": "world",
            "child_frame": "base_footprint",
            "odom_body_frame": "LINK_TORSO_YAW",
        }],
    )

    navigation = include(
        "rl_nav_policy",
        "rl_nav_policy.launch.py",
        {
            "policy_path": LaunchConfiguration("policy_path"),
            "cloud_topic": "/navigation/lidar/points",
            "odom_topic": "/lio/robo/odom",
            "goal_topic": "/goal_pose",
            "cmd_vel_topic": "/cmd_vel",
            "joystick_cmd_vel_topic": LaunchConfiguration(
                "joystick_cmd_vel_topic"
            ),
            "joy_topic": "/joy",
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "world",
                default_value="empty.world",
                description=(
                    "Simulator world file: empty.world or ground.world"
                ),
            ),
            # DeclareLaunchArgument(
            #     "rviz",
            #     default_value="true",
            #     description="Start the Super-LIO RViz configuration",
            # ),
            DeclareLaunchArgument(
                "gazebo_master_uri",
                default_value="http://127.0.0.1:11358",
            ),
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.82"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "lidar_update_rate", default_value="10.0"
            ),
            DeclareLaunchArgument(
                "lidar_horizontal_samples", default_value="600"
            ),
            DeclareLaunchArgument(
                "lidar_vertical_samples", default_value="64"
            ),
            DeclareLaunchArgument(
                "lidar_visualize",
                default_value="false",
                description="Show the source Mid360 rays in Gazebo",
            ),
            DeclareLaunchArgument(
                "walking_start_delay",
                default_value="9.0",
                description="Delay walking policy until controllers settle",
            ),
            DeclareLaunchArgument(
                "enable_joystick",
                default_value="true",
                description="Start joystick input and reset-button nodes",
            ),
            DeclareLaunchArgument(
                "joystick_cmd_vel_topic",
                default_value="/cmd_vel_joystick",
                description="Joystick input forwarded in MODE+X mode",
            ),
            DeclareLaunchArgument(
                "navigation_start_delay",
                default_value="10.0",
                description="Delay navigation actor startup",
            ),
            DeclareLaunchArgument(
                "navigation_mode_button", default_value="6"
            ),
            DeclareLaunchArgument(
                "navigation_disable_button",
                default_value="2",
                description="X button used with MODE to disable navigation",
            ),
            DeclareLaunchArgument(
                "navigation_enable_button",
                default_value="3",
                description="Y button used with MODE to enable navigation",
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
            walking,
            pointcloud_preprocessor,
            lio_robo_tf,
            # localization,
            TimerAction(
                period=LaunchConfiguration("navigation_start_delay"),
                actions=[navigation],
            ),
        ]
    )
