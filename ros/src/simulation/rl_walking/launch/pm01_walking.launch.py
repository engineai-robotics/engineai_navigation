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

"""Launch Gazebo and the PM01 C++ MNN walking policy."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_dir = get_package_share_directory('rl_walking')
    simulator_dir = get_package_share_directory('simulator')

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                simulator_dir, 'launch', 'pm01_gazebo.launch.py'
            )
        ),
        launch_arguments={
            'gui': LaunchConfiguration('gui'),
            'world': LaunchConfiguration('world'),
            'gazebo_master_uri': LaunchConfiguration('gazebo_master_uri'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'yaw': LaunchConfiguration('yaw'),
            'enable_sim_sensors': 'true',
            'enable_lidar': LaunchConfiguration('enable_lidar'),
            'lidar_update_rate': LaunchConfiguration('lidar_update_rate'),
            'lidar_horizontal_samples': LaunchConfiguration(
                'lidar_horizontal_samples'
            ),
            'lidar_vertical_samples': LaunchConfiguration(
                'lidar_vertical_samples'
            ),
            'lidar_visualize': LaunchConfiguration('lidar_visualize'),
            'policy_imu_topic': LaunchConfiguration('policy_imu_topic'),
            'controller_reset_service': '/pm01_walking_policy/reset',
        }.items(),
    )

    walking_policy = Node(
        package='rl_walking',
        executable='pm01_walking_policy_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'config_file': os.path.join(
                package_dir, 'config', 'rl_walking.yaml'
            ),
            'model_path': os.path.join(
                package_dir,
                'config',
                'policy',
                '251009_221907_41000.mnn',
            ),
            'command_timeout': ParameterValue(
                LaunchConfiguration('command_timeout'), value_type=float
            ),
            'sensor_timeout': ParameterValue(
                LaunchConfiguration('sensor_timeout'), value_type=float
            ),
            'max_tilt': ParameterValue(
                LaunchConfiguration('max_tilt'), value_type=float
            ),
            'simulation_action_clip': ParameterValue(
                LaunchConfiguration('simulation_action_clip'),
                value_type=float,
            ),
            'max_target_step': ParameterValue(
                LaunchConfiguration('max_target_step'), value_type=float
            ),
            'reset_settle_time': ParameterValue(
                LaunchConfiguration('reset_settle_time'), value_type=float
            ),
            'control_arms': ParameterValue(
                LaunchConfiguration('control_arms'), value_type=bool
            ),
            'imu_topic': LaunchConfiguration('policy_imu_topic'),
        }],
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        condition=IfCondition(LaunchConfiguration('enable_joystick')),
        output='screen',
        parameters=[{
            'device_id': ParameterValue(
                LaunchConfiguration('joy_device_id'), value_type=int
            ),
            'deadzone': ParameterValue(
                LaunchConfiguration('joy_deadzone'), value_type=float
            ),
            'autorepeat_rate': ParameterValue(
                LaunchConfiguration('joy_autorepeat_rate'), value_type=float
            ),
        }],
    )

    joy_velocity = Node(
        package='rl_walking',
        executable='joy_velocity_node.py',
        condition=IfCondition(LaunchConfiguration('enable_joystick')),
        output='screen',
        parameters=[{
            'joy_topic': '/joy',
            'cmd_vel_topic': LaunchConfiguration('joy_cmd_vel_topic'),
            'linear_axis': ParameterValue(
                LaunchConfiguration('joy_linear_axis'), value_type=int
            ),
            'lateral_axis': ParameterValue(
                LaunchConfiguration('joy_lateral_axis'), value_type=int
            ),
            'angular_axis': ParameterValue(
                LaunchConfiguration('joy_angular_axis'), value_type=int
            ),
            'linear_scale': ParameterValue(
                LaunchConfiguration('joy_linear_scale'), value_type=float
            ),
            'lateral_scale': ParameterValue(
                LaunchConfiguration('joy_lateral_scale'), value_type=float
            ),
            'angular_scale': ParameterValue(
                LaunchConfiguration('joy_angular_scale'), value_type=float
            ),
            'timeout': ParameterValue(
                LaunchConfiguration('joy_timeout'), value_type=float
            ),
            'reset_button': ParameterValue(
                LaunchConfiguration('joy_reset_button'), value_type=int
            ),
            'reset_service': '/pm01/reset_stand',
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value='empty.world',
            description=(
                'Simulator world file: empty.world or ground.world'
            ),
        ),
        DeclareLaunchArgument(
            'gazebo_master_uri',
            default_value='http://127.0.0.1:11358',
        ),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.82'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('enable_lidar', default_value='true'),
        DeclareLaunchArgument('lidar_update_rate', default_value='10.0'),
        DeclareLaunchArgument(
            'lidar_horizontal_samples', default_value='600'
        ),
        DeclareLaunchArgument('lidar_vertical_samples', default_value='64'),
        DeclareLaunchArgument(
            'lidar_visualize',
            default_value='false',
            description='Show Mid360 rays in the Gazebo client',
        ),
        DeclareLaunchArgument('policy_imu_topic', default_value='/pm01/imu'),
        DeclareLaunchArgument(
            'start_delay',
            default_value='9.0',
            description='Wait for simulator controllers and initial pose',
        ),
        DeclareLaunchArgument(
            'command_timeout',
            default_value='0.5',
            description='Stop command when cmd_vel becomes stale',
        ),
        DeclareLaunchArgument(
            'sensor_timeout',
            default_value='0.25',
            description='Maximum joint-state or IMU age',
        ),
        DeclareLaunchArgument(
            'max_tilt',
            default_value='1.0',
            description='Halt policy above this base tilt in radians',
        ),
        DeclareLaunchArgument(
            'simulation_action_clip',
            default_value='0.0',
            description='Optional raw-action safety clip; 0 uses SDK limit',
        ),
        DeclareLaunchArgument(
            'max_target_step',
            default_value='0.0',
            description='Optional target slew limit in rad/cycle; 0 disables',
        ),
        DeclareLaunchArgument(
            'reset_settle_time',
            default_value='0.0',
            description='Optional nominal-pose hold after reset',
        ),
        DeclareLaunchArgument(
            'control_arms',
            default_value='true',
            description='Apply all 22 SDK policy actions including the arms',
        ),
        DeclareLaunchArgument('enable_joystick', default_value='true'),
        DeclareLaunchArgument(
            'joy_cmd_vel_topic',
            default_value='/cmd_vel',
            description='Velocity output topic for joystick commands',
        ),
        DeclareLaunchArgument('joy_device_id', default_value='0'),
        DeclareLaunchArgument('joy_deadzone', default_value='0.08'),
        DeclareLaunchArgument('joy_autorepeat_rate', default_value='50.0'),
        DeclareLaunchArgument('joy_linear_axis', default_value='7'),
        DeclareLaunchArgument('joy_lateral_axis', default_value='0'),
        DeclareLaunchArgument('joy_angular_axis', default_value='3'),
        DeclareLaunchArgument('joy_linear_scale', default_value='1.0'),
        DeclareLaunchArgument('joy_lateral_scale', default_value='1.0'),
        DeclareLaunchArgument('joy_angular_scale', default_value='1.0'),
        DeclareLaunchArgument('joy_timeout', default_value='0.5'),
        DeclareLaunchArgument(
            'joy_reset_button',
            default_value='7',
            description='Joy button index for PM01 Gazebo stand reset',
        ),
        simulator,
        joy_node,
        joy_velocity,
        TimerAction(
            period=LaunchConfiguration('start_delay'),
            actions=[walking_policy],
        ),
    ])
