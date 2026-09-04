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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    package_share = get_package_share_directory("mid360_preprocessor")
    config_file = os.path.join(
        package_share, "config", "mid360_preprocessor.yaml"
    )

    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    source_frame = LaunchConfiguration("source_frame_override")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic", default_value="/livox/lidar"
            ),
            DeclareLaunchArgument(
                "output_topic", default_value="/navigation/lidar/points"
            ),
            DeclareLaunchArgument(
                "source_frame_override", default_value=""
            ),
            Node(
                package="mid360_preprocessor",
                executable="mid360_preprocessor_node",
                name="mid360_preprocessor",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "input_type": "custom_msg",
                        "input_topic": input_topic,
                        "output_topic": output_topic,
                        "source_frame_override": source_frame,
                    },
                ],
            ),
        ]
    )
