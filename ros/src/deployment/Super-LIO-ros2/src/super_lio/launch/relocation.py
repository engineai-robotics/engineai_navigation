import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context, *args, **kwargs):
    pkg_super_lio = get_package_share_directory('super_lio')
    config_yaml = os.path.join(pkg_super_lio, 'config', 'livox_360.yaml')
    rviz_config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'rviz', 'myconfig.rviz'
    )

    map_path = os.path.expanduser(LaunchConfiguration('map').perform(context))
    map_dir = os.path.dirname(map_path)
    map_name = os.path.basename(map_path)
    if not map_dir:
        map_dir = 'map'
    if not map_name:
        map_name = 'map.pcd'

    super_lio_node = Node(
        package='super_lio',
        executable='relocation_node',
        name='relocation_node',
        output='screen',
        parameters=[
            config_yaml,
            {
                'lio.map.save_map': ParameterValue(
                    LaunchConfiguration('save_map'),
                    value_type=bool,
                ),
                'lio.map.save_map_dir': map_dir,
                'lio.map.map_name': map_name,
            },
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='super_lio',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return [super_lio_node, rviz2_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Whether to start RVIZ2'
        ),
        DeclareLaunchArgument(
            'save_map',
            default_value='false',
            description='Cache scans while running and write map.pcd on Ctrl-C'
        ),
        DeclareLaunchArgument(
            'map',
            default_value='map/map.pcd',
            description=(
                'PCD map file. Relative paths are from the Super-LIO source '
                'directory; absolute paths are used as-is.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
