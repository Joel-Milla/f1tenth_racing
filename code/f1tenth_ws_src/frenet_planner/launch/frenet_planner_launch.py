import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('frenet_planner')
    params_file = LaunchConfiguration('params_file')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'frenet_planner.yaml'),
            description='Planner parameters'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open a second RViz with the planner displays added '
                        '(the sim launch already opens one without them)'),

        Node(
            package='frenet_planner',
            executable='frenet_planner_node.py',
            name='frenet_planner',
            output='screen',
            parameters=[params_file]),
        Node(
            package='rviz2',
            executable='rviz2',
            name='frenet_rviz',
            arguments=['-d', os.path.join(share, 'rviz', 'frenet_planner.rviz')],
            condition=IfCondition(rviz)),
    ])
