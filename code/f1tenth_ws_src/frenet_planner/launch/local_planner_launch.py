"""Local planner, selectable: the Frenet sampling planner (default) or the
team's RRT node. Both publish /drive themselves through PurePursuitController.

    ros2 launch frenet_planner local_planner_launch.py                    # Frenet, sim
    ros2 launch frenet_planner local_planner_launch.py planner:=rrt       # RRT, sim
    ros2 launch frenet_planner local_planner_launch.py use_sim:=false     # Frenet, car (config/car_overrides.yaml)
"""
import os

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

PLANNERS = ('frenet', 'rrt')


def launch_setup(context):
    planner = context.launch_configurations['planner']
    use_sim = context.launch_configurations['use_sim'].lower() == 'true'
    params_file = context.launch_configurations['params_file']
    rviz = context.launch_configurations['rviz'].lower() == 'true'
    if planner not in PLANNERS:
        raise RuntimeError(f'planner:={planner}; expected one of {PLANNERS}')
    share = get_package_share_directory('frenet_planner')

    nodes = []
    if planner == 'frenet':
        if use_sim:
            layers = [params_file, os.path.join(share, 'config', 'sim_overrides.yaml')]
        else:
            layers = [params_file, os.path.join(share, 'config', 'car_overrides.yaml')]
        nodes.append(Node(package='frenet_planner', executable='frenet_planner_node.py', name='frenet_planner',
                          output='screen', parameters=layers))
    else:
        # rrt_node.py imports pure_pursuit_node, which is installed in the
        # pure_pursuit package's lib directory, not on PYTHONPATH
        pp_lib = os.path.join(get_package_prefix('pure_pursuit'), 'lib', 'pure_pursuit')
        pythonpath = os.pathsep.join(filter(None, [pp_lib, os.environ.get('PYTHONPATH', '')]))
        with open(params_file) as f:
            raceline = yaml.safe_load(f)['frenet_planner']['ros__parameters']['raceline']['file']
        nodes.append(Node(package='motion_planning', executable='rrt_node.py', name='rrt_node', output='screen',
                          parameters=[{'use_sim': use_sim, 'waypoints_file': raceline}],
                          additional_env={'PYTHONPATH': pythonpath}))
    if rviz:
        nodes.append(Node(package='rviz2', executable='rviz2', name='frenet_rviz',
                          arguments=['-d', os.path.join(share, 'rviz', 'frenet_planner.rviz')]))
    return nodes


def generate_launch_description():
    share = get_package_share_directory('frenet_planner')
    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value='frenet', description='frenet or rrt'),
        DeclareLaunchArgument('use_sim', default_value='true',
                              description='true: sim odometry and config/sim_overrides.yaml; '
                                          'false: particle filter on the car'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'frenet_planner.yaml'),
                              description='Frenet planner parameters (its raceline file is also given to RRT)'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Open a second RViz with the planner displays'),
        OpaqueFunction(function=launch_setup),
    ])
