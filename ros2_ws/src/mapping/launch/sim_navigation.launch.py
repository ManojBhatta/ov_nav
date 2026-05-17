from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, TimerAction
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("robot_simulation"),
                    "launch",
                    "launch_gazebo.launch.py",
                )
            ]
        ),
    )

    map_to_odom = TimerAction(
        period=7.0,  # Delay of 7 seconds
        actions=[Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='static_tf_map_to_odom',
            arguments=['0.0', '0', '0.0', '0.0', '0.0', '0.0', 'map', 'odom'],
            output='screen'
        )]
    )

    # Navigation stack after TF is published
    nav = TimerAction(
        period=10.0,  # Delay of 10 seconds
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
            ]),
            launch_arguments={
                'params_file': os.path.join(get_package_share_directory('mapping'), 'config', 'nav2_params_changed.yaml'),
                'use_sim_time': 'true',
                'autostart': 'true',
                'log_level': 'info',
                'map': get_package_share_directory('mapping') + '/maps/first_map.yaml'
            }.items()
        )]
    )

    # RViz after Nav2
    rviz_nav = TimerAction(
        period=12.0,  # Delay of 12 seconds
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', get_package_share_directory('nav2_bringup') + '/rviz/nav2_default_view.rviz']
        )]
    )

    return LaunchDescription([
        gazebo,
        map_to_odom,
        nav,
        rviz_nav
    ])