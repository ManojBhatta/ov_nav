#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
import xacro

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    sim_package_name = 'robot_simulation'
    use_sim_time = True

    world_path = os.path.join(
        get_package_share_directory(sim_package_name),
        'worlds',
        'living_room_fixed.sdf'
        # 'empty.sdf'
    )
    bridge_config = os.path.join(
        get_package_share_directory(sim_package_name),
        'config',
        'gz_bridge.yaml'
    )

    # Start Gazebo Harmonic / Ignition
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': world_path,
            'on_exit_shutdown': 'true'
        }.items()
    )

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('robot_description'))
    xacro_file = os.path.join(pkg_path,'urdf','robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    # process xacro file to get robot description
    robot_description_config = robot_description_config.toxml()

    # Create a robot_state_publisher node
    params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',

        parameters=[params]
    )

    # Spawn robot from robot_description topic
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            # '-file', '/home/manoj/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger_cam/model.sdf',
            '-name', 'my_bot',
            '-z', '0.1'
        ],
        output='screen'
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p', 
            f'config_file:={bridge_config}',
            # '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        ],
    )
    ros_gz_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/camera/image_raw', '/depth_camera/depth'
            ],
    )
    # Delay spawn slightly so Gazebo is ready
    delayed_spawn = TimerAction(
        period=6.0,
        actions=[spawn_entity]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        delayed_spawn,
        ros_gz_bridge,
        ros_gz_image_bridge
    ])