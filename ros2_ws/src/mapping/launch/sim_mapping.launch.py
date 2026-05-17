from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

import os


def generate_launch_description():
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

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("mapping"), "rviz", "rviz_mapping.rviz"
            ),
            "--ros-args",
            "-p",
            "use_sim_time:=True",
        ],
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("slam_toolbox"),
                    "launch",
                    "online_async_launch.py",
                )
            ]
        ),
        launch_arguments={
            "scan_topic": "/scan",
            "odom_topic": "/odom",
            "odom_frame": "odom",
            "base_frame": "base_link",
            "map_frame": "map",
            "use_sim_time": "true",
        }.items(),
    )

    return LaunchDescription([
            gazebo,
            # odom_to_camera,
            # body_to_base,
            rviz_node,
            slam
        ])
