# robot_description
ros2 package containing urdf and supporting files for a four_wheeled skid-steering (diff drive) robot.  

Download zip or clone the repo inside the src folder of your workspace.  
build the package using  

`colcon build --packages-select robot_description`  

and then source your workspace and launch the urdf using:  

`ros2 launch robot_description display.launch.py`
