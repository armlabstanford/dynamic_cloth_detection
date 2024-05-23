from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dynamixel_fingers',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen',
            parameters=[
                {'camera_index': 2},  # Replace with desired camera index
                {'filename': 'src/dynamixel_fingers/tests/testingrun'}
            ]
        )
    ])
