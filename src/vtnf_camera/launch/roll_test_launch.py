# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import ExecuteProcess, OpaqueFunction
# import os
# import subprocess


# def launch_setup(context, *args, **kwargs):
#     # Start the rosbag recording process
#     rosbag_process = subprocess.Popen(
#         ['ros2', 'bag', 'record', '/RunCamera/force'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     # Create the flag monitor node with the rosbag process handle
#     flag_monitor_node = Node(
#         package='vtnf_camera',
#         executable='flag_monitor_node',
#         name='flag_monitor_node',
#         output='screen',
#         parameters=[{'rosbag_process': rosbag_process}]
#     )

#     camera_node_1 = Node(
#         package='vtnf_camera',
#         executable='dtv2_cam_pub',
#         name='dtv2_node',
#         parameters=[{'camera_id': '1'}],
#         output='screen'
#     ),

#     camera_node_2 = Node(
#         package='vtnf_camera',
#         executable='dtv2_cam_pub',
#         name='dtv2_node',
#         parameters=[{'camera_id': '3'}],
#         output='screen'
#     ),

#     rubbing_motion_node = Node(
#         package='dynamixel_fingers',
#         executable='rubbing_motion',
#         name='dynamixel_finger_node',
#         output='screen'
#     ),
        
#     force_bag = ExecuteProcess(
#         cmd=['ros2', 'bag', 'record', '/RunCamera/force'],
#         output='screen'
#     ),


#     bridge_process = ExecuteProcess(
#         cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
#         # output='screen'
#     ),

#     return [
        
#         flag_monitor_node, camera_node_1, camera_node_2, rubbing_motion_node, bridge_process, force_bag
#     ]

# def generate_launch_description():
#     return LaunchDescription([
#         OpaqueFunction(function=launch_setup)
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # Declare the launch arguments
    id1_arg = DeclareLaunchArgument(
        'id1',
        default_value='0',
        description='camera id for sensor 1 (calibrated)'
    )
    id2_arg = DeclareLaunchArgument(
        'id2',
        default_value='2',
        description='camera id for sensor 2'
    )
    num_layers_arg = DeclareLaunchArgument(
        'num_layers',
        default_value='0',
        description='number of layers of cloth'
    )
    trial_num_arg = DeclareLaunchArgument(
        'trial_num',
        default_value='0',
        description='trial number'
    )

    id1 = LaunchConfiguration('id1')
    id2 = LaunchConfiguration('id2')
    num_layers = LaunchConfiguration('num_layers')
    trial_num = LaunchConfiguration('trial_num')

    return LaunchDescription([
        id1_arg,
        id2_arg,
        num_layers_arg,
        trial_num_arg,

        Node(
            package='vtnf_camera',
            executable='dtv2_cam_pub',
            name='dtv2_node',
            parameters=[
                {'camera_id': id1}, 
                {'sensornum': '1'},
                {'num_layers': num_layers}, 
                {'trial_num': trial_num},
            ],
            output='screen'
        ),
        # Node(
        #     package='vtnf_camera',
        #     executable='dtv2_cam_pub',
        #     name='dtv2_node',
        #     parameters=[{'camera_id': id2}, {'sensornum': '3'}],
        #     output='screen'
        # ),
        Node(
            package='dynamixel_fingers',
            executable='rubbing_motion',
            name='dynamixel_finger_node',
            output='screen'
        ),
        # ExecuteProcess(
        #     cmd=['python3', '/home/armlab/Documents/soft_manipulation/src/vtnf_camera/vtnf_camera/rosbag_manager.py'],
        #     output='screen'
        # ),
        ExecuteProcess(
            cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
            output='screen'
        ),
    ])
