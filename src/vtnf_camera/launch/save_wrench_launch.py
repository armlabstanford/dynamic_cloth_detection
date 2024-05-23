from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # Declare the launch arguments
    trial_num_arg = DeclareLaunchArgument(
        'trial_num',
        default_value='0',
    )

    bagfile_arg = DeclareLaunchArgument(
        'bagfile',
        default_value='rosbag2',
    )

    trial_num = LaunchConfiguration('trial_num')
    bagfile = LaunchConfiguration('bagfile')

    return LaunchDescription([
        trial_num_arg,
        bagfile_arg,

        Node(
            package='dynamixel_fingers',
            executable='save_wrench_data',
            name='save_wrench_data_node',
            parameters=[{'trial_num': trial_num}],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bagfile],
            output='screen'
        ),
        
        # ExecuteProcess(
        #     cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
        #     output='screen'
        # ),
    ])
