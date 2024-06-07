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
    id0_arg = DeclareLaunchArgument(
        'id0',
        default_value='2',
        description='camera id for sensor 0'
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
    id0 = LaunchConfiguration('id0')
    num_layers = LaunchConfiguration('num_layers')
    trial_num = LaunchConfiguration('trial_num')

    return LaunchDescription([
        id1_arg,
        id0_arg,
        Node(
            package='vtnf_camera',
            executable='dtv2_cam_pub',
            name='dtv2_node',
            parameters=[{'camera_id': id1}, 
                        {'sensornum': '1'},
                        {'num_layers': num_layers}, 
                        {'trial_num': trial_num},],
            output='screen'
        ),
        # Node(
        #     package='vtnf_camera',
        #     executable='dtv2_cam_pub',
        #     name='dtv2_node',
        #     parameters=[{'camera_id': id0}, {'sensornum': '0'}],
        #     output='screen'
        # ),
        
        #### ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=9090
        ExecuteProcess(
            cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
            # output='screen'
        )
    ])