from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

# def generate_launch_description():

#     # Declare the launch arguments
#     num_layers_arg = DeclareLaunchArgument(
#         'num_layers',
#         default_value='0',
#         description='number of layers of cloth'
#     )
#     trial_num_arg = DeclareLaunchArgument(
#         'trial_num',
#         default_value='0',
#         description='trial number'
#     )
#     # bagfile_arg = DeclareLaunchArgument(
#     #     'bagfile',
#     #     default_value='rosbag2',
#     # )
    
#     num_layers = LaunchConfiguration('num_layers')
#     trial_num = LaunchConfiguration('trial_num')
#     # bagfile = LaunchConfiguration('bagfile')

#     return LaunchDescription([
#         num_layers_arg,
#         trial_num_arg,
#         # bagfile_arg,

#         Node(
#             package='dynamixel_fingers',
#             executable='save_wrench_data',
#             name='save_wrench_data_node',
#             parameters=[{'trial_num': trial_num}],
#             output='screen'
#         ),

#         ExecuteProcess(
#             cmd=['ros2', 'bag', 'play', f'/bag_files/{num_layers}_layer/bag_{trial_num}'],
#             output='screen'
#         ),
        
#         # ExecuteProcess(
#         #     cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
#         #     output='screen'
#         # ),
#     ])

def launch_setup(context, *args, **kwargs):
    num_layers = LaunchConfiguration('num_layers').perform(context)
    trial_num = LaunchConfiguration('trial_num').perform(context)

    save_wrench_node = Node(
        package='dynamixel_fingers',
        executable='save_wrench_data',
        name='save_wrench_data_node',
        parameters=[
            {'num_layers': num_layers}, 
            {'trial_num': trial_num},
        ],
        output='screen'
    )

    play_bag_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', f'/home/armlab/Documents/soft_manipulation/bag_files/' + num_layers + '_layer/bag_' + trial_num],
        output='screen'
    )

    return [save_wrench_node, 
            play_bag_process]

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            'num_layers',
            default_value='0',
            description='number of layers of cloth'
        ),
        DeclareLaunchArgument(
            'trial_num',
            default_value='0',
            description='trial number'
        ),
        OpaqueFunction(function=launch_setup)    
    ])
