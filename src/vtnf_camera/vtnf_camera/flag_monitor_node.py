import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import signal

class FlagMonitorNode(Node):
    def __init__(self, rosbag_process):
        super().__init__('flag_monitor_node')
        self.rosbag_process = rosbag_process
        self.subscription = self.create_subscription(
            Bool,
            'shutdown_flag',
            self.flag_callback,
            10)
        self.subscription  # prevent unused variable warning

    def flag_callback(self, msg):
        if msg.data:
            self.get_logger().info('Shutdown flag received, stopping rosbag...')
            self.rosbag_process.send_signal(signal.SIGINT)
            self.get_logger().info('Rosbag stopped.')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # This assumes the rosbag process is already started and passed to the node
    rosbag_process = None  # This should be replaced with the actual rosbag process handle
    
    node = FlagMonitorNode(rosbag_process)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
