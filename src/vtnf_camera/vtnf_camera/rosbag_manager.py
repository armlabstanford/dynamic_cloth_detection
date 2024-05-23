import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import signal
import threading

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

def start_rosbag():
    return subprocess.Popen(
        ['ros2', 'bag', 'record', '/RunCamera/force', '/vtnf/depth'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    rclpy.init()
    
    # Start the rosbag recording
    rosbag_process = start_rosbag()

    # Initialize the flag monitor node
    flag_monitor_node = FlagMonitorNode(rosbag_process)

    # Start spinning the node in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(flag_monitor_node,))
    thread.start()

    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        flag_monitor_node.destroy_node()
        rclpy.shutdown()
        rosbag_process.terminate()
        rosbag_process.wait()

if __name__ == '__main__':
    main()
