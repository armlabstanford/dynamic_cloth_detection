import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class ShutdownPublisher(Node):
    def __init__(self):
        super().__init__('shutdown_publisher')
        self.publisher_ = self.create_publisher(Bool, 'shutdown_flag', 10)
        self.timer = self.create_timer(5.0, self.timer_callback)  # Publish after 5 seconds

    def timer_callback(self):
        msg = Bool()
        msg.data = True
        self.publisher_.publish(msg)
        self.get_logger().info('Shutdown flag published')
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ShutdownPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
