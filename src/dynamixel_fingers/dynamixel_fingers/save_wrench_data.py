#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import csv
import os

class WrenchSaver(Node):
    def __init__(self):
        super().__init__('wrench_saver')
        
        # Declare parameters for CSV saving
        self.declare_parameter('num_layers', '0')  # default value
        self.declare_parameter('trial_num', '0')  # default value
        
        # Get parameters
        self.num_layers = self.get_parameter('num_layers').get_parameter_value().string_value
        self.trial_num = self.get_parameter('trial_num').get_parameter_value().string_value
        self.csv_filename = "/home/armlab/Documents/soft_manipulation/wrench_data/" + self.num_layers + "_layer/wrench_data_" + self.trial_num + ".csv"
        
        # Create subscriber to WrenchStamped topic
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/RunCamera/force',
            self.wrench_callback,
            10
        )
        
        # Open the CSV file and set up the writer
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write the header row
        self.csv_writer.writerow(['timestamp', 'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z'])
        
        self.get_logger().info(f'Wrench saver node initialized, saving to {self.csv_filename}')
        
        # Register shutdown hook
        # self.on_shutdown(self.on_shutdown)

    def wrench_callback(self, msg):
        # Extract data from WrenchStamped message
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        force_x = msg.wrench.force.x
        force_y = msg.wrench.force.y
        force_z = msg.wrench.force.z
        torque_x = msg.wrench.torque.x
        torque_y = msg.wrench.torque.y
        torque_z = msg.wrench.torque.z
        
        # Write data to CSV file
        self.csv_writer.writerow([timestamp, force_x, force_y, force_z, torque_x, torque_y, torque_z])
    
    def on_shutdown(self):
        # Close the CSV file on shutdown
        self.get_logger().info('Shutting down, closing CSV file')
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    
    node = WrenchSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Explicitly destroy the node and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
