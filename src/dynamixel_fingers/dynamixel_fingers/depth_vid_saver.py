#! /usr/bin/python3
####!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        
        # Declare parameters for video saving
        self.declare_parameter('video_filename', 'output.avi')
        self.declare_parameter('fps', 10)
        
        # Get parameters
        self.video_filename = self.get_parameter('video_filename').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Create subscriber to image topic
        self.subscription = self.create_subscription(
            Image,
            '/vtnf/depth',
            self.image_callback,
            10
        )
        
        # Variables for video writer
        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        
        self.get_logger().info('Image saver node initialized')

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, '8UC1')
        
        # Initialize video writer if not already done
        if self.video_writer is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (self.frame_width, self.frame_height))
            self.get_logger().info(f'Video writer initialized: {self.video_filename}')
        
        # Write frame to video file
        self.video_writer.write(frame)

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        super().destroy_node()
        
    def on_shutdown(self):
        # Release video writer on shutdown
        self.get_logger().info('Shutting down')
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info('Video writer released')

def main(args=None):
    rclpy.init(args=args)
    
    node = ImageSaver()
    
    rclpy.spin(node)
    
    rclpy.shutdown()
    node.destroy_node()

if __name__ == '__main__':
    main()
