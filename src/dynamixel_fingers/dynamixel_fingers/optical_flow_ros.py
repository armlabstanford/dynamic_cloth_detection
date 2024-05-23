# import rclpy
# from rclpy.node import Node
# import cv2
# import numpy as np

# class OpticalFlowNode(Node):
#     def __init__(self):
#         super().__init__('optical_flow_node')
#         # Declare and get parameters
#         self.declare_parameter('camera_index', 0)  # Default value is 0
#         self.declare_parameter('filename', 'output')  # Default value is 'output'
        
#         camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
#         filename = self.get_parameter('filename').get_parameter_value().string_value
        
#         self.get_logger().info(f'Starting optical flow with camera index: {camera_index} and filename: {filename}')
        

#         self.cap = cv2.VideoCapture(camera_index)
#         self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         #self.out = cv2.VideoWriter(f'{filename}_{camera_index}.avi', self.fourcc, 10.0, (1600, 1200))
#         self.out_raw = cv2.VideoWriter(f'{filename}_{camera_index}_raw.avi', self.fourcc, 10.0, (1600, 1200))
#         self.ret, self.frame = self.cap.read()
#         #self.old_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        

#         self.camera_index = camera_index
#         self.filename = filename
#         self.timer = self.create_timer(0.1, self.process_video)  # Process at 10 Hz
        
#         #self.optical_flow_cam = self.dense_optical_flow(self.camera_index, self.filename)
        

#     def process_video(self):
#         self.dense_optical_flow(self.camera_index, self.filename)

#     def dense_optical_flow(self, camera_index=0, filename="output"):

#         now = self.get_clock().now()
#         self.get_logger().info('Current time: {0}'.format(now.to_msg()))
#         #mask = np.zeros_like(self.frame)
#         #mask[..., 1] = 255

#         self.ret, self.frame = self.cap.read()
#         #
#         #
#         #

#         # if not self.ret:
#         #     break
#         #frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

#         #flow = cv2.calcOpticalFlowFarneback(self.old_gray, frame_gray,
#                                             # None,
#                                             # 0.5,
#                                             # 2,
#                                             # 8,
#                                             # 2,
#                                             # 5,
#                                             # 1.1,
#                                             # 0)


#         # Computes the magnitude and angle of the 2D vectors
#         #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         # Sets image hue according to the optical flow  direction
#         #mask[..., 0] = ang * 180 / np.pi

#         #mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

#         #bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

#         #frame_gray_3d = cv2.cvtColor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)
#         # img = cv2.add(frame, bgr)
#         #img = cv2.add(frame_gray_3d, bgr)

#         # cv2.imshow('frame2', bgr)
#         #mask = np.zeros_like(self.frame)
#         #mask[..., 1] = 255

#         #self.old_gray = frame_gray.copy()
#         # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

#         #imgb = cv2.resize(img, (1600, 1200))
#         #self.get_logger().info(self.frame)
#         # print(self.frame.shape)
#         self.out_raw.write(self.frame)
#         #self.out_raw.write(cv2.resize(self.frame, (1600, 1200)))
#         #self.out.write(imgb)
#         # cv2.resizeWindow("frame", (1600, 1200))
#         #cv2.imshow("frame", imgb)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#     # out.release()

#     # cap.release()
#     # 
#     # cv2.destroyAllWindows()
    

# def main(args=None):
#     rclpy.init(args=args)
#     optical_flow_node = OpticalFlowNode()
#     rclpy.spin(optical_flow_node)
#     optical_flow_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from threading import Thread
from queue import Queue

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('filename', 'output')
        
        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        filename = self.get_parameter('filename').get_parameter_value().string_value
        
        self.cap = cv2.VideoCapture(camera_index)
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Capture actual frame rate
        self.fps = 30.0
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_raw = cv2.VideoWriter(f'{filename}_{camera_index}_raw.avi', self.fourcc, self.fps, self.frame_size)
        
        self.frame_queue = Queue()
        self.writer_thread = Thread(target=self.process_video)
        self.writer_thread.start()
        self.timer = self.create_timer(1/self.fps, self.capture_frame)
        self.get_logger().info(f'{self.fps}')
        #self.get_logger().info(self.frame)
        

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_queue.put(frame)

    def process_video(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.out_raw.write(frame)

def main(args=None):
    rclpy.init(args=args)
    optical_flow_node = OpticalFlowNode()
    rclpy.spin(optical_flow_node)
    optical_flow_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
