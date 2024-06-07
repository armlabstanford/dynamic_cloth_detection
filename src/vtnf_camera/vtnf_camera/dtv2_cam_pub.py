

#### code for publishing the webcam feed as well as the DTv2 depth / color images
import sys

# print(sys.exec_prefix)
# sys.path.append('/home/wkdo/miniconda3/envs/dtros2/lib/python3.10/site-packages')

import torch

import rclpy
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import numpy as np
import time
from rclpy.node import Node 
import os, re
import threading
from geometry_msgs.msg import WrenchStamped, Vector3
from sensor_msgs.msg import JointState
import rosbag2_py
from rclpy.serialization import serialize_message, deserialize_message
import pandas as pd
import csv
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.serde import deserialize_cdr

# from .utils.utils import get_video_device_number
from .Img2Depth.img2depthforce import getDepth, getForce
from .Img2Depth.networks.DenseNet import DenseDepth
from .Img2Depth.networks.STForce import DenseNet_Force

import datetime

class camPublisher(Node):
    def __init__(self):
        super().__init__('cam_publisher_dt')

        path_base = os.getcwd()
        os.chdir(os.path.join(path_base, "src/vtnf_camera/vtnf_camera/"))

        self.get_logger().info(f"Initializing DT V2 Cam Publisher Node...")

        self.declare_parameter('camera_id', '0')  # Default value if not provided
        self.declare_parameter('sensornum', '0')  # Default value if not provided
        self.declare_parameter('num_layers', '0')  # Default value if not provided
        self.declare_parameter('trial_num', '0')  # Default value if not provided

        self.camera_id = int(self.get_parameter('camera_id').get_parameter_value().string_value)
        self.sensornum = int(self.get_parameter('sensornum').get_parameter_value().string_value)
        self.num_layers = int(self.get_parameter('num_layers').get_parameter_value().string_value)
        self.trial_num = int(self.get_parameter('trial_num').get_parameter_value().string_value)
        print (f"dtv2_device_number: {self.camera_id}")

        # rosbag directory
        self.rosbag_save_dir = f'/home/armlab/Documents/soft_manipulation/bag_files/{self.num_layers}_layer'
        if not os.path.exists(self.rosbag_save_dir):
            os.makedirs(self.rosbag_save_dir)

        # rosbag writer
        self._writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py._storage.StorageOptions(uri = self.rosbag_save_dir + f'/bag_{self.trial_num}',
                                                             storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self._writer.open(storage_options, converter_options)
        force_topic_info = rosbag2_py._storage.TopicMetadata(name='/RunCamera/force',
                                                             type="geometry_msgs/msg/WrenchStamped",
                                                             serialization_format='cdr')
        self._writer.create_topic(force_topic_info)        

        joint_state_topic_info = rosbag2_py._storage.TopicMetadata(name='/dxl_joint_states',
                                                                   type='sensor_msgs/msg/JointState',
                                                                   serialization_format='cdr')
        self._writer.create_topic(joint_state_topic_info)

        self.joint_states_sub = self.create_subscription(
            JointState,
            '/dxl_joint_states',
            self.joint_states_callback,
            10)
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = ['joint1', 'joint2', 'joint3', 'joint4']
        joint_state_msg.position = [np.NaN, np.NaN, np.NaN, np.NaN]
        self.curr_joint_state_msg = joint_state_msg

        queue_size = 1
        self.pub_dtv2 = self.create_publisher(Image, 'vtnf/camera_{}'.format(self.camera_id), queue_size)
        if self.sensornum == 1: # publish the depth image if it is sensor 1
            self.pub_dtv2_depth = self.create_publisher(Image, 'vtnf/depth', queue_size)
            self.force_pub = self.create_publisher(WrenchStamped, '/RunCamera/force', queue_size)


        self.br = CvBridge()

        ##################### depth estimation 

        #sensornum = 1 # 101 # 2
        # array for determining whether the sensor can do position estimation and force estimation or not
        sen_pf = np.array([[1,1,1],
                        [2,1,1],
                        [3,0,1],
                        [4,0,0],
                        [5,0,1],
                        [6,1,0],
                        [101,1,0],
                        [102,1,0]])
        # whether it use pos or force
        if self.sensornum:
            self.ispos = sen_pf[sen_pf[:,0]==self.sensornum][0,1]
            self.isforce = sen_pf[sen_pf[:,0]==self.sensornum][0,2]
            self.isforce = 1 # disable force for now

            self.imgidx = np.load('Img2Depth/calib_idx/mask_idx_{}.npy'.format(self.sensornum))
            self.radidx = np.load('Img2Depth/calib_idx/pts_2ndmask_{}_80deg.npy'.format(self.sensornum))

            self.cen_x, self.cen_y, self.exposure = self.get_sensorinfo(self.sensornum)
        self.flag = 0
        self.camopened = True
        self.netuse = True
        # Params
        self.image = None
        self.img_noncrop = np.zeros((768,1024))
        self.maxrad = 16.88
        self.minrad = 12.23
        self.input_width = 640
        self.imgsize = int(self.input_width/2)

        # FOR DEBUGGING
        self.start_time = time.time()
        self.num_frames = 0

        if self.netuse and self.sensornum == 1: # only use the net if it is sensor 1
            ######## model setting ######
            if self.ispos == 1:
                self.model_pos = DenseDepth(max_depth = 256, pretrained = False)
                modelname = 'Img2Depth/position_sensor_{}.pth'.format(self.sensornum)
                print(modelname)
                checkpoint_pos = torch.load(modelname)
                self.ispf = 'pos'
                self.model_pos = torch.nn.DataParallel(self.model_pos)
                self.model_pos.load_state_dict(checkpoint_pos['model'])
                self.model_pos.eval()
                self.model_pos.cuda()
                # self.imgDepth = self.img2Depth(np.ones((640,640,3)))

            if self.isforce == 1:
                self.model_force = DenseNet_Force(pretrained= False)
                modelname = 'Img2Depth/force_sensor_{}.pth'.format(self.sensornum)
                print(modelname)
                checkpoint_force = torch.load(modelname)
                self.ispf = 'force'
                self.model_force = torch.nn.DataParallel(self.model_force)
                self.model_force.load_state_dict(checkpoint_force['model'])
                self.model_force.eval()
                self.model_force.cuda()
                # self.imgForce = self.img2Force(np.ones((640,640,3)))


        ##################### depth estimation done

        use_timer = True
        # should we done with timer for ensuring stable frame rate?
        # 25 frame rate
        if use_timer:
            timerspeed = 0.04
            self.timer = self.create_timer(timerspeed, self.timer_callback)
            print("OpenCV Version: {}".format(cv2.__version__))
                
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            if not (self.cap.isOpened()):
                print("Cannot open the camera")

            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.cap.set(cv2.CAP_PROP_APERTURE, 150)
            commands = [
                ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c auto_exposure=3"),
                ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c auto_exposure=1"),
                ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c exposure_time_absolute="+str(150)),
        ]
            for c in commands: 
                os.system(c)


            print(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
            fourcc = int(fourcc)
            print(self.cap.get(cv2.CAP_PROP_FOURCC))
            print(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
            print(self.cap.get(cv2.CAP_PROP_SETTINGS))
            print(self.cap.get(cv2.CAP_PROP_FPS))

            # current_time = datetime.datetime.now()
            # timestamp = current_time.strftime('%m%d_%H%M%S')
            self.out = cv2.VideoWriter(f'/home/armlab/Documents/soft_manipulation/output_videos/{self.num_layers}_layer/output_{self.trial_num}.avi', fourcc, 10.0, (1024, 768))

        else:
            self.capture_thread = threading.Thread(target=self.capture_and_publish)
            self.capture_thread.start()

        self.subscription = self.create_subscription(
            Bool,
            'shutdown_flag',
            self.flag_callback,
            10)
        self.subscription  # prevent unused variable warning

    def joint_states_callback(self, msg):
        self.curr_joint_state_msg = msg

    def flag_callback(self, msg):
        if msg.data:
            # close rosbag writer
            self._writer.close()
            del self._writer

            # save data from rosbag to csv
            bag_name = self.rosbag_save_dir + f'/bag_{self.trial_num}'
            force_csv_filename = f'/home/armlab/Documents/soft_manipulation/wrench_data/{self.num_layers}_layer/wrench_data_{self.trial_num}.csv'
            joint_states_csv_filename = f'/home/armlab/Documents/soft_manipulation/joint_state_data/{self.num_layers}_layer/joint_states_{self.trial_num}.csv'
            self.read_rosbag_to_csv(bag_name, '/RunCamera/force', force_csv_filename)
            self.read_rosbag_to_csv(bag_name, '/dxl_joint_states', joint_states_csv_filename)

            # shutdown node
            self.get_logger().info('Shutdown flag received, shutting down...')
            self.get_logger().info('Frames captured: {}'.format(self.num_frames))
            self.get_logger().info(f'Time elapsed: {time.time() - self.start_time} seconds')
            rclpy.shutdown()
            
    def timer_callback(self):
        """
            callback function for the publishing and reading the sensor img
        """
        now = self.get_clock().now()

        ret, frame = self.cap.read()
        if ret:
            self.num_frames += 1
            # self.out.write(cv2.resize(frame, (1600, 1200)))
            self.out.write(frame)
            msg = self.br.cv2_to_imgmsg(frame)
            self.pub_dtv2.publish(msg)

            if self.sensornum == 1: # publish the depth image if it is sensor 1
                rectImg = self.rectifyimg(frame)

                depthImg = getDepth(self.model_pos, rectImg)
                msg_depth = self.br.cv2_to_imgmsg(depthImg, "8UC1")
                self.pub_dtv2_depth.publish(msg_depth)
                if self.isforce == 1:
                    forceEst = getForce(self.model_force, rectImg)
                    wrench_stamped_msg = WrenchStamped()
                    wrench_stamped_msg.header.stamp = now.to_msg()
                    wrench_stamped_msg.wrench.force = Vector3(x=forceEst[0], y=forceEst[1], z=forceEst[2])
                    wrench_stamped_msg.wrench.torque = Vector3(x=forceEst[3], y=forceEst[4], z=forceEst[5])
                    self.force_pub.publish(wrench_stamped_msg)

                    # Write wrench data to rosbag
                    self._writer.write('/RunCamera/force',
                                        serialize_message(wrench_stamped_msg),
                                        self.get_clock().now().nanoseconds)
                    
                    # Write joint state data to rosbag
                    self._writer.write('/dxl_joint_states',
                                        serialize_message(self.curr_joint_state_msg),
                                        self.get_clock().now().nanoseconds)
        else:
            self.get_logger().error('Failed to capture frame')

    def get_sensorinfo(self, calibnum):
        """
        get center of each sensor.
        """
        brightness = 160
        senarr = np.array([[6, 520, 389, 150, 320],
                            [1, 522, 343, 100, 314],
                            [2, 520, 389, 150, 322],
                            [3, 522, 343, 100, 316],
                            [4, 522, 343, 100, 307],
                            [5, 547, 384, 159, 303],
                            [101, 512, 358, 100, 298],
                            [102, 545, 379, 100, 300],
                            [103, 522, 343, 100, 300]])

        cen_x = senarr[senarr[:,0]==calibnum][0,1]
        cen_y = senarr[senarr[:,0]==calibnum][0,2]
        brightness = senarr[senarr[:,0]==calibnum][0,3]
        self.radius = senarr[senarr[:,0]==calibnum][0,4]
        return cen_x, cen_y, brightness

    def rectifyimg(self, frame2):
        '''
            function for rectifying the image based on the given camera node 
            Now the function manually get the center of each circular shape and match the function. 

            Key is to match the center of pixel correctly so that we can get the right match process with original sensor.

        '''
        beforeRectImg2 = frame2.copy()
        (h, w) = beforeRectImg2.shape[:2]

        img_reshape = beforeRectImg2.reshape(w*h, 3)
        mask = np.ones(img_reshape.shape[0], dtype=bool)
        mask[self.imgidx[self.radidx]] = False
        img_reshape[mask, :] = np.array([0, 0, 0])
        img2 = img_reshape.reshape(h, w, 3)

        beforeRectImg2 = img2[self.cen_y-self.imgsize:self.cen_y+self.imgsize,self.cen_x-self.imgsize:self.cen_x+self.imgsize]
        
        rectImg2 = beforeRectImg2
        return rectImg2

    def capture_and_publish(self):
        print("OpenCV Version: {}".format(cv2.__version__))
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        # cap = cv2.VideoCapture(self.camera_id, cv2.CAP_GSTREAMER)
        if not (cap.isOpened()):
            print("Cannot open the camera")


        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.set(cv2.CAP_PROP_APERTURE, 150)
        commands = [
            ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c auto_exposure=3"),
            ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c auto_exposure=1"),
            ("v4l2-ctl --device /dev/video"+str(self.camera_id)+" -c exposure_time_absolute="+str(150)),
       ]
        for c in commands: 
            os.system(c)


        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # show fourcc code in interpretable way
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fourcc = int(fourcc)
        fourcc = fourcc.to_bytes(4, 'little').decode()
        print(fourcc)
        print(cap.get(cv2.CAP_PROP_FOURCC))
        print(cap.get(cv2.CAP_PROP_AUTOFOCUS))
        print(cap.get(cv2.CAP_PROP_SETTINGS))
        print(cap.get(cv2.CAP_PROP_FPS))


        while rclpy.ok():
            # print out how much time it takes to capture and publish
            start_time = time.time()

            
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error('Failed to capture frame')
                break
            msg = self.br.cv2_to_imgmsg(frame)
            self.pub_dtv2.publish(msg)

            rectImg = self.rectifyimg(frame)

            depthImg = getDepth(self.model_pos, rectImg)
            # print("depthImg: ", depthImg.shape)
            msg_depth = self.br.cv2_to_imgmsg(depthImg, "8UC1")
            self.pub_dtv2_depth.publish(msg_depth)
            print('sth')
            if self.isforce == 1:
                forceEst = getForce(self.model_force, rectImg)
                wrench_stamped_msg = WrenchStamped()
                    # Set the force and torque values in the message
                wrench_stamped_msg.header.stamp = rclpy.get_clock().now().to_msg()
                wrench_stamped_msg.wrench.force = Vector3(*forceEst[:3])
                wrench_stamped_msg.wrench.torque = Vector3(*forceEst[3:])
                self.force_pub.publish(wrench_stamped_msg)

            end_time = time.time()


    def read_rosbag_to_csv(self, bag_file, topic_name, output_csv):

        # Initialize the rosbag2 reader
        storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic information
        topics_and_types = reader.get_all_topics_and_types()
        topic_type = None
        for topic in topics_and_types:
            if topic.name == topic_name:
                topic_type = topic.type
                break

        if topic_type is None:
            self.get_logger().info(f"Topic {topic_name} not found in the bag file.")
            return

        # Get the message type
        msg_type = get_message(topic_type)

        # Prepare the CSV file
        data = []
        if topic_name == '/RunCamera/force':
            self.get_logger().info(f"Writing {topic_name} data to {output_csv}")
            columns = ['timestamp', 'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']

            while reader.has_next():
                (topic, data_raw, t) = reader.read_next()
                if topic == '/RunCamera/force':
                    msg = deserialize_message(data_raw, msg_type)
                    timestamp = t
                    force_x = msg.wrench.force.x
                    force_y = msg.wrench.force.y
                    force_z = msg.wrench.force.z
                    torque_x = msg.wrench.torque.x
                    torque_y = msg.wrench.torque.y
                    torque_z = msg.wrench.torque.z

                    data.append([timestamp, force_x, force_y, force_z, torque_x, torque_y, torque_z])

        elif topic_name == '/dxl_joint_states':
            self.get_logger().info(f"Writing {topic_name} data to {output_csv}")
            columns = ['timestamp', 'joint_pos_1', 'joint_pos_2', 'joint_pos_3', 'joint_pos_4']
        
            while reader.has_next():
                (topic, data_raw, t) = reader.read_next()
                if topic == '/dxl_joint_states':
                    msg = deserialize_message(data_raw, msg_type)
                    timestamp = t
                    joint_pos_1 = msg.position[0]
                    joint_pos_2 = msg.position[1]
                    joint_pos_3 = msg.position[2]
                    joint_pos_4 = msg.position[3]

                    data.append([timestamp, joint_pos_1, joint_pos_2, joint_pos_3, joint_pos_4])

        # Save to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)
        self.get_logger().info(f"{topic_name} data has been written to {output_csv}")


    # read rosbag of wrench data and save to csv
    # def read_rosbag_to_csv(self, rosbag_dir, topic, csv_file):
    #     # self._readerSetup()
    #     typestore = get_typestore(Stores.LATEST)
    #     with Reader(rosbag_dir) as ros2_reader:
    #         with open(csv_file, mode='w', newline='') as file:
    #             csv_writer = csv.writer(file)
    #             # Write the header row
    #             csv_writer.writerow(['timestamp', 'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z'])

    #             ros2_conns = [x for x in ros2_reader.connections]
    #             ros2_messages = ros2_reader.messages(connections=ros2_conns)
    #             for m, msg in enumerate(ros2_messages):
    #                 (connection, timestamp, rawdata) = msg
    #                 if (connection.topic == topic):
    #                     data = typestore.deserialize_cdr(rawdata, connection.msgtype)
    #                     force = data.wrench.force
    #                     torque = data.wrench.torque
                        
    #                     # Write the data to the CSV file
    #                     csv_writer.writerow([
    #                         timestamp,
    #                         force.x, force.y, force.z,
    #                         torque.x, torque.y, torque.z
    #                     ])
                    


def main(args=None):
    rclpy.init(args=args)
    # get camera_id among ros2 parameters
    

    
    cam_pub = camPublisher()

    rclpy.spin(cam_pub)
    # cam_pub.capture_thread.join()
    cam_pub.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()