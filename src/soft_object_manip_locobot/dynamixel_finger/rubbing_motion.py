#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import numpy as np
from dynamixel_client import *
import time
"""
Recommended to only query when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.
0 is flat out for first joint, 360 for the second joing, increasing angle is closing more and more.
"""

class DynamixelFinger(Node):
    """
    """
    def __init__(self):
        super().__init__('dynamixel_finger_node')
        #parameters
        self.kP = 600
        self.kI = 0
        self.kD = 500
        self.curr_lim = 350  #500 normal, 350 for lite
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [1, 2, 3, 4]  # [lower blue, upper blue, lower brown, upper brown]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyACM0', 57600)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 57600)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM5', 57600)
                self.dxl_client.connect()

        # self.zero_pose = np.array([1.53401154e-03 , -4.65869983e+00 , 3.37474756e-02 , 1.57233022e+00])
        self.zero_pose = np.array([0.0, 0.0, 0.0, 0.0])
        self.prev_pos = self.pos = self.curr_pos = self.zero_pose

        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        #Max at current in mA so don't overheat / grip too hard 
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)

    #Receive pose and control finger
    def set_pose(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose) 
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos + self.zero_pose)

    #read position
    def read_pos(self):
        # return self.dxl_client.read_pos()
        return self.dxl_client.read_pos() - self.zero_pose    
    
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

    #calibrate current position as zero position (both fingers fully extended)
    def calibrate_zero(self):
        self.zero_pose = self.dxl_client.read_pos()
        self.curr_pos = np.zeros(4)

    #move smoothly to specified pose
    def move_to_pose(self, pose):
        des_pose = np.array(pose)
        while np.max(np.abs(des_pose - self.curr_pos)) > 0.05:
            self.set_pose(self.curr_pos + 0.03*(des_pose - self.curr_pos))
    
    #go to zero position (both fingers fully extended)
    def return_to_zero(self):
        self.move_to_pose(np.zeros(4))

    #go to open position
    def open(self):
        self.move_to_pose([-0.9, 1.8, 0.9, -1.8])
    
    #go to grasping position
    def grasp(self):
        self.move_to_pose([-0.45, 1.8, 0.45, -1.8])


    # def roll(self, duration):
    #     timeout = time.time() + duration
    #     original_pose = self.curr_pos
    #     A = 0.3
    #     while True:
    #         self.move_to_pose(original_pose + np.array([0, A, 0, A]))
    #         self.move_to_pose(original_pose - np.array([0, A, 0, A]))
    #         if time.time() > timeout:
    #             break
    
    #roll upper joints back and forth (opp directions on each finger) for specified duration (in seconds)
    def roll(self, duration):
        start_time = time.time()
        original_pose = self.curr_pos
        #sin wave parameters
        A = 0.15
        T = 2
        while True:
            curr_time = time.time() - start_time
            self.set_pose(original_pose + np.array([0, A*np.sin((6.28/T)*curr_time), 0, A*np.sin((6.28/T)*curr_time)]))
            if curr_time > duration:
                break


#init the node
def main(**kwargs):
    rclpy.init()
    finger = DynamixelFinger()
    print("Position: " + str(finger.read_pos()))
    finger.calibrate_zero()
    print("Position: " + str(finger.read_pos()))

    # while True:
    finger.open()
    time.sleep(1.0)
    finger.grasp()
    time.sleep(2.0)
    finger.roll(10.0)
    time.sleep(2.0)
    finger.return_to_zero()
    time.sleep(2.0)


if __name__ == "__main__":
    main()
