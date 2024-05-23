#! /usr/bin/python3
####!/usr/bin/env python


import struct
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt
import filters
import math
from matplotlib import gridspec
import argparse

def denseOpticalFlow(camera_index=0, filename="output"):
    # define a video capture object
    cap = cv2.VideoCapture(camera_index)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{filename}_{camera_index}.avi', fourcc, 10.0, (1600, 1200))
    out_raw = cv2.VideoWriter(f'{filename}_{camera_index}_raw.avi', fourcc, 10.0, (1600, 1200))
    #out = cv2.VideoWriter(f'{filename}_{camera_index}.mp4', fourcc, 10.0, (1600, 1200))

    # Create random colors
    color = np.random.randint(0, 255, (300, 3))
    ##### To keep follow same feature
    # p0, mask, lk_params, old_gray = lukas(cap, color)
    ret, frame = cap.read()

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    # print(len(frames))
    while True:

        ##### to define new feature for each cycle

        # Read new frame
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        ret, frame = cap.read()

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4.1
        # inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        # flow = inst.calc(old_gray, frame_gray, None)

        # Calculates dense optical flow by Farneback method
        # flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray,
        #                                    None,
        #                                    0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray,
                                           None,
                                           0.5,
                                            2,
                                            8,
                                            2,
                                            5,
                                            1.1,
                                            0)


        # Computes the magnitude and angle of the 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow  direction
        mask[..., 0] = ang * 180 / np.pi
        # print(ang * 180 / np.pi / 2)
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        frame_gray_3d = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)
        # img = cv2.add(frame, bgr)
        img = cv2.add(frame_gray_3d, bgr)

        # cv2.imshow('frame2', bgr)
        mask = np.zeros_like(frame)
        mask[..., 1] = 255

        old_gray = frame_gray.copy()
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

        imgb = cv2.resize(img, (1600, 1200))
        
        out_raw.write(cv2.resize(frame, (1600, 1200)))
        out.write(imgb)
        cv2.resizeWindow("frame", (1600, 1200))
        #cv2.imshow("frame", imgb)

        # k = cv2.waitKey(25) & 0xFF
        # if k == 27:
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()

    cap.release()
    cv2.destroyAllWindows()

def main(args):
   # opticalFlow()
   # Create the parser and add the argument for a single number
   parser = argparse.ArgumentParser(description="Read a single number from the command line.")
   parser.add_argument("number", type=int, help="A single integer number")
   parser.add_argument('file_name', type=str, help="CSV file containing input and output file pairs for optical_flow.")

   # Parse the argument
   args = parser.parse_args()
   denseOpticalFlow(args.number, args.file_name)

if __name__ == '__main__':
    main(sys.argv)