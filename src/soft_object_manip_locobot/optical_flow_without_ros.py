#! /usr/bin/python3
####!/usr/bin/env python


# print ('abc')
####################
##  [python 3 /2 problems](https://answers.ros.org/question/345942/modulenotfounderror-no-module-named-netifaces/)
####################


import struct
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt
import filters
import math
from matplotlib import gridspec
import argparse


# from random import randint

def DenseFlow(frames):
    """Draw dense flow from consecutive images
    """

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGRA2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    mask[..., 1] = 255
    # print(len(frames))

    for i, frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow =  cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.702, 5, 10, 2, 7, 1.5,
        #                                      cv2.OPTFLOW_FARNEBACK_GAUSSIAN )

        # Computes the magnitude and angle of the 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow  direction
        mask[..., 0] = ang * 180 / np.pi
        # print(ang * 180 / np.pi / 2)
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        img = cv2.add(frame, bgr)
        # cv2.imshow('frame2', bgr)
        mask = np.zeros_like(old_frame)
        mask[..., 1] = 255

        old_gray = frame_gray.copy()

    return img

def DISFlow(frames):
    """Draw dense flow from consecutive images
    """

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGRA2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    mask[..., 1] = 255
    # print(len(frames))

    for i, frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # 3.4.4.19
        # inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)

        # 4.1
        inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

        flow = inst.calc(old_gray, frame_gray, None)

        # Computes the magnitude and angle of the 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow  direction
        mask[..., 0] = ang * 180 / np.pi
        # print(ang * 180 / np.pi / 2)
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        img = cv2.add(frame, bgr)
        # cv2.imshow('frame2', bgr)
        mask = np.zeros_like(old_frame)
        mask[..., 1] = 255

        old_gray = frame_gray.copy()

    return img

def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    return img


def cleanup(self):
    print
    "Shutting down vision node."
    cv2.destroyAllWindows()



indx = 2000

def lukas(cap, color):

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=indx, qualityLevel=0.2, minDistance=8, blockSize=8)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(30, 30),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    return p0, mask, lk_params, old_gray

# fourcc = cv2.VideoWriter_fourcc(*'X264')
# fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 10.0, (1600, 1200))

def opticalFlow():
    # define a video capture object
    cap = cv2.VideoCapture(5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_opticalflow_2.avi', fourcc, 10.0, (1600, 1200))
    # Create random colors
    color = np.random.randint(0, 255, (indx, 3))
    ##### To keep follow same feature
    # p0, mask, lk_params, old_gray = lukas(cap, color)

    while True:

        ##### to define new feature for each cycle
        p0, mask, lk_params, old_gray = lukas(cap, color)

        # Read new frame
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Display the demo
        img = cv2.add(frame, mask)
        # cv2.imshow("frame", img)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

        imgb = cv2.resize(img, (1600, 1200))
        out.write(imgb)

        cv2.resizeWindow("frame", (1600, 1200))
        cv2.imshow("frame", imgb)

        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

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
        cv2.imshow("frame", imgb)

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