#! /usr/bin/python3
####!/usr/bin/env python


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
import os

# Params for data analysis
num_max = 10        # number of top magnitudes to take average of for plotting
percentile = 99.9   # percentile of magnitudes for plotting
cap_reps = 7        # number of times to repeat frame capture before performing optical flow
comparison_rate = 10/cap_reps  # frame-rate of optical flow comparison (Hz)


def denseOpticalFlow(file_path, filename="output"):
    # define a video capture object
    cap = cv2.VideoCapture(file_path)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f'{filename}_postprocess.avi', fourcc, 10.0, (1600, 1200))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'postprocess_vids/{filename}_postprocess.mp4', fourcc, 10.0, (1600, 1200))
    
    #out = cv2.VideoWriter(f'{filename}_{camera_index}.mp4', fourcc, 10.0, (1600, 1200))
    #out_raw = cv2.VideoWriter(f'{filename}_{camera_index}_raw.avi', fourcc, 10.0, (1600, 1200))

    # Create random colors
    color = np.random.randint(0, 255, (300, 3))
    ##### To keep follow same feature
    # p0, mask, lk_params, old_gray = lukas(cap, color)
    ret, frame = cap.read()
    # print(ret)
    # print(frame)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    # print(len(frames))

    # List of max magnitudes of each frame
    max_list = []
    quartile_list = []
    while True:

        ##### to define new feature for each cycle

        # Read new frame
        # Repeat this line n times to only compare every nth frame
        for i in range(cap_reps):
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

        # Get first num_max largest magnitudes
        indices = np.argsort(mag.flatten())[::-1][:num_max]
        max_vals = mag[np.unravel_index(indices, mag.shape)]
        max_list.append(np.mean(max_vals))
        quartile_list.append(np.percentile(mag, percentile))

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
        
        #out_raw.write(cv2.resize(frame, (1600, 1200)))
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

    max_arr = np.array(max_list)
    quartile_arr = np.array(quartile_list)
    return max_arr, quartile_arr


def main(args):
    parser = argparse.ArgumentParser(description="Read a folder path and name from the command line.")
    parser.add_argument("folder_path", type=str, help="Path folder of to raw RGB video from DT.")

    # Parse the argument
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"The path '{folder_path}' is not a directory.")
        return

    # Loop through each file in the directory
    max_combined_list = []
    quartile_combined_list = []
    filename_list = []
    for file_name in os.listdir(args.folder_path):
        # Get the full path of the file
        print(file_name)
        file_path = os.path.join(args.folder_path, file_name)
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            filename_list.append(file_name)
            max_arr, quartile_arr = denseOpticalFlow(file_path, file_name)
            max_combined_list.append(max_arr)
            quartile_combined_list.append(quartile_arr)
            # print(max_arr.shape)

    min_length = min(len(arr) for arr in max_combined_list)
    max_combined_arr = np.vstack([arr[:min_length] for arr in max_combined_list]) # shape: num vids x num frames + 1
    np.savetxt('max_mag.csv', max_combined_arr, delimiter=',')
    quartile_combined_arr = np.vstack([arr[:min_length] for arr in quartile_combined_list])
    
    # Plotting
    frames = np.arange(max_combined_arr.shape[1])
    if len(filename_list) == 1:
        plt.figure(figsize=(10,6))
        for i in range(max_combined_arr.shape[0]):
            plt.plot(frames, max_combined_arr[i,:], label=f'Average of max {num_max} magnitudes for {filename_list[i][:-10]}')
            plt.plot(frames, quartile_combined_arr[i,:], label=f'{percentile}% percentile magnitude for {filename_list[i][:-10]}')
        plt.title(f'{filename_list[i][:-10]} \nFrame Comparison Rate: {comparison_rate:.2f} Hz')
        plt.show()

    else: 
    # Plot average of specified number of maximum magnitudes
        plt.figure(figsize=(10,6))
        for i in range(max_combined_arr.shape[0]):
            plt.plot(frames, max_combined_arr[i,:], label=filename_list[i][:-10])
        plt.title(f'Average of Max {num_max} Magnitudes \nFrame Comparison Rate: {comparison_rate:.2f} Hz')
        plt.xlabel('Frame')
        plt.ylabel('Optical Flow Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show() 

        # Plot specified percentile magnitudes
        plt.figure(figsize=(10,6))
        for i in range(max_combined_arr.shape[0]):
            plt.plot(frames, quartile_combined_arr[i,:], label=filename_list[i][:-10])
        plt.title(f'{percentile}% percentile Magnitude \nFrame Comparison Rate: {comparison_rate:.2f} Hz')
        plt.xlabel('Frame')
        plt.ylabel('Optical Flow Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show() 


if __name__ == '__main__':
    main(sys.argv)