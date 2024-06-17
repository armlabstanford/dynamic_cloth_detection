import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_quiver(num, quiver, cap):


    # ret, frame2 = cap.read()

    for i in range(7):
        ret, frame2 = cap.read()

        if not ret:
            print("Failed to read frame from video stream.")
            # animation.event_source.stop()
            # ani.event_source.stop()
            return quiver

    # Convert frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray[0], gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray[0] = gray2  # Update the previous frame

    # Subsample the flow for plotting
    u, v = flow[::step, ::step, 0], flow[::step, ::step, 1]

    # Update quiver data
    quiver.set_UVC(u, v)
    # return quiver,
    return quiver

# Open video
# filename = 'tests_silk/four_layer_0_raw'
# file_save_dir = "data_6_12_tighter_pose"
# file_save_dir = "data_6_13"
# l = '0_layer'
# f = 'output_1'

l = '3_layer'
f = 'output_0'
file_save_dir = "threshold_1.4_period_4.0"

# filename = f'/home/armlab/Documents/soft_manipulation/output_videos/{l}/{f}.avi'
filename = f'/home/armlab/Documents/soft_manipulation/{file_save_dir}/output_videos/calibrated_cam/{l}/{f}.avi'
# cap = cv2.VideoCapture(f'videos/{filename}.avi')  # Update the path to your video file
cap = cv2.VideoCapture(filename)  # Update the path to your video file
ret, frame1 = cap.read()
if not ret:
    print("Failed to read first frame.")
    cap.release()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total number of frames: ", frame_count)

# Convert the first frame to grayscale
prev_gray = [cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)]

# Parameters for subsampling
step = 32

# Set up initial plot on a black background
fig, ax = plt.subplots()
ax.imshow(np.zeros_like(frame1), aspect='auto')  # Display a black image
y, x = np.mgrid[0:frame1.shape[0]:step, 0:frame1.shape[1]:step]
quiver = ax.quiver(x, y, np.zeros_like(x), np.zeros_like(y), color='r', angles='xy', scale_units='xy', scale=1, width=0.002)
plt.axis('off')

# Create animation
ani = animation.FuncAnimation(fig, update_quiver, fargs=(quiver, cap), interval=100, blit=False)

# Specify the writer
FFMpegWriter = animation.writers['ffmpeg']
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800) # check if this fps is right

# Save the animation
# ani.save(f'quiver_plots/{l}/{f}.mp4', writer=writer)
ani.save(f'/home/armlab/Documents/soft_manipulation/quiver_plots/threshold_1.4_period_4.0_plots/{l}/{f}.mp4', writer=writer)

# plt.show()
cap.release()
