import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to load video and calculate optical flow
def load_and_compute_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flows = []
    print("Processing video...")
    while True:
        print("Processing frame", len(flows)+1)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev_gray = gray
    
    cap.release()
    return flows

# Update function for animation
def update_flow(num, flows, ax):
    flow = flows[num]
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ax.clear()
    ax.imshow(rgb)
    ax.set_title(f"Frame {num+1}")

# Main part of the script
video_path = 'videos/tests_silk/no_layer_0_raw.avi'  # Change this to your video file path
flows = load_and_compute_optical_flow(video_path)

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update_flow, frames=len(flows), fargs=(flows, ax), interval=50)
plt.show()
