import cv2
import time
import argparse

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -50)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video capture script.')
    parser.add_argument('--index', type=int, default=0, help='Video capture index (default: 0)')
    args = parser.parse_args()

    # Initialize the USB webcam feed
    cap = cv2.VideoCapture(args.index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # Set camera resolution (width, height)
    set_camera_resolution(cap, 640, 480)  # Example resolution; modify as needed

    # Variables to calculate the actual frame rate
    frame_count = 0
    start_time = time.time()

    # Continuously capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Calculate frame rate
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            actual_frame_rate = frame_count / elapsed_time
        else:
            actual_frame_rate = 0

        # Display the frame rate on the frame
        cv2.putText(frame, f'FPS: {actual_frame_rate:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Stream', frame)

        # Press 'q' on the keyboard to exit the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
