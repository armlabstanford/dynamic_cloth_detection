import cv2

def display_video_frames(video_path, vid_name):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")
    
    frame_count = 0
    
    while cap.isOpened():
        # Read one frame at a time
        ret, frame = cap.read()
        print(frame_count)
        
        if not ret:
            print("Reached the end of the video or there was an error reading the frame.")
            break
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Wait for a key press
        key = cv2.waitKey(0)
        
        if key == ord('s'):
            # Save the current frame
            frame_filename = f'{vid_name}_frame_{frame_count}.jpg'
            cv2.imwrite(frame_filename, frame)
            print(f"Frame saved as {frame_filename}")
        
        # Exit on 'q' key
        elif key == ord('q'):
            print("Exiting...")
            break
        
        frame_count += 1
        
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid_name = 'filtered_video_85_uncal'
    video_path = f'{vid_name}.avi'
    display_video_frames(video_path, vid_name)
