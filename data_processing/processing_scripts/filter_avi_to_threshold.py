import pandas as pd
import cv2
import os

def filter_data(input_file, threshold=-1.25):
    # Load the CSV file
    data = pd.read_csv(input_file)
    
    # Find the first occurrence where force_x exceeds the threshold
    index_threshold = data[data['force_x'] > threshold].index[0]
    
    return index_threshold

def filter_csv_data(input_file, output_file, threshold=-1.25):
    # Load the CSV file
    data = pd.read_csv(input_file)
    
    # Find the first occurrence where force_x exceeds the threshold
    index_threshold = data[data['force_x'] > threshold].index[0]
    
    # Keep all rows after this index
    filtered_data = data.iloc[index_threshold:]
    
    # Save the filtered data to a new CSV file
    filtered_data.to_csv(output_file, index=False)

def process_video(input_video, output_video, start_frame):
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write frames starting from the start_frame
        if frame_count >= start_frame:
            out.write(frame)
        
        frame_count += 1

    print(f'Frames of input video: {frame_count}')
    print(f'Frames of output video: {frame_count - start_frame}')
    
    # Release everything if job is finished
    cap.release()
    out.release()

def process_files(input_dir, output_dir, wrench_dir, wrench_prefix='wrench_data_', video_prefix='output_', output_video_prefix='filtered_video_', threshold=-1.25):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    trial_num = 20
    while True:
        input_csv = os.path.join(wrench_dir, f'{wrench_prefix}{trial_num}.csv')
        input_video = os.path.join(input_dir, f'{video_prefix}{trial_num}.avi')
        if not os.path.exists(input_csv) or not os.path.exists(input_video):
            print(f'Files not found: {input_csv}, {input_video}')
            break
        output_video = os.path.join(output_dir, f'{output_video_prefix}{trial_num}.avi')
        start_frame = filter_data(input_csv, threshold)
        process_video(input_video, output_video, start_frame)
        print(f'Processed {input_video} -> {output_video}')
        trial_num += 1

# Example usage
layer_str = 'two_layers'
wrench_dir = f'tests_5_23/wrench_data/{layer_str}'
input_dir = f'tests_5_23/output_videos/{layer_str}'
output_dir = f'output_tests_5_23/video_data/{layer_str}'
process_files(input_dir, output_dir, wrench_dir)
