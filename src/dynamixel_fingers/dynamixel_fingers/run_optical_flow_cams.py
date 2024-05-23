import subprocess
import threading
import time
import sys 
import signal
import os

processes = []
# Function to run optical_flow.py with a specific set of arguments
def run_optical_flow_instance(camera_index, filename):
    args = ['python3', 'src/dynamixel_fingers/dynamixel_fingers/optical_flow_without_ros.py', camera_index, filename]
    print(f"Running optical_flow with {camera_index}...")
    # result = subprocess.Popen(args, capture_output=True, text=True)
    result = subprocess.Popen(args)
    processes.append(result)

# Main function to launch threads for running the two commands in parallel
def main():
    # Define the video input and output pairs
    arguments = [
        ("2", "src/dynamixel_fingers/tests/run"),
        ("3", "src/dynamixel_fingers/tests/run")
    ]

    threads = []

    # Create and start threads for each command
    for camera_index, filename in arguments:
        thread = threading.Thread(target=run_optical_flow_instance, args=(camera_index, filename))
        threads.append(thread)
        thread.start()
    
    time.sleep(2.0)
    thread = subprocess.run(['python3', 'src/dynamixel_fingers/dynamixel_fingers/dynamixel_finger/rubbing_motion.py'])
    threads.append(thread)
    
    print("Rubbing Complete")
    
    for process in processes:
        process.kill()


if __name__ == "__main__":
    main()


