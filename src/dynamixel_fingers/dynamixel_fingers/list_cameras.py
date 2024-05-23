import cv2

def list_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        print(f"CAMERA INDEX {index}")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

# Example usage:
camera_indices = list_cameras()
print(f"Available cameras: {camera_indices}")
