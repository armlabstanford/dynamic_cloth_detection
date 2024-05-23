

import sys

# print(sys.exec_prefix)
# sys.path.append('/home/wkdo/miniconda3/envs/dtros2/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np 
import os
from datetime import datetime
import threading
from pynput import keyboard
from skimage.metrics import structural_similarity as ssim
# from skimage import measure

#TODO: add function to check if dtv2 is pressed or not

def calculate_psnr(img1, img2):
    # The function calculates the PSNR between two images.
    # Note: Both images need to be in the same dimensions and data type.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # Means the two images are identical
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        # Subscribers
        camera_0_sub = message_filters.Subscriber(self, Image, '/vtnf/camera_0')
        camera_2_sub = message_filters.Subscriber(self, Image, '/vtnf/camera_2')
        depth_sub = message_filters.Subscriber(self, Image, '/vtnf/depth')

        # Time Synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([camera_0_sub, camera_2_sub, depth_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.folder_path = self.create_directory()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        # placeholder for the last received messages
        self.touchrgb_img = None
        self.rgb_img = None
        self.depth_img = None
        self.count = 0

        # Buffer for images
        self.buffer_size = 10
        self.touchrgb_buffer = []
        self.rgb_buffer = []
        self.depth_buffer = []

        self.auto_save = False


    def add_to_buffer(self, buffer, image):
        if len(buffer) >= self.buffer_size:
            buffer.pop(0)  # Remove the oldest image
        buffer.append(image)

    def is_similar(self, new_image, buffer, threshold=0.9):
        for img in buffer:
            s = ssim(new_image, img, multichannel=True,  channel_axis=-1)
            # s = measure.compare_ssim(new_image, img)

            if s > threshold:  # Adjust the threshold as needed
                # print(f"Similarity: {s}, and threshold: {threshold}")
                return True
        # print("Not similar images")
        return False
    
    def is_similar_psnr(self, new_image, buffer, threshold=32.8):
        # Assuming a default threshold value of 30dB, which can be adjusted.
        # Higher PSNR values indicate higher similarity.
        for img in buffer:
            if calculate_psnr(new_image, img) > threshold:
                # print(f"similar images -PSNR: {calculate_psnr(new_image, img)}, and threshold: {threshold}")
                return True
        # print("Not similar images")
        return False

    def callback(self, camera_0_msg, camera_2_msg, depth_msg):
        # print('Capturing synchronized images...')
        # folder_path = self.create_directory()

        # update the msg

        self.touchrgb_img = self.bridge.imgmsg_to_cv2(camera_0_msg, desired_encoding='passthrough')
        self.rgb_img = self.bridge.imgmsg_to_cv2(camera_2_msg, desired_encoding='passthrough')
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if len(self.touchrgb_buffer) < self.buffer_size:
            print("Buffer not full yet")
            self.add_to_buffer(self.touchrgb_buffer, self.touchrgb_img)
            self.add_to_buffer(self.rgb_buffer, self.rgb_img)
            self.add_to_buffer(self.depth_buffer, self.depth_img)
        if self.auto_save:
            self.save_images()

    def create_directory(self):
        base_path = 'src/vtnf_camera/capture_data'
        date_time_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_path = os.path.join(base_path, date_time_folder)
        
        os.makedirs(folder_path, exist_ok=True)
        # make subdirectory webcam, dtv2, depth
        os.makedirs(os.path.join(folder_path, 'touchrgb'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'webcam'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'touchdepth'), exist_ok=True)

        return folder_path


    def is_image_blurry(self, image, threshold=180):
        """
        Check if an image is blurry by applying the Laplacian operator and
        calculating the variance of the result. If the variance is below a
        certain threshold, the image is considered blurry.

        Parameters:
        - image_path: Path to the image file.
        - threshold: Variance threshold for determining if an image is blurry.

        Returns:
        - A boolean indicating if the image is blurry.
        """

        if image is None:
            raise ValueError("Image not found or unable to read.")
        # print(image.shape)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the Laplacian operator
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check if the variance is below the threshold
        if laplacian_var < threshold:
            print('Blurry image detected!')
            return True  # The image is blurry
        else:
            return False  # The image is not blurry

    def save_images(self):
        if not self.is_image_blurry(self.touchrgb_img) and \
        not self.is_similar_psnr(self.touchrgb_img, self.touchrgb_buffer[:-1]) and \
        not self.is_similar_psnr(self.rgb_img, self.rgb_buffer[:-1]) and \
        not self.is_similar_psnr(self.depth_img, self.depth_buffer[:-1]):
        #     print(f"The image is blurry.")
        #     blurstr = 'blurry'
        # else:
        #     print(f"The image is not blurry.")
        #     blurstr = 'not_blurry'
        
            print("Saving {}-th images...".format(self.count))
            name_dtv2 = 'touchrgb/touchrgb_{}'.format(self.count) + '.jpg'
            name_webcam = 'webcam/rgb_{}'.format(self.count) + '.jpg'
            name_depth = 'touchdepth/touchdepth_{}'.format(self.count) + '.jpg'
    
            cv2.imwrite(os.path.join(self.folder_path, name_dtv2), self.touchrgb_img)
            cv2.imwrite(os.path.join(self.folder_path, name_webcam), self.rgb_img)
            cv2.imwrite(os.path.join(self.folder_path, name_depth), self.depth_img)
            # Add images to their respective buffers
            self.add_to_buffer(self.rgb_buffer, self.rgb_img)
            self.add_to_buffer(self.touchrgb_buffer, self.touchrgb_img)
            self.add_to_buffer(self.depth_buffer, self.depth_img)

            self.count += 1

    def on_press(self, key):

        if key == keyboard.KeyCode.from_char('s'):
            # print("Button 's' pressed! Capturing images...")
            # before capturing, filter out the blurry image from the last_camera_0_msg files 

            self.save_images()

            
        # if key 'a' is pressed, print which img is being saved
        elif key == keyboard.KeyCode.from_char('a'):
            print("Button 'a' pressed! toggling the auto-saving ftn...")
            self.auto_save = not self.auto_save
            print(f"Auto-saving is set to {self.auto_save}")

        elif key == keyboard.Key.esc:
            # Stop listener
            return False


def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    print("Image saver node started. Press 's' to capture images, 'ESC' to quit.")
    rclpy.spin(image_saver)  # This will keep your node running

    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()