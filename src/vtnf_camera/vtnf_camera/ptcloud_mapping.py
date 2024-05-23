## extract transformation matrices from images.bin file, and combine all extracted pointcloud 

import numpy as np
from utils.read_write_model import read_images_binary, qvec2rotmat

def extract_transformation_matrices(images_bin_path):
    """
    Extract transformation matrices from COLMAP's images.bin file.
    
    :param images_bin_path: Path to the images.bin file.
    :return: A dictionary with image names as keys and transformation matrices as values.
    """
    images = read_images_binary(images_bin_path)
    transformation_matrices = {}

    for image_id, image in images.items():
        # Convert quaternion to rotation matrix
        rotmat = qvec2rotmat(image.qvec)
        # Combine rotation matrix and translation vector to form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotmat
        transformation_matrix[:3, 3] = image.tvec
        # Store the transformation matrix with the image name as the key
        transformation_matrices[image.name] = transformation_matrix

    return transformation_matrices

# Example usage
images_bin_path = '/home/wkdo/colcon_ws/src/vtnf_camera/capture_data/test_withtactile_2/result/sparse/2/images.bin'
transformation_matrices = extract_transformation_matrices(images_bin_path)

for image_name, matrix in transformation_matrices.items():
    print(f"Image: {image_name}, Transformation Matrix:\n{matrix}\n")
