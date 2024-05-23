import open3d as o3d
import json
import numpy as np

# Load transforms
with open('touchnerf_092723/touchnerf_touch_1_092723/touch/transforms_train.json', 'r') as file:
    transforms = json.load(file)

combined_cloud = o3d.geometry.PointCloud()
# print(transforms.keys())
print(transforms['frames'][0])

total_frames = len(transforms['frames'])

# Initialize an array to store normal vectors and their starting points
normals = []
points = []

# Function to generate a unique color for each frame
# Function to generate a unique color for each frame
def generate_color(index):
    # Simple method to generate distinct colors
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1]   # Cyan
    ]
    return colors[index % len(colors)]

for i, element in enumerate(transforms['frames']):
    # Load point cloud

    # Load point cloud
    np_points = np.load('touchnerf_092723/touchnerf_touch_1_092723/touch/' + (element['file_path'].replace('jpg', 'npy')))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_points)

    

    # Apply transformation
    transform_matrix = np.array(element['transform_matrix'])
    point_cloud.transform(transform_matrix)

    # Extract the rotation matrix
    rotation_matrix = transform_matrix[:3, :3]

    # The normal vector is typically the third column of the rotation matrix
    normal_vector = rotation_matrix[:, 2]

    centroid = np.mean(np_points, axis=0)

    # Transform the centroid using the transformation matrix
    homogeneous_centroid = np.append(centroid, 1)  # Append 1 for homogeneous coordinates
    transformed_centroid = transform_matrix @ homogeneous_centroid
    transformed_centroid = transformed_centroid[:3]  # Convert back to 3D coordinates

    # Store the normal vector and the transformed centroid
    normals.append(normal_vector)
    points.append(transformed_centroid)

    # Assign color
    color = generate_color(i)
    # print(np.shape(np_points))
    point_cloud.colors = o3d.utility.Vector3dVector([color] * len(np_points))

    # Combine
    combined_cloud += point_cloud


# Let's try to remove outliers 
# Apply statistical outlier removal
cl, ind = combined_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# cl is the inlier cloud, and ind are the indices of inliers
inlier_cloud = combined_cloud.select_by_index(ind)
outlier_cloud = combined_cloud.select_by_index(ind, invert=True)

# Visualize inliers and outliers
# o3d.visualization.draw_geometries([inlier_cloud])
# o3d.visualization.draw_geometries([outlier_cloud])

combined_cloud = inlier_cloud
# # Save or visualize combined cloud
o3d.io.write_point_cloud("combined_cloud.ply", combined_cloud)
# o3d.visualization.draw_geometries([combined_cloud])



####### visualize normals
# Create an Open3D point cloud for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# # Create a point cloud from centroids
# centroid_cloud = o3d.geometry.PointCloud()
# centroid_cloud.points = o3d.utility.Vector3dVector(points)

# # Apply statistical outlier removal
# inlier_cloud, ind = centroid_cloud.remove_statistical_outlier(nb_neighbors=2, std_ratio=.05)

# # Filter centroids and normals based on inliers
# filtered_centroids = np.array(points)[ind]
# filtered_normals = np.array(normals)[ind]
# centroids_cloud = filtered_centroids
# normals = filtered_normals

# # filtered pcd
# filtered_pcd = o3d.geometry.PointCloud()
# filtered_pcd.points = o3d.utility.Vector3dVector(filtered_centroids)

filtered_centroids = points
filtered_normals = normals
filtered_pcd = pcd

# Create line set for normals visualization
lines = []
colors = []
for i in range(len(normals)):
    lines.append([i, i + len(normals)]) # Lines from point to point + normal
    colors.append([1, 0, 0]) # Red color for lines

# Add the normal vectors as points
normals_p = normals
normals = .005*np.array(normals_p) + np.array(filtered_centroids) # Translate normal vectors to start from centroids
normals_reverse = -.005*np.array(normals_p) + np.array(filtered_centroids) # Translate normal vectors to start from centroids
# normals = normals_reverse

all_points = np.vstack((filtered_centroids, normals)) # Combine points and normals
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

# Visualize
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])

o3d.visualization.draw_geometries([filtered_pcd, line_set, combined_cloud, coordinate_frame])

# 

# export the point cloud to a file as a numpy array
np_points = np.asarray(combined_cloud.points)
np.save('output/combined_cloud.npy', np_points)

# now save normals 
np_normals = np.asarray(normals)
np.save('output/combined_cloud_normals.npy', np_normals)

# now save normals_reverse
np_normals_reverse = np.asarray(normals_reverse)
np.save('output/combined_cloud_normals_reverse.npy', np_normals_reverse)