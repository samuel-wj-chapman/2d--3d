import numpy as np
import os
from stl import Mesh
import torch

def preprocess_stl_to_pointcloud(stl_path, target_num_points=1024):
    your_mesh = Mesh.from_file(stl_path)
    points = np.array(your_mesh.points).reshape(-1, 3)

    # Downsampling or Upsampling
    current_num_points = len(points)
    if current_num_points > target_num_points:
        points = points[np.random.choice(current_num_points, target_num_points, replace=False)]
    elif current_num_points < target_num_points:
        extra_points_idx = np.random.choice(current_num_points, target_num_points - current_num_points, replace=True)
        extra_points = points[extra_points_idx]
        points = np.vstack([points, extra_points])

    return points

# Path settings
stl_directory = 'path/to/stldirectory'
output_directory = 'path/to/outputdirectory'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each STL file
for filename in os.listdir(stl_directory):
    if filename.endswith('.stl'):
        stl_path = os.path.join(stl_directory, filename)
        point_cloud = preprocess_stl_to_pointcloud(stl_path)
        output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.pt')
        torch.save(torch.tensor(point_cloud).float(), output_path)

print("All STL files have been processed and saved.")