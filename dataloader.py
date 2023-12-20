import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from stl import Mesh # or any other preferred library for handling STL files
import numpy as np
from PIL import Image
import os
import random

class ImageSTLDataset(Dataset):
    def __init__(self, image_folder, stl_folder, processed_stl_folder, transform=None):
        self.image_folder = image_folder
        self.stl_folder = stl_folder
        self.processed_stl_folder = processed_stl_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Extract object name from image filename
        object_name = image_name.split('_angle')[0]
        processed_point_cloud_path = os.path.join(self.processed_stl_folder, f"{object_name}.pt")

        if os.path.exists(processed_point_cloud_path):
            # Load preprocessed point cloud
            point_cloud = torch.load(processed_point_cloud_path)
        else:
            # Process STL file
            stl_path = os.path.join(self.stl_folder, f"{object_name}.stl")
            point_cloud = torch.tensor(self.stl_to_pointcloud(stl_path)).float()

        
        point_cloud = self.apply_random_rotation(point_cloud)

        return image, point_cloud




    def stl_to_pointcloud(self, stl_path, target_num_points=1024):
        # Load STL and convert to point cloud
        your_mesh = Mesh.from_file(stl_path)
        points = np.array(your_mesh.points).reshape(-1, 3)
        #points = self.apply_random_rotation(points)

        current_num_points = len(points)

        if current_num_points > target_num_points:
            # Downsample: Randomly select 'target_num_points' from the point cloud
            points = points[np.random.choice(current_num_points, target_num_points, replace=False)]
        elif current_num_points < target_num_points:
            # Upsample: Randomly duplicate points until we reach 'target_num_points'
            extra_points_idx = np.random.choice(current_num_points, target_num_points - current_num_points, replace=True)
            extra_points = points[extra_points_idx]
            points = np.vstack([points, extra_points])

        points_tensor = torch.from_numpy(points).float()

        return points



    def apply_random_rotation(self, points_tensor):
        # Create a random rotation matrix
        theta = np.radians(random.uniform(0, 360))
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])

        # Apply rotation to all points
        rotated_points = torch.matmul(points_tensor, torch.tensor(rotation_matrix).float())
        return rotated_points
