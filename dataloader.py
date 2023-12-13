import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from stl import mesh  # or any other preferred library for handling STL files
import numpy as np
from PIL import Image
import os

class ImageSTLDataset(Dataset):
    def __init__(self, image_folder, stl_folder, transform=None):
        self.image_folder = image_folder
        self.stl_folder = stl_folder
        self.transform = transform

        # Assuming image filenames and STL filenames have a matching pattern
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)

        # Load image
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        # Extract object name from image filename
        # Assuming the filename format is 'objectname_angleX_angleY_imageID.png'
        object_name = image_name.split('_')[0]  # Adjust this based on actual filename format
        stl_name = f"{object_name}.stl"  # Construct STL filename
        stl_path = os.path.join(self.stl_folder, stl_name)
        point_cloud = self.stl_to_pointcloud(stl_path)

        return image, point_cloud

    def stl_to_pointcloud(self, stl_path):
        # Implement STL to point cloud conversion
        # This is an example using numpy-stl
        your_mesh = mesh.Mesh.from_file(stl_path)
        points = np.array(your_mesh.points).reshape(-1, 3)
        return points
