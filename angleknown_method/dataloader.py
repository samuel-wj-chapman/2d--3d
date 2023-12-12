import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def get_angles_from_filename(filename):
    # Split by underscore and get the angles, remove the extension and convert to float
    angles = filename.split('_')[-3:-1]
    angleX, angleY, angleZ = map(float, angles)
    return angleX, angleY, angleZ

def rotate_point_cloud(points, angleX, angleY, angleZ):
    """ Rotate the point cloud around all three axes """
    thetaX = np.radians(angleX)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(thetaX), -np.sin(thetaX)],
        [0, np.sin(thetaX), np.cos(thetaX)]
    ])

    thetaY = np.radians(angleY)
    rotation_matrix_y = np.array([
        [np.cos(thetaY), 0, np.sin(thetaY)],
        [0, 1, 0],
        [-np.sin(thetaY), 0, np.cos(thetaY)]
    ])

    thetaZ = np.radians(angleZ)
    rotation_matrix_z = np.array([
        [np.cos(thetaZ), -np.sin(thetaZ), 0],
        [np.sin(thetaZ), np.cos(thetaZ), 0],
        [0, 0, 1]
    ])

    rotated_points = points.dot(rotation_matrix_x).dot(rotation_matrix_y).dot(rotation_matrix_z)
    return rotated_points




class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root_dir, self.folders[idx])
            
        # Convert STL to point cloud
        stl_file = [f for f in os.listdir(folder_path) if f.endswith('.stl')][0]
        point_cloud = stl_to_pointcloud(os.path.join(folder_path, stl_file))
            
        # Load images and rotate the point cloud based on the angles from the image filename
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images = []
        for img in image_files:
            angleX, angleY, angleZ = get_angles_from_filename(img)
            rotated_point_cloud = rotate_point_cloud(point_cloud, angleX, angleY, angleZ)
            
            image = Image.open(os.path.join(folder_path, img)).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            images.append((image, rotated_point_cloud))
                
        return images  # Returns a list of (image, point cloud) pairs

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = CustomDataset(root_dir='path_to_dataset_directory', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)