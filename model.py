import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import resnet50
from dataloader import ImageSTLDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
from tqdm import tqdm


import torch
from torchvision.models import resnet50
import time

def download_model_with_retry(model_func, attempts=3, sleep_interval=5):
    for attempt in range(attempts):
        try:
            print(f"Attempt {attempt + 1} of {attempts}")
            return model_func(pretrained=True)
        except RuntimeError as e:
            print(f"Failed to download on attempt {attempt + 1}: {e}")
            if attempt < attempts - 1:
                time.sleep(sleep_interval)
            else:
                raise



class ImageInterpretationModel(nn.Module):
    def __init__(self):
        super(ImageInterpretationModel, self).__init__()
        # Load a pre-trained ResNet-50 model
        #resnet = resnet50(pretrained=True)
        #resnet = download_model_with_retry(resnet50, attempts=10, sleep_interval=5)
        resnet = resnet50(pretrained=False)

        weights_path = './resnet50-0676ba61.pth'  # Update with your path
        if os.path.exists(weights_path):
            resnet.load_state_dict(torch.load(weights_path))
        else:
            raise RuntimeError(f"Failed to load weights from {weights_path}")
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # Forward pass through the network
        x = self.features(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        return x


class PointCloudGenerationModel(nn.Module):
    def __init__(self, num_points=1024):
        super(PointCloudGenerationModel, self).__init__()
        self.num_points = num_points
        
        # Example architecture: a series of fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_points * 3)  # x, y, z for each point
        )
        
    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(-1, self.num_points, 3)  # Reshape to point cloud format
        return x

class ImageToPointCloud(nn.Module):
    def __init__(self):
        super(ImageToPointCloud, self).__init__()
        self.image_model = ImageInterpretationModel()
        self.point_cloud_model = PointCloudGenerationModel()

    def forward(self, x):
        x = self.image_model(x)
        x = self.point_cloud_model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_points=1024):
        super(Discriminator, self).__init__()
        self.num_points = num_points

        # A simple discriminator model
        self.model = nn.Sequential(
            nn.Linear(num_points * 3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the point cloud
        return self.model(x)


from stl import mesh

def save_point_cloud_as_stl(point_cloud, filename):
    # Simple example: creating a mesh assuming point_cloud is Nx3
    # This is a placeholder. In practice, you need a proper algorithm to create a mesh.
    data = np.zeros(len(point_cloud), dtype=mesh.Mesh.dtype)
    for i, p in enumerate(point_cloud):
        data['vectors'][i] = np.array([p, p, p])  # Dummy triangle, replace with real logic

    m = mesh.Mesh(data)
    m.save(filename)



image_folder = './images'
stl_folder = './stldataset'

# Define any required transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    # Add any additional transformations as needed
])

dataset = ImageSTLDataset(image_folder=image_folder, stl_folder=stl_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize models
num_epochs = 10 
save_interval = 5  # Save every 5 epochs
generator = ImageToPointCloud().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# Define a fixed sample for consistent evaluation (adjust as needed)
fixed_sample = torch.randn(4, 3, 224, 224).to(device)

# Training Loop
import torch.nn.functional as F

# Training Loop
# Training Loop
for epoch in range(num_epochs):
    total_d_loss, total_g_loss = 0, 0

    dataloader_with_progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for images, real_point_clouds in dataloader_with_progress:
        # Move data to the device (GPU or CPU)
        #print('here')
        images = images.to(device).float() 
        real_point_clouds = real_point_clouds.to(device).float() 
        
        # Real labels (1s) and fake labels (0s)
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        
        # Real point clouds
        real_loss = F.binary_cross_entropy(discriminator(real_point_clouds), real_labels)
        
        # Generate fake point clouds
        fake_point_clouds = generator(images)

        
        # Fake point clouds
        fake_loss = F.binary_cross_entropy(discriminator(fake_point_clouds.detach()), fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        total_d_loss += d_loss.item()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = F.binary_cross_entropy(discriminator(fake_point_clouds), real_labels)
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()
        dataloader_with_progress.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

    # Save generated point clouds periodically
    if (epoch + 1) % save_interval == 0:
        with torch.no_grad():
            generator.eval()  # Set the generator to eval mode for sample generation
            sample_point_cloud = generator(fixed_sample).cpu().numpy()  # Generate a sample
            save_point_cloud_as_stl(sample_point_cloud[0], f"generated_epoch_{epoch+1}.stl")
            generator.train()  # Set the generator back to train mode

    avg_d_loss = total_d_loss / len(dataloader)
    avg_g_loss = total_g_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average D Loss: {avg_d_loss}, Average G Loss: {avg_g_loss}")


