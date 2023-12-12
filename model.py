import torch
import torch.nn as nn
from torchvision.models import resnet50


class ImageInterpretationModel(nn.Module):
    def __init__(self):
        super(ImageInterpretationModel, self).__init__()
        # Load a pre-trained ResNet-50 model
        resnet = resnet50(pretrained=True)
        
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

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize models
generator = ImageToPointCloud().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    for images, real_point_clouds in dataloader:
        # Generate point clouds
        fake_point_clouds = generator(images)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)
        real_loss = nn.BCELoss()(discriminator(real_point_clouds), real_labels)
        fake_loss = nn.BCELoss()(discriminator(fake_point_clouds.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = nn.BCELoss()(discriminator(fake_point_clouds), real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
