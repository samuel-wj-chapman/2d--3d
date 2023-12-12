import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 2D convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_points=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_points * 3)  # 3 for (x, y, z) coordinates for each point
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(batch_size, self.num_points, 3)
        return x

class Generator(nn.Module):
    def __init__(self, num_points=1024):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the generator
generator = Generator()
print(generator)
