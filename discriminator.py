class Discriminator(nn.Module):
    def __init__(self, num_points=1024):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_points * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

discriminator = Discriminator()
print(discriminator)
