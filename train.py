# Hyperparameters
lr = 0.0002
epochs = 1000
batch_size = 32
real_label = 1
fake_label = 0

# Initialize Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Assume dataloader is a DataLoader object that provides batches of real point clouds
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, real_point_clouds in enumerate(dataloader):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizerD.zero_grad()
        
        # Real point clouds
        output = discriminator(real_point_clouds)
        errD_real = bce_loss(output, torch.full((batch_size, 1), real_label))
        errD_real.backward()
        
        # Fake point clouds
        noise = torch.randn(batch_size, 3, 64, 64)
        fake_point_clouds = generator(noise)
        output = discriminator(fake_point_clouds.detach())
        errD_fake = bce_loss(output, torch.full((batch_size, 1), fake_label))
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        
        optimizerG.zero_grad()
        
        output = discriminator(fake_point_clouds)
        errG = bce_loss(output, torch.full((batch_size, 1), real_label))  # We want the discriminator to believe the fake point cloud is real
        
        # Reconstruction loss
        recon_loss = chamfer_distance(real_point_clouds, fake_point_clouds)
        
        # Combine losses
        err_total = errG + recon_loss
        err_total.backward()
        optimizerG.step()

        print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {errD.item()}] [G loss: {errG.item()}, recon loss: {recon_loss.item()}]")
