# Binary Cross Entropy Loss for GAN training
bce_loss = torch.nn.BCELoss()

# Chamfer Distance for point cloud reconstruction loss
def chamfer_distance(pc1, pc2):
    '''
    Compute the Chamfer Distance between two point clouds pc1 and pc2.
    pc1, pc2: [B, N, 3] and [B, M, 3] tensors
    '''
    B, N, _ = pc1.size()
    _, M, _ = pc2.size()
    
    pc1 = pc1.view(B, N, 1, 3)
    pc2 = pc2.view(B, 1, M, 3)
    
    dist = pc1 - pc2
    dist = torch.norm(dist, dim=3)  # [B, N, M]
    
    min_dist1, _ = dist.min(dim=2)  # [B, N]
    min_dist2, _ = dist.min(dim=1)  # [B, M]
    
    chamfer_dist = min_dist1.mean(dim=1) + min_dist2.mean(dim=1)
    return chamfer_dist.mean()
