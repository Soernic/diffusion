import torch

def distChamfer(a, b):
    """
    Compute the Chamfer distance between two point clouds.
    
    Args:
        a (torch.Tensor): First point cloud of shape (batch_size, num_points, points_dim).
        b (torch.Tensor): Second point cloud of shape (batch_size, num_points, points_dim).
    
    Returns:
        tuple: Two tensors containing the minimum distances from each point in a to b and from each point in b to a.
    """
    x, y = a, b
    bs, num_points, points_dim = x.size()
    
    # Compute pairwise distance matrices
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    
    # Compute the diagonal indices
    diag_ind = torch.arange(0, num_points).to(a).long()
    
    # Compute rx and ry for pairwise distances
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    
    # Compute pairwise distances P
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    
    # Get the minimum distances
    min_dist_a_to_b = P.min(1)[0]
    min_dist_b_to_a = P.min(2)[0]
    
    return min_dist_a_to_b, min_dist_b_to_a

# Example usage
if __name__ == "__main__":
    point_cloud_1 = torch.rand(1, 2048, 3)  # Batch of 2 point clouds with 5 points in 3D space
    point_cloud_2 = torch.rand(1, 2048, 3)  # Another batch of 2 point clouds with 5 points in 3D space

    min_dist_a_to_b, min_dist_b_to_a = distChamfer(point_cloud_1, point_cloud_2)
    print("Minimum distances from each point in a to b:", min_dist_a_to_b)
    print("Minimum distances from each point in b to a:", min_dist_b_to_a)
