import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def wasserstein_distance(a, b):
    """
    Compute the Wasserstein distance (Earth Mover's Distance) between two point clouds.
    
    Args:
        a (torch.Tensor): First point cloud of shape (batch_size, num_points, points_dim).
        b (torch.Tensor): Second point cloud of shape (batch_size, num_points, points_dim).
    
    Returns:
        tuple: Two tensors containing the Wasserstein distances for each batch.
    """
    a_np = a.numpy()
    b_np = b.numpy()
    batch_size, num_points, points_dim = a_np.shape
    wasserstein_distances_a_to_b = torch.zeros((batch_size, num_points))
    wasserstein_distances_b_to_a = torch.zeros((batch_size, num_points))

    for i in range(batch_size):
        # Extract the point clouds for the current batch
        point_cloud_1 = a_np[i]
        point_cloud_2 = b_np[i]

        # Compute the pairwise distance matrix
        distance_matrix = cdist(point_cloud_1, point_cloud_2, metric='euclidean')
        
        # Solve the optimal transport problem
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # Compute the Wasserstein distance for the current batch
        wasserstein_dist_a_to_b = distance_matrix[row_ind, col_ind]
        
        # Store the distances in the results tensors
        wasserstein_distances_a_to_b[i] = torch.tensor(wasserstein_dist_a_to_b)


    return wasserstein_distances_a_to_b

# Example usage
if __name__ == "__main__":
    point_cloud_1 = torch.rand(1, 2048, 3)  # Batch of 1 point cloud with 2048 points in 3D space
    point_cloud_2 = torch.rand(1, 2048, 3)  # Another batch of 1 point cloud with 2048 points in 3D space

    min_dist_a_to_b = wasserstein_distance(point_cloud_1, point_cloud_2)
    print("Minimum distances from each point in a to b:", min_dist_a_to_b)

