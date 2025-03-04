import torch

# Example 2D tensor
diff_feat_same_cluster_dist = torch.tensor([
    [0.0, 0.5, 0.0],
    [0.2, 0.0, 0.3],
    [0.0, 0.4, 0.0]
])

# Extract non-zero values
non_zero_values = diff_feat_same_cluster_dist[diff_feat_same_cluster_dist != 0]
print("Non-zero values:", non_zero_values)

# Get indices and values
indices = torch.nonzero(diff_feat_same_cluster_dist, as_tuple=True)
values = diff_feat_same_cluster_dist[indices]
print("Indices:", indices)
print("Values at indices:", values)
