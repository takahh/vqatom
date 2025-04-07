import numpy as np
import matplotlib.pyplot as plt

# Simulate: For each k, you have
#   cluster_centroids_list[k] = (k, D)
#   cluster_assignments_list[k] = (N,)
# And latent_vectors = (N, D)

wcss_list = []

for k in range(1, max_k + 1):
    centroids = cluster_centroids_list[k]
    assignments = cluster_assignments_list[k]

    # Compute WCSS manually
    wcss = 0.0
    for i in range(len(latent_vectors)):
        centroid = centroids[assignments[i]]
        diff = latent_vectors[i] - centroid
        wcss += np.dot(diff, diff)  # ||x - c||²

    wcss_list.append(wcss)

# Plot
plt.plot(range(1, max_k + 1), wcss_list, marker='o')
plt.title('Elbow Method using Precomputed Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
