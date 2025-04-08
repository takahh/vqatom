import numpy as np
import matplotlib.pyplot as plt

# Simulate: For each k, you have
#   cluster_centroids_list[k] = (k, D)
#   cluster_assignments_list[k] = (N,)
# And latent_vectors = (N, D)


def calc_wcss(latents, centroid):
    # Compute WCSS manually
    wcss = 0.0
    diff = latents - centroid
    wcss += np.dot(diff, diff)  # ||x - c||²
    return wcss

exp_list = ['1000_64', '1000_128', '1000_256', '1500_64', '1500_128', '1500_256', '2000_64', '2000_128', '2000_256']


def get_data(exp_name):
    "/Users/taka/Documents/data_for_elbow/1000_64/cb_1.npz"
    cb = np.load(f"/Users/taka/Documents/data_for_elbow/{exp_name}/cb_1.npz")['arr_0']
    latent = np.load(f"/Users/taka/Documents/data_for_elbow/{exp_name}/latents_mol_1.npz")['arr_0']
    return latent, cb

wcss_list = []
for exp in exp_list:
    latent, cb = get_data(exp)
    wcss_list.append(calc_wcss(latent, cb))


# Plot
plt.plot(exp_list, wcss_list, marker='o')
plt.title('Elbow Method using Precomputed Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
