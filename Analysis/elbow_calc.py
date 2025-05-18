import numpy as np
import matplotlib.pyplot as plt

# Simulate: For each k, you have
#   cluster_centroids_list[k] = (k, D)
#   cluster_assignments_list[k] = (N,)
# And latent_vectors = (N, D)


def calc_wcss(latents, centroid):
    # Compute WCSS manually
    wcss = 0.0
    print(f"latents: {latents.shape}, centroid: {centroid.shape}")
    for i in range(len(latents)):
        diff = latents[i] - centroid[i]
        wcss += np.dot(diff, diff)  # ||x - c||²
    return wcss

exp_list = ['15000_64', '10000_64', '5000_64', '15000_128', '10000_128', '5000_128', '15000_32', '10000_32', '5000_32']
non_list = ['5000_16', '20000_128']

def run():
    used_exp_list = []
    wcss_list = []
    for dim in [16, 32, 64, 128]:
        for cb_size in [5000, 10000, 15000, 20000]:
            exp_name = f"{cb_size}_{dim}"
            if exp_name in non_list:
                continue
            cb = np.load(f"/Users/taka/Documents/data_for_elbow/{exp_name}/quantized_1.npz")['arr_0']
            latent = np.load(f"/Users/taka/Documents/data_for_elbow/{exp_name}/latents_mol_1.npz")['arr_0']
            print(f"cb {cb.shape}, latent {latent.shape}")
            wcss_list.append(calc_wcss(latent, cb))
            used_exp_list.append(exp_name)

    return wcss_list, used_exp_list


wcss_list, used_exp_list = run()


# Plot

plt.figure(dpi=350)
plt.plot(used_exp_list, wcss_list, marker='o')
plt.title('Elbow Method using Precomputed Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.xticks(fontsize=7, rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()
