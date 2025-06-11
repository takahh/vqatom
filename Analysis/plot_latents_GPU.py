import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cuml.manifold import UMAP as cumlUMAP
from umap import UMAP as cpuUMAP
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

sns.set(style="white", rc={"figure.figsize": (8, 8)})

def filter_zero_distance_neighbors(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Exclude distance to self
    mask = ~np.all(distances == 0, axis=1)
    return X[mask], mask

def run_dbscan_dedup(X, eps=1e-8):
    scaled = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=2).fit(scaled)
    labels = db.labels_

    # keep first point of each cluster, remove rest
    mask = labels == -1
    for lbl in set(labels):
        if lbl == -1:
            continue
        idx_lbl = np.where(labels == lbl)[0]
        mask[idx_lbl[1:]] = False
    print(f"[INFO] DBSCAN deduplicated: {np.sum(~mask)} near-duplicates removed")
    return X[mask], mask

def plot_latents(latent_arr, cb_arr, epoch, save_path):
    n_neighbors = 15
    min_dist = 0.1

    print(f"[INFO] Starting UMAP for epoch {epoch}")

    cb_arr_sample = cb_arr[:3000]
    latent_sample = latent_arr[:19000]
    combined_arr = np.concatenate([cb_arr_sample, latent_sample], axis=0)
    labels = np.array([0] * len(cb_arr_sample) + [1] * len(latent_sample))

    print(f"[INFO] Combined array shape: {combined_arr.shape}")

    # Step 1: DBSCAN deduplication
    combined_arr, mask_dbscan = run_dbscan_dedup(combined_arr)
    labels = labels[mask_dbscan]

    # Step 2: Remove rows with all zero-distance neighbors
    combined_arr, mask_nbr = filter_zero_distance_neighbors(combined_arr, n_neighbors)
    labels = labels[mask_nbr]

    print(f"[INFO] Remaining after filtering zero-distance neighbors: {combined_arr.shape[0]}")

    # Step 3: Try cuML UMAP
    try:
        umap = cumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        emb = umap.fit_transform(combined_arr)
        print("[INFO] cuML UMAP succeeded.")
    except Exception as e:
        print(f"[WARN] cuML UMAP failed: {e}")
        print("[INFO] Falling back to CPU UMAP...")
        umap = cpuUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        emb = umap.fit_transform(combined_arr)

    # Step 4: Plot
    plt.figure(figsize=(8, 8))
    palette = ["red", "blue"]
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette=palette, s=5, alpha=0.7)
    plt.title(f"UMAP at Epoch {epoch}")
    plt.legend(title="Type", labels=["CB", "Latent"])
    plt.axis("off")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"latent_plot_epoch{epoch}.png"))
    plt.close()
    print(f"[INFO] Plot saved to {save_path}/latent_plot_epoch{epoch}.png")

def main():
    for epoch in [1]:  # Add more if needed
        print(f"Processing epoch {epoch}")
        latent_path = f"latent_epoch{epoch}.pkl"
        cb_path = f"cb_epoch{epoch}.pkl"

        with open(latent_path, "rb") as f:
            latent_arr = pickle.load(f)

        with open(cb_path, "rb") as f:
            cb_arr = pickle.load(f)

        print("latent_arr.shape")
        print(latent_arr.shape)
        print(cb_arr.shape)

        save_path = "plots"
        plot_latents(latent_arr, cb_arr, epoch, save_path)

if __name__ == "__main__":
    main()
