import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import cudf
from cuml.manifold import UMAP as cumlUMAP
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Configuration
DATA_PATH = "/40000_16/"
DIMENSION = 16
EPOCH_START = 1
EPOCH_END = EPOCH_START + 1
SAMPLE_LATENT = 3000000
sns.set(style="white", rc={"figure.figsize": (8, 8)})


def load_npz_array(filename):
    arr = np.load(filename, allow_pickle=True)
    return np.squeeze(arr["arr_0"])


def load_npz_array_multi(filename):
    arr0 = np.load(filename, allow_pickle=True)
    all_arr = [arr0[name].tolist() for name in arr0.files]
    return np.squeeze(np.array(sum(all_arr, [])))


def filter_zero_distance_neighbors(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return X[np.any(distances[:, 1:] > 0, axis=1)]


def run_dbscan_dedup(X, eps=1e-8):
    scaled = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=2).fit(scaled)
    labels = db.labels_

    mask = labels == -1
    for lbl in set(labels):
        if lbl != -1:
            mask[np.where(labels == lbl)[0][1:]] = False
    print(f"[INFO] DBSCAN deduplicated: {np.sum(~mask)} removed")
    return X[mask], mask


def check_duplicates(arr):
    rounded = np.round(arr, decimals=6)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    return np.sum(counts > 1)

import os
import cupy as cp
import numpy as np
import cudf
import seaborn as sns
import matplotlib.pyplot as plt
from cuml.manifold import UMAP as cumlUMAP

from sklearn.metrics import pairwise_distances

def filter_close_points(arr, epsilon=1e-6):
    """Remove rows that are too close to all others."""
    dists = pairwise_distances(arr, metric="euclidean")
    mask = ~(np.all(dists < epsilon, axis=1))
    return arr[mask]

def plot_latents(latent_arr, cb_arr, epoch, save_path):
    n_neighbors, min_dist = 5, 0.1
    print(f"[INFO] Starting UMAP for epoch {epoch}")

    cb_sample, latent_sample = cb_arr[:3000], latent_arr[:19000]
    combined = np.concatenate([cb_sample, latent_sample], axis=0)
    labels = np.array([0] * len(cb_sample) + [1] * len(latent_sample))

    print(f"[INFO] Combined array shape: {combined.shape}")

    combined, db_mask = run_dbscan_dedup(combined)
    labels = labels[db_mask]

    combined = filter_zero_distance_neighbors(combined, n_neighbors)
    labels = labels[:len(combined)]

    print(f"[INFO] After filtering: {combined.shape[0]} samples")

    df = cudf.DataFrame(combined.astype(np.float32))
    arr = df.to_numpy()

    # Drop zero rows and duplicates
    df = df[~(df == 0).all(axis=1)]
    df = df.drop_duplicates()

    # Further filter: remove rows too close to all others
    arr = df.to_numpy()
    arr = filter_close_points(arr, epsilon=1e-6)

    print(f"Zero rows: {np.sum(np.all(arr == 0, axis=1))}")
    print(f"Duplicates: {check_duplicates(arr)}")

    # Convert to GPU (CuPy) array and apply stronger jitter
    df_cp = cp.asarray(arr)
    noise = cp.random.normal(0, 1e-4, df_cp.shape).astype(cp.float32)
    df_jittered_cp = df_cp + noise

    # Back to cuDF for cuML UMAP
    df_jittered = cudf.DataFrame.from_records(cp.asnumpy(df_jittered_cp))
    from sklearn.neighbors import NearestNeighbors

    def filter_isolated_points(X, n_neighbors=5, epsilon=1e-6):
        """
        Remove rows that have fewer than n_neighbors neighbors with distance > epsilon.
        """
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        # Exclude self-distance (0.0) at index 0
        valid = (distances[:, 1:] > epsilon).sum(axis=1) >= n_neighbors
        print(f"[FILTER] Removing {np.sum(~valid)} isolated or nearly-identical points")
        return X[valid]

    # Before converting to GPU:
    df_jittered = filter_close_points(df_jittered, epsilon=1e-6)
    df_jittered = filter_isolated_points(df_jittered, n_neighbors=5, epsilon=1e-6)

    umap = cumlUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42, verbose=True)
    embedding = umap.fit_transform(df_jittered).to_numpy()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette=["red", "blue"], s=5, alpha=0.7)
    plt.title(f"GPU UMAP at Epoch {epoch}")
    plt.axis("off")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"latent_plot_epoch{epoch}.png"))
    plt.close()
    print(f"[INFO] Saved plot to {save_path}/latent_plot_epoch{epoch}.png")


def process_epoch(epoch):
    cb_arr = load_npz_array(f"{DATA_PATH}used_cb_vectors.npz")
    print("loaded...cb")
    latent_arr = load_npz_array_multi(f"{DATA_PATH}latents_all_{epoch}.npz")[:SAMPLE_LATENT]
    print("loaded...latents")
    cb_arr = np.unique(cb_arr, axis=0).reshape(-1, DIMENSION)
    print(f"latent_arr.shape: {latent_arr.shape}\ncb_arr.shape: {cb_arr.shape}")

    plot_latents(latent_arr, cb_arr, epoch, save_path="plots")


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch)


if __name__ == '__main__':
    main()
