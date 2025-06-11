import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import cuml
import cudf
from cuml.manifold import UMAP as cumlUMAP
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "/40000_16/"
# DATA_PATH = "/Users/taka/Documents/vqatom_train_output/bothloss_40000_16/"
DIMENSION = 16
# BATCH_SIZE = 8000
EPOCH_START = 1
SAMPLE_LATENT = 3000000
EPOCH_END = EPOCH_START + 1
# MODE = "umap"  # Choose between "tsne" and "umap"
MODE = "tsne"  # Choose between "tsne" and "umap"

sns.set(style="white", rc={"figure.figsize": (8, 8)})


def load_npz_array(filename):
    """Load and return the array from a .npz file."""
    arr = np.load(filename, allow_pickle=True)
    arr = arr["arr_0"]
    return np.squeeze(arr)

def load_npz_array_multi(filename):
    """Load and return the array from a .npz file."""
    arr0 = np.load(filename, allow_pickle=True)
    arr_all = []
    for names in arr0.files:
        arr = arr0[names].tolist()
        arr_all.extend(arr)
    final_arr = np.array(arr_all)
    return np.squeeze(final_arr)

def filter_zero_distance_neighbors(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # exclude distance to self
    mask = np.any(distances > 0, axis=1)
    return X[mask], mask

def run_dbscan_dedup(X, eps=1e-8):
    scaled = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=2).fit(scaled)
    labels = db.labels_

    mask = labels == -1
    for lbl in set(labels):
        if lbl == -1:
            continue
        idx_lbl = np.where(labels == lbl)[0]
        mask[idx_lbl[1:]] = False
    print(f"[INFO] DBSCAN deduplicated: {np.sum(~mask)} removed")
    return X[mask], mask

def plot_latents(latent_arr, cb_arr, epoch, save_path):
    n_neighbors = 5
    min_dist = 0.1

    print(f"[INFO] Starting UMAP for epoch {epoch}")
    cb_arr_sample = cb_arr[:3000]
    latent_sample = latent_arr[:19000]
    combined_arr = np.concatenate([cb_arr_sample, latent_sample], axis=0)
    labels = np.array([0] * len(cb_arr_sample) + [1] * len(latent_sample))

    print(f"[INFO] Combined array shape: {combined_arr.shape}")

    combined_arr, mask_dbscan = run_dbscan_dedup(combined_arr)
    labels = labels[mask_dbscan]

    combined_arr, mask_nbr = filter_zero_distance_neighbors(combined_arr, n_neighbors)
    labels = labels[mask_nbr]

    print(f"[INFO] After filtering: {combined_arr.shape[0]} samples")

    # Convert to cuDF DataFrame for GPU
    df = cudf.DataFrame(combined_arr.astype(np.float32))

    umap = cumlUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        verbose=True
    )
    emb = umap.fit_transform(df).to_numpy()

    # Plotting
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette=["red", "blue"], s=5, alpha=0.7)
    plt.title(f"GPU UMAP at Epoch {epoch}")
    plt.axis("off")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"latent_plot_epoch{epoch}.png"))
    plt.close()
    print(f"[INFO] Saved plot to {save_path}/latent_plot_epoch{epoch}.png")

def main():
    for epoch in [1]:  # add more epochs if needed
        print(f"Processing epoch {epoch}")
        with open(f"latent_epoch{epoch}.pkl", "rb") as f:
            latent_arr = pickle.load(f)
        with open(f"cb_epoch{epoch}.pkl", "rb") as f:
            cb_arr = pickle.load(f)

        print("latent_arr.shape:", latent_arr.shape)
        print("cb_arr.shape:", cb_arr.shape)

        plot_latents(latent_arr, cb_arr, epoch, save_path="plots")


def process_epoch(epoch):
    """Load data and plot visualization for a single epoch."""
    codebook_file = f"{DATA_PATH}used_cb_vectors.npz"
    # codebook_file = "/Users/taka/PycharmProjects/vqatom/Analysis/kmeans_centers.npy"
    # codebook_file = '/Users/taka/Documents/best_codebook_40000_16/init_codebook_1.npz'
    latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"

    cb_arr = load_npz_array(codebook_file)
    latent_arr = load_npz_array_multi(latent_file)
    print("latent_arr.shape")
    print(latent_arr.shape)
    latent_arr = latent_arr[:SAMPLE_LATENT]
    print(cb_arr.shape)

    cb_arr = np.unique(cb_arr, axis=0).reshape(-1, DIMENSION)
    cb_size = cb_arr.shape[0]

    plot_latents(latent_arr, cb_arr, epoch, save_path="plots")


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch)


if __name__ == '__main__':
    main()
