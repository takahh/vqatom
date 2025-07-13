import os
from pickletools import markobject

import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf)

# DATA_PATH = "/Users/taka/Documents/1_infer_for_uk_dynamic_epo1/10000_16/"
DATA_PATH = "/Users/taka/Downloads/"
OPATH = "/Users/taka/Documents/"
SAMPLES = 2000000
# DATA_PATH = "/"
DIMENSION = 16
N_NEIGHBORS = 10
MIN_DIST = 0.01
SPREAD = 1
# BATCH_SIZE = 8000
EPOCH_START = 2
EPOCH_END = EPOCH_START + 4
MODE = "umap"  # Choose between "tsne" and "umap"
# MODE = "tsne"  # Choose between "tsne" and "umap"

def load_npz_array(filename):
    """Load and return the array from a .npz file."""
    arr = np.load(filename, allow_pickle=True)
    print(arr)
    arr = arr["embed"]
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(cb_arr, latent_arr, epoch, perplexity, cb_size):
    """
    Plots t-SNE visualization of codebook and latent vectors with progressive zoom.

    Args:
        cb_arr (np.ndarray): Codebook vectors, shape [cb_size, D]
        latent_arr (np.ndarray): Latent vectors, shape [N, D]
        epoch (int): Current epoch (for labeling)
        perplexity (float): t-SNE perplexity
        cb_size (int): Number of codebook vectors (same as cb_arr.shape[0])
    """
    assert cb_arr.shape[0] == cb_size, "cb_size mismatch with cb_arr shape"
    title = f"T-SNE: perplex {perplexity}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, n_iter=250)
    print("Fitting t-SNE...")
    embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_arr), axis=0))
    print("Fitting complete.")

    # Split embeddings
    cb_emb = embedding[:cb_size]
    latent_emb = embedding[cb_size:]

    # Zoom levels (percentile range around median)
    for zoom_percent in [100, 50, 20, 15, 10, 7, 5, 3, 2]:
        x_range = np.percentile(cb_emb[:, 0], [50 - zoom_percent / 2, 50 + zoom_percent / 2])
        y_range = np.percentile(cb_emb[:, 1], [50 - zoom_percent / 2, 50 + zoom_percent / 2])

        # Mask points in zoomed region
        latent_mask = (
            (latent_emb[:, 0] >= x_range[0]) & (latent_emb[:, 0] <= x_range[1]) &
            (latent_emb[:, 1] >= y_range[0]) & (latent_emb[:, 1] <= y_range[1])
        )
        cb_mask = (
            (cb_emb[:, 0] >= x_range[0]) & (cb_emb[:, 0] <= x_range[1]) &
            (cb_emb[:, 1] >= y_range[0]) & (cb_emb[:, 1] <= y_range[1])
        )

        zoomed_latent = latent_emb[latent_mask]
        zoomed_cb = cb_emb[cb_mask]

        print(f"\n[Zoom {zoom_percent}%] Latent pts: {len(zoomed_latent)}, CB pts: {len(zoomed_cb)}")

        bins = 100
        for i in range(2):
            plt.figure(figsize=(10, 8))
            if i == 0:
                # Heatmap of latent density + CB overlay
                plt.hist2d(
                    zoomed_latent[:, 0], zoomed_latent[:, 1],
                    bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
                    cmap="Greys"
                )
                plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.9, marker='x')
            else:
                # Optional: visualize just the scatter points
                plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=5, alpha=0.4, c='black')
                plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.9, marker='x')

            plt.xlim(x_range)
            plt.ylim(y_range)
            plt.title(title + f" | Zoom {zoom_percent}%")
            if i == 0:
                plt.colorbar(label='Density')
            plt.tight_layout()
            plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

def plot_umap(cb_arr, latent_arr, epoch, pca_dim=16):
    # 1. Concatenate for PCA
    combined = np.concatenate((latent_arr, cb_arr), axis=0)
    # combined_pca = PCA(n_components=pca_dim).fit_transform(combined)
    combined_pca = combined

    # Split PCA-transformed latent and codebook
    latent_pca = combined_pca[:latent_arr.shape[0]]
    cb_pca = combined_pca[latent_arr.shape[0]:]

    # Try different UMAP spreads and min_dists
    for zoom in [10]:
        for spread, min_dist in [[1, 0.1]]:
            reducer = umap.UMAP(
                n_neighbors=N_NEIGHBORS,
                min_dist=min_dist,
                spread=spread,
                n_components=2,
                n_epochs=100,
                init='random',
                low_memory=True,
                metric='euclidean',
                verbose=True,
                n_jobs=-1,
                # random_state=42
            ).fit(combined_pca)

            latent_emb = reducer.embedding_[:latent_arr.shape[0]]
            cb_emb = reducer.embedding_[latent_arr.shape[0]:]

            # Zoom window
            lo, hi = max(0, 50 - zoom), min(100, 50 + zoom)
            x_range = np.percentile(cb_emb[:, 0], [lo, hi])
            y_range = np.percentile(cb_emb[:, 1], [lo, hi])

            # Apply zoom masks
            latent_mask = (
                (latent_emb[:, 0] >= x_range[0]) & (latent_emb[:, 0] <= x_range[1]) &
                (latent_emb[:, 1] >= y_range[0]) & (latent_emb[:, 1] <= y_range[1])
            )
            cb_mask = (
                (cb_emb[:, 0] >= x_range[0]) & (cb_emb[:, 0] <= x_range[1]) &
                (cb_emb[:, 1] >= y_range[0]) & (cb_emb[:, 1] <= y_range[1])
            )

            zoomed_latent = latent_emb[latent_mask]
            zoomed_cb = cb_emb[cb_mask]

            # Plotting
            title = f"UMAP: neighbors={N_NEIGHBORS}, min_dist={min_dist}, spread={spread},\n samples {SAMPLES}, zoom={int(50/zoom)}"
            bins = 100
            save_dir = f"{OPATH}/distri_images"
            os.makedirs(save_dir, exist_ok=True)

            for i in range(2):
                plt.figure(figsize=(6, 5))
                plt.hist2d(
                    zoomed_latent[:, 0], zoomed_latent[:, 1],
                    bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
                    cmap="Blues"
                )
                plt.colorbar(label='Density')
                if i == 0:
                    plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=20, c='red', alpha=0.5, marker='x')

                plt.xlim(x_range)
                plt.ylim(y_range)
                plt.title(title + " (Zoomed)")
                fname = f"{save_dir}/n{10}_s{spread}_z{zoom}_mindist{min_dist}_epo{epoch}_{i}.png"
                print(fname)
                plt.savefig(fname)
                plt.show()
                plt.close()


def process_epoch(epoch, samples):
    """Load data and plot visualization for a single epoch."""
    # codebook_file = f"{DATA_PATH}used_cb_vectors.npz"
    # codebook_file = "/Users/taka/PycharmProjects/vqatom/Analysis/kmeans_centers.npy"
    # latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"

    codebook_file = f'{DATA_PATH}naked_embed_{epoch}.npz'
    # latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"
    # codebook_file = f'{DATA_PATH}used_cb_vectors_{epoch}.npz'
    latent_file = f"{DATA_PATH}naked_latent_{epoch}.npz"
    cb_arr = load_npz_array(codebook_file)
    print(cb_arr.shape)
    latent_arr = load_npz_array_multi(latent_file)
    print("latent_arr.shape")
    print(latent_arr.shape)

    # Shuffle
    print("latent_arr.shape before")
    print(latent_arr.shape)
    # np.random.seed(0)  # For reproducibility, optional
    np.random.shuffle(latent_arr)
    latent_arr = latent_arr[:samples]
    print("latent_arr.shape")
    print(latent_arr.shape)
    print("cb_arr.shape")
    print(cb_arr.shape)

    cb_arr = np.unique(cb_arr, axis=0).reshape(-1, DIMENSION)
    cb_size = cb_arr.shape[0]
    print("cb_arr.shape")
    print(cb_arr.shape)

    if MODE == "tsne":
        plot_tsne(cb_arr, latent_arr, epoch, perplexity=10, cb_size=cb_size)
    elif MODE == "umap":
        plot_umap(cb_arr, latent_arr, epoch, 16)
        #cb_arr, latent_arr, epoch, pca_dim=16

def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch, SAMPLES)


if __name__ == '__main__':
    main()
