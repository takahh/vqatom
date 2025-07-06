import os
from pickletools import markobject

import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf)

DATA_PATH = "/Users/taka/Downloads/"
OPATH = "/Users/taka/Documents/"
SAMPLES = 500000
# DATA_PATH = "/"
DIMENSION = 16
N_NEIGHBORS = 2
MIN_DIST = 0.01
SPREAD = 1
# BATCH_SIZE = 8000
EPOCH_START = 10
EPOCH_END = EPOCH_START + 1
MODE = "umap"  # Choose between "tsne" and "umap"
# MODE = "tsne"  # Choose between "tsne" and "umap"

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


def plot_umap(cb_arr, latent_arr, epoch, n_neighbors, min_dist, cb_size, zoom, samples):
    # 1. Concatenate before PCA
    combined = np.concatenate((latent_arr, cb_arr), axis=0)

    # 2. Apply PCA to combined array
    combined_pca = PCA(n_components=DIMENSION).fit_transform(combined)
    latent_pca = combined_pca[:latent_arr.shape[0]]
    cb_pca = combined_pca[latent_arr.shape[0]:]

    # for N_NEIGHBORS in [2, 10, 20, 40]:
    for N_NEIGHBORS in [10]:
        for zoom in [2]:
            for SPREAD, min_dist in [[10, 0], [5, 0.1], [5, 1]]:
                reducer = umap.UMAP(
                    n_neighbors=N_NEIGHBORS,
                    min_dist=min_dist,
                    spread=SPREAD,
                    n_components=2,
                    n_epochs=30,
                    random_state=None,
                    init='random',
                    low_memory=True,
                    verbose=True,
                    metric='euclidean',
                    n_jobs=-1
                ).fit(combined_pca)
                # 3. Separate embeddings
                latent_emb = reducer.embedding_[:latent_arr.shape[0]]
                cb_emb = reducer.embedding_[latent_arr.shape[0]:]

                print("*** reducer setup done")
                # latent_emb = reducer.transform(latent_emb)
                print("*** latent transform done")
                # cb_emb = reducer.transform(cb_emb)
                print("*** cb transform done")
                x_range = np.percentile(cb_emb[:, 0], [50 - zoom, 50 + zoom])
                y_range = np.percentile(cb_emb[:, 1], [50 - zoom, 50 + zoom])

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

                bins = 100
                title = f"UMAP: n_neighbors {N_NEIGHBORS}, min_dist {min_dist}, \n spread {SPREAD}, zoom {zoom} samples {samples}"

                for i in range(2):
                    plt.figure()
                    # plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=3, c='black')
                    plt.hist2d(
                        zoomed_latent[:, 0], zoomed_latent[:, 1],
                        bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
                        cmap="Blues"
                    )
                    plt.colorbar(label='Density')
                    if i == 0:
                        plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=20, c='red', alpha=0.9, marker='x')
                    plt.xlim(x_range)
                    plt.ylim(y_range)
                    # plt.xlim(-30, 30)
                    # plt.ylim(-30, 30)

                    plt.title(title + " (Zoomed)")
                    if not os.path.exists(f"{OPATH}/distri_images/"):
                        os.mkdir(f"{OPATH}/distri_images/")
                    plt.savefig(f"{OPATH}/distri_images/n{N_NEIGHBORS}_s{SPREAD}_z{zoom}_mindist{min_dist}_epo_{epoch}_{i}.png")
                    # plt.savefig(f"/{samples}/n{N_NEIGHBORS}_s{SPREAD}_z{zoom}_{i}.png")


def process_epoch(epoch, samples):
    """Load data and plot visualization for a single epoch."""
    # codebook_file = f"{DATA_PATH}used_cb_vectors.npz"
    # codebook_file = "/Users/taka/PycharmProjects/vqatom/Analysis/kmeans_centers.npy"
    # latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"

    codebook_file = f'{DATA_PATH}used_cb_vectors_{epoch}.npz'
    latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"
    cb_arr = load_npz_array(codebook_file)
    print(cb_arr.shape)
    latent_arr = load_npz_array_multi(latent_file)
    print("latent_arr.shape")
    print(latent_arr.shape)
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
        # for zoom in [50, 20, 15, 10, 7, 5, 3, 2]:
        for zoom in [10]:
            plot_umap(cb_arr, latent_arr, epoch, n_neighbors=10, min_dist=1.0, cb_size=cb_size, zoom=zoom, samples=samples)


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch, SAMPLES)


if __name__ == '__main__':
    main()
