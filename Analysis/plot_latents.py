import os
from pickletools import markobject

import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

DATA_PATH = "/Users/taka/Downloads/tmp/40000_16/"
SAMPLES = 500000
# DATA_PATH = "/"
DIMENSION = 16
N_NEIGHBORS = 2
MIN_DIST = 0.01
SPREAD = 1
# BATCH_SIZE = 8000
EPOCH_START = 2
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
    print(arr0.files)
    for names in arr0.files:
        print(arr0[names].shape)
        arr = arr0[names].tolist()
        arr_all.extend(arr)
    final_arr = np.array(arr_all)
    return np.squeeze(final_arr)

def plot_tsne(cb_arr, latent_arr, epoch, perplexity, cb_size):
    title = f"T-SNE: perplex {perplexity}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"
    tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, n_iter=250)
    print("fitting start")
    embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_arr), axis=0))
    print("fitting done")
    for zoom in [50, 20, 15, 10, 7, 5, 3, 2]:
        cb_emb = embedding[:cb_size]
        latent_emb = embedding[cb_size:cb_size]
        x_range = np.percentile(cb_emb[:, 0], [50 - zoom, 50 + zoom])
        y_range = np.percentile(cb_emb[:, 1], [50 - zoom, 50 + zoom])
        zoom = float(50/int(zoom))

        # Mask both latent and cb to zoom-in range
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
        for i in range(2):
            plt.figure(figsize=(10, 8))
            plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=20, c='black', alpha=0.6)
            # plt.hist2d(
            #     zoomed_latent[:, 0], zoomed_latent[:, 1],
            #     bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            #     cmap="Greys"
            # )
            plt.xlim(x_range)
            plt.ylim(y_range)

            if i == 0:
                plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.6, marker='x')
            plt.title(title + f" (Zoomed {zoom}, epoch {epoch})")
            plt.colorbar(label='Density')
            plt.show()

def plot_umap(cb_arr, latent_arr, epoch, n_neighbors, min_dist, cb_size, zoom, samples):
    for N_NEIGHBORS in [10]:
        for zoom in [5]:
            for SPREAD in [1]:
                # print("reducer setup")
                reducer = umap.UMAP(
                    n_neighbors=N_NEIGHBORS,
                    min_dist=MIN_DIST,
                    spread=SPREAD,
                    n_components=2,
                    n_epochs=100,
                    random_state=42
                ).fit(latent_arr)
                # print("reducer setup done")
                latent_emb = reducer.transform(latent_arr)
                # print("latent transform done")
                cb_emb = reducer.transform(cb_arr)
                # print("cb transform done")
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
                title = f"UMAP: n_neighbors {N_NEIGHBORS}, min_dist {MIN_DIST}, \n spread {SPREAD}, zoom {zoom} samples {samples}"

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
                        plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=20, c='red', alpha=0.2, marker='x')
                    plt.xlim(x_range)
                    plt.ylim(y_range)
                    # plt.xlim(-30, 30)
                    # plt.ylim(-30, 30)

                    plt.title(title + " (Zoomed)")
                    # if not os.path.exists(f"/{samples}"):
                    #     os.mkdir(f"/{samples}")
                    plt.savefig(f"./n{N_NEIGHBORS}_s{SPREAD}_z{zoom}_epo_{epoch}_{i}.png")
                    # plt.savefig(f"/{samples}/n{N_NEIGHBORS}_s{SPREAD}_z{zoom}_{i}.png")


def process_epoch(epoch, samples):
    """Load data and plot visualization for a single epoch."""
    # codebook_file = f"{DATA_PATH}used_cb_vectors.npz"
    # codebook_file = "/Users/taka/PycharmProjects/vqatom/Analysis/kmeans_centers.npy"
    # latent_file = f"{DATA_PATH}latents_all_{epoch}.npz"

    codebook_file = f'{DATA_PATH}used_cb_vectors.npz'
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
