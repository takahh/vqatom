DATA_PATH = "/40000_16/"

import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

DIMENSION = 16
N_NEIGHBORS = 30
MIN_DIST = 0.1
EPOCH_START = 1
SAMPLE_LATENT = 8700000
PERPLEXITY = 1
EPOCH_END = EPOCH_START + 1
# MODE = "umap"  # Choose between "tsne" and "umap"
MODE = "tsne"  # Choose between "tsne" and "umap"

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

def plot_tsne(cb_arr, latent_arr, latent_to_fit, epoch, perplexity, cb_size):
    title = f"T-SNE: perplex {perplexity}, "
    tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, max_iter=250)
    print("fitting start")
    embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_to_fit), axis=0))
    print("fitting done")
    for zoom in [50, 20, 15, 10, 7, 5, 3, 2, 1, 0.5, 0.2, 0.1]:
        cb_emb = embedding[:cb_size]
        latent_emb = embedding[cb_size:]
        zoom_pct = f"{float(50 - zoom)}_{float(50 + zoom)}"
        x_range = np.percentile(cb_emb[:, 0], [50 - zoom, 50 + zoom])
        y_range = np.percentile(cb_emb[:, 1], [50 - zoom, 50 + zoom])

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

        for i in range(2):
            plt.figure(figsize=(10, 8))
            plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=5, c='black')
            # plt.hist2d(
            #     zoomed_latent[:, 0], zoomed_latent[:, 1],
            #     bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            #     cmap="Greys"
            # )
            plt.xlim(x_range)
            plt.ylim(y_range)

            if i == 0:
                plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.6, marker='x')
            plt.title(title + f" Zoom {zoom_pct}")
            print(f"saving file to {DATA_PATH}/zoom_{zoom_pct}_{i}.png")
            plt.savefig(f"{DATA_PATH}/zoom_{zoom_pct}_{i}.png")

def plot_umap(cb_arr, latent_arr, latent_to_fit, epoch, n_neighbors, min_dist, cb_size):
    print("reducer setup")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_epochs=250,
        random_state=42
    ).fit(latent_to_fit)
    print("reducer setup done")
    latent_emb = reducer.transform(latent_arr)
    print("latent transform done")
    cb_emb = reducer.transform(cb_arr)
    print("cb transform done")

    for zoom in [50, 20, 15, 10, 7, 5, 3, 2, 1, 0.5, 0.2, 0.1]:
        zoom_pct = f"{float(50 - zoom)}_{float(50 + zoom)}"
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

        title = f"UMAP: n_neighbors {n_neighbors}, min_dist {min_dist}, zoom {zoom_pct}"

        for i in range(2):
            plt.figure()
            plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=1, c='black')
            # plt.hist2d(
            #     zoomed_latent[:, 0], zoomed_latent[:, 1],
            #     bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            #     cmap="Blues"
            # )
            plt.xlim(x_range)
            plt.ylim(y_range)

            if i == 0:
                plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.6, marker='x')
            plt.title(title + " (Zoomed)")
            print(f"saving file to {DATA_PATH}/zoom_{zoom_pct}_{i}.png")
            plt.savefig(f"{DATA_PATH}/zoom_{zoom_pct}_{i}.png")


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
    latent_arr_to_fit = latent_arr[:SAMPLE_LATENT]
    print(cb_arr.shape)

    cb_arr = np.unique(cb_arr, axis=0).reshape(-1, DIMENSION)
    cb_size = cb_arr.shape[0]
    print("cb_size")
    print(cb_size)

    if MODE == "tsne":
        plot_tsne(cb_arr, latent_arr, latent_arr_to_fit, epoch, perplexity=PERPLEXITY, cb_size=cb_size)
    elif MODE == "umap":
        plot_umap(cb_arr, latent_arr, latent_arr_to_fit, epoch, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, cb_size=cb_size)


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch)


if __name__ == '__main__':
    main()
