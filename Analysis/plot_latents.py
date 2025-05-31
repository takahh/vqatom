import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

DATA_PATH = "/Users/taka/Downloads/40000_16/"
# DATA_PATH = "/Users/taka/Documents/vqatom_train_output/bothloss_40000_16/"
DIMENSION = 16
BATCH_SIZE = 8000
EPOCH_START = 1
SAMPLE_LATENT = 5000
EPOCH_END = EPOCH_START + 1
MODE = "tsne"  # Choose between "tsne" and "umap"


def load_npz_array(filename):
    """Load and return the array from a .npz file."""
    arr = np.load(filename, allow_pickle=True)
    print(arr.files)
    arr = arr["arr_0"]
    return np.squeeze(arr)

def plot_tsne(cb_arr, latent_arr, epoch, perplexity, cb_size, batch_size):
    title = f"T-SNE: perplex {perplexity}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"
    tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, n_iter=5000)
    embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_arr), axis=0))

    cb_emb = embedding[:cb_size]
    latent_emb = embedding[cb_size:cb_size + batch_size]
    x_range = np.percentile(cb_emb[:, 0], [44, 56])
    y_range = np.percentile(cb_emb[:, 1], [44, 56])

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
        plt.hist2d(
            zoomed_latent[:, 0], zoomed_latent[:, 1],
            bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            cmap="Blues"
        )
        plt.xlim(x_range)
        plt.ylim(y_range)

        if i == 0:
            plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=3, c='purple', alpha=0.6)
        plt.title(title + " (Zoomed)")
        plt.colorbar(label='Density')
        plt.show()

def plot_umap(cb_arr, latent_arr, epoch, n_neighbors, min_dist, cb_size):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_epochs=5000,
        random_state=42
    ).fit(latent_arr)

    latent_emb = reducer.transform(latent_arr)
    cb_emb = reducer.transform(cb_arr)
    x_range = np.percentile(cb_emb[:, 0], [2, 98])
    y_range = np.percentile(cb_emb[:, 1], [2, 98])

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

    bins = 200
    title = f"UMAP: n_neighbors {n_neighbors}, min_dist {min_dist}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"

    for i in range(2):
        plt.figure()
        plt.hist2d(
            zoomed_latent[:, 0], zoomed_latent[:, 1],
            bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            cmap="Blues"
        )
        plt.xlim(x_range)
        plt.ylim(y_range)

        if i == 0:
            plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=2, c='red', alpha=0.6)
        plt.title(title + " (Zoomed)")
        plt.colorbar(label='Density')
        plt.show()


def process_epoch(epoch):
    """Load data and plot visualization for a single epoch."""
    codebook_file = f"{DATA_PATH}init_codebook_{epoch}.npz"
    latent_file = f"{DATA_PATH}latents_{epoch}.npz"

    cb_arr = load_npz_array(codebook_file)
    latent_arr = load_npz_array(latent_file)
    print(latent_file)
    print(latent_arr.shape)
    latent_arr = latent_arr[:SAMPLE_LATENT]
    print(latent_arr.shape)

    cb_arr = np.unique(cb_arr, axis=0).reshape(-1, DIMENSION)
    cb_size = cb_arr.shape[0]

    if MODE == "tsne":
        plot_tsne(cb_arr, latent_arr, epoch, perplexity=10, cb_size=cb_size, batch_size=BATCH_SIZE)
    elif MODE == "umap":
        plot_umap(cb_arr, latent_arr, epoch, n_neighbors=10, min_dist=1.0, cb_size=cb_size)


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch)


if __name__ == '__main__':
    main()
