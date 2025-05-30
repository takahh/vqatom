import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

DATA_PATH = "/Users/taka/Documents/vqatom_train_output/bothloss_40000_16/"
DIMENSION = 16
BATCH_SIZE = 8000
EPOCH_START = 2
EPOCH_END = EPOCH_START + 1
MODE = "tsne"  # Choose between "tsne" and "umap"


def load_npz_array(filename):
    """Load and return the array from a .npz file."""
    arr = np.load(filename, allow_pickle=True)["arr_0"]
    return np.squeeze(arr)


def plot_tsne(cb_arr, latent_arr, epoch, perplexity, cb_size, batch_size):
    """Plot 2D visualization using t-SNE."""
    title = f"T-SNE: perplex {perplexity}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"
    tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, n_iter=5000)
    embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_arr), axis=0))

    cb_emb = embedding[:cb_size]
    latent_emb = embedding[cb_size:cb_size + batch_size]
    padding = 5  # adjust this to control zoom tightness
    x_range = (cb_emb[:, 0].min() - padding, cb_emb[:, 0].max() + padding)
    y_range = (cb_emb[:, 1].min() - padding, cb_emb[:, 1].max() + padding)

    bins = 100

    for i in range(2):
        plt.figure(figsize=(10, 8))
        plt.hist2d(
            latent_emb[:, 0], latent_emb[:, 1],
            bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            cmap="Blues"
        )
        plt.xlim(x_range)
        plt.ylim(y_range)

        plt.title(title)
        if i == 0:
            plt.scatter(cb_emb[:, 0], cb_emb[:, 1], s=3, c='purple', alpha=0.4)
        plt.show()


def plot_umap(cb_arr, latent_arr, epoch, n_neighbors, min_dist, cb_size):
    """Plot 2D visualization using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_epochs=5000,
        random_state=42
    ).fit(latent_arr)

    latent_emb = reducer.transform(latent_arr)
    cb_emb = reducer.transform(cb_arr)
    padding = 5  # adjust this to control zoom tightness
    x_range = (cb_emb[:, 0].min() - padding, cb_emb[:, 0].max() + padding)
    y_range = (cb_emb[:, 1].min() - padding, cb_emb[:, 1].max() + padding)

    # x_range = (latent_emb[:, 0].min(), latent_emb[:, 0].max())
    # y_range = (latent_emb[:, 1].min(), latent_emb[:, 1].max())
    bins = 200

    title = f"UMAP: n_neighbors {n_neighbors}, min_dist {min_dist}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"

    for i in range(2):
        plt.figure()
        plt.hist2d(
            latent_emb[:, 0], latent_emb[:, 1],
            bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
            cmap="Blues"
        )
        plt.xlim(x_range)
        plt.ylim(y_range)

        plt.title(title)
        if i == 0:
            plt.scatter(cb_emb[:, 0], cb_emb[:, 1], s=1, c='red', alpha=1)
        plt.colorbar(label='Density')
        plt.show()


def process_epoch(epoch):
    """Load data and plot visualization for a single epoch."""
    codebook_file = f"{DATA_PATH}init_codebook_{epoch}.npz"
    latent_file = f"{DATA_PATH}latents_{epoch}.npz"

    cb_arr = load_npz_array(codebook_file)
    latent_arr = load_npz_array(latent_file)[:2000]

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
