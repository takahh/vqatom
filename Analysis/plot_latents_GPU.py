from cuml.manifold import UMAP
import cupy as cp
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

DATA_PATH = "/40000_16/"
# DATA_PATH = "/Users/taka/Documents/vqatom_train_output/bothloss_40000_16/"
DIMENSION = 16
# BATCH_SIZE = 8000
EPOCH_START = 1
SAMPLE_LATENT = 3000000
SAMPLE_LATENT = 300
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

# def plot_tsne(cb_arr, latent_arr, epoch, perplexity, cb_size):
#     title = f"T-SNE: perplex {perplexity}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"
#     tsne = TSNE(n_components=2, random_state=44, perplexity=perplexity, n_iter=250)
#     print("fitting start")
#     embedding = tsne.fit_transform(np.concatenate((cb_arr, latent_arr), axis=0))
#     print("fitting done")
#     for zoom in [50, 20, 15, 10, 7, 5, 3, 2]:
#         cb_emb = embedding[:cb_size]
#         latent_emb = embedding[cb_size:cb_size]
#         x_range = np.percentile(cb_emb[:, 0], [50 - zoom, 50 + zoom])
#         y_range = np.percentile(cb_emb[:, 1], [50 - zoom, 50 + zoom])
#         zoom = float(50/int(zoom))
#
#         # Mask both latent and cb to zoom-in range
#         latent_mask = (
#             (latent_emb[:, 0] >= x_range[0]) & (latent_emb[:, 0] <= x_range[1]) &
#             (latent_emb[:, 1] >= y_range[0]) & (latent_emb[:, 1] <= y_range[1])
#         )
#         cb_mask = (
#             (cb_emb[:, 0] >= x_range[0]) & (cb_emb[:, 0] <= x_range[1]) &
#             (cb_emb[:, 1] >= y_range[0]) & (cb_emb[:, 1] <= y_range[1])
#         )
#         zoomed_latent = latent_emb[latent_mask]
#         zoomed_cb = cb_emb[cb_mask]
#
#         bins = 100
#         for i in range(2):
#             plt.figure(figsize=(10, 8))
#             plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=20, c='black', alpha=0.6)
#             # plt.hist2d(
#             #     zoomed_latent[:, 0], zoomed_latent[:, 1],
#             #     bins=[np.linspace(*x_range, bins), np.linspace(*y_range, bins)],
#             #     cmap="Greys"
#             # )
#             plt.xlim(x_range)
#             plt.ylim(y_range)
#
#             if i == 0:
#                 plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.6, marker='x')
#             plt.title(title + f" (Zoomed {zoom}, sample {SAMPLE_LATENT})")
#             plt.colorbar(label='Density')
#             plt.show()

def plot_umap(cb_arr, latent_arr, epoch, n_neighbors, min_dist, cb_size):
    print("reducer setup")

    reducer = UMAP(n_neighbors=n_neighbors,
                   n_components=2,
                   min_dist=min_dist,
                   random_state=42).fit(latent_arr)
    # reducer = umap.UMAP(
    #     n_neighbors=n_neighbors,
    #     min_dist=min_dist,
    #     n_components=2,
    #     n_epochs=5000,
    #     random_state=42
    # ).fit(latent_arr)
    print("reducer setup done")
    latent_emb = reducer.transform(latent_arr)
    print("latent transform done")
    cb_emb = reducer.transform(cb_arr)
    print("cb transform done")
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
        plt.scatter()
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.scatter(zoomed_latent[:, 0], zoomed_latent[:, 1], s=20, c='black', alpha=0.6)
        if i == 0:
            plt.scatter(zoomed_cb[:, 0], zoomed_cb[:, 1], s=30, c='red', alpha=0.6, marker='x')
        plt.title(title + " (Zoomed)")
        plt.colorbar(label='Density')
        plt.show()


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

    if MODE == "tsne":
        plot_tsne(cb_arr, latent_arr, epoch, perplexity=10, cb_size=cb_size)
    elif MODE == "umap":
        plot_umap(cb_arr, latent_arr, epoch, n_neighbors=10, min_dist=1.0, cb_size=cb_size)


def main():
    for epoch in range(EPOCH_START, EPOCH_END):
        print(f"Processing epoch {epoch}")
        process_epoch(epoch)


if __name__ == '__main__':
    main()

# # ----------------------------------------
#
# # 1. Load sample data
# X, y = load_digits(return_X_y=True)
#
# # 2. Move data to GPU
# X_gpu = cp.asarray(X)
#
# # 3. Run UMAP on GPU
# umap = UMAP(n_neighbors=15, n_components=2, random_state=42)
# X_embedded_gpu = umap.fit_transform(X_gpu)
#
# # 4. Bring result back to CPU for plotting
# X_embedded = cp.asnumpy(X_embedded_gpu)
#
# # 5. Plot the result
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='Spectral', s=10)
# plt.colorbar(scatter, label='Digit Label')
# plt.title("UMAP Projection of Digits Dataset")
#
# # 6. Save the plot to file
# plt.savefig("umap_digits.png", dpi=300)
# plt.show()
