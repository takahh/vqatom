
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

path = "/Users/taka/Documents/vqgraph_0217_cb700_3hop_impled/"
# path = "/Users/taka/Downloads/"
# LMIN = 2.5
# LMAX = 7.5

def plot_graph(cb_arr, latent_arr, mode, epoch, param, cb_size, batch_size, param2=None):
    # Initialize UMAP or TSNE with custom parameters
    parameter_names = None
    embedding = None
    heatmap_colors = ["Blues", "binary", "BuGn", "Purples"]
    cb_colors = ["blue", "black", "green", "orange"]

    if mode == "tsne":
        perplex = param
        n_iter = 5000
        tsne = TSNE(n_components=2, random_state=44, perplexity=perplex, n_iter=n_iter)
        data = np.concatenate((cb_arr, latent_arr), axis=0)
        title = f"T-SNE: perplex {param}, epoch {epoch}, cb {cb_size}, dim {latent_arr.shape[-1]}"

        # -------------------------------------
        # put all data into tsne
        # -------------------------------------
        embedding = tsne.fit_transform(data)
        print(f"embedding.shape {embedding.shape}")

        # -------------------------------------
        # make two lists
        # -------------------------------------
        cb_list, latent_list = [], []
        for i in range(1):
            cb_list.append(embedding[cb_size * i: cb_size * (i + 1)])
            latent_list.append(embedding[cb_size:][batch_size * i: batch_size * (i + 1)])

        # -------------------------------------
        # plot three pairs of data
        # -------------------------------------
        for i in range(1):
            plt.figure()
            # Define bin edges to control the size of the bins
            x_min = min(min(cb_list[i][:, 0]), min(latent_list[i][:, 0]))
            x_max = max(max(cb_list[i][:, 0]), max(latent_list[i][:, 0]))
            y_min = min(min(cb_list[i][:, 1]), min(latent_list[i][:, 1]))
            y_max = max(max(cb_list[i][:, 1]), max(latent_list[i][:, 1]))
            x_range = (x_min, x_max)  # Range for the x-axis
            y_range = (y_min, y_max)  # Range for the y-axis
            n_bins = 100  # Number of bins for both axes

            # cb_size = 1201

            plt.hist2d(
                latent_list[i][:, 0], latent_list[i][:, 1],
                bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
                cmap=heatmap_colors[i]
            )
            # Overlay scatter plot
            # plt.scatter(cb_list[i][:, 0], cb_list[i][:, 1], s=1, c="red", alpha=1)

            # plt.colorbar(label='Density')
            plt.title(title)
            plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=3, c='purple', alpha=0.4)
            plt.show()
            # plt.savefig(f"./plot_epoch{epoch}")

    elif mode == "umap":
        n_neibogher = param
        min_dist = param2
        n_epochs = 5000
        # reducer = umap.UMAP(n_neighbors=n_neibogher, metric='cosine', min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42)
        reducer = umap.UMAP(n_neighbors=n_neibogher, min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42).fit(latent_arr)
        embedding_latent = reducer.transform(latent_arr)
        embedding_quantized = reducer.transform(cb_arr)
        parameter_names = f"umap: n_neiboughers {n_neibogher}, min_dist {param2}, epoch {epoch}\n n_epochs {n_epochs}"

        plt.figure()
        # Define bin edges to control the size of the bins
        x_range = (min(embedding_latent[:, 0]), max(embedding_latent[:, 0]))  # Range for the x-axis
        y_range = (min(embedding_latent[:, 1]), max(embedding_latent[:, 1]))  # Range for the y-axis

        # x_range = (LMIN, LMAX)  # Range for the x-axis
        # y_range = (LMIN, LMAX)  # Range for the y-axis
        n_bins = 200  # Number of bins for both axes
        # cb_size = 1201
        plt.hist2d(
            embedding_latent[:, 0], embedding_latent[:, 1],
            bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
            cmap='Blues'
        )

        plt.colorbar(label='Density')

        title = f"UMAP: n_neibogher {param}, epoch {epoch}, min_dist {param2}, cb {cb_size}, dim {latent_arr.shape[-1]}"
        plt.title(title)
        # Overlay scatter plot
        plt.scatter(embedding_quantized[:, 0], embedding_quantized[:, 1], s=1, c='red', alpha=1)
        # plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=3, c='purple', alpha=1)
        plt.show()
        # plt.savefig(f"./plot_epoch{epoch}")

        plt.figure()
        # Define bin edges to control the size of the bins
        # x_range = (LMIN, LMAX)  # Range for the x-axis
        # y_range = (LMIN, LMAX)  # Range for the y-axis
        x_range = (min(embedding_latent[:, 0]), max(embedding_latent[:, 0]))  # Range for the x-axis
        y_range = (min(embedding_latent[:, 1]), max(embedding_latent[:, 1]))  # Range for the y-axis
        n_bins = 200  # Number of bins for both axes
        # cb_size = 1201
        plt.hist2d(
            embedding_latent[:, 0], embedding_latent[:, 1],
            bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
            cmap='Blues'
        )

        plt.colorbar(label='Density')

        title = f"UMAP: n_neibogher {param}, epoch {epoch}, min_dist {param2}, cb {cb_size}, dim {latent_arr.shape[-1]}"
        plt.title(title)
        # Overlay scatter plot
        # plt.scatter(embedding_quantized[:, 0], embedding_quantized[:, 1], s=1, c='red', alpha=1)
        # plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=3, c='purple', alpha=1)
        plt.show()
        # plt.savefig(f"./plot_epoch{epoch}")


def getdata(filename):
    # filename = "out_emb_list.npz"
    arr = np.load(f"{filename}", allow_pickle=True)["arr_0"]
    arr = np.squeeze(arr)
    return arr


def main():
    arr_list = []
    DIMENSION = 64
    BATCH = 8000
    EPOCH = 12
    EPOCH2 = EPOCH + 1

    # MODE = "tsne"
    MODE = "umap"
    for epoch in range(EPOCH, EPOCH2):
        arr = None
        print(f"epoch {epoch}")
        namelist = [f"{path}init_codebook_{epoch}.npz", f"{path}latents_{epoch}.npz"]
        # namelist = [f"{path}codebook_{epoch}.npz", f"{path}init_codebook_{epoch}.npz", f"{path}latent_train_{epoch}.npz"]
        for names in namelist:
            arr = getdata(names)
            if "book" in names:
                print(arr.shape)
                cb_arr = np.unique(arr, axis=0)
                cb_size = cb_arr.shape[0]
                print(f"cb_size {cb_size}")
                cb_arr = np.reshape(cb_arr, (-1, DIMENSION))
                print(f"cb_arr.shape {cb_arr.shape}")
            else:
                latent_arr = arr[:2000]
                print(f"arr.shape {arr.shape}")

        for param in [10]:
            if MODE == "tsne":
                plot_graph(cb_arr, latent_arr, MODE, epoch, param, cb_size, BATCH)
            else:
                # for param2 in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for param2 in [1.0]:
                    plot_graph(cb_arr, latent_arr, MODE, epoch, param, cb_size, BATCH, param2)

if __name__ == '__main__':
    main()