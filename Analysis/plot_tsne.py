#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

# ----------------------------
# helpers
# ----------------------------
def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_key(k):
    return str(k).replace("/", "_").replace(" ", "_")

def _kde_on_grid(Y_lat, grid_size=300, bw_method=None, pad_ratio=0.02):
    """
    Evaluate KDE on a regular grid.
    Returns: (Z, xmin, xmax, ymin, ymax, Xg, Yg)
    """
    xy = np.vstack([Y_lat[:, 0], Y_lat[:, 1]])
    kde = gaussian_kde(xy, bw_method=bw_method)

    xmin, ymin = Y_lat.min(axis=0)
    xmax, ymax = Y_lat.max(axis=0)

    # small padding for nicer framing
    xpad = (xmax - xmin) * pad_ratio if xmax > xmin else 1e-3
    ypad = (ymax - ymin) * pad_ratio if ymax > ymin else 1e-3
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    xgrid = np.linspace(xmin, xmax, grid_size)
    ygrid = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xgrid, ygrid)

    grid_xy = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(grid_xy).reshape(grid_size, grid_size)

    return Z, xmin, xmax, ymin, ymax, Xg, Yg


def plot_density_heatmap(
    Y_lat,      # [N, 2]
    Y_ctr,      # [K, 2]
    title,
    out_png,
    *,
    grid_size=300,
    bw_method=None,        # None (Scott default) or float like 0.2
    use_log=False,         # log density for high dynamic range
    add_contours=False,    # add contour lines
    contour_levels=12,
    show_centers=True,
    centers_size=40,
    cmap="viridis",
    xlab="t-SNE-1",
    ylab="t-SNE-2",
):
    """
    True heatmap (continuous), not scatter.
    """
    Z, xmin, xmax, ymin, ymax, Xg, Yg = _kde_on_grid(
        Y_lat, grid_size=grid_size, bw_method=bw_method
    )

    if use_log:
        Z = np.log(Z + 1e-12)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        Z,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        aspect="auto",
    )

    if add_contours:
        plt.contour(
            Xg, Yg, Z,
            levels=contour_levels,
            colors="white",
            linewidths=0.6,
            alpha=0.6,
        )

    if show_centers and (Y_ctr is not None) and (len(Y_ctr) > 0):
        plt.scatter(
            Y_ctr[:, 0], Y_ctr[:, 1],
            marker="x",
            c="black",
            s=centers_size,
            linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("log density" if use_log else "density")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_density_by_assign(
    Y_lat,      # [N, 2]
    asg,        # [N] int
    Y_ctr,      # [K, 2] (optional)
    title,
    out_png,
    *,
    grid_size=300,
    bw_method=None,
    use_log=False,
    min_points_per_cluster=50,   # skip tiny clusters
    normalize_per_cluster=False, # if True, each cluster map scaled to max=1 before summing
    show_centers=True,
    cmap="viridis",
    xlab="t-SNE-1",
    ylab="t-SNE-2",
):
    """
    Assign別 density を1枚に合成。
    - 各クラスタの点のみで KDE→足し合わせ
    - normalize_per_cluster=True で「形」優先（サイズ差を潰す）
    """
    asg = asg.astype(int)
    K = int(asg.max()) + 1 if asg.size > 0 else 0

    # common extent from all points
    Z0, xmin, xmax, ymin, ymax, Xg, Yg = _kde_on_grid(
        Y_lat, grid_size=grid_size, bw_method=bw_method
    )
    Z_sum = np.zeros_like(Z0)

    for c in range(K):
        mask = (asg == c)
        if mask.sum() < min_points_per_cluster:
            continue
        Yc = Y_lat[mask]
        Zc, *_ = _kde_on_grid(
            Yc, grid_size=grid_size, bw_method=bw_method
        )
        if normalize_per_cluster:
            m = Zc.max()
            if m > 0:
                Zc = Zc / m
        Z_sum += Zc

    if use_log:
        Z_sum = np.log(Z_sum + 1e-12)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        Z_sum,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        aspect="auto",
    )

    if show_centers and (Y_ctr is not None) and (len(Y_ctr) > 0):
        plt.scatter(
            Y_ctr[:, 0], Y_ctr[:, 1],
            marker="x",
            c="black",
            s=40,
            linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("log density (sum)" if use_log else "density (sum)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_voronoi_regions(
    Y_ctr,      # [K, 2]
    title,
    out_png,
    *,
    grid_size=500,
    pad_ratio=0.05,
    show_centers=True,
    xlab="t-SNE-1",
    ylab="t-SNE-2",
):
    """
    2D center の Voronoi（最近傍領域）を描く。
    """
    if Y_ctr is None or len(Y_ctr) == 0:
        return

    xmin, ymin = Y_ctr.min(axis=0)
    xmax, ymax = Y_ctr.max(axis=0)
    xpad = (xmax - xmin) * pad_ratio if xmax > xmin else 1e-3
    ypad = (ymax - ymin) * pad_ratio if ymax > ymin else 1e-3
    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad

    xgrid = np.linspace(xmin, xmax, grid_size)
    ygrid = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    P = np.stack([Xg, Yg], axis=-1)               # [H,W,2]
    C = Y_ctr[None, None, :, :]                   # [1,1,K,2]
    dist2 = ((P[..., None, :] - C) ** 2).sum(-1)  # [H,W,K]
    regions = dist2.argmin(axis=-1)               # [H,W]

    plt.figure(figsize=(8, 7))
    plt.imshow(
        regions,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        interpolation="nearest",
    )

    if show_centers:
        plt.scatter(
            Y_ctr[:, 0], Y_ctr[:, 1],
            marker="x",
            c="black",
            s=50,
            linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# main
# ----------------------------
def plot_tsne_from_dump(
    pt_path,
    out_dir="tsne_plots",
    keys=None,                  # None = all keys
    max_points_per_key=15000,   # t-SNE is heavy
    random_seed=42,

    # ---- t-SNE knobs ----
    tsne_perplexity=30,         # will be clamped by N
    tsne_iter=1000,
    tsne_init="pca",
    tsne_learning_rate="auto",

    # ---- density heatmap knobs ----
    grid_size=320,
    bw_method=None,             # try 0.2/0.3 to tighten blobs
    use_log=True,
    add_contours=True,
    contour_levels=12,

    # ---- extras ----
    make_assign_density=True,
    assign_min_points=80,
    assign_normalize_per_cluster=False,
    make_voronoi=True,
):
    os.makedirs(out_dir, exist_ok=True)

    dump = torch.load(pt_path, map_location="cpu")
    assert isinstance(dump, dict)

    rng = np.random.default_rng(random_seed)

    if keys is None:
        keys = list(dump.keys())
    else:
        keys = [k for k in keys if k in dump]

    for k in keys:
        entry = dump[k]

        lat = to_np(entry.get("latents"))
        ctr = to_np(entry.get("centers"))
        asg = to_np(entry.get("assign"))

        if lat is None or ctr is None or asg is None:
            print(f"[skip] {k}: missing data")
            continue

        if lat.shape[0] != asg.shape[0]:
            print(f"[skip] {k}: N mismatch")
            continue

        N, D = lat.shape
        K = ctr.shape[0]
        total = N + K

        if total < 5:
            print(f"[skip] {k}: too few points for t-SNE (N={N}, K={K})")
            continue

        print(f"[t-SNE] key={k}  N={N}  D={D}  K={K}")

        # ----------------------------
        # subsample latents (speed)
        # ----------------------------
        if N > max_points_per_key:
            idx = rng.choice(N, size=max_points_per_key, replace=False)
            lat_s = lat[idx]
            asg_s = asg[idx].astype(int)
        else:
            lat_s = lat
            asg_s = asg.astype(int)

        # ----------------------------
        # stack latents + centers
        # ----------------------------
        X = np.concatenate([lat_s, ctr], axis=0)

        # perplexity must be < N (and typically < (N-1)/3)
        perplx = min(tsne_perplexity, (X.shape[0] - 1) // 3)
        perplx = max(perplx, 2)

        tsne = TSNE(
            n_components=2,
            perplexity=perplx,
            learning_rate=tsne_learning_rate,
            init=tsne_init,
            random_state=random_seed,
            n_iter=tsne_iter,
            verbose=0,
        )

        Y = tsne.fit_transform(X)

        Y_lat = Y[: lat_s.shape[0]]
        Y_ctr = Y[lat_s.shape[0] :]

        safe_k = _safe_key(k)

        # ---- 1) true density heatmap ----
        out_png = os.path.join(out_dir, f"tsne_HEATMAP_{safe_k}.png")
        plot_density_heatmap(
            Y_lat=Y_lat,
            Y_ctr=Y_ctr,
            title=f"VQ-Atom t-SNE heatmap: {k}",
            out_png=out_png,
            grid_size=grid_size,
            bw_method=bw_method,
            use_log=use_log,
            add_contours=add_contours,
            contour_levels=contour_levels,
            show_centers=True,
            xlab="t-SNE-1",
            ylab="t-SNE-2",
        )
        print(f"  -> saved {out_png}")

        # ---- 2) assign-wise density (optional) ----
        if make_assign_density:
            out_png2 = os.path.join(out_dir, f"tsne_ASSIGN_DENSITY_{safe_k}.png")
            plot_density_by_assign(
                Y_lat=Y_lat,
                asg=asg_s,
                Y_ctr=Y_ctr,
                title=f"VQ-Atom t-SNE assign-density: {k}",
                out_png=out_png2,
                grid_size=grid_size,
                bw_method=bw_method,
                use_log=use_log,
                min_points_per_cluster=assign_min_points,
                normalize_per_cluster=assign_normalize_per_cluster,
                show_centers=True,
                xlab="t-SNE-1",
                ylab="t-SNE-2",
            )
            print(f"  -> saved {out_png2}")

        # ---- 3) Voronoi regions from centers (optional) ----
        if make_voronoi:
            out_png3 = os.path.join(out_dir, f"tsne_VORONOI_{safe_k}.png")
            plot_voronoi_regions(
                Y_ctr=Y_ctr,
                title=f"VQ-Atom t-SNE Voronoi (centers): {k}",
                out_png=out_png3,
                grid_size=500,
                pad_ratio=0.05,
                show_centers=True,
                xlab="t-SNE-1",
                ylab="t-SNE-2",
            )
            print(f"  -> saved {out_png3}")


# ----------------------------
# run
# ----------------------------
if __name__ == "__main__":
    # pt_path = "init_kmeans_final_ep1_chunkNone_20251218_025637.pt"
    pt_path = "init_kmeans_final_ep10_chunkNone_20251218_074830.pt"


    plot_tsne_from_dump(
        pt_path=pt_path,
        out_dir="tsne_plots",
        keys=None,
        max_points_per_key=15000,
        random_seed=42,

        # ---- t-SNE knobs ----
        tsne_perplexity=30,   # auto-clamped by N
        tsne_iter=1000,

        # ---- density heatmap knobs ----
        grid_size=320,
        bw_method=None,       # try 0.2 or 0.3 if you want tighter blobs
        use_log=True,         # recommended
        add_contours=True,    # recommended for figures
        contour_levels=12,

        # ---- extras ----
        make_assign_density=True,
        assign_min_points=80,
        assign_normalize_per_cluster=False,
        make_voronoi=True,
    )
