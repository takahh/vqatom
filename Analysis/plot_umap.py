#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

# ----------------------------
# KDE grid utilities (IMPORTANT: fixed grid!)
# ----------------------------
def make_common_grid(Y_lat, grid_size=300, pad_ratio=0.02):
    """
    Build ONE common grid for all KDE evaluations.
    """
    xmin, ymin = Y_lat.min(axis=0)
    xmax, ymax = Y_lat.max(axis=0)

    xpad = (xmax - xmin) * pad_ratio if xmax > xmin else 1e-3
    ypad = (ymax - ymin) * pad_ratio if ymax > ymin else 1e-3

    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad

    xgrid = np.linspace(xmin, xmax, grid_size)
    ygrid = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    return Xg, Yg, xmin, xmax, ymin, ymax

def kde_on_given_grid(Y_pts, Xg, Yg, bw_method=None):
    """
    KDE evaluated ON THE GIVEN GRID (Xg,Yg). This keeps coordinates consistent.
    """
    xy = np.vstack([Y_pts[:, 0], Y_pts[:, 1]])
    kde = gaussian_kde(xy, bw_method=bw_method)
    grid_xy = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(grid_xy).reshape(Xg.shape)
    return Z

# ----------------------------
# plots
# ----------------------------
def plot_density_heatmap(
    Y_lat,      # [N,2]
    Y_ctr,      # [K,2]
    title,
    out_png,
    *,
    grid_size=320,
    pad_ratio=0.02,
    bw_method=None,
    use_log=True,
    add_contours=True,
    contour_levels=12,
    show_centers=True,
    centers_size=40,
    cmap="viridis",
):
    """
    True density heatmap for ALL latents (single KDE).
    """
    Xg, Yg, xmin, xmax, ymin, ymax = make_common_grid(
        Y_lat, grid_size=grid_size, pad_ratio=pad_ratio
    )
    Z = kde_on_given_grid(Y_lat, Xg, Yg, bw_method=bw_method)

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

    if show_centers and (Y_ctr is not None) and len(Y_ctr) > 0:
        plt.scatter(
            Y_ctr[:, 0], Y_ctr[:, 1],
            marker="x", c="black",
            s=centers_size, linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("log density" if use_log else "density")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_assign_density_sum_FIXEDGRID(
    Y_lat,      # [N,2]
    asg,        # [N] int
    Y_ctr,      # [K,2]
    title,
    out_png,
    *,
    grid_size=320,
    pad_ratio=0.02,
    bw_method=None,
    use_log=True,
    min_points_per_cluster=80,
    normalize_per_cluster=False,
    add_contours=False,
    contour_levels=12,
    show_centers=True,
    cmap="viridis",
):
    """
    Assign-wise density SUM on a FIXED grid. (This fixes the 'totally off' bug.)
    """
    asg = asg.astype(int)
    K = int(asg.max()) + 1 if asg.size > 0 else 0

    # ONE common grid for all clusters
    Xg, Yg, xmin, xmax, ymin, ymax = make_common_grid(
        Y_lat, grid_size=grid_size, pad_ratio=pad_ratio
    )
    Z_sum = np.zeros_like(Xg, dtype=float)

    for c in range(K):
        mask = (asg == c)
        n_c = int(mask.sum())
        if n_c < min_points_per_cluster:
            continue

        Yc = Y_lat[mask]
        Zc = kde_on_given_grid(Yc, Xg, Yg, bw_method=bw_method)

        if normalize_per_cluster:
            Zc = Zc / (Zc.max() + 1e-12)

        Z_sum += Zc

    if use_log:
        Z_plot = np.log(Z_sum + 1e-12)
    else:
        Z_plot = Z_sum

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        Z_plot,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        aspect="auto",
    )

    if add_contours:
        plt.contour(
            Xg, Yg, Z_plot,
            levels=contour_levels,
            colors="white",
            linewidths=0.6,
            alpha=0.6,
        )

    if show_centers and (Y_ctr is not None) and len(Y_ctr) > 0:
        plt.scatter(
            Y_ctr[:, 0], Y_ctr[:, 1],
            marker="x", c="black",
            s=20, linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("log density (sum)" if use_log else "density (sum)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_voronoi_regions(
    Y_ctr,      # [K,2]
    title,
    out_png,
    *,
    grid_size=500,
    pad_ratio=0.05,
    show_centers=True,
):
    """
    Voronoi regions for centers (nearest-center assignment in 2D).
    """
    if Y_ctr is None or len(Y_ctr) == 0:
        return

    Xg, Yg, xmin, xmax, ymin, ymax = make_common_grid(
        Y_ctr, grid_size=grid_size, pad_ratio=pad_ratio
    )

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
            marker="x", c="black",
            s=50, linewidths=1.2,
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# main
# ----------------------------
def plot_umap_from_dump(
    pt_path,
    out_dir="umap_plots",
    keys=None,
    max_points_per_key=20000,
    random_seed=42,
    umap_n_neighbors=30,
    umap_min_dist=0.05,
    umap_metric="euclidean",

    # ---- heatmap knobs ----
    grid_size=320,
    pad_ratio=0.02,
    bw_method=None,          # None or float like 0.2
    use_log=True,
    add_contours=True,
    contour_levels=12,

    # ---- extra plots ----
    make_assign_density=True,
    assign_min_points=80,
    assign_normalize_per_cluster=False,
    assign_add_contours=False,

    make_voronoi=True,
):
    import umap  # pip install umap-learn

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
            print(f"[skip] {k}: N mismatch (lat {lat.shape[0]} vs asg {asg.shape[0]})")
            continue

        N, D = lat.shape
        K = ctr.shape[0]
        print(f"[UMAP] key={k}  N={N}  D={D}  K={K}")

        # subsample latents (speed)
        if N > max_points_per_key:
            idx = rng.choice(N, size=max_points_per_key, replace=False)
            lat_s = lat[idx]
            asg_s = asg[idx].astype(int)
        else:
            lat_s = lat
            asg_s = asg.astype(int)

        # UMAP on latents + centers (so centers align)
        X = np.concatenate([lat_s, ctr], axis=0)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_seed,
        )
        Y = reducer.fit_transform(X)

        Y_lat = Y[: lat_s.shape[0]]
        Y_ctr = Y[lat_s.shape[0] :]

        safe_k = _safe_key(k)

        # ---- 1) all-latents heatmap ----
        out1 = os.path.join(out_dir, f"umap_HEATMAP_{safe_k}.png")
        plot_density_heatmap(
            Y_lat=Y_lat,
            Y_ctr=Y_ctr,
            title=f"VQ-Atom UMAP heatmap: {k}",
            out_png=out1,
            grid_size=grid_size,
            pad_ratio=pad_ratio,
            bw_method=bw_method,
            use_log=use_log,
            add_contours=add_contours,
            contour_levels=contour_levels,
            show_centers=True,
        )
        print(f"  -> saved {out1}")

        # ---- 2) assign-density SUM (FIXED GRID) ----
        if make_assign_density:
            out2 = os.path.join(out_dir, f"umap_ASSIGN_DENSITY_FIXED_{safe_k}.png")
            plot_assign_density_sum_FIXEDGRID(
                Y_lat=Y_lat,
                asg=asg_s,
                Y_ctr=Y_ctr,
                title=f"VQ-Atom UMAP assign-density (fixed grid): {k}",
                out_png=out2,
                grid_size=grid_size,
                pad_ratio=pad_ratio,
                bw_method=bw_method,
                use_log=use_log,
                min_points_per_cluster=assign_min_points,
                normalize_per_cluster=assign_normalize_per_cluster,
                add_contours=assign_add_contours,
                contour_levels=contour_levels,
                show_centers=True,
            )
            print(f"  -> saved {out2}")

        # ---- 3) Voronoi (centers) ----
        if make_voronoi:
            out3 = os.path.join(out_dir, f"umap_VORONOI_{safe_k}.png")
            plot_voronoi_regions(
                Y_ctr=Y_ctr,
                title=f"VQ-Atom UMAP Voronoi (centers): {k}",
                out_png=out3,
                grid_size=500,
                pad_ratio=0.05,
                show_centers=True,
            )
            print(f"  -> saved {out3}")


# ----------------------------
# run
# ----------------------------
if __name__ == "__main__":
    pt_path = "init_kmeans_final_ep1_chunkNone_20251218_025637.pt"

    plot_umap_from_dump(
        pt_path=pt_path,
        out_dir="umap_plots",
        keys=None,
        max_points_per_key=20000,
        random_seed=42,
        umap_n_neighbors=30,
        umap_min_dist=0.05,
        umap_metric="euclidean",

        grid_size=120,
        pad_ratio=0.02,
        bw_method=None,          # try 0.2 / 0.3 if you want tighter blobs
        use_log=True,
        add_contours=True,
        contour_levels=12,

        make_assign_density=True,
        assign_min_points=80,
        assign_normalize_per_cluster=False,
        assign_add_contours=False,

        make_voronoi=True,
    )
