import torch.distributed as distributed
from einops import rearrange, repeat, pack, unpack
from utils import CBDICT
from torch import nn, einsum

from args import get_args


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x

from einops import rearrange, repeat

def l2norm(t, dim=-1, eps=1e-8):
    return F.normalize(t, dim=dim, eps=eps)

import torch

def noop(x):
    return x

def batched_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    # x: [H, N] int64 on CUDA
    H, N = x.shape
    out = torch.zeros(H, minlength, device=x.device, dtype=torch.int64)
    one = torch.ones_like(x, dtype=torch.int64)
    out.scatter_add_(1, x, one)            # all CUDA if x is CUDA
    return out

import torch
from typing import Optional, Sequence, Dict, Tuple


def kmeans(
    samples: torch.Tensor,          # [H, N, D]
    num_clusters: int,
    use_cosine_sim: bool = False,
    all_reduce_fn = lambda x: None, # in-place sum for DDP (noop by default)
    eps: float = 1e-12,
    max_iters: int = 20,
    tol: float = 0.0,
    n_block: int = 131072,          # tile size over N (points)
    k_block: int = 4096,            # tile size over K (centers)
    element_names=None,             # optional labels for heads (e.g., ["C","N","O",...])
):
    """
    Lloyd K-Means w/ K-Means++ init, streaming/blocked (no [H,N,K] allocation).

    Returns:
        means:          [H, K, D]
        bins:           [H, K]        # counts per cluster
        used_per_head:  [H]           # number of non-empty clusters per head
        used_per_label: dict[str,int] # only if element_names provided, else None
    """
    import math
    import torch

    # ----------------------------
    # device / shape
    # ----------------------------
    samples = samples.to("cuda", non_blocking=True)
    H, N, D = samples.shape
    device, dtype = samples.device, samples.dtype

    if max_iters <= 0:
        base = 20
        extra = int(10 * max(0.0, math.log10(max(N, 1)) - 2.0))
        max_iters = min(100, base + extra)

    K = int(min(num_clusters, N))
    if K <= 0:
        raise ValueError("No samples to cluster.")

    if use_cosine_sim:
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)

    # ----------------------------
    # deterministic helpers
    # ----------------------------
    def _determinism_enabled() -> bool:
        # True if torch.use_deterministic_algorithms(True) has been set
        return torch.are_deterministic_algorithms_enabled()

    def _with_nondet_allowed(fn, *args, **kwargs):
        """
        Run fn with deterministic algorithms temporarily disabled (only if they are enabled),
        to avoid hard errors on ops like CUDA cumsum.
        """
        prev = _determinism_enabled()
        if not prev:
            return fn(*args, **kwargs)
        try:
            torch.use_deterministic_algorithms(False)
            return fn(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(True)

    # ----------------------------
    # KMeans++ init (blockwise, deterministic-safe)
    # ----------------------------
    @torch.no_grad()
    def kmeanspp_init_blockwise(
        X: torch.Tensor,                 # [H,N,D] on GPU
        K: int,
        cosine: bool = False,
        eps: float = 1e-12,
        sample_block_elems: int = 262_144,
        dtype_prob: torch.dtype = torch.float32,
        deterministic: str = "auto",     # "cpu_cumsum" | "gpu_cumsum" | "gpu_cumsum_nondet_ok" | "auto"
    ) -> torch.Tensor:
        """
        Memory-safe KMeans++ initializer with deterministic-safe sampling.

        Modes:
          - "cpu_cumsum": fully deterministic, does sampling cumsum/searchsorted on CPU.
          - "gpu_cumsum": uses GPU cumsum (will ERROR if deterministic_algorithms=True).
          - "gpu_cumsum_nondet_ok": uses GPU cumsum but temporarily disables determinism just for cumsum.
          - "auto": if determinism enabled -> cpu_cumsum else -> gpu_cumsum
        """
        assert deterministic in ("cpu_cumsum", "gpu_cumsum", "gpu_cumsum_nondet_ok", "auto")
        if deterministic == "auto":
            deterministic = "cpu_cumsum" if _determinism_enabled() else "gpu_cumsum"

        H, N, D = X.shape
        dev = X.device
        C = torch.empty((H, K, D), device=dev, dtype=X.dtype)

        if cosine:
            Xwork = torch.nn.functional.normalize(X, p=2, dim=-1)
            x2 = None
        else:
            Xwork = X
            x2 = (X ** 2).sum(-1).to(dtype_prob)  # [H,N] fp32

        # first seed per head
        idx0 = torch.randint(0, N, (H, 1), device=dev)
        C[:, 0, :] = X.gather(1, idx0.unsqueeze(-1).expand(H, 1, D)).squeeze(1)

        def dist_to_center(x_all: torch.Tensor, c_one: torch.Tensor) -> torch.Tensor:
            # x_all: [H,N,D], c_one: [H,D] -> [H,N] (dtype_prob)
            if cosine:
                d = (1.0 - (x_all * c_one.unsqueeze(1)).sum(-1)).clamp_min_(0)
            else:
                c2 = (c_one ** 2).sum(-1)                 # [H]
                xc = (x_all * c_one.unsqueeze(1)).sum(-1) # [H,N]
                d = (x2 + c2.unsqueeze(1) - 2.0 * xc).clamp_min_(0)
            return d.to(dtype_prob)

        # closest squared distance to chosen centers so far
        closest = dist_to_center(Xwork, C[:, 0, :])  # [H,N] fp32

        def _sample_block_gpu_cumsum(prob: torch.Tensor) -> torch.Tensor:
            """
            prob: [H,N] nonnegative, fp32
            Return idx: [H] long on GPU
            """
            totals = prob.sum(dim=1)                         # [H]
            zero_mask = totals <= 0
            u = torch.rand(H, device=dev, dtype=prob.dtype) * torch.clamp(totals, min=eps)

            idx_out = torch.empty(H, device=dev, dtype=torch.long)
            cum = torch.zeros(H, device=dev, dtype=prob.dtype)
            found = torch.zeros(H, device=dev, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = prob[:, start:end]                  # [H,B]
                block_sum = block.sum(dim=1)                # [H]

                target = (~found) & (cum + block_sum >= u)
                if target.any():
                    h_idx = target.nonzero(as_tuple=False).squeeze(1)
                    block_h = block[h_idx]                  # [Hsel,B]
                    need = (u[h_idx] - cum[h_idx]).unsqueeze(1)

                    # ---- THIS is the deterministic trouble-maker on CUDA ----
                    block_cum = block_h.cumsum(dim=1)
                    pos = torch.searchsorted(block_cum, need, right=False).squeeze(1)
                    pos = pos.clamp_max(block_h.size(1) - 1)

                    idx_out[h_idx] = start + pos
                    found[h_idx] = True

                nf = ~found
                if nf.any():
                    cum[nf] = cum[nf] + block_sum[nf]

                start = end
                if found.all():
                    break

            if (~found).any():
                idx_out[~found] = N - 1
            if zero_mask.any():
                idx_out[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device=dev)
            return idx_out

        def _sample_block_cpu_cumsum(prob: torch.Tensor) -> torch.Tensor:
            """
            Deterministic sampling on CPU:
            prob: [H,N] on GPU fp32 -> moves one head-block at a time as needed.
            Returns idx: [H] on GPU long
            """
            # Move whole prob to CPU can be expensive; but N is big and sampling happens K times.
            # We'll do it blockwise on CPU to keep memory manageable.
            totals = prob.sum(dim=1).detach().cpu()  # [H]
            zero_mask = (totals <= 0)
            # u on CPU for deterministic path
            u = (torch.rand(H, dtype=torch.float64) * torch.clamp(totals.to(torch.float64), min=float(eps))).to(torch.float64)

            idx_out_cpu = torch.empty(H, dtype=torch.long)
            cum = torch.zeros(H, dtype=torch.float64)
            found = torch.zeros(H, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = prob[:, start:end].detach().cpu().to(torch.float64)  # [H,B]
                block_sum = block.sum(dim=1)                                 # [H]

                target = (~found) & (cum + block_sum >= u)
                if target.any():
                    h_idx = target.nonzero(as_tuple=False).squeeze(1)
                    block_h = block[h_idx]                                   # [Hsel,B]
                    need = (u[h_idx] - cum[h_idx]).unsqueeze(1)              # [Hsel,1]
                    block_cum = block_h.cumsum(dim=1)                        # CPU cumsum deterministic
                    pos = torch.searchsorted(block_cum, need, right=False).squeeze(1)
                    pos = torch.clamp(pos, max=block_h.size(1) - 1)
                    idx_out_cpu[h_idx] = start + pos
                    found[h_idx] = True

                nf = ~found
                if nf.any():
                    cum[nf] = cum[nf] + block_sum[nf]

                start = end
                if found.all():
                    break

            if (~found).any():
                idx_out_cpu[~found] = N - 1
            if zero_mask.any():
                # random but deterministic algorithms demand doesn't constrain RNG reproducibility unless you seed;
                # still fine. If you want fully repeatable, seed torch before calling kmeans.
                idx_out_cpu[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device="cpu")

            return idx_out_cpu.to(device=dev)

        # choose sampler
        def sample_idx(prob: torch.Tensor) -> torch.Tensor:
            if deterministic == "cpu_cumsum":
                return _sample_block_cpu_cumsum(prob)
            elif deterministic == "gpu_cumsum":
                return _sample_block_gpu_cumsum(prob)  # may error if deterministic_algorithms=True
            elif deterministic == "gpu_cumsum_nondet_ok":
                return _with_nondet_allowed(_sample_block_gpu_cumsum, prob)
            else:
                raise RuntimeError("unreachable")

        # pick K-1 more centers
        for k in range(1, K):
            # stabilized probs from closest distances
            prob = closest
            prob = (prob - prob.amin(dim=1, keepdim=True)).clamp_min_(0) + eps
            prob = prob * prob  # standard kmeans++ uses d^2 weighting

            idxk = sample_idx(prob)  # [H] long on GPU
            C[:, k, :] = X[torch.arange(H, device=dev), idxk, :]

            dk = dist_to_center(Xwork, C[:, k, :])  # [H,N]
            closest = torch.minimum(closest, dk)

        if cosine:
            C = torch.nn.functional.normalize(C, p=2, dim=-1)
        return C

    # ---- IMPORTANT CHANGE ----
    # If determinism is enabled, default to CPU sampling (no crash).
    # If you prefer "allow nondet only for cumsum", set deterministic="gpu_cumsum_nondet_ok".
    means = kmeanspp_init_blockwise(
        samples,
        K,
        cosine=use_cosine_sim,
        eps=eps,
        deterministic="gpu_cumsum_nondet_ok",
    )

    # ----------------------------
    # Lloyd steps (streaming/blocked)
    # ----------------------------
    def assign_pass(X, C):
        """
        Returns buckets [H,N] (long)
        Streaming over K tiles and N tiles to find argmin per point.
        """
        H, N, D = X.shape
        _, K, _ = C.shape
        buckets = torch.empty(H, N, device=device, dtype=torch.long)

        for h in range(H):
            if use_cosine_sim:
                best_val = torch.full((N,), -float("inf"), device=device, dtype=X.dtype)
            else:
                best_val = torch.full((N,), float("inf"), device=device, dtype=X.dtype)
            best_idx = torch.zeros((N,), device=device, dtype=torch.long)

            if not use_cosine_sim:
                x2_full = (X[h] ** 2).sum(-1)  # [N]

            for k0 in range(0, K, k_block):
                k1 = min(k0 + k_block, K)
                Ck = C[h, k0:k1]  # [kb,D]

                if use_cosine_sim:
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        sims = X[h, n0:n1] @ Ck.T            # [nb,kb]
                        vals, idxs = sims.max(dim=1)         # [nb]
                        update = vals > best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, idxs + k0, best_idx[n0:n1])
                else:
                    c2 = (Ck ** 2).sum(-1)                   # [kb]
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        xc = X[h, n0:n1] @ Ck.T              # [nb,kb]
                        d2 = (x2_full[n0:n1].unsqueeze(1) + c2.unsqueeze(0) - 2 * xc).clamp_min_(0)
                        vals, idxs = d2.min(dim=1)           # [nb]
                        update = vals < best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, idxs + k0, best_idx[n0:n1])

            buckets[h] = best_idx

        return buckets

    def update_pass(X, buckets, K, old_means):
        """
        Accumulates sums and counts in chunks of N.
        Returns:
          new_means [H,K,D], bins [H,K]
        """
        H, N, D = X.shape
        new_means = torch.zeros(H, K, D, device=device, dtype=X.dtype)
        bins = torch.zeros(H, K, device=device, dtype=torch.long)

        ones = None
        for h in range(H):
            for n0 in range(0, N, n_block):
                n1 = min(n0 + n_block, N)
                b = buckets[h, n0:n1]                       # [nb]
                x = X[h, n0:n1]                             # [nb,D]
                if ones is None or ones.numel() != b.numel():
                    ones = torch.ones_like(b, dtype=torch.long, device=device)
                bins[h].index_add_(0, b, ones)
                new_means[h].index_add_(0, b, x)

        all_reduce_fn(bins)
        all_reduce_fn(new_means)

        zero = (bins == 0)
        denom = bins.clamp_min(1).unsqueeze(-1)             # [H,K,1]
        new_means = new_means / denom

        # keep old center for empty bins
        new_means = torch.where(zero.unsqueeze(-1), old_means, new_means)

        if use_cosine_sim:
            new_means = torch.nn.functional.normalize(new_means, p=2, dim=-1)

        return new_means, bins

    prev_means = None
    for it in range(max_iters):
        buckets = assign_pass(samples, means)                 # [H,N]
        new_means, bins = update_pass(samples, buckets, K, means)
        if tol > 0.0:
            prev_means = means
        means = new_means
        if tol > 0.0:
            shift = (means - prev_means).pow(2).sum(-1).sqrt().mean()
            if float(shift) <= tol:
                break

    # final counts matched to final means
    buckets = assign_pass(samples, means)
    _, bins = update_pass(samples, buckets, K, means)

    used_per_head = (bins > 0).sum(dim=1)

    used_per_label = None
    if element_names is not None:
        if len(element_names) != H:
            raise ValueError(f"element_names must have length H={H}, got {len(element_names)}")
        used_per_label = {element_names[h]: int(used_per_head[h].item()) for h in range(H)}

    return means, bins, used_per_head, used_per_label



def compact_clusters(means: torch.Tensor, bins: torch.Tensor, pad_to_max: bool = True):
    """
    Filter out empty clusters head-by-head.

    Args:
        means: [H, K, D]
        bins:  [H, K]
        pad_to_max: if True, returns padded tensors [H, K_used_max, ...];
                    if False, returns Python lists per head (ragged).

    Returns:
        If pad_to_max:
            used_means:  [H, K_used_max, D]
            used_bins:   [H, K_used_max]
            used_mask:   [H, K] boolean (where original clusters were used)
        Else:
            used_means_list: list of H tensors with shapes [K_h, D]
            used_bins_list:  list of H tensors with shapes [K_h]
            used_mask:       [H, K] boolean
    """
    H, K, D = means.shape
    used_mask = bins > 0

    if not pad_to_max:
        means_list, bins_list = [], []
        for h in range(H):
            m = means[h, used_mask[h]]
            b = bins[h, used_mask[h]]
            means_list.append(m)
            bins_list.append(b)
        return means_list, bins_list, used_mask

    max_used = int(used_mask.sum(dim=1).max().item()) if H > 0 else 0
    used_means = means.new_zeros((H, max_used, D))
    used_bins = bins.new_zeros((H, max_used), dtype=bins.dtype)
    for h in range(H):
        idx = used_mask[h].nonzero(as_tuple=False).squeeze(-1)
        k_h = idx.numel()
        if k_h > 0:
            used_means[h, :k_h] = means[h, idx]
            used_bins[h, :k_h] = bins[h, idx]
    return used_means, used_bins, used_mask


def batched_embedding(indices, embed):
    embed = embed.squeeze(0)              # (K, D)
    indices = indices.view(-1).long()     # Ensure shape (B,) and dtype long
    quantized = F.embedding(indices, embed)  # (B, D)
    return quantized.unsqueeze(0)         # (1, B, D)


import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, latent_dim, atom_feat_dim, margin=0.5, temperature=0.1, init_sigmoid_base=0.5):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(margin))
        self.temperature = temperature
        self.sigmoid_base = nn.Parameter(torch.tensor(init_sigmoid_base))
        self.layer_norm_z = nn.LayerNorm(latent_dim)
        self.layer_norm_atom = nn.LayerNorm(latent_dim)
        args = get_args()
        if args.dynamic_threshold:
            self.use_dynamic_threshold = True
        else:
            self.use_dynamic_threshold = False

    def sample_cap(self, t, max_n=200_000, with_replacement=False):
        """
        t: 1D/任意形状 Tensor（flattenして扱う場合は呼び出し側で調整）
        max_n: 取り出す最大サンプル数
        with_replacement: True なら重複あり（超省メモリ）
        """
        import torch
        n = t.numel()
        if n <= max_n:
            return t

        if with_replacement:
            # 重複あり：最小メモリ（長さ max_n のみ確保）
            idx = torch.randint(0, n, (max_n,), device=t.device)
            return t[idx]

        # 重複なし：GPU OOM回避のため CPU で randperm
        idx_cpu = torch.randperm(n, device='cpu')[:max_n]
        idx = idx_cpu.to(t.device, non_blocking=True)
        return t[idx]

    def forward(self, z, chunk, logger, codebook, key):
        import torch
        import torch.nn.functional as F

        device = z.device
        z = z.squeeze()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B, D = z.shape
        if B <= 1:
            return 0, 0, 0, 0, 0

        # ------------------------------------------------------------
        # 0) NORMALIZATION SWITCH (make geometry consistent everywhere)
        # ------------------------------------------------------------
        use_unit_norm = bool(getattr(self, "repel_use_unit_norm", True))
        # If you prefer to follow your VQ setting:
        # use_unit_norm = bool(getattr(self, "use_cosine_sim", False))

        if use_unit_norm:
            # keep gradients; eps avoids NaNs when norm ~ 0
            z_used = F.normalize(z, p=2, dim=-1, eps=1e-8)
            cb_used = F.normalize(codebook, p=2, dim=-1, eps=1e-8) if codebook is not None else None
        else:
            z_used = z
            cb_used = codebook

        N = z_used.shape[0]

        # ---- 0. Subsample for pdist to avoid O(N^2) explosion ----
        max_pdist_points = int(getattr(self, "max_pdist_points", 4096))

        if N > max_pdist_points:
            perm = torch.randperm(N, device=device)
            idx = perm[:max_pdist_points]
            z_for_pdist = z_used.index_select(0, idx)
        else:
            z_for_pdist = z_used

        # pairwise distances in the SAME geometry as everything else
        pdist_z = torch.pdist(z_for_pdist, p=2)  # 1D

        # （巨大時）サンプルを間引き
        sample = pdist_z
        if sample.numel() > 1_000_000:
            sample = self.sample_cap(sample, max_n=200_000)

        with torch.no_grad():
            dynamic_threshold = torch.quantile(sample, 0.10)
            lower_thresh = torch.quantile(sample, 0.25)
            upper_thresh = torch.quantile(sample, 0.75)
            center = (lower_thresh + upper_thresh) / 2

        # ---- ログ（負荷・転送を抑えて）----
        if chunk % 32 == 0:
            with torch.no_grad():
                s = sample
                if s.numel() > 100_000:
                    ridx = torch.randperm(s.numel(), device=s.device)[:100_000]
                    s = s.index_select(0, ridx)
                s_cpu = s.detach().to("cpu", dtype=torch.float32).flatten()
                hist = torch.histc(s_cpu, bins=10, min=0.0, max=1.0)
                vals = hist.tolist()
                # logger.info(f"[VQ_REPEL_HIST] key={key} {vals}")

        # ------------------------------------------------------------
        # 1) latent repel (blockwise) in the SAME geometry
        # ------------------------------------------------------------
        def latent_repel_mid_chunked(
            z_in,
            low,
            high,
            center,
            sigma=3.0,
            sharp=20.0,
            row_block=0,
            col_block=0,
            detach_weight=True,
            use_checkpoint=True,
            stream_backward=False,
            eps=1e-8,
        ):
            import torch
            import torch.utils.checkpoint as cp

            if (not z_in.requires_grad) or (not z_in.is_leaf and z_in.grad_fn is None):
                # encoder freeze / no_grad / detach のときは、この損失は意味がないので 0 にする
                return z_in.new_zeros(())
            assert not (use_checkpoint and stream_backward), \
                "checkpoint と streaming backward は同時に使えません。"

            B, D = z_in.shape
            device, dtype = z_in.device, z_in.dtype

            if self.training:
                assert z_in.requires_grad, "INPUT ERROR: z.requires_grad=False"

            band = float(abs(high - low))
            sigma_val = max(1e-4, 0.20 * band)
            sharp_val = float(sharp) if sharp is not None else 10.0

            low_t = torch.as_tensor(low, device=device, dtype=dtype)
            high_t = torch.as_tensor(high, device=device, dtype=dtype)
            center_t = torch.as_tensor(center, device=device, dtype=dtype)
            sigma_t = torch.as_tensor(sigma_val, device=device, dtype=dtype)

            if row_block <= 0 or row_block > B:
                row_block = B
            if col_block <= 0 or col_block > B:
                col_block = B

            def count_blocks(B, rb, cb):
                cnt = 0
                i = 0
                while i < B:
                    j = i
                    while j < B:
                        cnt += 1
                        j += cb
                    i += rb
                return max(cnt, 1)

            n_blocks_total = count_blocks(B, row_block, col_block)

            def block_loss(zi, zj, low_t, high_t, center_t, sigma_t):
                if self.training:
                    assert zi.requires_grad and zj.requires_grad

                if zi.numel() == 0 or zj.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

                # IMPORTANT: do distances in float32 for stability
                zi32 = zi.float()
                zj32 = zj.float()

                low32 = low_t.float()
                high32 = high_t.float()
                center32 = center_t.float()
                sigma32 = torch.clamp(sigma_t.float(), min=1e-6)

                d = torch.cdist(zi32, zj32, p=2)

                mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
                d = d[mask]
                if d.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

                x1 = (sharp_val * (d - low32)).clamp(-40, 40)
                x2 = (sharp_val * (high32 - d)).clamp(-40, 40)
                w = torch.sigmoid(x1) * torch.sigmoid(x2)
                if detach_weight:
                    w = w.detach()

                exp_arg = -((d - center32) ** 2) / (2 * sigma32 * sigma32)
                exp_arg = exp_arg.clamp(-60, 0)
                bell = torch.exp(exp_arg)

                out = (w * bell).sum() / w.sum().clamp_min(eps)
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                if self.training:
                    assert out.requires_grad
                return out

            def block_loss_ckpt(zi, zj, low_t, high_t, center_t, sigma_t):
                return block_loss(zi, zj, low_t, high_t, center_t, sigma_t)

            total = z_in.new_zeros(())
            i = 0
            while i < B:
                bi = min(row_block, B - i)
                zi = z_in[i:i + bi]

                j = i
                while j < B:
                    bj = min(col_block, B - j)
                    zj = z_in[j:j + bj]

                    if use_checkpoint:
                        lb = cp.checkpoint(
                            block_loss_ckpt,
                            zi, zj, low_t, high_t, center_t, sigma_t,
                            use_reentrant=False,
                        )
                    else:
                        lb = block_loss(zi, zj, low_t, high_t, center_t, sigma_t)

                    total = total + lb
                    j += col_block
                i += row_block

            out = total / n_blocks_total
            if self.training:
                assert out.requires_grad
            return out

        latent_repel_loss_mid = latent_repel_mid_chunked(
            z_used,
            low=lower_thresh, high=upper_thresh, center=center,
            sigma=1.0, sharp=20.0,
            row_block=1024, col_block=1024,
            detach_weight=True,
            use_checkpoint=True,
            stream_backward=False,
        )

        # ------------------------------------------------------------
        # 2) codebook repel (recommend normalize if use_unit_norm)
        # ------------------------------------------------------------
        def repel_codebooks_chunked(cb, sigma=1.0, block=4096):
            """
            cb: [K, D]
            returns avg exp(-d^2/(2*sigma^2)) over all pairs
            """
            K = cb.size(0)
            sum_val = cb.new_zeros(())
            cnt = 0

            for i in range(0, K, block):
                ci = cb[i:i + block]

                if ci.size(0) >= 2:
                    dij = torch.pdist(ci.float(), p=2)
                    if dij.numel() > 0:
                        li = torch.exp(-dij.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + li.sum()
                        cnt += li.numel()
                    del dij

                for j in range(i + block, K, block):
                    cj = cb[j:j + block]
                    if ci.numel() > 0 and cj.numel() > 0:
                        d = torch.cdist(ci.float(), cj.float(), p=2)
                        lj = torch.exp(-d.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + lj.sum()
                        cnt += lj.numel()
                        del d, lj
                    del cj
                del ci

            if cnt == 0:
                return sum_val
            return sum_val / cnt

        cb_loss = repel_codebooks_chunked(cb_used) if cb_used is not None else z_used.new_zeros(())

        # ------------------------------------------------------------
        # 3) auxiliary terms (same geometry)
        # ------------------------------------------------------------
        repel_from_2 = torch.exp(- (pdist_z - 2.0) ** 2 / (2 * 3.0 ** 2)).mean()

        def repel_from_zero_1d(d, margin):
            return torch.relu(margin - d).mean()

        latent_repel_loss = repel_from_zero_1d(pdist_z, lower_thresh) + cb_loss

        repel_weight = 1.0
        final_loss = repel_weight * latent_repel_loss_mid
        neg_loss = z.new_tensor(1.0)

        return final_loss, neg_loss, repel_from_2, cb_loss, latent_repel_loss


import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange

# 必要なら
# from utils import CORE_ELEMENTS
# from your_kmeans_module import kmeans
# from your_sampling_module import batched_sample_vectors, sample_vectors_distributed
# from your_distributed_module import distributed, noop
# from your_config_module import get_args
# from your_cb_dict import CBDICT

import re
import torch
import torch.nn as nn
from einops import rearrange

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=True,
        kmeans_iters=100,
        sync_kmeans=True,
        decay=0.1,
        eps=1e-5,
        threshold_ema_dead_code=2,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0,
    ):
        super().__init__()

        self.decay = float(decay)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        args = get_args()
        self.epoch_at_mode_shift = args.epoch_at_mode_shift
        self.train_or_infer = args.train_or_infer
        self.use_checkpoint = args.use_checkpoint

        self.cb_dict = CBDICT  # {element_key(str or int): K_e}

        assert not (
            use_ddp and num_codebooks > 1 and kmeans_init
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        )
        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and sync_kmeans else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # kmeans 済みかどうか
        self.register_buffer("initted", torch.tensor([not kmeans_init], dtype=torch.bool))

        # --- cluster_size / embed_avg (float32 固定) ---
        for elem in self.cb_dict.keys():
            K_e = int(self.cb_dict[elem])
            elem_str = str(elem)
            self.register_buffer(
                f"cluster_size_{elem_str}",
                torch.zeros(K_e, dtype=torch.float32),
            )
            self.register_buffer(
                f"embed_avg_{elem_str}",
                torch.zeros(K_e, dim, dtype=torch.float32),
            )

        self.learnable_codebook = learnable_codebook

        # element ごとのコードブック本体（ParameterDict のキーは「安全キー」に変換）
        self.embed = nn.ParameterDict()
        self.key_to_safe = {}  # original_key(str) -> safe_key(str)
        self.safe_to_key = {}  # safe_key(str) -> original_key(str)

        for key in self.cb_dict.keys():
            orig = str(key)
            K_e = int(self.cb_dict[key])
            safe = self._get_or_create_safe_key(orig, K_e, dim, device="cpu")
        self.ss_max_total_latent_count = args.ss_max_total_latent_count
        self.latent_size_sum = 0
        self.embed_ind_dict = {}
        self.quantize_dict = {}

    # ------------------------------------------------------------------
    # key まわりのユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_key(k: str) -> str:
        """ParameterDict 用の安全なキーに変換（属性名として有効な文字列）。"""
        s = re.sub(r"[^0-9A-Za-z_]", "_", k)
        if not s or not (s[0].isalpha() or s[0] == "_"):
            s = "k_" + s
        return s

    def ensure_from_state_dict(self, state_dict, prefix: str = ""):
        """
        state_dict に含まれる codebook 関連の buffer/parameter を事前に登録して、
        load_state_dict(strict=True) が通るようにする。

        prefix 例:
          - model 側で呼ぶなら: prefix="vq._codebook."
          - EuclideanCodebook 自身に渡すなら: prefix=""（推奨）
        """
        import torch
        import re
        from torch import nn

        # 1) cluster_size_*, embed_avg_* から orig key を拾って buffers を作る
        cs_pat = re.compile(rf"^{re.escape(prefix)}cluster_size_(.+)$")
        ea_pat = re.compile(rf"^{re.escape(prefix)}embed_avg_(.+)$")

        # まず embed_avg の D を拾いやすいので embed_avg を優先
        for k, v in state_dict.items():
            m = ea_pat.match(k)
            if not m:
                continue
            orig = m.group(1)  # 例: "6_-1_3_0_0_0"
            if not torch.is_tensor(v):
                continue
            K, D = int(v.shape[0]), int(v.shape[1])

            # buffers (orig 名)
            buf_cs = f"cluster_size_{orig}"
            buf_ea = f"embed_avg_{orig}"

            if not hasattr(self, buf_cs):
                self.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))
            if not hasattr(self, buf_ea):
                self.register_buffer(buf_ea, torch.zeros((K, D), dtype=torch.float32))

            # embed parameter (safe 名)
            safe = self._get_or_create_safe_key(orig, K_e=K, D=D, device="cpu")
            # ここで self.embed[safe] が無ければ作られる

        # embed_avg が無いケース向けに cluster_size だけでも拾う
        for k, v in state_dict.items():
            m = cs_pat.match(k)
            if not m:
                continue
            orig = m.group(1)
            if not torch.is_tensor(v):
                continue
            K = int(v.shape[0])

            buf_cs = f"cluster_size_{orig}"
            if not hasattr(self, buf_cs):
                self.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))

            # embed_avg が state に無い場合、D が不明なので embed はここでは作らない
            # （embed 側 key が state にあれば後段で作れる）

        # 2) embed.<safe> から safe key を拾って足りない Parameter を作る
        #    （orig が逆引きできない場合もあるので、とりあえず safe=orig 扱いで作る）
        emb_pat = re.compile(rf"^{re.escape(prefix)}embed\.(.+)$")
        for k, v in state_dict.items():
            m = emb_pat.match(k)
            if not m:
                continue
            safe = m.group(1)  # 例: "k_6__1_3_0_0_0"
            if not torch.is_tensor(v):
                continue

            # v の shape は [K, D] を想定（あなたの実装）
            if v.ndim != 2:
                continue
            K, D = int(v.shape[0]), int(v.shape[1])

            # safe をすでに持っていなければ Parameter を作る
            if safe not in self.embed:
                init = torch.randn((K, D), device="cpu") * 0.01
                self.embed[safe] = nn.Parameter(init, requires_grad=True)

            # mapping も最低限埋める（orig が分からなければ safe を orig 扱いに）
            if safe not in self.safe_to_key:
                self.safe_to_key[safe] = safe
            if self.safe_to_key[safe] not in self.key_to_safe:
                self.key_to_safe[self.safe_to_key[safe]] = safe

    def _get_or_create_safe_key(self, skey: str, K_e=None, D=None, device=None) -> str:
        """
        original key (skey) から safe_key を取得。
        必要なら Parameter を作成（K_e, D, device が与えられている場合）。
        """
        skey = str(skey)
        if skey in self.key_to_safe:
            safe = self.key_to_safe[skey]
        else:
            safe = self._safe_key(skey)
            self.key_to_safe[skey] = safe
            self.safe_to_key[safe] = skey

        if safe not in self.embed and K_e is not None and D is not None and device is not None:
            init = torch.randn(K_e, D, device=device) * 0.01
            self.embed[safe] = nn.Parameter(init, requires_grad=True)

        return safe

    # ------------------------------------------------------------------
    # ユーティリティ / 初期化系
    # ------------------------------------------------------------------
    def reset_kmeans(self):
        self.initted.data.fill_(False)

    def copy_codebook_(self, embed: torch.Tensor, init: torch.Tensor, fill="data", data=None):
        """
        embed: [K, D] or [D, K] current codebook
        init : [K_used, D] or [D, K_used] initializer
        fill : "data" | "repeat" | "randn"
        data : [N, D] latents to sample from if fill=="data"
        """
        def to_KD(t, D_expected):
            if t.shape[-1] == D_expected:
                return t  # [K_used, D]
            if t.shape[0] == D_expected:
                return t.t().contiguous()
            raise ValueError(f"init shape {tuple(t.shape)} not compatible with D={D_expected}")

        if embed.shape[-1] < embed.shape[0]:  # [K, D] とみなす
            K, D = embed.shape
            initKD = to_KD(init, D)
            K_used = initKD.shape[0]
            n = min(K, K_used)
            embed[:n].copy_(initKD[:n])

            if K > n:
                if fill == "data" and data is not None and data.numel() > 0:
                    idx = torch.randint(0, data.size(0), (K - n,), device=embed.device)
                    embed[n:K].copy_(data[idx])
                elif fill == "repeat" and n > 0:
                    reps = (K - n + n - 1) // n
                    embed[n:K].copy_(initKD[:n].repeat((reps, 1))[: K - n])
                elif fill == "randn":
                    embed[n:K].normal_(0, 1e-3)
                else:
                    embed[n:K].copy_(embed[:1])
        else:  # [D, K]
            D, K = embed.shape
            initKD = to_KD(init, D)
            initDK = initKD.t().contiguous()  # [D, K_used]
            n = min(K, initDK.shape[1])
            embed[:, :n].copy_(initDK[:, :n])
            if K > n:
                if fill == "randn":
                    embed[:, n:K].normal_(0, 1e-3)
                else:
                    embed[:, n:K].copy_(embed[:, :1])


    @staticmethod
    def silhouette_score_torch(
        X,
        labels,
        row_block: int = 8192,
        device: str | torch.device | None = None,
    ) -> float:
        """
        Compute Silhouette score in (mini)blocks, on GPU if available.

        Parameters
        ----------
        X : Tensor or nn.Module or nn.ParameterDict
            - Tensor: shape [N, D]
            - nn.Module: must have `.weight` or `.embed` tensor
            - nn.ParameterDict: values are [K, D] parameters → concatenated to [N, D]
        labels : Tensor
            Cluster labels of shape [N] (int).
        row_block : int
            Block size along N for pairwise distance computation.
        device : str or torch.device or None
            Target device. If None: "cuda" if available else "cpu".
        """

        # ---- 0. 正しいテンソル X を取り出す ------------------------------------
        # ParameterDict (or類似) の場合：中身を縦に concat
        if isinstance(X, nn.ParameterDict):
            # 各 value が [K, D] を想定
            X = torch.cat([p.view(p.shape[0], -1) for p in X.values()], dim=0)  # [N, D]

        # nn.Module の場合：.embed または .weight を拾う
        elif isinstance(X, nn.Module):
            weight = getattr(X, "weight", None)
            embed = getattr(X, "embed", None)
            if embed is not None:
                X = embed
            elif weight is not None:
                X = weight
            else:
                raise TypeError("X is a module without .embed/.weight tensor.")

        # ここまで来たら X は Tensor のはず
        if not torch.is_tensor(X):
            raise TypeError(
                f"X must be a Tensor or Module/ParameterDict of Tensors, but got {type(X)}"
            )

        # ---- 1. デバイス決定 & 型整形 -----------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # X: float32, labels: long に統一
        X = X.detach().to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.detach().to(device=device, dtype=torch.long, non_blocking=True)

        # 余計な次元を削る (e.g. [N, D, 1] → [N, D])
        X = X.squeeze()
        labels = labels.squeeze()

        # N チェック
        if X.ndim == 1:
            X = X.unsqueeze(1)  # [N] → [N, 1]

        N = X.shape[0]
        if N <= 1:
            return 0.0

        # ---- 2. ラベルを 0..K-1 にリマップ（空クラスタがあっても無視される） ----
        uniq, inv = labels.unique(sorted=True, return_inverse=True)  # inv: [N] in 0..K-1
        K = uniq.numel()
        if K <= 1:
            return 0.0

        counts = torch.bincount(inv, minlength=K).to(device)  # >0 by construction
        counts_f = counts.float()

        # ---- 3. ブロック毎に Silhouette を計算 --------------------------------
        sil_sum = 0.0
        processed = 0

        for start in range(0, N, row_block):
            end = min(start + row_block, N)
            B = end - start

            # このブロックのサンプル
            Xb = X[start:end]  # [B, D]

            # [B, N] の距離行列
            d_block = torch.cdist(Xb, X, p=2)  # Euclid

            # クラスタごとの距離総和を計算
            inv_index = inv.view(1, N).expand(B, N)  # [B, N]
            sums_per_cluster = torch.zeros(B, K, device=device, dtype=d_block.dtype)
            sums_per_cluster.scatter_add_(1, inv_index, d_block)  # [B, K]

            # クラスタごとの平均距離
            means_per_cluster = sums_per_cluster / counts_f.view(1, K)  # [B, K]

            # ----- a(i): 同じクラスタ内の平均距離（自分自身を除外） -------------
            k_block = inv[start:end]  # [B]
            a_counts = counts[k_block] - 1  # 自分自身を除いた個数

            # とりあえず "含自分" の平均距離
            a_mean = means_per_cluster.gather(1, k_block[:, None]).squeeze(1)  # [B]

            # クラスタサイズが 1 の場合は 0、>1 の場合だけ補正係数を載せる
            scale = torch.zeros_like(a_counts, dtype=a_mean.dtype)
            mask_gt1 = a_counts > 0
            if mask_gt1.any():
                # n / (n - 1) で「自分自身 (距離0) を除いた平均」に補正
                n = counts[k_block][mask_gt1].float()
                scale[mask_gt1] = n / (n - 1.0)
            a_i = a_mean * scale  # [B]; クラスタサイズ 1 のものは 0 のまま

            # ----- b(i): 他クラスタの平均距離の最小値 ---------------------------
            # 自分のクラスタの列は無限大にして除外
            means_per_cluster.scatter_(1, k_block[:, None], float("inf"))
            b_i, _ = means_per_cluster.min(dim=1)  # [B]

            # ----- Silhouette: (b - a) / max(a, b) -------------------------------
            denom = torch.maximum(a_i, b_i)
            sil_block = (b_i - a_i) / denom
            sil_block = torch.nan_to_num(
                sil_block, nan=0.0, posinf=0.0, neginf=0.0
            )

            sil_sum += sil_block.sum().item()
            processed += B

            # メモリ解放
            del Xb, d_block, sums_per_cluster, means_per_cluster, a_mean, a_i, b_i, sil_block

        return float(sil_sum / max(processed, 1))

    import math
    import torch

    @torch.jit.ignore
    def compute_cluster_metrics(self, y: torch.Tensor, K_req: int = None, topk: int = 5):
        """
        y: int64 labels, shape [N] (already compressed to 0..nuniq-1 recommended)
        Returns:
          N, K_req, K_eff, max_frac, topk_frac, entropy, perplexity,
          singleton_ratio (clusters), singleton_point_ratio (points)
        """
        # y must be 1D long
        if y is None or y.numel() == 0:
            return {
                "N": 0,
                "K_req": int(K_req) if K_req is not None else None,
                "K_eff": 0,
                "max_frac": 0.0,
                "topk_frac": 0.0,
                "entropy": 0.0,
                "perplexity": 1.0,
                "singleton_ratio": 0.0,
                "singleton_point_ratio": 0.0,
            }

        y = y.long().view(-1)
        N = int(y.numel())

        # compress labels to 0..nuniq-1 (robust even if already compressed)
        _, y = torch.unique(y, return_inverse=True)

        bc = torch.bincount(y)  # shape [nuniq]
        nz_mask = (bc > 0)
        bc_nz = bc[nz_mask]
        K_eff = int(bc_nz.numel())

        # fractions
        bc_float = bc_nz.float()
        denom = float(max(1, N))
        p = (bc_float / denom).clamp_min(0.0)  # sum to 1 over active clusters

        # max_frac
        max_frac = float(p.max().item()) if K_eff > 0 else 0.0

        # topk_frac
        k = int(min(max(1, topk), K_eff)) if K_eff > 0 else 1
        if K_eff > 0:
            topk_vals = torch.topk(p, k=k, largest=True).values
            topk_frac = float(topk_vals.sum().item())
        else:
            topk_frac = 0.0

        # entropy (natural log) and perplexity
        # H = -sum p log p
        # perplexity = exp(H)
        import math
        if K_eff > 0:
            H = float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())
            ppl = float(math.exp(H))
        else:
            H, ppl = 0.0, 1.0

        # singleton ratios
        singletons = (bc_nz == 1).sum()
        singleton_ratio = float((singletons.float() / max(1.0, float(K_eff))).item())  # clusters
        singleton_point_ratio = float((bc_nz[bc_nz == 1].sum().float() / denom).item())  # points

        out = {
            "N": N,
            "K_req": int(K_req) if K_req is not None else None,
            "K_eff": K_eff,
            "max_frac": max_frac,
            "topk_frac": topk_frac,
            "entropy": H,
            "perplexity": ppl,
            "singleton_ratio": singleton_ratio,
            "singleton_point_ratio": singleton_point_ratio,
        }
        return out

    @torch.jit.ignore
    def init_embed_(self, data, logger, epoch, mask_dict=None, use_cosine_sim: bool = False):
        """
        Initialize per-element codebooks (absolute K from self.cb_dict).

        Guarantees:
          - kmeans init happens ONLY at epoch==1
            (epoch>1: if params/buffers missing -> raise error immediately)
          - cluster metrics computed every epoch
          - metrics/argmin are computed over ALL K_req centers (no active filtering)
          - DIAG stats never produce NaN due to N==1 (uses correction=0 / guarded std)
        """
        import os
        import torch
        import torch.nn as nn

        assert mask_dict is not None, "mask_dict is required"
        device = data.device
        D = int(data.shape[-1])

        # -------------------------
        # helpers
        # -------------------------
        def _get_from_dict(d, k):
            if k in d:
                return d[k]
            return d.get(str(k), None)

        def _get_absK(d, k):
            v = _get_from_dict(d, k)
            return None if v is None else int(v)

        def _stats(x_nd: torch.Tensor):
            """Robust data stats for padding. Returns (mu, sigma) or None."""
            if x_nd is None or x_nd.numel() == 0:
                return None
            mu = x_nd.mean(dim=0)
            with torch.no_grad():
                # robust "radius" median
                dev = (x_nd - mu).pow(2).sum(dim=1).sqrt()
                sigma = torch.quantile(dev, 0.5)
            return (mu, sigma)

        def _pad_to_K(means_1kd, counts_1k, K_req: int, data_stats=None):
            """
            means_1kd: [1, K_run, D]
            counts_1k: [1, K_run]
            return:
              means_kd: [K_req, D]
              counts_k: [K_req]
            """
            H, K_run, Dd = means_1kd.shape
            assert H == 1 and Dd == D

            means_kd = means_1kd[0]  # (K_run, D)
            counts_k = counts_1k[0]  # (K_run,)

            if K_req <= K_run:
                return means_kd[:K_req].contiguous(), counts_k[:K_req].contiguous()

            K_pad = K_req - K_run
            out_means = torch.empty((K_req, D), device=means_kd.device, dtype=means_kd.dtype)
            out_counts = torch.zeros((K_req,), device=counts_k.device, dtype=counts_k.dtype)

            out_means[:K_run].copy_(means_kd)
            out_counts[:K_run].copy_(counts_k)

            if data_stats is not None:
                mu, sigma = data_stats
                # small noise around mu (sigma can be 0 for tiny sets)
                noise = torch.randn((K_pad, D), device=means_kd.device, dtype=means_kd.dtype) * (0.01 * sigma + 1e-6)
                out_means[K_run:] = mu + noise
            else:
                out_means[K_run:].zero_()

            return out_means, out_counts

        def _ensure_param(safe: str, shape):
            """Ensure self.embed[safe] exists with exact shape (recreate if mismatch)."""
            if (safe not in self.embed) or (tuple(self.embed[safe].shape) != tuple(shape)):
                self.embed[safe] = nn.Parameter(
                    torch.zeros(shape, device=device, dtype=torch.float32),
                    requires_grad=True,
                )

        def _safe_var_mean(x: torch.Tensor):
            """Mean variance over dim=0, never NaN for N==1."""
            if x is None or x.numel() == 0:
                return float("nan")
            if x.shape[0] < 2:
                return 0.0
            return x.float().var(dim=0, correction=0).mean().item()

        def _safe_std(x: torch.Tensor):
            """Std over a 1D tensor, never NaN for len==1."""
            if x is None or x.numel() == 0:
                return float("nan")
            if x.numel() < 2:
                return 0.0
            return x.float().std(correction=0).item()

        # -------------------------
        # main loop
        # -------------------------
        for raw_key in sorted(mask_dict.keys(), key=lambda x: str(x)):
            skey = str(raw_key)

            idx = _get_from_dict(mask_dict, raw_key)
            if idx is None:
                idx = _get_from_dict(mask_dict, skey)
            if idx is None:
                continue

            masked = data[0][idx]  # (N_i, D)
            N_i = int(masked.shape[0])
            if N_i == 0:
                continue

            # optional dump for UMAP/debug
            if skey == "6_0_3_1_1_0":
                os.makedirs("dumps", exist_ok=True)
                path = f"dumps/masked_latents_epoch{epoch}_{skey}.pt"
                torch.save(
                    {
                        "epoch": int(epoch) if epoch is not None else None,
                        "key": skey,
                        "masked_latents": masked.detach().to("cpu", dtype=torch.float16),
                        "N": int(masked.shape[0]),
                        "D": int(masked.shape[1]),
                    },
                    path,
                )

            # K settings
            K_req = _get_absK(self.cb_dict, skey)
            if K_req is None or K_req <= 0:
                K_req = 1
            K_run = min(K_req, N_i)

            # safe key and buffer names
            safe = self._get_or_create_safe_key(skey, K_req, D, device=device)
            buf_name_cs = f"cluster_size_{skey}"
            buf_name_ea = f"embed_avg_{skey}"

            # -------------------------
            # 1) Initialize ONLY at epoch 1
            #    epoch>1: if missing -> hard error
            # -------------------------
            missing_param = (safe not in self.embed)
            missing_bufs = (not hasattr(self, buf_name_cs)) or (not hasattr(self, buf_name_ea))

            if epoch == 1:
                need_init = True
            else:
                if missing_param or missing_bufs:
                    raise RuntimeError(
                        f"[init_embed_] missing at epoch={epoch} key={skey} "
                        f"missing_param={missing_param} missing_bufs={missing_bufs} safe={safe}"
                    )
                need_init = False

            if need_init:
                means_1kd, counts_1k, *_ = kmeans(
                    masked.unsqueeze(0).to(device),
                    num_clusters=K_run,
                    use_cosine_sim=use_cosine_sim,
                )

                ds = _stats(masked)
                means_kd, counts_k = _pad_to_K(means_1kd, counts_1k, K_req, data_stats=ds)

                # ensure parameter exists with correct shape, then copy
                _ensure_param(safe, (K_req, D))
                self.embed[safe].data.copy_(means_kd.detach().to(device=device, dtype=self.embed[safe].dtype))

                # init EMA buffers (float32)
                cs = counts_k.to(device=device, dtype=torch.float32)  # (K_req,)
                ea = means_kd.detach().to(device=device, dtype=torch.float32) * cs.view(-1, 1)  # (K_req, D)

                # replace buffers safely
                if hasattr(self, buf_name_cs):
                    delattr(self, buf_name_cs)
                if hasattr(self, buf_name_ea):
                    delattr(self, buf_name_ea)
                self.register_buffer(buf_name_cs, cs)
                self.register_buffer(buf_name_ea, ea)

                nz = int((cs > 0).sum().item())
                logger.info(f"[init_embed_] Z={skey} N={N_i} K_req={K_req} K_run={K_run} K_used={nz}/{K_req}")

            # -------------------------
            # 2) Cluster metrics every epoch (NO SS)
            # -------------------------
            with torch.no_grad():
                centers_clst = self.embed[safe].detach()  # (K_req, D)

                # active count is just for logging/diagnostics
                cs_now = getattr(self, buf_name_cs, None)
                active = (cs_now > 0) if cs_now is not None else None
                K_active = int(active.sum().item()) if active is not None else -1

                max_n = int(getattr(self, "clst_max_n", 20000))

                # subsample X first (avoid full argmin on huge N)
                if masked.shape[0] > max_n:
                    perm = torch.randperm(masked.shape[0], device=masked.device)[:max_n]
                    x_eval = masked[perm]
                else:
                    x_eval = masked

                # optional diagnostics (safe stats; never NaN from N==1)
                def diag(tag: str, X: torch.Tensor, centers: torch.Tensor):
                    cn = centers.float().norm(dim=-1)
                    print(
                        f"[DIAG]{tag} "
                        f"X shape={tuple(X.shape)} centers shape={tuple(centers.shape)} "
                        f"X finite={torch.isfinite(X).all().item()} "
                        f"C finite={torch.isfinite(centers).all().item()} "
                        f"X var={_safe_var_mean(X):.3e} "
                        f"C var={_safe_var_mean(centers):.3e} "
                        f"C mean-norm={cn.mean().item():.3e} "
                        f"C std-norm={_safe_std(cn):.3e}"
                    )

                diag(" clst", x_eval, centers_clst)

                # assign labels using ALL centers (K_req)
                y_eval = self.argmin_dist_blockwise_l2(x_eval, centers_clst, k_block=1024)

                # compute metrics from labels only
                metrics = self.compute_cluster_metrics(y_eval, K_req=K_req, topk=5)

                logger.info(
                    f"[CLST] key={skey} epoch={epoch} "
                    f"N={metrics['N']} K_req={metrics['K_req']} K_eff={metrics['K_eff']} "
                    f"max_frac={metrics['max_frac']:.4f} top5_frac={metrics['topk_frac']:.4f} "
                    f"H={metrics['entropy']:.4f} ppl={metrics['perplexity']:.2f} "
                    f"singleton_ratio={metrics['singleton_ratio']:.4f} "
                    f"singleton_point_ratio={metrics['singleton_point_ratio']:.4f} "
                    f"(K_active={K_active})"
                )

    def _normalize_mask_dict(self, mask_dict, device=None):
        import numpy as np

        if mask_dict is None:
            return None
        norm = {}
        for k, v in mask_dict.items():
            k_int = int(k) if isinstance(k, str) and k.isdigit() else k
            k_str = str(k_int)

            if isinstance(v, (list, tuple, np.ndarray)):
                v = torch.as_tensor(v, dtype=torch.long, device=device)
            elif isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                v = torch.nonzero(v.flatten(), as_tuple=False).flatten().long().to(device)

            norm[k_int] = v
            norm[k_str] = v
        return norm

    import torch

    @torch.no_grad()
    def argmin_dist_blockwise_l2(self, x: torch.Tensor, code: torch.Tensor, k_block: int = 1024):
        x_f = x.float()
        c_f = code.float()

        # 早期診断（必要ならコメントアウト）
        if not torch.isfinite(x_f).all():
            bad = (~torch.isfinite(x_f)).sum().item()
            print(f"[argmin] non-finite x: {bad}")
        if not torch.isfinite(c_f).all():
            bad = (~torch.isfinite(c_f)).sum().item()
            print(f"[argmin] non-finite code: {bad}")

        x2 = (x_f * x_f).sum(dim=1, keepdim=True)  # [N,1]

        best_val = torch.full((x_f.size(0),), float("inf"), device=x.device, dtype=torch.float32)
        best_idx = torch.zeros((x_f.size(0),), device=x.device, dtype=torch.long)

        K = c_f.size(0)
        for k0 in range(0, K, k_block):
            k1 = min(k0 + k_block, K)
            ck = c_f[k0:k1]  # [kb,D]
            c2 = (ck * ck).sum(dim=1).unsqueeze(0)  # [1,kb]

            xc = x_f @ ck.t()  # [N,kb]
            d2 = x2 + c2 - 2.0 * xc

            # ★ NaN/Inf を潰す（ここが本命）
            d2 = torch.nan_to_num(d2, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            d2.clamp_min_(0.0)

            vals, idxs = d2.min(dim=1)

            # vals が inf の点は更新されないので、そのままだと0固定になり得る
            update = vals < best_val
            best_val = torch.where(update, vals, best_val)
            best_idx = torch.where(update, idxs + k0, best_idx)

        # 追加の診断：一度も更新できてない点があるなら warning
        if torch.isinf(best_val).any():
            n_bad = torch.isinf(best_val).sum().item()
            print(f"[argmin] WARNING: {n_bad} points never updated (all distances non-finite).")

        return best_idx

    @torch.no_grad()
    def argmax_sim_blockwise_cos(self, x: torch.Tensor, code: torch.Tensor, k_block: int = 2048, eps: float = 1e-12):
        """
        cosine similarity argmax without allocating [N,K]
        """
        x_f = torch.nn.functional.normalize(x.float(), p=2, dim=1, eps=eps)
        c_f = torch.nn.functional.normalize(code.float(), p=2, dim=1, eps=eps)

        best_val = torch.full((x_f.size(0),), -float("inf"), device=x.device, dtype=torch.float32)
        best_idx = torch.zeros((x_f.size(0),), device=x.device, dtype=torch.long)

        K = c_f.size(0)
        for k0 in range(0, K, k_block):
            k1 = min(k0 + k_block, K)
            ck = c_f[k0:k1]  # [kb,D]
            sims = x_f @ ck.t()  # [N,kb]
            vals, idxs = sims.max(dim=1)
            update = vals > best_val
            best_val = torch.where(update, vals, best_val)
            best_idx = torch.where(update, idxs + k0, best_idx)
            del ck, sims, vals, idxs, update

        return best_idx

    # @torch.no_grad()
    # def split_the_winner_ema(
    #         self,
    #         embed,  # (K, D)  centers tensor (writeable view)
    #         ema_sum,  # (K, D)  EMA sum of assigned latents (same shape as embed)
    #         ema_count,  # (K,)    EMA count
    #         usage_ema,  # (K,)    EMA usage (for split decision)
    #         batch_counts,  # (K,)    batch histogram (float)
    #         batch_counts2,
    #         split_thr=0.15,
    #         prune_src_thr=0.005,  # prefer donor codes with p < this
    #         noise_scale=0.02,
    #         cooldown=None,  # (K,) int/long buffer
    #         cooldown_steps=2000,
    #         eps=1e-8,
    # ):

    import torch

    @torch.no_grad()
    def split_the_winner_ema(
            self,
            *,
            embed,  # (K,D)  <-- IMPORTANT: this is MEANS (centers), writable view
            ema_sum,  # (K,D)  <-- EMA SUM buffer (embed_avg_*), writable
            ema_count,  # (K,)   <-- EMA COUNT buffer (cluster_size_*), writable
            usage_ema,  # (K,)   <-- usage EMA (already updated outside), writable
            batch_counts,  # (K,)   <-- current minibatch histogram (float)
            batch_counts2=None,  # unused
            # PCA-guided split inputs (per-key, per-minibatch)
            X_batch=None,  # (N_i, D) latents for THIS key in THIS minibatch
            assign=None,  # (N_i,) hard assignment in [0..K-1] for THIS key

            split_thr=0.15,
            prune_src_thr=0.005,
            noise_scale=0.02,
            cooldown=None,  # (K,) long
            cooldown_steps=2000,
            eps=1e-8,
            min_points=32,  # PCA needs some points; raise from 8 to be safer
            pca_power=0.5,  # step ~ pca_power * std along top PC
            fallback_to_noise=True,
    ):
        """
        Split-the-winner (PCA-guided) that **persists** under EMA recomputation.

        Contract:
          - `embed` is interpreted as MEANS (centers): embed[k] = ema_sum[k] / (ema_count[k] + eps)
          - This function will update BOTH:
              (A) embed[winner], embed[donor]  (means)
              (B) ema_sum / ema_count          so that next `means = ema_sum/ema_count` preserves the split
              (C) usage_ema                    split 50/50 so donor doesn't inherit full popularity
              (D) cooldown (optional)

        Winner selection is based on usage_ema distribution p = usage_ema / sum(usage_ema).
        Donor is chosen among low-usage codes if possible.
        """

        import torch

        K, D = embed.shape
        device = embed.device

        # -------------------------
        # 0) cooldown tick
        # -------------------------
        if cooldown is not None:
            cooldown.sub_(1).clamp_(min=0)

        # -------------------------
        # 1) compute p from usage_ema
        # -------------------------
        total = usage_ema.sum()
        if float(total.item()) <= 0.0:
            return False

        p = usage_ema / (total + float(eps))

        # -------------------------
        # 2) eligibility mask
        # -------------------------
        if cooldown is not None:
            eligible = (cooldown <= 0)
        else:
            eligible = torch.ones(K, dtype=torch.bool, device=device)

        if not bool(eligible.any()):
            return False

        # -------------------------
        # 3) pick winner among eligible
        # -------------------------
        p_eligible = torch.where(eligible, p, torch.full_like(p, -float("inf")))
        winner = int(torch.argmax(p_eligible).item())
        winner_p = float(p[winner].item())

        if (not torch.isfinite(p_eligible[winner])) or (winner_p <= float(split_thr)):
            return False

        # -------------------------
        # 4) pick donor
        #     prefer low-usage among eligible and != winner
        # -------------------------
        low = (p < float(prune_src_thr)) & eligible
        low[winner] = False

        if bool(low.any()):
            p_low = torch.where(low, p, torch.full_like(p, float("inf")))
            donor = int(torch.argmin(p_low).item())
        else:
            elig2 = eligible.clone()
            elig2[winner] = False
            if not bool(elig2.any()):
                return False
            p_elig2 = torch.where(elig2, p, torch.full_like(p, float("inf")))
            donor = int(torch.argmin(p_elig2).item())

        if donor == winner:
            return False
        # -------------------------
        # 4.5) guard: too few points for a meaningful split
        # -------------------------
        if (X_batch is None) or (assign is None):
            return False  # ← X_batch 無しで split させたくないならこう
        # ここでの N_eff は「この key のミニバッチ中の点数」
        N_eff = int(X_batch.shape[0])
        K_eff = int(K)  # embed.shape[0]

        if N_eff < max(8, 2 * K_eff):
            return False

        # -------------------------
        # 5) determine split direction (PCA if possible)
        # -------------------------
        did_pca = False

        cw = embed[winner].clone()  # current winner center (mean)

        if (X_batch is not None) and (assign is not None):
            if X_batch.ndim == 2 and assign.ndim == 1 and X_batch.shape[0] == assign.shape[0]:
                aw = (assign == int(winner))
                n_w = int(aw.sum().item())
                if n_w >= int(min_points):
                    Xw = X_batch[aw].to(device=device, dtype=torch.float32)  # (n_w, D)
                    mu = Xw.mean(dim=0, keepdim=True)
                    Y = Xw - mu

                    # guard: if variance is almost zero, PCA is meaningless
                    if torch.isfinite(Y).all() and float(Y.var(dim=0, unbiased=False).mean().item()) > 1e-12:
                        try:
                            # top principal direction via SVD
                            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                            v = Vh[0]  # (D,)
                            proj = (Y @ v)  # (n_w,)
                            std = proj.std(unbiased=False) + float(eps)
                            step = float(pca_power) * std

                            if torch.isfinite(v).all() and torch.isfinite(step) and float(step.item()) > 0.0:
                                # move winner and donor in opposite directions
                                embed[winner].copy_(cw - v * step)
                                embed[donor].copy_(cw + v * step)
                                did_pca = True
                        except Exception:
                            did_pca = False

        # fallback: winner + small noise
        if (not did_pca) and fallback_to_noise:
            # scale noise by center RMS so it is neither tiny nor huge
            rms = torch.sqrt((cw * cw).mean() + float(eps))
            noise = torch.randn(D, device=device, dtype=torch.float32) * (float(noise_scale) * rms)
            embed[winner].copy_(cw)  # keep winner as-is
            embed[donor].copy_(cw + noise)

        # -------------------------
        # 6) split EMA buffers so split persists
        # -------------------------
        if (ema_sum is not None) and (ema_count is not None):
            # ensure types
            if ema_sum.dtype != torch.float32:
                ema_sum = ema_sum.float()
            if ema_count.dtype != torch.float32:
                ema_count = ema_count.float()

            # if winner has ~0 count, splitting counts is pointless; still seed donor with small count
            cw_cnt = float(ema_count[winner].item())
            if cw_cnt <= 0.0:
                # seed a tiny mass so the new donor isn't erased instantly by division
                seed = 1.0
                ema_count[winner] = seed
                ema_sum[winner].copy_(embed[winner] * seed)
                cw_cnt = seed

            # 50/50 split counts
            cnt_w_new = ema_count[winner] * 0.5
            cnt_d_new = ema_count[winner] - cnt_w_new  # keep exact sum

            ema_count[winner] = cnt_w_new
            ema_count[donor] = cnt_d_new

            # reconstruct sums from the NEW means so consistency is exact:
            # ema_sum[k] should equal embed[k] * ema_count[k] (approximately)
            ema_sum[winner].copy_(embed[winner] * ema_count[winner].unsqueeze(0))
            ema_sum[donor].copy_(embed[donor] * ema_count[donor].unsqueeze(0))

        # -------------------------
        # 7) split usage_ema too (so donor survives)
        # -------------------------
        usage_w = usage_ema[winner].clone()
        usage_ema[winner].mul_(0.5)
        usage_ema[donor] = usage_w * 0.5

        # -------------------------
        # 8) cooldown both
        # -------------------------
        if cooldown is not None:
            cooldown[winner] = int(cooldown_steps)
            cooldown[donor] = int(cooldown_steps)

        return True

    # ------------------------------------------------------------------
    # Forward: ここで EMA update を dtype 安全 & safe-key 化
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Forward: dtype-safe, safe-key buffers, no crash on empty masks,
    #          entropy as METRIC (not differentiable in hard-argmin EMA),
    #          optional per-key logging.
    # ------------------------------------------------------------------
    import torch
    from einops import rearrange

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
            self,
            x,
            feature=None,
            mask_dict=None,
            logger=None,
            chunk_i=None,
            epoch=None,
            mode=None,
            is_last_batch: bool = False,
    ):
        """
        Per-element vector quantization codebook forward (L2norm unified + N-dependent split policy).

        Option A (recommended): keep BOTH assignment and EMA update in the SAME L2-normalized space.
          - assignment uses l2-normalized latents & l2-normalized centers
          - EMA updates use l2-normalized latents (NOT raw latents)
          - centers stored in codebook remain l2-normalized (or close; we explicitly normalize at write-back)
        """
        import os, time
        import torch
        from einops import rearrange

        # -----------------------------
        # small helpers
        # -----------------------------
        def _log(msg: str):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        def _safe_l2norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
            den = t.norm(dim=-1, keepdim=True).clamp_min(eps)
            t = t / den
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        def _assign_l2norm(masked_latents: torch.Tensor, code: torch.Tensor, k_block: int = 1024) -> torch.Tensor:
            # L2norm unified assignment (cosine-ish)
            ml = _safe_l2norm(masked_latents)
            cc = _safe_l2norm(code)
            return self.argmin_dist_blockwise_l2(ml, cc, k_block=k_block)

        # -----------------------------
        # N-dependent split policy
        # -----------------------------
        def _target_max_p(N: int) -> float:
            if N >= 1_000_000:
                return float(getattr(self, "split_thr_big", 0.05))
            if N >= 200_000:
                return float(getattr(self, "split_thr_mid", 0.08))
            if N >= 50_000:
                return float(getattr(self, "split_thr_small", 0.12))
            return float(getattr(self, "split_thr_default", 0.20))

        diag_keys = set(getattr(self, "diag_keys", []))
        k_block = int(getattr(self, "assign_k_block", 1024))

        do_split = bool(getattr(self, "do_split_the_winner", True))
        split_once_per_epoch = True
        split_only_last_batch = False

        usage_mom = float(getattr(self, "usage_ema_mom", 0.99))
        decay = float(getattr(self, "decay", 0.1))
        eps = float(getattr(self, "eps", 1e-8))
        prune_src_thr = float(getattr(self, "prune_src_thr", 0.005))

        # -----------------------------
        # 0) global offset management
        # -----------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0
        if chunk_i is not None and chunk_i == 0:
            self.latent_size_sum = 0

        # -----------------------------
        # 1) reshape -> flatten (1,B,D)
        # -----------------------------
        x = x.float()
        if x.ndim < 4:
            x = rearrange(x, "... -> 1 ...")
        flatten = x.view(x.shape[0], -1, x.shape[-1])  # (1,B,D)
        B, D = int(flatten.shape[1]), int(flatten.shape[2])

        # init embed (kept)
        if mode == "init_kmeans_final":
            self.init_embed_(flatten, logger, epoch, mask_dict=mask_dict)

        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        self.quantize_dict = {}
        self.embed_ind_dict = {}

        mask_dict = self._normalize_mask_dict(mask_dict, device=flatten.device)

        # -----------------------------
        # 2) safe-key <-> int id mapping
        # -----------------------------
        if not hasattr(self, "_safe2id"):
            self._safe2id = {}
        if not hasattr(self, "_id2safe"):
            self._id2safe = {}

        if not hasattr(self, "cb_dict"):
            self.cb_dict = {}

        def _safe_id(safe: str) -> int:
            sid = self._safe2id.get(safe)
            if sid is None:
                sid = len(self._safe2id)
                self._safe2id[safe] = sid
                self._id2safe[sid] = safe
            return sid

        def _get_safe_and_code(skey: str):
            K_e = int(self.cb_dict.get(skey, self.codebook_size))
            safe = self._get_or_create_safe_key(skey, K_e=K_e, D=D, device=flatten.device)
            code_param = self.embed[safe]  # (1,K,D) or (K,D)
            code = code_param.squeeze(0) if code_param.ndim == 3 else code_param  # (K,D)
            return safe, code_param, code

        # -----------------------------
        # 3) infer: IDs only
        # -----------------------------
        if mode == "infer":
            if mask_dict is None:
                raise ValueError("mode='infer' requires mask_dict (global indices).")

            key_id_full = torch.full((B,), -1, device=flatten.device, dtype=torch.int32)
            cluster_id_full = torch.full((B,), -1, device=flatten.device, dtype=torch.int32)

            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue
                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                masked_latents = flatten[0].index_select(0, idx_local)  # (Ni,D)
                if masked_latents.numel() == 0:
                    continue

                skey = str(key)
                safe, _code_param, code = _get_safe_and_code(skey)
                sid = _safe_id(safe)

                with torch.no_grad():
                    idx_code = _assign_l2norm(masked_latents, code, k_block=k_block).to(dtype=torch.int32)

                key_id_full.index_copy_(
                    0,
                    idx_local,
                    torch.full((idx_local.numel(),), sid, device=flatten.device, dtype=torch.int32),
                )
                cluster_id_full.index_copy_(0, idx_local, idx_code)
                self.embed_ind_dict[skey] = idx_code

            self.latent_size_sum = global_end
            torch.cuda.empty_cache()
            return key_id_full, cluster_id_full, dict(self._id2safe)

        # -----------------------------
        # 4) init_kmeans_final: dump
        # -----------------------------
        if mode == "init_kmeans_final":
            if mask_dict is None:
                return 0

            if not hasattr(self, "_kmeans_dump"):
                self._kmeans_dump = {}

            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue
                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                masked_latents = flatten[0].index_select(0, idx_local)
                if masked_latents.numel() == 0:
                    continue

                skey = str(key)
                safe, _code_param, code = _get_safe_and_code(skey)

                with torch.no_grad():
                    idx_code = _assign_l2norm(masked_latents, code, k_block=k_block)
                    quantize = code.index_select(0, idx_code)

                with torch.no_grad():
                    # centers used for assignment in init_kmeans_final
                    C0 = _safe_l2norm(code.detach().float())
                    # assignment histogram signature
                    bc0 = torch.bincount(idx_code.long(), minlength=C0.shape[0]).float()
                    sig0 = torch.stack([
                        bc0.sum(),  # Ni
                        (bc0 > 0).sum().float(),  # used clusters
                        bc0.max(),  # max count
                        (bc0 / bc0.sum().clamp_min(1)).max()  # max_p
                    ])
                    self._kmeans_sig = getattr(self, "_kmeans_sig", {})
                    self._kmeans_sig[skey] = {
                        "C_norm_mean": C0.norm(dim=1).mean().item(),
                        "sig": sig0.cpu(),
                    }

                save_latents, save_centers, save_assign, save_quantized = True, True, True, False
                lat_cpu = masked_latents.detach().to("cpu", dtype=torch.float16) if save_latents else None
                ctr_cpu = code.detach().to("cpu", dtype=torch.float16) if save_centers else None
                asg_cpu = idx_code.detach().to("cpu") if save_assign else None
                qnt_cpu = quantize.detach().to("cpu", dtype=torch.float16) if save_quantized else None

                entry = self._kmeans_dump.get(skey)
                if entry is None:
                    entry = {"latents": [], "centers": None, "assign": [], "quantize": []}
                    self._kmeans_dump[skey] = entry

                if save_latents:
                    entry["latents"].append(lat_cpu)
                if save_assign:
                    entry["assign"].append(asg_cpu)
                if save_quantized:
                    entry["quantize"].append(qnt_cpu)
                if save_centers and entry["centers"] is None:
                    entry["centers"] = ctr_cpu

            self.latent_size_sum = global_end
            torch.cuda.empty_cache()

            out = {}
            for k, v in self._kmeans_dump.items():
                out[k] = {
                    "latents": torch.cat(v["latents"], dim=0) if len(v["latents"]) else None,
                    "centers": v["centers"],
                    "assign": torch.cat(v["assign"], dim=0) if len(v["assign"]) else None,
                    "quantize": torch.cat(v["quantize"], dim=0) if len(v["quantize"]) else None,
                }

            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("dumps", f"init_kmeans_final_dump_{stamp}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(out, path)

            self._kmeans_dump = {}
            return 0

        # -----------------------------
        # 5) train/eval: quantize + EMA (+ N-dependent split)
        # -----------------------------
        ent_metric_total = torch.zeros((), device=flatten.device, dtype=torch.float32)
        do_ema = bool(self.training and (mode == "train") and (epoch is not None))

        # ---- epoch-wise max_p distribution collector ----
        if not hasattr(self, "_mp_epoch"):
            self._mp_epoch = None
        if not hasattr(self, "_mp_list"):
            self._mp_list = []
        if not hasattr(self, "_mp_list_keys"):
            self._mp_list_keys = []

        if epoch is not None:
            ep = int(epoch)
            if self._mp_epoch != ep:
                self._mp_epoch = ep
                self._mp_list = []
                self._mp_list_keys = []

        # ---- per-epoch split gate (key-wise at most once) ----
        if do_ema and split_once_per_epoch:
            if not hasattr(self, "_split_epoch"):
                self._split_epoch = None
            if not hasattr(self, "_split_done_epoch"):
                self._split_done_epoch = set()
            if self._split_epoch != int(epoch):
                self._split_epoch = int(epoch)
                self._split_done_epoch.clear()

        for key, idx_global in (mask_dict.items() if mask_dict is not None else []):
            if idx_global is None or idx_global.numel() == 0:
                continue

            gmask = (idx_global >= global_start) & (idx_global < global_end)
            if not bool(gmask.any()):
                continue

            idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
            masked_latents = flatten[0].index_select(0, idx_local)  # (Ni,D)
            if masked_latents.numel() == 0:
                continue

            skey = str(key)
            safe, code_param, code = _get_safe_and_code(skey)

            # ---- hard assignment (L2norm unified)
            with torch.no_grad():
                idx_code = _assign_l2norm(masked_latents, code, k_block=k_block)
                idx_code_long = idx_code.to(dtype=torch.long)

            if (epoch == 1) and (chunk_i == 0) and (skey in getattr(self, "_kmeans_sig", {})) and (skey in diag_keys):
                with torch.no_grad():
                    C = _safe_l2norm(code.detach().float())
                    bc = torch.bincount(idx_code_long, minlength=C.shape[0]).float()
                    sig = torch.stack([
                        bc.sum(),
                        (bc > 0).sum().float(),
                        bc.max(),
                        (bc / bc.sum().clamp_min(1)).max()
                    ]).cpu()
                    sig0 = self._kmeans_sig[skey]["sig"]
                    _log(f"[KMEANS-CHECK] key={skey} sig_init={sig0.tolist()} sig_train={sig.tolist()}")

            quantize = code.index_select(0, idx_code_long)
            self.quantize_dict[skey] = quantize
            self.embed_ind_dict[skey] = idx_code_long.to(torch.int32)

            # -------------------------
            # EMA + split (Option A): EMA in normalized space
            # -------------------------
            if do_ema:
                def _hist_sig(idx: torch.Tensor, K: int):
                    bc = torch.bincount(idx.long(), minlength=K).float()
                    tot = bc.sum().clamp_min(1.0)
                    used = (bc > 0).sum().item()
                    maxc = bc.max().item()
                    maxp = (bc / tot).max().item()
                    return used, maxc, maxp

                dev = code_param.device

                if code_param.ndim == 3:
                    K_e, D_e = int(code_param.shape[1]), int(code_param.shape[2])
                else:
                    K_e, D_e = int(code_param.shape[0]), int(code_param.shape[1])
                self.cb_dict[skey] = int(K_e)

                buf_name_cs = f"cluster_size_{safe}"
                buf_name_ea = f"embed_avg_{safe}"
                buf_name_ue = f"usage_ema_{safe}"
                buf_name_cd = f"split_cd_{safe}"

                def _get_buf(name, shape, dtype):
                    if hasattr(self, name):
                        t = getattr(self, name)
                        if t.dtype != dtype or tuple(t.shape) != tuple(shape) or t.device != dev:
                            t = torch.zeros(*shape, device=dev, dtype=dtype)
                            setattr(self, name, t)
                        return t
                    t = torch.zeros(*shape, device=dev, dtype=dtype)
                    self.register_buffer(name, t)
                    return t

                cs = _get_buf(buf_name_cs, (K_e,), torch.float32)
                ea = _get_buf(buf_name_ea, (K_e, D_e), torch.float32)
                ue = _get_buf(buf_name_ue, (K_e,), torch.float32)
                cd = _get_buf(buf_name_cd, (K_e,), torch.long)

                with torch.no_grad():
                    idx_dev = idx_code_long.to(device=dev)

                    # >>> INSERT HERE (EMA pre-check: is EMA buffer empty? are current centers kmeans-like?)
                    if (epoch == 1) and (chunk_i == 0) and (skey in diag_keys):
                        C_pre = code_param.detach()
                        C_pre = C_pre.squeeze(0) if C_pre.ndim == 3 else C_pre
                        C_pre = _safe_l2norm(C_pre.float())
                        cs_sum_pre = float(cs.sum().item())
                        # ea/cs only meaningful if cs has mass
                        ea_mean_norm_pre = float(
                            (ea / (cs.unsqueeze(-1) + eps)).norm(dim=1).mean().item()) if cs_sum_pre > 0 else -1.0
                        _log(
                            f"[EMA-PRE] ep={epoch} key={skey} cs_sum={cs_sum_pre:.3f} ea_mean_norm={ea_mean_norm_pre:.3f} C_pre_mean_norm={C_pre.norm(dim=1).mean().item():.3f}")
                    # <<< INSERT END

                    # batch_counts (K,)
                    batch_counts = torch.zeros((K_e,), device=dev, dtype=torch.float32)
                    batch_counts.index_add_(0, idx_dev, torch.ones_like(idx_dev, device=dev, dtype=torch.float32))

                    # ===== Option A change #1: normalize latents BEFORE accumulating =====
                    Xn = _safe_l2norm(masked_latents.to(device=dev, dtype=torch.float32))  # (Ni,D)

                    # batch_embed_sum (K,D) in normalized space
                    batch_embed_sum = torch.zeros((K_e, D_e), device=dev, dtype=torch.float32)
                    batch_embed_sum.index_add_(0, idx_dev, Xn)

                    # ---- metric: entropy (optional)
                    denom2 = batch_counts.sum()
                    if float(denom2.item()) > 0.0:
                        p_ = batch_counts / (denom2 + 1e-8)
                        entropy = -(p_ * (p_ + 1e-12).log()).sum()
                        ent_w = float(getattr(self, "entropy_weight", 1e-3))
                        ent_metric_total = ent_metric_total + ent_w * (-entropy)

                    # ---- update usage_ema
                    ue.mul_(usage_mom).add_(batch_counts, alpha=(1.0 - usage_mom))

                    # ---- EMA update
                    one_m = 1.0 - decay
                    cs.mul_(decay).add_(batch_counts, alpha=one_m)
                    ea.mul_(decay).add_(batch_embed_sum, alpha=one_m)

                    # ---- compute means, then normalize centers (Option A consistency)
                    means = ea / (cs.unsqueeze(-1) + eps)  # (K,D)
                    means = _safe_l2norm(means, eps=1e-12)  # <-- Option A change #2

                    # keep ea consistent with normalized means (optional but recommended)
                    # (so next means=ea/cs also stays on unit sphere)
                    ea.copy_(means * (cs.unsqueeze(-1) + eps))  # <-- Option A change #3

                    # >>> INSERT HERE (EMA post-mean)
                    if (epoch == 1) and (chunk_i == 0) and (skey in diag_keys):
                        used_u, maxc_u, maxp_u = _hist_sig(idx_dev, K_e)
                        _log(
                            f"[EMA-MEANS] ep={epoch} key={skey} cs_sum={float(cs.sum().item()):.1f} used={used_u} maxp_batch={maxp_u:.4f} means_mean_norm={means.norm(dim=1).mean().item():.3f}")
                    # <<< INSERT END

                    # =========================================================
                    # N-dependent split policy (uses usage_ema distribution)
                    # =========================================================
                    can_split_now = do_split and (not split_only_last_batch or bool(is_last_batch))
                    if can_split_now and (not split_once_per_epoch or (skey not in self._split_done_epoch)):
                        total_u = ue.sum().clamp_min(1e-8)
                        p_u = ue / total_u
                        max_p = float(p_u.max().item())
                        N_key = int(cs.sum().item())
                        thr = _target_max_p(N_key)

                        if skey in diag_keys and (chunk_i == 0):
                            _log(
                                f"[SPLIT-POLICY] epoch={epoch} key={skey} N~{N_key} max_p={max_p:.4f} thr={thr:.4f} K={K_e}")

                        if max_p > thr:
                            did = self.split_the_winner_ema(
                                embed=means,  # normalized means view
                                ema_sum=ea,
                                ema_count=cs,
                                usage_ema=ue,
                                batch_counts=batch_counts,
                                batch_counts2=batch_counts,
                                X_batch=Xn,  # <-- Option A: PCA uses normalized latents too
                                assign=idx_dev.to(device=dev, dtype=torch.long),
                                split_thr=float(thr),
                                prune_src_thr=float(prune_src_thr),
                                cooldown=cd,
                                cooldown_steps=int(getattr(self, "split_cooldown_steps", 2000)),
                                eps=float(eps),
                                min_points=int(getattr(self, "split_min_points", 32)),
                                pca_power=float(getattr(self, "split_pca_power", 0.5)),
                                noise_scale=float(getattr(self, "split_noise_scale", 0.02)),
                                fallback_to_noise=bool(getattr(self, "split_fallback_to_noise", True)),
                            )

                            if split_once_per_epoch:
                                self._split_done_epoch.add(skey)

                            if did:
                                # re-normalize after split to be safe + keep ea consistent
                                means.copy_(_safe_l2norm(means))
                                ea.copy_(means * (cs.unsqueeze(-1) + eps))

                                total_u2 = ue.sum().clamp_min(1e-8)
                                p_u2 = ue / total_u2
                                _log(
                                    f"[SPLIT][N-DEP] epoch={epoch} key={skey} did_split=True "
                                    f"N~{N_key} thr={thr:.4f} max_p_before={max_p:.4f} max_p_after={float(p_u2.max().item()):.4f}"
                                )

                    # ---- collect max_p_u per key
                    total_u = ue.sum().clamp_min(1e-8)
                    p_u = ue / total_u
                    max_p_u = float(p_u.max().item())
                    self._mp_list.append(max_p_u)
                    N_key = int(cs.sum().item())
                    self._mp_list_keys.append((skey, max_p_u, N_key, int(K_e)))

                    # ---- finally write centers into parameter (normalized)
                    if code_param.ndim == 3:
                        code_param.data.copy_(means.unsqueeze(0))
                    else:
                        code_param.data.copy_(means)
                    # >>> INSERT HERE (write-back drift)
                    if (epoch == 1) and (chunk_i == 0) and (skey in diag_keys):
                        C_post = code_param.detach()
                        C_post = C_post.squeeze(0) if C_post.ndim == 3 else C_post
                        C_post = _safe_l2norm(C_post.float())
                        delta = (C_post - C_pre).abs().mean().item() if 'C_pre' in locals() else float('nan')
                        _log(f"[EMA-WRITE] ep={epoch} key={skey} mean_abs_delta={delta:.3e}")
                    # <<< INSERT END

            del masked_latents, code, idx_code, quantize

        # -----------------------------
        # 6) rebuild quantized tensor in original atom order
        # -----------------------------
        quantize_full = torch.empty((B, D), device=flatten.device, dtype=flatten.dtype)

        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                skey = str(key)
                if skey not in self.quantize_dict:
                    continue
                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue
                idx_local = (idx_global[gmask] - global_start).to(device=quantize_full.device, dtype=torch.long)
                qk = self.quantize_dict[skey].to(device=quantize_full.device, dtype=quantize_full.dtype)
                if idx_local.numel() > 0:
                    quantize_full.index_copy_(0, idx_local, qk)

        all_local = []
        if mask_dict is not None:
            for idx in mask_dict.values():
                if idx is None or idx.numel() == 0:
                    continue
                in_chunk = idx[(idx >= global_start) & (idx < global_end)]
                if in_chunk.numel() > 0:
                    all_local.append(in_chunk - global_start)

        used = (
            torch.unique(torch.cat(all_local, dim=0))
            if len(all_local) > 0
            else torch.tensor([], dtype=torch.long, device=flatten.device)
        )
        unused = torch.ones(B, dtype=torch.bool, device=flatten.device)
        if used.numel() > 0:
            unused[used] = False
        if bool(unused.any()):
            quantize_full[unused] = flatten[0][unused]

        # straight-through
        quantize_st = flatten[0] + (quantize_full - flatten[0]).detach()
        quantize_st = quantize_st.unsqueeze(0)

        if chunk_i is not None:
            self.latent_size_sum = global_end

        # ---- end-of-epoch report (only on last batch) ----
        if (epoch is not None) and bool(is_last_batch) and (logger is not None):
            if len(self._mp_list) > 0:
                mp = torch.tensor(self._mp_list, dtype=torch.float32)
                qs = torch.quantile(mp, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
                logger.info(
                    f"[MAXP][epoch={int(epoch)}] "
                    f"n_keys={len(self._mp_list)} "
                    f"mean={mp.mean().item():.4f} std={mp.std(unbiased=False).item():.4f} "
                    f"p50={qs[0].item():.4f} p75={qs[1].item():.4f} "
                    f"p90={qs[2].item():.4f} p95={qs[3].item():.4f} p99={qs[4].item():.4f} "
                    f"max={mp.max().item():.4f}"
                )

                topK = int(getattr(self, "maxp_report_topK", 20))
                worst = sorted(self._mp_list_keys, key=lambda t: t[1], reverse=True)[:topK]
                for (k, v, nkey, Ke) in worst:
                    logger.info(f"[MAXP-WORST][epoch={int(epoch)}] key={k} max_p={v:.4f} N~{nkey} K={Ke}")

                if bool(getattr(self, "save_maxp_each_epoch", True)):
                    os.makedirs("dumps", exist_ok=True)
                    out = {"epoch": int(epoch), "maxp_list": self._mp_list, "detail": self._mp_list_keys}
                    torch.save(out, f"dumps/maxp_epoch_{int(epoch):04d}.pt")

        torch.cuda.empty_cache()
        return quantize_st, self.embed_ind_dict, self.embed, ent_metric_total


class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.8,
            eps=1e-5,
            kmeans_init=True,
            kmeans_iters=600,
            sync_kmeans=True,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            margin_weight=1,
            spread_weight=0.2,
            pair_weight=0.01,
            lamb_div_ele=1,
            lamb_div_bonds=1,
            lamb_div_aroma=1,
            lamb_div_ringy=1,
            lamb_div_h_num=1,
            lamb_div_equidist=1,
            lamb_div_elec_state=1,
            lamb_div_charge=1,
            commitment_weight=1,  # using
            codebook_weight=1,  # using
            lamb_sil=0.00001,           # using
            lamb_cb=0.01,           # using
            lamb_div=0.01,           # using
            lamb_equiv_atom=1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False,
            use_cosine=False,
            target_radius=3.0,  # soft cap for ||z||
            radius_weight=1e-3,  # weight for norm regularizer
            tau_init=0.5,  # starting temperature (distance scale)
            tau_min=0.05,
            tau_max=2.0,  # clamp for temperature
            tau_ema=0.9,  # EMA for temperature
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)  # use coocbook_dim if not None
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        # self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.tau_ema = 0.9
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight

        has_codebook_orthogonal_loss = margin_weight > 0

        self.margin_weight = margin_weight
        self.spread_weight = spread_weight
        self.lamb_div_ele = lamb_div_ele
        self.lamb_div_bonds = lamb_div_bonds
        self.lamb_div_aroma = lamb_div_aroma
        self.lamb_div_ringy = lamb_div_ringy
        self.lamb_div_h_num = lamb_div_h_num
        self.lamb_div_elec_state = lamb_div_elec_state
        self.lamb_div_charge = lamb_div_charge
        self.lamb_equiv_atom = lamb_equiv_atom
        self.lamb_sil = lamb_sil
        self.lamb_div = lamb_div
        self.lamb_cb = lamb_cb
        self.lamb_div_equidist = lamb_div_equidist
        self.pair_weight = pair_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        codebook_max_norm = None  # e.g., 4.0 to clamp code vectors
        codebook_class = EuclideanCodebook # if not use_cosine_sim else CosineSimCodebook
        self._codebook = codebook_class(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp,
        )
        args = get_args()
        self.epoch_at_mode_shift = args.epoch_at_mode_shift
        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.compute_contrastive_loss = ContrastiveLoss(dim, 136)

        self.use_cosine = use_cosine
        self.target_radius = float(target_radius)
        self.radius_weight = float(radius_weight)
        self.register_buffer("_tau", torch.tensor(float(tau_init)))
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.tau_ema = float(tau_ema)
        self.codebook_max_norm = codebook_max_norm

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def _cb_to_tensor(self, cb):
        """
        Normalize various "codebook" containers to a [K, D] Tensor.

        対応する型:
          - torch.Tensor
          - nn.Parameter
          - nn.Module (EuclideanCodebook 含む)
              * .weight を持つ場合 → それを使う
              * .embed を持つ場合 → embed.weight か embed 自体を使う
              * .codebook / .codes / .embedding を持つ場合 → それを使う
              * 上記が無くても、named_parameters / named_buffers から
                最初の 2 次元テンソル ([K, D]) を拾う
          - nn.ParameterDict（[K_i, D] を縦に concat）
          - dict[key -> (上記のいずれか)]（縦に concat）
        """
        import torch
        import torch.nn as nn

        # -------- 1. すでに Tensor/Parameter の場合 --------
        if isinstance(cb, torch.Tensor):
            return cb
        if isinstance(cb, nn.Parameter):
            return cb

        # -------- 2. nn.Module (EuclideanCodebook を含む) --------
        if isinstance(cb, nn.Module):
            # (a) .weight を持つモジュール（Embedding, Linear など）
            if hasattr(cb, "weight") and isinstance(cb.weight, torch.Tensor):
                return cb.weight

            # (b) EuclideanCodebook などで .embed を使っている場合
            if hasattr(cb, "embed"):
                e = cb.embed
                if isinstance(e, nn.Embedding):
                    return e.weight
                if isinstance(e, (torch.Tensor, nn.Parameter)):
                    return e

            # (c) その他ありがちな名前をチェック
            for attr in ("codebook", "codes", "embedding"):
                if hasattr(cb, attr):
                    t = getattr(cb, attr)
                    if isinstance(t, (torch.Tensor, nn.Parameter)):
                        return t

            # (d) 最後のフォールバック:
            #     モジュール内の parameter / buffer から
            #     「次元数が 2 のテンソル」を探して最初のものを使う
            #     (recurse=True なので embed.weight のような下位モジュールも拾える)
            for name, p in cb.named_parameters(recurse=True):
                if isinstance(p, torch.Tensor) and p.ndim == 2:
                    # print(f"[DEBUG] using parameter {name} from {type(cb)} as codebook")  # 必要ならログ
                    return p

            for name, b in cb.named_buffers(recurse=True):
                if isinstance(b, torch.Tensor) and b.ndim == 2:
                    # print(f"[DEBUG] using buffer {name} from {type(cb)} as codebook")
                    return b

            # ここまで来ると、2D テンソルが一つも見つからない本当にイレギュラーなケース
            raise TypeError(
                f"_cb_to_tensor: unsupported nn.Module type {type(cb)} "
                f"(no usable weight/embed/codebook/codes/embedding or 2D param/buffer)"
            )

        # -------- 3. nn.ParameterDict: values を全部縦に concat --------
        if isinstance(cb, nn.ParameterDict):
            tensors = []
            for k in sorted(cb.keys()):
                v = cb[k]
                if isinstance(v, (torch.Tensor, nn.Parameter)):
                    tensors.append(v)
                else:
                    raise TypeError(
                        f"_cb_to_tensor: unsupported leaf type {type(v)} "
                        f"in ParameterDict for key {k!r}"
                    )
            if not tensors:
                raise ValueError("_cb_to_tensor: empty ParameterDict")
            return torch.cat(tensors, dim=0)

        # -------- 4. dict: value ごとに再帰的にテンソル化して concat --------
        if isinstance(cb, dict):
            tensors = []
            for k in sorted(cb.keys()):
                v = cb[k]
                t = self._cb_to_tensor(v)  # 再帰
                tensors.append(t)
            if not tensors:
                raise ValueError("_cb_to_tensor: empty dict")
            return torch.cat(tensors, dim=0)

        # -------- 5. それ以外はサポート外 --------
        raise TypeError(f"_cb_to_tensor: unsupported type {type(cb)}")

    def get_codes_from_indices(self, indices):
        indices = indices.long()
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack([indices], 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes, = unpack(codes, ps, 'b * d')
        return codes

    import torch

    def fast_find_equivalence_groups(self, latents):
        from collections import defaultdict
        hash_map = defaultdict(list)
        for idx, vector in enumerate(latents):
            key = tuple(vector.tolist())  # Convert tensor to a hashable tuple
            hash_map[key].append(idx)
        equivalence_groups = [group for group in hash_map.values() if len(group) > 1]
        return equivalence_groups

    def _unwrap_codebook_entry(self, cb):
        """
        Accept: Tensor, nn.Parameter, dict/ParameterDict possibly holding 'embed'.
        Return: Tensor (or Parameter) with shape [K,D] or [1,K,D].
        """
        if isinstance(cb, (torch.Tensor, nn.Parameter)):
            return cb
        if isinstance(cb, (dict, nn.ParameterDict)):
            if 'embed' in cb:
                return self._unwrap_codebook_entry(cb['embed'])
        raise TypeError(f"Unsupported codebook entry type: {type(cb)}. "
                        "Expected Tensor/Parameter or dict/ParameterDict with 'embed'.")

    def commitment_loss(
            self,
            encoder_outputs,
            mask_dict,
            codebook,
            codebook_mod=None,
            logger=None,
            chunk_start=None,
            beta=0.25,
            temperature=None,
            use_cosine=False,
            embed_ind_dict=None,  # <-- add this
    ):
        """
        Per-element commitment + codebook + repel losses.

        戻り値:
            commit_loss, codebook_loss, repel_loss, cb_repel_loss

        - encoder_outputs : [B, D] （このチャンクに含まれる latent）
        - mask_dict       : { key -> グローバル index or bool mask }
        - codebook        :
            * Tensor [K, D]              … 全 key で共通
            * dict / nn.ParameterDict    … key ごとに [K_e, D]
            * None                       … self._codebook を使う
        - chunk_start     : このチャンクの global start index
        - beta            : commitment loss の係数
        - use_cosine      : True のときは cosine 距離で最も近いコードを選ぶ
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # ---- 0. encoder_outputs 形状そろえ ----
        encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
        device = encoder_outputs.device
        B, D = encoder_outputs.shape
        dtype = encoder_outputs.dtype

        if chunk_start is None:
            chunk_start = 0
        chunk_end = chunk_start + B

        # ---- 1. mask_dict が空なら全部 0 (勾配付き) を返す ----
        if mask_dict is None or len(mask_dict) == 0:
            zero = encoder_outputs.sum() * 0.0  # graph に繋がった 0
            # if logger is not None:
            #     logger.info(f"[VQ_COMMIT] mask_dict is empty → skip in chunk [{chunk_start},{chunk_end})")
            return zero, zero, zero, zero

        def _get_codebook_for_key(key, *, device, dtype):
            """
            `codebook` / `self._codebook` が以下のいずれでも動くように統一:
              - 単一 Tensor / Parameter / Embedding
              - dict[str -> Tensor/Parameter]
              - nn.ParameterDict (サブコードブック束ね)
              - dict[str -> nn.ParameterDict]（多段）
            戻り値は必ず [K, D] の Tensor。
            """
            cb_src = codebook if codebook is not None else getattr(self, "_codebook", None)
            if cb_src is None:
                raise RuntimeError("Codebook is not initialized (both `codebook` and `self._codebook` are None).")

            import torch
            import torch.nn as nn

            # ---------- dict / ModuleDict / ParameterDict 系 ----------
            if isinstance(cb_src, (dict, nn.ModuleDict, nn.ParameterDict)):
                t = None

                # 1) 完全一致
                if key in cb_src:
                    t = self._cb_to_tensor(cb_src[key])

                # 2) "k_..." をざっくり上位キーにマッピング
                if t is None and isinstance(key, str) and key.startswith("k_"):
                    head = key.split("_")[1]  # 例: k_6_0_3_1_1_... → "6"
                    if head in cb_src:
                        t = self._cb_to_tensor(cb_src[head])

                # 3) それでも無ければ「全部まとめて」連結して使う
                if t is None:
                    t = self._cb_to_tensor(cb_src)

            else:
                # ---------- 単一 Tensor / Parameter / Embedding 等 ----------
                t = self._cb_to_tensor(cb_src)

            # 念のため device/dtype をそろえる
            t = t.to(device=device, dtype=dtype)
            return t

        # ---- 2. 累積器: Ni, K_e で重み付け平均する ----
        total_latent = 0
        total_cb_count = 0

        commit_num   = encoder_outputs.new_zeros(())
        codebk_num   = encoder_outputs.new_zeros(())
        repel_num    = encoder_outputs.new_zeros(())
        cb_repel_num = encoder_outputs.new_zeros(())

        # ---- 3. main loop over mask_dict keys ----
        for k, g_idx in mask_dict.items():
            # -------------------------------
            # 3-1) list / numpy / tensor → Tensor に正規化
            # -------------------------------
            if isinstance(g_idx, (list, tuple)):
                if len(g_idx) == 0:
                    continue
                g_idx = torch.as_tensor(g_idx, device=device)
            elif not isinstance(g_idx, torch.Tensor):
                try:
                    g_idx = torch.as_tensor(g_idx, device=device)
                except Exception as e:
                    raise TypeError(f"mask_dict[{k}] has unsupported type {type(g_idx)}") from e
            else:
                g_idx = g_idx.to(device)

            if g_idx.numel() == 0:
                continue

            # -------------------------------
            # 3-2) bool mask → index に変換
            # -------------------------------
            if g_idx.dtype == torch.bool:
                g_idx = torch.nonzero(g_idx, as_tuple=False).view(-1)

            g_idx = g_idx.long()
            if g_idx.numel() == 0:
                continue

            # -------------------------------
            # 3-3) このチャンク [chunk_start, chunk_end) に属する index だけ抜き出し
            # -------------------------------
            in_chunk = (g_idx >= chunk_start) & (g_idx < chunk_end)
            if not torch.any(in_chunk):
                continue

            local_idx = (g_idx[in_chunk] - chunk_start).long()
            if local_idx.numel() == 0:
                continue

            # 対象 latent: z ∈ R^{N_i × D}
            z = encoder_outputs[local_idx]  # [N_i, D]
            N_i = z.shape[0]
            if N_i == 0:
                continue

            # -------------------------------
            # 3-4) この key 用コードブック取得 [K_e, D]
            #      MUST be identical to _codebook safe-key mapping
            # -------------------------------
            skey = str(k)

            K_e_default = int(
                getattr(self, "cb_dict", {}).get(
                    skey,
                    getattr(self, "codebook_size", 0)
                )
            )
            if codebook_mod is None:
                raise RuntimeError("[VQ_COMMIT] codebook_mod is required (pass self._codebook).")

            safe = codebook_mod._get_or_create_safe_key(
                skey,
                K_e=K_e_default,
                D=D,
                device=device
            )

            if safe not in codebook:
                raise KeyError(
                    f"[VQ_COMMIT] safe-key '{safe}' not in codebook "
                    f"(from skey='{skey}')."
                )

            cb_param = codebook[safe]
            cb = cb_param.squeeze(0) if cb_param.ndim == 3 else cb_param  # [K_e, D]

            K_e, Dk = cb.shape
            if Dk != D:
                raise RuntimeError(
                    f"[VQ_COMMIT] D mismatch for skey={skey}: latent D={D} vs cb D={Dk}"
                )

            # -------------------------------
            # 3-5) 最近傍コード index
            #      USE embed_ind_dict from _codebook (no NN recompute)
            # -------------------------------
            if embed_ind_dict is None:
                raise RuntimeError(
                    "[VQ_COMMIT] embed_ind_dict is required to keep "
                    "assignments consistent with EMA / entropy / split"
                )

            if skey not in embed_ind_dict:
                raise KeyError(f"[VQ_COMMIT] embed_ind_dict missing skey={skey}")

            nn_idx = embed_ind_dict[skey].to(device=device, dtype=torch.long).reshape(-1)

            if nn_idx.numel() != N_i:
                raise ValueError(
                    f"[VQ_COMMIT] nn_idx length mismatch skey={skey}: "
                    f"{nn_idx.numel()} vs {N_i}"
                )

            # ---- OOB guard BEFORE indexing (prevents CUDA device assert)
            min_i = int(nn_idx.min().item())
            max_i = int(nn_idx.max().item())
            if (min_i < 0) or (max_i >= K_e):
                raise RuntimeError(
                    f"[VQ_COMMIT][OOB] skey={skey} safe={safe} "
                    f"K_e={K_e} nn_idx range=[{min_i},{max_i}] "
                    f"cb.shape={tuple(cb.shape)}"
                )

            # -------------------------------
            # 3-6) gather codes and compute losses
            # -------------------------------
            e_star = cb.index_select(0, nn_idx)  # [N_i, D]

            commit_part = F.mse_loss(z, e_star.detach(), reduction="mean")
            codebk_part = F.mse_loss(e_star, z.detach(), reduction="mean")

            total_latent += N_i
            commit_num = commit_num + commit_part * N_i
            codebk_num = codebk_num + codebk_part * N_i

            # repel loss (your existing API)
            ret = self.compute_contrastive_loss(z, 0, logger, cb_param, k)
            repel_val = ret[0]
            cb_repel_val = ret[3]

            repel_num = repel_num + repel_val * N_i
            total_cb_count += K_e
            cb_repel_num = cb_repel_num + cb_repel_val * K_e

            # -------------------------------
            # 3-7) repel loss (contrastive)
            # -------------------------------
            # 旧版と同様に cb 自体を使っても良いし、codebook[k] があればそれを使う
            if isinstance(codebook, (dict, nn.ParameterDict)) and k in codebook:
                cb_for_contrast = codebook[k]
            else:
                cb_for_contrast = cb  # Tensor でも compute_contrastive_loss が対応していれば OK
            # compute_contrastive_loss は旧実装と同じインターフェースを仮定
            # 返り値: (repel_loss, ..., ..., cb_repel_loss, ...)
            ret = self.compute_contrastive_loss(z, 0, logger, cb_for_contrast, k)
            repel_val     = ret[0]
            cb_repel_val  = ret[3] # final_loss, neg_loss, repel_from_2, cb_loss, latent_repel_loss
            repel_num    = repel_num + repel_val * N_i
            total_cb_count += K_e
            cb_repel_num = cb_repel_num + cb_repel_val * K_e

        # ---- 4. 集計 ----
        if total_latent == 0:
            # このチャンクに該当サンプルなし → 全て 0 だが encoder_outputs に繋げる
            zero = encoder_outputs.sum() * 0.0
            if logger is not None:
                logger.info(f"[VQ_EMPTY] total_latent=0 in chunk [{chunk_start},{chunk_end})")
            return zero, zero, zero, zero

        commit_loss   = beta * (commit_num / float(total_latent))
        codebook_loss = codebk_num / float(total_latent)
        repel_loss    = repel_num / float(total_latent)

        if total_cb_count > 0:
            cb_repel_loss = cb_repel_num / float(total_cb_count)
        else:
            cb_repel_loss = cb_repel_num * 0.0  # guard

        # if logger is not None:
        #     logger.info(
        #         f"[VQ_COMMIT] chunk_start={chunk_start}, B={B}, "
        #         f"total_latent={int(total_latent)}, "
        #         f"commit_loss={commit_loss.item():.6f}"
        #     )

        return commit_loss, codebook_loss, repel_loss, cb_repel_loss

    import torch
    import torch.nn.functional as F
    from einops import rearrange

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
            self,
            x,
            feature=None,
            mask_dict=None,
            logger=None,
            chunk_i=None,
            epoch=None,
            mode=None,
    ):
        """
        Forward pass with per-element quantization and EMA update (commitment_loss 版).

        - encoder_outputs は必ず [B_total, D] に flatten される
        - mask_dict は collect_global_indices_compact 由来の「グローバル index」
        - chunk_start (= global_start) を commitment_loss に渡し、
          commitment_loss 内で global->local (global - chunk_start) 変換する前提

        Debug:
          self.debug_index = True のとき、mask_dict の index がこの chunk 範囲に入っているか検査。
        """
        import torch
        from einops import rearrange
        import math

        # 例：コードブックサイズ K が取れる場所で
        K = self.codebook_size  # or self.embed.shape[0] など
        batch_counts = torch.zeros(K, device=x.device, dtype=torch.float32)

        # --------------------------------------------------------------
        # 0) グローバル latent オフセット管理
        # --------------------------------------------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0

        # eval/test/init は毎回 0 始まりにする（chunk_i が渡らないケースでも安全）
        if mode in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = 0
        elif chunk_i is not None and chunk_i == 0:
            self.latent_size_sum = 0

        # --------------------------------------------------------------
        # 1) 入力整形: encoder_outputs を [B_total, D] にする
        # --------------------------------------------------------------
        x = x.float()

        if getattr(self, "accept_image_fmap", False) and x.ndim == 4:
            # (B, C, H, W) → (B*H*W, C)
            encoder_outputs = rearrange(x, "b c h w -> (b h w) c")
        else:
            # (B, D) / (1, B, D) / (B, N, D) / その他 → (B_total, D)
            if x.ndim == 2:
                encoder_outputs = x
            elif x.ndim == 3:
                encoder_outputs = x.reshape(-1, x.size(-1))
            else:
                encoder_outputs = x.reshape(-1, x.size(-1))

        B = int(encoder_outputs.size(0))
        device = encoder_outputs.device
        dtype = encoder_outputs.dtype

        # このチャンクのグローバル index 範囲
        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        if mode == "init_kmeans_final":
            _ = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode=mode,
            )
            # keep old behavior for kmeans init final
            z = torch.zeros((), device=device, dtype=dtype)
            return z, (z, z, z, z)

        if mode == "infer":
            out = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode=mode,
            )

            # Expect: (key_id_full, cluster_id_full, id2safe)
            if not (isinstance(out, (tuple, list)) and len(out) == 3):
                raise TypeError(
                    f"[vq.forward] mode='infer' expects _codebook() -> (key_id_full, cluster_id_full, id2safe), "
                    f"got {type(out)} with len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
                )

            key_id_full, cluster_id_full, id2safe = out

            # shape/dtype guarantees
            key_id_full = key_id_full.reshape(-1).long()
            cluster_id_full = cluster_id_full.reshape(-1).long()

            if key_id_full.numel() != cluster_id_full.numel():
                raise ValueError(
                    f"[vq.forward] infer length mismatch: key_id_full {key_id_full.shape} vs cluster_id_full {cluster_id_full.shape}"
                )

            return key_id_full, cluster_id_full, id2safe

        # --------------------------------------------------------------
        # 1.5) Debug: mask_dict の global index が chunk 範囲内か検査
        # --------------------------------------------------------------
        if getattr(self, "debug_index", False) and chunk_i == 0 and B > 0:
            # 最初の1原子だけ確認
            k = next(iter(mask_dict))
            gi = mask_dict[k][0]  # global index
            li = gi - global_start  # local index

            z_norm = encoder_outputs[li].norm().item()
            assert z_norm > 0 and not math.isnan(z_norm), \
                f"latent invalid at global_id={gi}, local_id={li}"

        if getattr(self, "debug_index", False) and (mask_dict is not None) and (B > 0):
            # 先頭 chunk なら global_start は 0 のはず（設計通りなら）
            if mode in ("eval", "test", "init_kmeans_final") or (chunk_i == 0):
                assert global_start == 0, f"[INDEX_MISMATCH] chunk head but global_start={global_start}"

            n_checked = 0
            n_oob = 0

            for k, idxs in mask_dict.items():
                if idxs is None:
                    continue
                if isinstance(idxs, torch.Tensor):
                    gi = idxs.to(device=device)
                else:
                    gi = torch.as_tensor(idxs, device=device)

                if gi.numel() == 0:
                    continue

                gi = gi.reshape(-1)
                # 重くしないように最大 256 だけ見る
                if gi.numel() > 256:
                    gi = gi[:256]

                in_chunk = (gi >= global_start) & (gi < global_end)
                n_oob += int((~in_chunk).sum().item())
                n_checked += int(gi.numel())
                if n_checked >= 1024:
                    break

            assert n_oob == 0, (
                f"[INDEX_MISMATCH] mask_dict contains out-of-chunk indices. "
                f"global_range=[{global_start},{global_end}), checked={n_checked}, oob={n_oob}. "
                f"Likely: start_atom_id/atom_offset chaining or chunk_start mismatch or ordering mismatch."
            )

        # --------------------------------------------------------------
        # 2) loss 初期化（“新しい Tensor を作る”のではなく、必要なら 0 を作る）
        #    ※ requires_grad は commit_loss の計算結果が持つはずなので、ここは0でOK
        # --------------------------------------------------------------
        commit_loss = torch.zeros((), device=device, dtype=dtype)
        codebook_loss = torch.zeros((), device=device, dtype=dtype)
        repel_loss = torch.zeros((), device=device, dtype=dtype)
        cb_repel_loss = torch.zeros((), device=device, dtype=dtype)
        ent_loss = torch.zeros((), device=device, dtype=dtype)

        # --------------------------------------------------------------
        # 3) commitment_loss 計算（mask_dict があるときだけ）
        # --------------------------------------------------------------
        if (mask_dict is not None) and (B > 0):
            # まず codebook を回して「割当」と「更新」を発生させる
            quantize_st, embed_ind_dict, _embed, ent_loss = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode="train",
            )

            # out = self.commitment_loss(
            #     encoder_outputs=encoder_outputs,
            #     mask_dict=mask_dict,
            #     codebook=self._codebook,  # あなたの実装に合わせる（dict or tensor/module）
            #     logger=logger,
            #     chunk_start=global_start,  # ★重要: global->local 変換の基準
            #     beta=getattr(self, "beta", 0.25),
            #     temperature=getattr(self, "temperature", None),
            #     use_cosine=getattr(self, "use_cosine", False),
            # )
            # commit_loss, codebook_loss, repel_loss, cb_repel
            commit_loss, codebook_loss, repel_loss, cb_repel_loss = self.commitment_loss(
                encoder_outputs=encoder_outputs,
                mask_dict=mask_dict,
                codebook=_embed,  # or the correct container of centers (see note below)
                codebook_mod=self._codebook,  # <-- ADD THIS
                logger=logger,
                chunk_start=global_start,
                beta=getattr(self, "beta", 0.25),
                temperature=getattr(self, "temperature", None),
                use_cosine=getattr(self, "use_cosine", False),
                embed_ind_dict=embed_ind_dict,
            )

            # total_loss = commit_loss + codebook_loss + repel_loss + cb_repel_loss + ent_loss
            #
            # # commitment_loss 側が tuple を返す想定に合わせて受ける
            # # 期待: (commit, cb, rep, cb_rep)
            # if isinstance(out, (tuple, list)) and len(out) == 4:
            #     commit_loss, codebook_loss, repel_loss, cb_repel_loss = out
            # else:
            #     # 想定外の返り値は落とす（静かに壊れるより良い）
            #     raise TypeError(f"commitment_loss must return 4-tuple, got {type(out)}: {out}")

        # --------------------------------------------------------------
        # 4) loss の “型/デバイス/次元” を揃える（0-dim Tensor）
        #    - 既存 Tensor はラップしない（detach もしない）
        #    - scalar/None だけ Tensor 化
        # --------------------------------------------------------------
        def _loss0(x):
            if isinstance(x, torch.Tensor):
                # 0次元に揃える（値のコピーは発生し得るが、wrap ではない）
                return x.to(device=device, dtype=dtype).reshape(())
            if x is None:
                return torch.zeros((), device=device, dtype=dtype)
            if isinstance(x, (float, int)):
                return torch.tensor(float(x), device=device, dtype=dtype)
            raise TypeError(f"_loss0: unsupported loss type {type(x)}")

        commit_loss = _loss0(commit_loss)
        codebook_loss = _loss0(codebook_loss)
        repel_loss = _loss0(repel_loss)
        cb_repel_loss = _loss0(cb_repel_loss)

        # --------------------------------------------------------------
        # 5) グローバル offset 更新（学習時のみ）
        # --------------------------------------------------------------
        if mode not in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = global_end

        # --------------------------------------------------------------
        # 6) return
        # --------------------------------------------------------------
        total_loss = commit_loss + codebook_loss + repel_loss + cb_repel_loss + ent_loss
        # now the mode is train
        return total_loss, (commit_loss, codebook_loss, repel_loss, cb_repel_loss, ent_loss)
