import torch.distributed as distributed
from einops import rearrange, repeat, pack, unpack
from utils import CBDICT
from torch import nn, einsum

from train_teacher import get_args


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
    element_names: Optional[Sequence[str]] = None,  # optional: labels for heads (e.g., ["C","N","O",...])
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, int]]]:
    """
    Lloyd K-Means w/ K-Means++ init, streaming/blocked (no [H,N,K] allocation).

    Returns:
        means:          [H, K, D]
        bins:           [H, K]        # counts per cluster
        used_per_head:  [H]           # number of non-empty clusters per head
        used_per_label: dict[str,int] # only if element_names provided, else None
    """
    H, N, D = samples.shape
    device, dtype = samples.device, samples.dtype

    # cap K to N
    K = int(min(num_clusters, N))
    if K <= 0:
        raise ValueError("No samples to cluster.")

    # optional cosine normalizations (for Lloyd steps; init handles inside)
    if use_cosine_sim:
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)

    @torch.no_grad()
    def kmeanspp_init_blockwise(
        X: torch.Tensor,
        K: int,
        cosine: bool = False,
        eps: float = 1e-12,
        square_prob: bool = True,
        sample_block_elems: int = 262_144,      # tune to your memory budget
        dtype_prob: torch.dtype = torch.float32,# keep distances/probs in fp32
        deterministic: str = "auto",            # "cpu_cumsum" | "gpu_scan" | "auto"
    ):
        """
        Memory-safe KMeans++ initializer with deterministic-safe sampling.
        """
        assert deterministic in ("cpu_cumsum", "gpu_scan", "auto")
        if deterministic == "auto":
            deterministic = "cpu_cumsum" if torch.are_deterministic_algorithms_enabled() else "gpu_scan"

        H, N, D = X.shape
        device = X.device
        C = torch.empty((H, K, D), device=device, dtype=X.dtype)

        # --- distance workspace ---
        if cosine:
            Xwork = torch.nn.functional.normalize(X, p=2, dim=-1)
            x2 = None
        else:
            Xwork = X
            x2 = (X ** 2).sum(-1).to(dtype_prob)  # [H, N]

        # 1) first seed per head
        idx0 = torch.randint(0, N, (H, 1), device=device)
        C[:, 0, :] = X.gather(1, idx0.unsqueeze(-1).expand(H, 1, D)).squeeze(1)

        def dist_to_center(x_all: torch.Tensor, c_one: torch.Tensor) -> torch.Tensor:
            # x_all: [H,N,D], c_one: [H,D] -> [H,N] (dtype_prob)
            if cosine:
                d = (1.0 - (x_all * c_one.unsqueeze(1)).sum(-1)).clamp_min_(0)
            else:
                c2 = (c_one ** 2).sum(-1)                # [H]
                xc = (x_all * c_one.unsqueeze(1)).sum(-1)# [H,N]
                d = (x2 + c2.unsqueeze(1) - 2.0 * xc).clamp_min(0)
            return d.to(dtype_prob)

        # 2) initialize closest distances
        closest = dist_to_center(Xwork, C[:, 0, :])  # [H, N], fp32

        # --- deterministic-safe samplers ---
        def _sample_block_cpu_cumsum(rp: torch.Tensor) -> torch.Tensor:
            totals = rp.sum(dim=1)  # [H]
            zero_mask = totals <= 0
            u = torch.rand(H, device=rp.device, dtype=dtype_prob) * torch.clamp(totals, min=eps)

            idx_out = torch.empty(H, device=rp.device, dtype=torch.long)
            cum = torch.zeros(H, device=rp.device, dtype=dtype_prob)
            found = torch.zeros(H, device=rp.device, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = rp[:, start:end]              # [H,B]
                block_sum = block.sum(dim=1)          # [H]

                target_in_block = (~found) & (cum + block_sum >= u)
                if target_in_block.any():
                    h_idx = target_in_block.nonzero(as_tuple=False).squeeze(1)

                    # CPU hop for deterministic cumsum
                    sub_cpu = block[h_idx].to("cpu", dtype=torch.float64, non_blocking=False)  # [H_sel,B]
                    sub_cum_cpu = sub_cpu.cumsum(dim=1)                                        # deterministic
                    need_cpu = (u[h_idx] - cum[h_idx]).to("cpu", dtype=torch.float64).unsqueeze(1)
                    pos_cpu = torch.searchsorted(sub_cum_cpu, need_cpu, right=False).squeeze(1) # [H_sel]
                    pos = pos_cpu.to(device=rp.device, dtype=torch.long, non_blocking=False)

                    idx_out[h_idx] = start + pos
                    found[h_idx] = True

                # advance cum for unresolved heads
                not_found = ~found
                if not_found.any():
                    cum[not_found] = cum[not_found] + block_sum[not_found]

                start = end
                if found.all():
                    break

            # finalize unresolved / degenerate
            if (~found).any():
                idx_out[~found] = N - 1
            if zero_mask.any():
                idx_out[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device=rp.device)
            return idx_out

        def _sample_block_gpu_scan(rp: torch.Tensor) -> torch.Tensor:
            totals = rp.sum(dim=1)  # [H]
            zero_mask = totals <= 0
            u = torch.rand(H, device=rp.device, dtype=dtype_prob) * torch.clamp(totals, min=eps)

            idx_out = torch.empty(H, device=rp.device, dtype=torch.long)
            cum = torch.zeros(H, device=rp.device, dtype=dtype_prob)
            found = torch.zeros(H, device=rp.device, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = rp[:, start:end]              # [H,B]
                block_sum = block.sum(dim=1)          # [H]

                target_in_block = (~found) & (cum + block_sum >= u)
                if target_in_block.any():
                    h_idx = target_in_block.nonzero(as_tuple=False).squeeze(1)
                    sub = block[h_idx]                # [H_sel,B]
                    need = (u[h_idx] - cum[h_idx])    # [H_sel]

                    # row-wise deterministic scan (no cumsum)
                    pos = torch.empty_like(h_idx, dtype=torch.long, device=rp.device)
                    for n in range(sub.size(0)):
                        row = sub[n]                  # [B]
                        running = torch.zeros((), device=rp.device, dtype=row.dtype)
                        j = 0
                        B = row.numel()
                        while j < B:
                            running = running + row[j]
                            if running >= need[n]:
                                break
                            j += 1
                        pos[n] = j if j < B else (B - 1)

                    idx_out[h_idx] = start + pos
                    found[h_idx] = True

                # advance cum for unresolved heads
                not_found = ~found
                if not_found.any():
                    cum[not_found] = cum[not_found] + block_sum[not_found]

                start = end
                if found.all():
                    break

            if (~found).any():
                idx_out[~found] = N - 1
            if zero_mask.any():
                idx_out[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device=rp.device)
            return idx_out

        # --- outer loop: pick K-1 more centers ---
        for k in range(1, K):
            # if k % 1000 == 0:
            #     print(f"{k},", end="")

            # build stabilized probabilities from current closest distances
            rp = closest
            rp = (rp - rp.amin(dim=1, keepdim=True)).clamp_min_(0) + eps
            rp = rp * rp  # squared prob is standard k-means++

            idxk = _sample_block_cpu_cumsum(rp) if deterministic == "cpu_cumsum" else _sample_block_gpu_scan(rp)

            # set new center
            C[:, k, :] = X[torch.arange(H, device=device), idxk, :]

            # update closest distances in one pass
            dk = dist_to_center(Xwork, C[:, k, :])  # [H,N]
            closest = torch.minimum(closest, dk)

        if cosine:
            C = torch.nn.functional.normalize(C, p=2, dim=-1)
        return C

    means = kmeanspp_init_blockwise(samples, K, use_cosine_sim, eps)   # [H,K,D]

    # ---- Lloyd steps (streaming/blocked) ----
    def assign_pass(X, C):
        """
        Returns buckets [H,N] (long)
        Streaming over K tiles and N tiles to find argmin per point.
        """
        H, N, D = X.shape
        _, K, _ = C.shape
        buckets = torch.empty(H, N, device=device, dtype=torch.long)

        for h in range(H):
            # running best per point in this head
            if use_cosine_sim:
                best_val = torch.full((N,), -float('inf'), device=device, dtype=X.dtype)
            else:
                best_val = torch.full((N,), float('inf'), device=device, dtype=X.dtype)
            best_idx = torch.zeros((N,), device=device, dtype=torch.long)

            # precompute norms of X[h] once (for L2)
            if not use_cosine_sim:
                x2_full = (X[h]**2).sum(-1)  # [N]

            # tile over K
            for k0 in range(0, K, k_block):
                k1 = min(k0 + k_block, K)
                Ck = C[h, k0:k1]                                   # [kb, D]
                if use_cosine_sim:
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        sims = X[h, n0:n1] @ Ck.T                  # [nb,kb]
                        vals, idxs = sims.max(dim=1)               # [nb]
                        update = vals > best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, (idxs + k0), best_idx[n0:n1])
                else:
                    c2 = (Ck**2).sum(-1)                           # [kb]
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        xc = X[h, n0:n1] @ Ck.T                    # [nb,kb]
                        d2 = (x2_full[n0:n1].unsqueeze(1) + c2.unsqueeze(0) - 2*xc).clamp_min(0)
                        vals, idxs = d2.min(dim=1)                 # [nb]
                        update = vals < best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, (idxs + k0), best_idx[n0:n1])

            buckets[h] = best_idx

        return buckets

    def update_pass(X, buckets, K):
        """
        Accumulates sums and counts in chunks of N.
        Returns:
          new_means [H,K,D], bins [H,K]
        """
        H, N, D = X.shape
        new_means = torch.zeros(H, K, D, device=device, dtype=X.dtype)
        bins      = torch.zeros(H, K,     device=device, dtype=torch.long)

        for h in range(H):
            for n0 in range(0, N, n_block):
                n1 = min(n0 + n_block, N)
                b = buckets[h, n0:n1]                              # [nb]
                x = X[h, n0:n1]                                    # [nb,D]
                # counts
                bins[h].index_add_(0, b, torch.ones_like(b, dtype=torch.long, device=device))
                # sums
                new_means[h].index_add_(0, b, x)

        all_reduce_fn(bins)
        all_reduce_fn(new_means)

        # safe divide
        zero_mask = (bins == 0)
        denom = bins.clamp_min(1).unsqueeze(-1)                    # [H,K,1]
        new_means = new_means / denom
        # keep old center for empty bins
        new_means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

        if use_cosine_sim:
            new_means = torch.nn.functional.normalize(new_means, p=2, dim=-1)

        return new_means, bins

    prev_means = None
    for it in range(max_iters):
        buckets = assign_pass(samples, means)                       # [H,N]
        new_means, bins = update_pass(samples, buckets, K)          # [H,K,D], [H,K]
        if tol > 0.0:
            prev_means = means
        means = new_means
        if tol > 0.0:
            shift = (means - prev_means).pow(2).sum(-1).sqrt().mean()
            if float(shift) <= tol:
                break

    # final counts matched to final means
    buckets = assign_pass(samples, means)
    _, bins = update_pass(samples, buckets, K)

    # ---- NEW: report actually-used codebook size per head ----
    used_per_head = (bins > 0).sum(dim=1)   # [H]

    used_per_label: Optional[Dict[str, int]] = None
    if element_names is not None:
        if len(element_names) != H:
            raise ValueError(f"element_names must have length H={H}, got {len(element_names)}")
        used_per_label = {element_names[h]: int(used_per_head[h].item()) for h in range(H)}
        # neat printout
        print("\n[Used codebook size per element]")
        # for name, cnt in used_per_label.items():
        #     print(f"  {name:>4s} : {cnt}")

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
        device = z.device
        eps = 1e-8
        z = z.squeeze()
        # ---- 1) 距離の一次統計（1Dで扱う）----
        # z: [B, D]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # print(f"z {z.shape}")
        if z.shape[0] == 1:
            print(f"latent count is only 1. Not calculating losses.")
            return 0, 0, 0, 0, 0
        pdist_z = torch.pdist(z, p=2)  # [B*(B-1)/2], 1D

        # （巨大時）サンプルを間引き
        sample = pdist_z
        if sample.numel() > 1_000_000:
            sample = self.sample_cap(sample, max_n=200_000)

        # 分位点などは学習に使う重みの境界値なので勾配不要
        with torch.no_grad():
            dynamic_threshold = torch.quantile(sample, 0.10)  # tensor
            lower_q, upper_q = 0, 0.9
            lower_thresh = torch.quantile(sample, 0.25)  # tensor
            upper_thresh = torch.quantile(sample, 0.75)  # tensor
            center = (lower_thresh + upper_thresh) / 2  # tensor

        # ---- ログ（負荷・転送を抑えて）----
        if chunk % 32 == 0:
            with torch.no_grad():
                s = sample
                if s.numel() > 100_000:
                    idx = torch.randperm(s.numel(), device=s.device)[:100_000]
                    s = s.index_select(0, idx)
                s_cpu = s.detach().to('cpu', dtype=torch.float32).flatten()
                hist = torch.histc(s_cpu, bins=10, min=0.0, max=1.0)
                vals = hist.tolist()
                print(f"{key} {vals}")
                # logger.info(vals)

        def latent_repel_mid_chunked(
                z,
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
            """
            z: [B, D], requires_grad=True
            low/high/center: no-grad tensors on the same device as z (OK)
            """
            import torch
            import torch.utils.checkpoint as cp

            # --- Safety: mutually exclusive modes ---
            assert not (use_checkpoint and stream_backward), \
                "checkpoint と streaming backward は同時に使えません。"

            # --- quick diagnostic sampling ---
            with torch.no_grad():
                idx = torch.randperm(z.shape[0], device=z.device)[:4096]
                d = torch.cdist(z[idx], z[idx], p=2)
                mask = (d > low) & (d < high)
                if mask.any():
                    d_in = d[mask]
                    m = float(d_in.mean())
                    c = float(center)
                    s = float(sigma)
                    rel = float(((d_in - c).abs().mean() / max(1e-8, (high - low))))
                    # print(f"[repel diag] in-window pairs={mask.sum().item()} "
                    #       f"mean(d)={m:.4f} center={c:.4f} "
                    #       f"mean|d-center|={float((d_in - c).abs().mean()):.4f} "
                    #       f"rel_to_band={rel:.3f}")
            if self.training:
                assert z.requires_grad, "z.requires_grad=False（上流で detach されている可能性）"
            B, D = z.shape
            device, dtype = z.device, z.dtype

            # --- parameters ---
            band = (high - low).abs()
            sigma = torch.clamp(0.20 * band, min=1e-4)
            sharp = 10  # not 20 to start

            # --- threshold tensors ---
            low = torch.as_tensor(low, device=device, dtype=dtype)
            high = torch.as_tensor(high, device=device, dtype=dtype)
            center = torch.as_tensor(center, device=device, dtype=dtype)

            # --- block counting for normalization ---
            def count_blocks(B, rb, cb):
                cnt, i = 0, 0
                while i < B:
                    j = i + rb
                    while j < B:
                        cnt += 1
                        j += cb
                    i += rb
                return max(cnt, 1)

            row_block = max(1, min(row_block, B))
            col_block = max(1, min(col_block, B))

            n_blocks_total = count_blocks(B, row_block, col_block)

            # --- inner block loss ---
            def block_loss(
                    zi, zj,
                    low_t, high_t, center_t,
                    sigma, sharp, eps, detach_weight, diagonal=True,
                    gamma: float = 1.0,
            ):
                import torch

                if zi.numel() == 0 or zj.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

                # fp32 for stability
                zi32, zj32 = zi.float(), zj.float()
                low32 = torch.as_tensor(low_t, device=zi.device, dtype=torch.float32)
                high32 = torch.as_tensor(high_t, device=zi.device, dtype=torch.float32)
                center32 = torch.as_tensor(center_t, device=zi.device, dtype=torch.float32)
                sigma32 = torch.clamp(torch.as_tensor(sigma, device=zi.device, dtype=torch.float32), min=1e-6)

                d = torch.cdist(zi32, zj32, p=2)  # [bi, bj]
                mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
                d = d[mask]  # keep i<j only

                # gating window
                x1 = (sharp * (d - low32)).clamp(-40.0, 40.0)
                x2 = (sharp * (high32 - d)).clamp(-40.0, 40.0)
                w = torch.sigmoid(x1) * torch.sigmoid(x2)
                if detach_weight:
                    w = w.detach()

                # bell curve weighting
                exp_arg = -((d - center32) ** 2) / (2.0 * sigma32 * sigma32)
                exp_arg = exp_arg.clamp(min=-60.0, max=0.0)
                bell = torch.exp(exp_arg)
                if gamma != 1.0:
                    bell = bell ** gamma

                num = (w * bell).sum()
                den = w.sum().clamp_min(eps)

                out = num / den
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                # gradient safety
                if self.training:
                    if not out.requires_grad:
                        out = out + 0.0 * (zi.sum() + zj.sum())
                    else:
                        pass

                return out
            # print("--- 0 ---")
            i = 0
            total = z.new_zeros(())
            while i < B:
                bi = min(row_block, B - i)
                zi = z[i:i + bi]

                j = i  # include diagonal blocks
                while j < B:
                    bj = min(col_block, B - j)
                    zj = z[j:j + bj]
                    diagonal = (i == j)  # inside block, keep only i<j if diagonal
                    lb = cp.checkpoint(block_loss, zi, zj, low, high, center, sigma, sharp, eps, detach_weight,
                                        diagonal=diagonal,use_reentrant=False)
                    # lb = block_loss(zi, zj, ..., diagonal=diagonal)
                    total = total + lb
                    j += col_block
                i += row_block

            out = total / n_blocks_total

            # print("--- 1 ---")
            return out

        # ---- 3) コードブック反発（チャンク & no-backward）----
        def repel_codebooks_chunked(cb, sigma=1.0, block=4096):
            """
            cb: [K, D]
            返り値は全ペア（上三角＋ブロック間）の平均（スカラーTensor）。
            """
            K = cb.size(0)
            sum_val = cb.new_zeros(())
            cnt = 0

            for i in range(0, K, block):
                ci = cb[i:i + block]  # [bi, D]

                # 同ブロック内の上三角（pdist）
                if ci.size(0) >= 2:
                    dij = torch.pdist(ci, p=2)  # [bi*(bi-1)/2]
                    if dij.numel() > 0:
                        li = torch.exp(-dij.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + li.sum()
                        cnt += li.numel()
                    del dij

                # ブロック間（cdist）
                for j in range(i + block, K, block):
                    cj = cb[j:j + block]  # [bj, D]
                    if ci.size(0) > 0 and cj.size(0) > 0:
                        d = torch.cdist(ci, cj, p=2)  # [bi, bj]
                        lj = torch.exp(-d.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + lj.sum()
                        cnt += lj.numel()
                        del d, lj
                    del cj

                del ci

            if cnt == 0:
                return sum_val  # 0
            return sum_val / cnt

        # ---- 4) そのほかの軽量ロス ----
        def repel_from_zero_1d(d, margin):
            # 0 から margin までの距離にペナルティ（平均）
            return torch.relu(margin - d).mean()

        # ---- 5) ロス計算 ----
        latent_repel_loss_mid = latent_repel_mid_chunked(
            z,
            low=lower_thresh, high=upper_thresh, center=center,
            sigma=1.0, sharp=20.0,
            row_block=1024, col_block=1024,
            detach_weight=True,
            use_checkpoint=True,  # 推奨: True
            stream_backward=False,  # forward 内では False
        )
        # print(f"latent_repel_loss_mid {latent_repel_loss_mid}")
        # ほかの損失と合算して最後に一度だけ backward()
        # 参考指標（元式に相当）
        repel_from_2 = torch.exp(- (pdist_z - 2.0) ** 2 / (2 * 3.0 ** 2)).mean()

        cb_loss = repel_codebooks_chunked(codebook)

        # “ゼロからのマージン反発”を 1D 距離で
        latent_repel_loss = repel_from_zero_1d(pdist_z, lower_thresh) + cb_loss

        # 必要に応じて attract を使うならここで足す
        # attract_weight = 1.0
        repel_weight = 1.0

        final_loss = repel_weight * latent_repel_loss_mid
        neg_loss = z.new_tensor(1.0)  # プレースホルダ（元コードの返却形に合わせて）

        # （注意）empty_cache は通常は不要。OOM 調査中だけ使うのが無難
        # del pdist_z, sample
        # torch.cuda.empty_cache()
        # final_loss and cb_loss are used
        return final_loss, neg_loss, repel_from_2, cb_loss, latent_repel_loss


import torch.nn.functional as F


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
            sample_codebook_temp=0
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        args = get_args()
        self.samples_latent_in_kmeans = args.samples_latent_in_kmeans
        self.epoch_at_mode_shift = args.epoch_at_mode_shift
        self.train_or_infer = args.train_or_infer
        # if args.train_or_infer == "infer":
        #     self.kmeans_iters = 200
        # else:
        #     self.kmeans_iters = 30
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        self.use_checkpoint = args.use_checkpoint
        self.cb_dict = CBDICT
        # key 1, code torch.Size([47, 16]), embed_k torch.Size([19, 16])
        assert not (
                    use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        for elem in self.cb_dict.keys():
            self.register_buffer(f"cluster_size_{elem}", torch.zeros(self.cb_dict[elem]))
            self.register_buffer(f"embed_avg_{elem}", torch.zeros(self.cb_dict[elem], dim))
        self.learnable_codebook = learnable_codebook
        self.embed = nn.ParameterDict()
        # self.embed_avg = nn.ParameterDict()
        for key in self.cb_dict.keys():
            # Make a fresh tensor copy per element
            K_e = self.cb_dict[key]  # e.g. 4360 for carbon
            D = dim
            init = torch.randn(K_e, D) * 0.01  # initial latents does not matter cause overwritten in init_emb
            self.embed[str(key)] = nn.Parameter(init, requires_grad=True)
            # self.embed_avg[str(key)] = nn.Parameter(embed.clone().detach(), requires_grad=True)
        self.latent_size_sum = 0
        self.embed_ind_dict = {}
        self.quantize_dict = {}

    def reset_kmeans(self):
        self.initted.data.copy_(torch.Tensor([False]))

    def copy_codebook_(self, embed: torch.Tensor, init: torch.Tensor, fill="data", data=None):
        """
        embed: [K, D] or [D, K] current codebook
        init : [K_used, D] or [D, K_used] initializer
        fill : "data" | "repeat" | "randn"
        data : [N, D] latents to sample from if fill=="data"
        """
        # --- align orientation so last dim is D ---
        def to_KD(t, D_expected):
            # try [K, D] first
            if t.shape[-1] == D_expected:
                return t  # [K_used, D]
            # else try transpose from [D, K_used]
            if t.shape[0] == D_expected:
                return t.t().contiguous()  # -> [K_used, D]
            raise ValueError(f"init shape {tuple(t.shape)} not compatible with D={D_expected}")

        # make embed [K, D] view
        if embed.shape[-1] < embed.shape[0]:  # heuristics: typical is [K, D]
            K, D = embed.shape
            initKD = to_KD(init, D)
            K_used = initKD.shape[0]
            # copy overlap
            n = min(K, K_used)
            embed[:n].copy_(initKD[:n])

            # fill remaining codes
            if K > n:
                if fill == "data" and data is not None and data.numel() > 0:
                    idx = torch.randint(0, data.size(0), (K - n,), device=embed.device)
                    embed[n:K].copy_(data[idx])
                elif fill == "repeat" and n > 0:
                    reps = (K - n + n - 1) // n
                    embed[n:K].copy_(initKD[:n].repeat((reps, 1))[:K - n])
                elif fill == "randn":
                    embed[n:K].normal_(0, 1e-3)
                else:
                    embed[n:K].copy_(embed[:1])  # fallback
        else:
            # codebook is [D, K]; do the same with transposes
            D, K = embed.shape
            initKD = to_KD(init, D)  # [K_used, D]
            initDK = initKD.t().contiguous()  # [D, K_used]
            n = min(K, initDK.shape[1])
            embed[:, :n].copy_(initDK[:, :n])
            if K > n:
                if fill == "randn":
                    embed[:, n:K].normal_(0, 1e-3)
                else:
                    embed[:, n:K].copy_(embed[:, :1])

    @torch.jit.ignore
    def init_embed_(self, data, mask_dict=None, use_cosine_sim: bool = False):
        """
        Initialize per-element codebooks using absolute K from self.cb_dict.
        - self.cb_dict[skey] is absolute K_req (int).
        - We DO NOT cap the final codebook size: embed and cluster_size buffers
          are created with shape (K_req, D) and (K_req,), even if K_req > N_i.
        - K-Means is executed with K_run = min(K_req, N_i) and then padded.
        """
        assert mask_dict is not None, "mask_dict is required"
        print("++++++++++++++++ RUNNING init_embed (ABS K, NO FINAL CAP) +++++++++++++")

        device = data.device
        D = data.shape[-1]

        from utils import CORE_ELEMENTS  # e.g. {"6","7","8",...} as strings

        def get_idx(md, k):
            return md.get(k, md.get(str(k), md.get(str(k))))

        def get_absK(d, k):
            v = d.get(k, d.get(str(k), d.get(str(k))))
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                raise ValueError(f"cb_dict[{k}] must be an integer (got {type(v)}: {v})")

        # simple padding helper: pad to K_req with zero-count, small-noise means
        def _pad_to_K(means_1kd, counts_1k, K_req: str, data_stats=None):
            """
            means_1kd: [1, K_run, D]
            counts_1k: [1, K_run]
            returns means_pad [K_req, D], counts_pad [K_req]
            """
            H, K_run, Dd = means_1kd.shape
            assert H == 1 and Dd == D
            means_kd = means_1kd[0]  # [K_run, D]
            counts_k = counts_1k[0]  # [K_run]

            if K_req <= K_run:
                return means_kd[:K_req].contiguous(), counts_k[:K_req].contiguous()

            # allocate output
            K_pad = K_req - K_run
            out_means = torch.empty((K_req, D), device=means_kd.device, dtype=means_kd.dtype)
            out_counts = torch.zeros((K_req,), device=counts_k.device, dtype=counts_k.dtype)

            # copy existing
            out_means[:K_run].copy_(means_kd)
            out_counts[:K_run].copy_(counts_k)

            # init extra slots: small noise around data mean (or zeros as fallback)
            if data_stats is not None:
                mu, sigma = data_stats  # [D], scalar
                noise = torch.randn((K_pad, D), device=means_kd.device, dtype=means_kd.dtype) * (0.01 * sigma + 1e-6)
                out_means[K_run:] = mu + noise
            else:
                out_means[K_run:] = 0

            # counts remain zero for padded slots
            return out_means, out_counts

        # precompute global stats per element batch (for better pad init)
        def _stats(x_nd):
            # x_nd: [N_i, D]
            if x_nd.numel() == 0:
                return None
            mu = x_nd.mean(dim=0)
            # robust scalar scale for noise: median absolute deviation proxy
            with torch.no_grad():
                dev = (x_nd - mu).pow(2).sum(dim=1).sqrt()
                sigma = torch.quantile(dev, 0.5)  # median L2 radius
            return mu, sigma

        for raw_key in sorted(mask_dict.keys(), key=lambda x: str(x)):
            skey = str(raw_key)
            if skey not in self.cb_dict.keys():
                continue

            idx = get_idx(mask_dict, skey)
            if idx is None:
                print(f"[init_embed_] skip Z={skey}: no indices in mask_dict")
                continue

            masked = data[0][idx]  # (N_i, D)
            N_i = masked.shape[0]
            if N_i == 0:
                print(f"[init_embed_] skip Z={skey}: N_i=0")
                continue

            # ---- absolute K requested (you can set bigger values for 6/7/8 in cb_dict) ----
            K_req = get_absK(self.cb_dict, skey)
            if K_req is None or K_req <= 0:
                print(f"[init_embed_] warn Z={skey}: invalid K in cb_dict -> default to 1")
                K_req = 1

            # K-Means can only produce up to N_i distinct centers; run with K_run
            K_run = min(K_req, N_i)

            # run kmeans (on CUDA); returns [1,K_run,D], [1,K_run]
            means_1kd, counts_1k, used_per_head, used_per_label = kmeans(
                masked.unsqueeze(0).to(device),
                num_clusters=K_run,
                use_cosine_sim=use_cosine_sim,
            )

            # compact if you have dead centers (optional, keeps K_run ≤ produced)
            # If your compact_clusters always pads to max of its input, you can skip it here.
            # Otherwise, do:
            # means_1kd, counts_1k, used_mask = compact_clusters(means_1kd, counts_1k, pad_to_max=True)

            # pad up to K_req (NO CAP on final size)
            means_kd, counts_k = _pad_to_K(means_1kd, counts_1k, K_req, data_stats=_stats(masked))

            # allocate/resize destination params & buffers to EXACTLY K_req
            if skey not in self.embed or self.embed[skey].shape != (K_req, D):
                self.embed[skey] = torch.nn.Parameter(
                    means_kd.detach().to(device=device, dtype=means_kd.dtype),
                    requires_grad=True
                )
            else:
                self.embed[skey].data.copy_(means_kd)

            buf_name = f"cluster_size_{skey}"
            if not hasattr(self, buf_name) or getattr(self, buf_name).shape[0] != K_req:
                self.register_buffer(buf_name, counts_k.detach().to(device=device))
            else:
                getattr(self, buf_name).data.copy_(counts_k)

            nz = int((counts_k > 0).sum().item())
            print(f"[init_embed_] Z={skey} N={N_i} K_req={K_req} K_run={K_run} K_used={nz}/{K_req}")

    def replace(self, batch_samples, batch_mask):
        self.initted.data.copy_(torch.Tensor([True]))
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    import torch

    @torch.inference_mode()
    def silhouette_score_torch(self,
                               X: torch.Tensor,
                               labels: torch.Tensor,
                               row_block: int = 8192,
                               device: str | torch.device | None = None) -> float:
        """
        Silhouette score on GPU if available, otherwise CPU.
        - X: [N, D] (any device / dtype)
        - labels: [N] (ints; any device / dtype)
        - row_block: chunk size for memory control
        - device: force 'cuda'/'cpu' or torch.device; default: auto
        """
        # Accept modules with .embed/.weight
        if isinstance(X, torch.nn.Module):
            X = getattr(X, "embed", getattr(X, "weight", None))
            if X is None:
                raise TypeError("X is a module without .embed/.weight tensor.")

        # Choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prepare tensors
        X = X.detach().to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.detach().to(device=device, dtype=torch.long, non_blocking=True)
        X = X.squeeze()
        labels = labels.squeeze()

        N = X.shape[0]
        if N <= 1:
            return 0.0

        # Map labels -> compact 0..K-1 (this automatically ignores any vacant codebook IDs)
        uniq, inv = labels.unique(sorted=True, return_inverse=True)  # inv: [N] in 0..K-1
        K = uniq.numel()
        if K <= 1:
            return 0.0

        counts = torch.bincount(inv, minlength=K).to(device)  # >0 by construction
        counts_f = counts.float()

        sil_sum = 0.0
        processed = 0

        for start in range(0, N, row_block):
            end = min(start + row_block, N)
            B = end - start

            Xb = X[start:end]  # [B, D]

            # Pairwise distances for the block
            d_block = torch.cdist(Xb, X, p=2)  # [B, N]

            # Sum distances to each active cluster
            inv_index = inv.view(1, N).expand(B, N)  # [B, N]
            sums_per_cluster = torch.zeros(B, K, device=device, dtype=d_block.dtype)
            sums_per_cluster.scatter_add_(1, inv_index, d_block)  # [B, K]

            # Mean distance to each cluster
            means_per_cluster = sums_per_cluster / counts_f.view(1, K)  # [B, K]

            # a(i): mean intra-cluster (exclude self)
            k_block = inv[start:end]  # [B]
            a_counts = counts[k_block] - 1
            a_mean = means_per_cluster.gather(1, k_block[:, None]).squeeze(1)  # [B]

            scale = torch.zeros_like(a_counts, dtype=a_mean.dtype)
            mask_gt1 = a_counts > 0
            if mask_gt1.any():
                # scale = n / (n - 1) to remove self-distance (0) from the mean
                n = counts[k_block][mask_gt1].float()
                scale[mask_gt1] = n / (n - 1.0)
            a_i = a_mean * scale  # [B]; 0 where cluster size == 1

            # b(i): min mean distance to any other cluster
            means_per_cluster.scatter_(1, k_block[:, None], float('inf'))
            b_i, _ = means_per_cluster.min(dim=1)  # [B]

            denom = torch.maximum(a_i, b_i)
            sil_block = (b_i - a_i) / denom
            sil_block = torch.nan_to_num(sil_block, nan=0.0, posinf=0.0, neginf=0.0)

            sil_sum += sil_block.sum().item()
            processed += B

            del Xb, d_block, sums_per_cluster, means_per_cluster, a_mean, a_i, b_i, sil_block

        return float(sil_sum / max(processed, 1))

    def _normalize_mask_dict(self, mask_dict, device=None):
        import torch, numpy as np
        if mask_dict is None:
            return None
        norm = {}
        for k, v in mask_dict.items():
            # unify key types
            k_int = int(k) if isinstance(k, str) and k.isdigit() else k
            k_str = str(k_int)
            # convert value to tensor
            if isinstance(v, (list, tuple, np.ndarray)):
                v = torch.as_tensor(v, dtype=torch.long, device=device)
            elif isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                v = torch.nonzero(v.flatten(), as_tuple=False).flatten().long().to(device)
            norm[k_int] = v
            norm[k_str] = v
        return norm

    def _as_index_tensor(self, x, N=None, device=None):
        """
        Convert x -> LongTensor[NumIdx] of indices (not boolean).
        - list/tuple/np.ndarray/np.int64 → tensor.long()
        - bool tensor → nonzero indices
        - long/int tensor → .long()
        - None → None
        If N is provided and x is a bool tensor of length N, convert to indices.
        """
        import torch
        import numpy as np

        if x is None:
            return None

        if isinstance(x, torch.Tensor):
            t = x.to(device)
            if t.dtype == torch.bool:
                if N is None:
                    N = t.numel()
                idx = torch.nonzero(t.view(-1), as_tuple=False).flatten()
                return idx.long()
            return t.long()

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            # allow nested np.int64, etc.
            return torch.as_tensor([int(i) for i in x], dtype=torch.long, device=device)

        if isinstance(x, np.ndarray):  # type: ignore[name-defined]
            if x.dtype == np.bool_:  # boolean mask
                x = torch.as_tensor(x, device=device)
                idx = torch.nonzero(x.view(-1), as_tuple=False).flatten()
                return idx.long()
            return torch.as_tensor(x, dtype=torch.long, device=device)

        # single scalar index
        return torch.as_tensor([int(x)], dtype=torch.long, device=device)

    def _global_to_local_indices(self, global_idx, global_start, local_size):
        """
        Given global indices and the current chunk window [global_start, global_start+local_size),
        return local indices within [0, local_size).
        """
        import torch
        lo = global_start
        hi = global_start + local_size
        in_window = (global_idx >= lo) & (global_idx < hi)
        if not torch.any(in_window):
            return global_idx.new_empty((0,), dtype=torch.long)
        return (global_idx[in_window] - lo).long()

    @torch.amp.autocast('cuda', enabled=False)
    #               data, features, mask_dict, logger, chunk_i, epoch, mode
    def forward(self, x, feature, mask_dict=None, logger=None, chunk_i=None, epoch=None, mode=None):
        """Forward pass with per-element quantization and EMA update."""
        # 0. prepare input の少し上 or 直下あたりに
        if mode != "init_kmeans_final" and chunk_i is not None and chunk_i == 0:
            self.latent_size_sum = 0

        # ------------------------------------------------------------------
        # 0. prepare input
        # ------------------------------------------------------------------
        x = x.float()
        if x.ndim < 4:
            x = rearrange(x, '... -> 1 ...')  # (1, B, D)
        flatten = x.view(x.shape[0], -1, x.shape[-1])  # (1, B, D)
        B, D = flatten.shape[1], flatten.shape[2]

        # clear per-call buffers
        self.quantize_dict = {}
        self.embed_ind_dict = {}

        # ------------------------------------------------------------------
        # 1. initialization phase (K-Means embedding)
        # ------------------------------------------------------------------
        if mode == "init_kmeans_final":
            self.init_embed_(flatten, mask_dict)
            print("init_embed is done")
        mask_dict = self._normalize_mask_dict(mask_dict)
        # ------------------------------------------------------------------
        # 2. per-element quantization loop
        # ------------------------------------------------------------------
        for key in {int(k) for k in mask_dict.keys() if str(k).isdigit()}:
            skey = str(key)
            from utils import CORE_ELEMENTS
            if skey not in CORE_ELEMENTS:
                continue
            print(f" feat in ecuclid forward {feature.shape}")
            print(f" flatten in ecuclid forward {flatten.shape}")
            # -------------------- select latents for this element --------------------
            if mode == "init_kmeans_final":
                masked_latents = flatten[0][mask_dict[key]]  # global pass
            else:  # train
                # slice current minibatch range
                gmask = (mask_dict[key] >= self.latent_size_sum) & (
                        mask_dict[key] < self.latent_size_sum + B
                )
                loc = mask_dict[key][gmask] - self.latent_size_sum
                masked_latents = flatten[0][loc]

            if masked_latents.numel() == 0:
                continue

            code = self.embed[str(key)]
            code = code.squeeze(0) if code.ndim == 3 else code  # [K_e, D]

            # -------------------- nearest code indices (no grad) --------------------
            with torch.no_grad():
                dist = torch.cdist(masked_latents, code, p=2).pow(2)
                idx = dist.argmin(dim=-1)  # [Ni]
                del dist
            quantize = code.index_select(0, idx)  # [Ni, D]

            self.quantize_dict[str(key)] = quantize
            self.embed_ind_dict[str(key)] = idx.to(torch.int32)
            # ========================================================================
            #        silhouette (init only, on CPU)
            # ========================================================================
            if mode == "init_kmeans_final":
                try:
                    torch.save(code.detach().cpu(), f"./naked_embed_{epoch}_{key}.pt")
                    torch.save(flatten.detach().cpu(), f"./naked_latent_{epoch}_{key}.pt")
                except Exception as e:
                    if logger: logger.warning(f"Save failed for key {key}: {e}")

                try:
                    from sklearn.utils import resample
                    n = min(self.samples_latent_in_kmeans, masked_latents.shape[0])
                    if n > 1:
                        xs, ys = resample(
                            masked_latents.cpu().numpy(),
                            idx.cpu().numpy(),
                            n_samples=n,
                            random_state=42,
                        )
                        sil = self.silhouette_score_torch(
                            torch.from_numpy(xs).float(), torch.from_numpy(ys).long()
                        )
                        msg = (f"Silhouette Score (subsample): {key} {sil:.4f}, "
                               f"sample size {masked_latents.shape[0]}, K_e {code.shape[0]}")
                        print(msg)
                        logger.info(msg)
                    else:
                        print("n <= 1 !!!!!!!!!")
                except Exception as e:
                    print(f"Silhouette failed for {key}: {e}")
                    if logger: logger.warning(f"Silhouette failed for {key}: {e}")
            # elif mode is None: # training
            #     # ========================================================================
            #     # ここで repel ロス計算。Sil score と違い合計計算必要
            #     # ========================================================================
            #     inds = self.embed_ind_dict[key]
            #     n = int(inds.numel()) if torch.is_tensor(inds) else (len(inds) if inds is not None else 0)
            #     if n < 2:
            #         continue
            #     sil_k = 0
            #     repel_k, div_nega_loss, repel_loss_from_2, cb_repel_k, repel_loss_mid_high = \
            #         (self.compute_contrastive_loss(x, chunk_i, logger, self.embed[str(key)]))
            #     weight_by_counts = True
            #     w = float(n) if weight_by_counts else 1.0
            #     repel_wsum = repel_wsum + w * repel_k
            #     cb_repel_wsum = cb_repel_wsum + w * cb_repel_k
            #     sil_wsum = sil_wsum + w * sil_k
            #     w_total += w
            #     contributed_keys.append((key, n))


            # -------------------- EMA codebook update (hard-EMA) --------------------
            if self.training and epoch is not None and epoch < 30:
                with torch.no_grad():
                    ea = getattr(self, f"embed_avg_{key}")  # [K_e, D]
                    cs = getattr(self, f"cluster_size_{key}")  # [K_e]
                    eps = getattr(self, "eps", 1e-6)
                    decay = float(self.decay)

                    ea.mul_(decay)
                    cs.mul_(decay)

                    one = torch.ones_like(idx, dtype=cs.dtype)
                    cs.index_add_(0, idx, one * (1.0 - decay))
                    ea.index_add_(0, idx, masked_latents.to(ea.dtype) * (1.0 - decay))

                    means = ea / (cs.unsqueeze(-1) + eps)

                    code_param = self.embed[str(key)]
                    code_param.data.copy_(
                        means.unsqueeze(0) if code_param.ndim == 3 else means
                    )

            del masked_latents, code, idx, quantize

        # ------------------------------------------------------------------
        # 3. build full quantized tensor aligned to minibatch
        # ------------------------------------------------------------------
        quantize_full = torch.empty((B, D), device=flatten.device, dtype=flatten.dtype)

        mask_dict = self._normalize_mask_dict(mask_dict)
        for key in {int(k) for k in mask_dict.keys() if str(k).isdigit()}:

            skey = str(key)
            from utils import CORE_ELEMENTS
            if skey not in CORE_ELEMENTS:
                continue

            gmask = (mask_dict[key] >= self.latent_size_sum) & (
                    mask_dict[key] < self.latent_size_sum + B
            )
            idx_global = mask_dict[key][gmask]
            if idx_global.numel() == 0:
                continue
            idx_local = (idx_global - self.latent_size_sum).to(torch.long)
            qk = self.quantize_dict[str(key)].to(flatten.device)
            # 前提：quantize_full は CUDA 側で作る
            device = quantize_full.device

            # idx_local はインデックスなので int64（long）かつ同じ device
            idx_local = idx_local.to(device=device, dtype=torch.long)

            # qk も dtype/device を合わせる（autocast中でもOK）
            qk = qk.to(device=device, dtype=quantize_full.dtype)

            # 空配列ガード（要らなければ削ってOK）
            if idx_local.numel() > 0:
                quantize_full.index_copy_(0, idx_local, qk)

            quantize_full.index_copy_(0, idx_local, qk)

        # fill unused with original latents
        all_local = torch.cat([
            (mask_dict[k][(mask_dict[k] >= self.latent_size_sum) &
                          (mask_dict[k] < self.latent_size_sum + B)]) - self.latent_size_sum
            for k in mask_dict.keys()
            if mask_dict[k].numel() > 0
        ], dim=0)
        used = torch.unique(all_local)
        unused = torch.ones(B, dtype=torch.bool, device=flatten.device)
        unused[used] = False
        if unused.any():
            quantize_full[unused] = flatten[0][unused]

        # ------------------------------------------------------------------
        # 4. Straight-Through estimator & bookkeeping
        # ------------------------------------------------------------------
        quantize_st = flatten[0] + (quantize_full - flatten[0]).detach()
        quantize_st = quantize_st.unsqueeze(0)  # restore (1, B, D)
        self.latent_size_sum += B

        # ------------------------------------------------------------------
        # 5. return
        # ------------------------------------------------------------------
        if mode == "init_kmeans_final":
            return 0
        else:
            torch.cuda.empty_cache()
            return quantize_st, self.embed_ind_dict, self.embed


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

    def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, temperature=1.0, margin=0.1):
        device = embeddings.device
        batch_size = embeddings.size(0)
        # Get soft cluster assignments with temperature control
        if embed_ind.dim() == 1:
            embed_ind = embed_ind.unsqueeze(1)  # (N, 1)
        logits = embed_ind.expand(-1, num_clusters)  # (N, K)
        cluster_assignments = F.softmax(logits / temperature, dim=-1)  # (N, K)
        hard_assignments = torch.zeros_like(cluster_assignments).scatter_(
            1, cluster_assignments.argmax(dim=1, keepdim=True), 1.0
        )
        cluster_sums = hard_assignments.T @ embeddings  # (K, D)
        cluster_sizes = hard_assignments.sum(dim=0, keepdim=True).T  # (K, 1)
        cluster_sizes = cluster_sizes.clamp(min=1.0)  # Avoid division by very small numbers
        centroids = cluster_sums / cluster_sizes  # (K, D)
        assigned_clusters = cluster_assignments.argmax(dim=1)  # (N,)
        assigned_centroids = centroids[assigned_clusters]  # (N, D)
        a = torch.norm(embeddings - assigned_centroids, dim=1)  # (N,)
        mask = torch.ones((batch_size, num_clusters), device=device)
        mask.scatter_(1, assigned_clusters.unsqueeze(1), 0)
        all_distances = torch.cdist(embeddings, centroids)  # (N, K)
        masked_distances = all_distances * mask + (1 - mask) * 1e6  # Set assigned cluster distance high
        b, _ = torch.min(masked_distances, dim=1)  # (N,)
        max_dist = torch.max(a, b) + 1e-6
        silhouette = (b - a) / max_dist - margin
        loss = 1 - torch.mean(torch.tanh(silhouette))
        return loss

    def fast_find_equivalence_groups(self, latents):
        from collections import defaultdict
        hash_map = defaultdict(list)
        for idx, vector in enumerate(latents):
            key = tuple(vector.tolist())  # Convert tensor to a hashable tuple
            hash_map[key].append(idx)
        equivalence_groups = [group for group in hash_map.values() if len(group) > 1]
        return equivalence_groups

    def orthogonal_loss_fn(
            self, mask_dict, embed_ind_dict, codebook, init_feat, x, quantize_dict, logger, epoch, chunk=0
        ):
        import torch
        dev = x.device if hasattr(x, "device") else "cpu"

        # Accumulators
        repel_wsum = torch.zeros((), device=dev)
        cb_repel_wsum = torch.zeros((), device=dev)
        sil_wsum = torch.zeros((), device=dev)
        w_total = 0.0  # total weight of contributing elements
        # OPTIONAL: collect debug info
        contributed_keys = []
        # print(f"[ORTHO] epoch={epoch} chunk={chunk} type(embed_ind_dict)={type(embed_ind_dict).__name__}")
        # if hasattr(embed_ind_dict, "__len__"):
        #     print(f"[ORTHO] keys={list(embed_ind_dict.keys())[:8]} len={len(embed_ind_dict)}")
        # else:
        #     print("[ORTHO] embed_ind_dict has no __len__")
        #
        # # If dict-like but empty, say why we return zeros:
        # if not embed_ind_dict:
        #     print(f"[ORTHO SKIP] epoch={epoch} chunk={chunk}: embed_ind_dict is empty")

        # Iterate per element
        for key in sorted(embed_ind_dict.keys()):
            inds = embed_ind_dict[key]
            # inds may be tensor or list; normalize length
            n = int(inds.numel()) if torch.is_tensor(inds) else (len(inds) if inds is not None else 0)
            # print(f"n {n}")
            # Need >=2 for pairwise losses; skip otherwise
            if n < 2:
                continue
            # Fetch per-element latents/embeds as you already do
            # Example (adapt to your variables):
            # z_k = quantize_dict[key]  # [n, D]
            # cb_k = codebook[str(key)]  # [K, D] or similar

            # --- compute your per-element losses here (examples as placeholders) ---
            # repel_k = ...
            # cb_repel_k = ...
            # sil_k = ...
            sil_k = 0
            repel_k, div_nega_loss, repel_loss_from_2, cb_repel_k, repel_loss_mid_high = \
                (self.compute_contrastive_loss(x, chunk, logger, codebook[str(key)]))
            # print(f"repel_k {repel_k}")
            weight_by_counts = True
            # Weight: either by count or uniform per contributing element
            w = float(n) if weight_by_counts else 1.0

            # Accumulate weighted sums
            repel_wsum = repel_wsum + w * repel_k
            cb_repel_wsum = cb_repel_wsum + w * cb_repel_k
            sil_wsum = sil_wsum + w * sil_k

            w_total += w
            contributed_keys.append((key, n))
            # print(f"w_total {w_total}")

        if w_total == 0.0:
            # No contributors in this chunk → return clean zeros (and optionally log)
            if logger is not None:
                logger.info(f"[orthogonal_loss_fn] chunk {chunk}: no contributing elements; returning zeros.")
            zero = torch.zeros((), device=dev)
            return {
                "repel_loss": zero,
                "cb_repel_loss": zero,
                "sil_loss": zero,
                "contrib": [],
            }

        # Safe averages
        repel_avg = repel_wsum / w_total
        cb_repel_avg = cb_repel_wsum / w_total
        sil_avg = sil_wsum / w_total
        # print(f"repel_avg {repel_avg}")  # this is nonzero
        return {
            "repel_loss": repel_avg,
            "cb_repel_loss": cb_repel_avg,
            "sil_loss": sil_avg,
            "contrib": contributed_keys,  # for debugging
        }

    # def orthogonal_loss_fn(self, embed_ind_dict, codebook, init_feat, latents, quantized_dict, logger, epoch, chunk=0):
    #     # embed_ind_dict.to("cuda")
    #     codebook.to("cuda")
    #     init_feat.to("cuda")
    #     latents.to("cuda")
    #     quantized_dict.to("cuda")
    #     latent_len_sum = 0
    #     two_repel_loss_weighted_sum = 0
    #     cb_loss_weighted_sum = 0
    #     for key in embed_ind_dict.keys():
    #         import torch
    #         latents_for_sil = torch.squeeze(latents)
    #         latents_size = latents_for_sil.shape[0]
    #         latent_len_sum += latents_size
    #         # final_loss, neg_loss, repel_from_2, cb_loss, latent_repel_loss
    #         two_repel_loss, div_nega_loss, repel_loss_from_2, cb_loss, repel_loss_mid_high = (
    #             self.compute_contrastive_loss(latents_for_sil, chunk, logger, codebook[str(key)]))
    #
    #         two_repel_loss_weighted_sum += latents_size * two_repel_loss
    #         cb_loss_weighted_sum += latents_size * cb_loss
    #     two_repel_loss_avg = two_repel_loss_weighted_sum / latent_len_sum
    #     cb_loss_avg = cb_loss_weighted_sum / latent_len_sum
    #
    #     return (two_repel_loss_avg, cb_loss_avg)

    import torch
    import torch.nn.functional as F

    def pairwise_sq_dists(self, x, y):
        # x: [B, D], y: [K, D]
        # computes ||x - y||^2 without sqrt (better gradients / numerics)
        x2 = (x ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        y2 = (y ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        # Clamp to avoid tiny negatives from fp errors
        d2 = torch.clamp(x2 + y2 - 2.0 * x @ y.t(), min=0.0)
        return d2

    @staticmethod
    def pairwise_sq_dists(x, y):  # [B,D], [K,D] -> [B,K]
        # robust and efficient squared Euclidean
        x2 = (x*x).sum(dim=-1, keepdim=True)       # [B,1]
        y2 = (y*y).sum(dim=-1).unsqueeze(0)        # [1,K]
        xy = x @ y.t()                              # [B,K]
        d2 = x2 + y2 - 2*xy
        return torch.clamp(d2, min=0.0)

    def _latent_radius_loss(self, z):
        # Penalize norms above a target radius (no penalty inside the ball)
        norms = torch.linalg.norm(z, dim=-1)
        excess = F.relu(norms - self.target_radius)
        return (excess * excess).mean()

    def _center_batch(self, z):
        # small mean-shift to keep z centered (doesn’t block gradients)
        return z - z.mean(dim=0, keepdim=True)

    import torch
    import torch.nn.functional as F
    from torch import nn
    import torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _as_index_tensor(self, idx_raw, N, device):
        """Make a 1D LongTensor of indices on device. Accepts list/np/tensor/bool-mask."""
        if idx_raw is None:
            return torch.arange(N, device=device)
        if torch.is_tensor(idx_raw):
            if idx_raw.dtype == torch.bool:
                return torch.nonzero(idx_raw, as_tuple=False).squeeze(-1).to(device)
            return idx_raw.to(device).long().view(-1)
        # list / numpy
        return torch.as_tensor(idx_raw, device=device).long().view(-1)

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

    def _squeeze_01(self, x):
        # [1, K, D] -> [K, D]; leave [K, D] unchanged
        return x.squeeze(0) if x.dim() == 3 and x.size(0) == 1 else x

    def _normalize_mask_dict(self, mask_dict):
        """mask_dict のキーを str / int / torch.scalar の同義語に正規化して返す（非破壊）"""
        if mask_dict is None:
            return None
        norm = dict(mask_dict)  # shallow copy
        for k, v in list(mask_dict.items()):
            # int alias
            if isinstance(k, str) and k.isdigit():
                norm.setdefault(int(k), v)
            # str alias
            if isinstance(k, int):
                norm.setdefault(str(k), v)
            # torch scalar alias
            if hasattr(k, "item") and callable(k.item):
                try:
                    ki = int(k.item())
                    norm.setdefault(ki, v)
                    norm.setdefault(str(ki), v)
                except Exception:
                    pass
        return norm

    # x.squeeze(), mask_dict, self._codebook.embed
    def commitment_loss(
            self,
            encoder_outputs,  # [B, D]  … 現在の“チャンク”だけ
            mask_dict,  # dict[str|int -> 1D indices or bool-mask]（グローバルindex想定）
            codebook,  # dict-like per element or single tensor/param
            logger,
            chunk_start=None,  # このチャンクのグローバル先頭位置（例: self.latent_size_sum）
            beta=0.25,
            temperature=None,  # 未使用ならそのまま（EMAは別処理）
            use_cosine=False,
    ):
        """
        重要: mask_dict の indices は「グローバル index」を想定。
             本関数内でチャンク境界 [chunk_start, chunk_start+B) に入るものだけを残し、
             ローカル index (= global - chunk_start) に変換してから使用する。
        """
        total_cb_count = 0
        encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
        assert encoder_outputs.dim() == 2, f"encoder_outputs must be [B,D], got {tuple(encoder_outputs.shape)}"
        device = encoder_outputs.device
        B, D = encoder_outputs.shape
        print(f"B = {B}, D = {D}")

        # chunk_start が未指定なら 0（= 既にローカル index を渡しているケースに対応）
        if chunk_start is None:
            chunk_start = 0
        chunk_end = chunk_start + B

        mask_dict = self._normalize_mask_dict(mask_dict)

        # 累積器（平均の二重計算を避けるため、Ni 加重和で持つ）
        total_latent = 0
        commit_num = encoder_outputs.new_zeros(())
        codebk_num = encoder_outputs.new_zeros(())
        repel_num = encoder_outputs.new_zeros(())
        cb_repel_num = encoder_outputs.new_zeros(())

        # イテレーション準備（要素別コードブック or 共有コードブック）
        if isinstance(codebook, (dict, nn.ParameterDict)):
            items = list(codebook.items())
        else:
            items = [(None, codebook)]

        # --- ヘルパ: グローバル index → （このチャンク内の）ローカル index へ ---
        def _select_by_global_idx(encoder_outputs, global_idx, start, end, *, name=""):
            """
            encoder_outputs: [B, D]（現チャンク）
            global_idx: 1D LongTensor（グローバル index）
            戻り値: (z_local [Ni, D], idx_local [Ni])
            """
            if global_idx.numel() == 0:
                return encoder_outputs.new_zeros((0, encoder_outputs.size(1))), global_idx

            # チャンクに属する index だけを残す
            mask = (global_idx >= start) & (global_idx < end)
            idx_in = global_idx[mask]
            if idx_in.numel() == 0:
                return encoder_outputs.new_zeros((0, encoder_outputs.size(1))), idx_in

            idx_local = (idx_in - start).to(torch.long)

            # device-side assert を防ぐため手前チェック
            max_idx = int(idx_local.max().item())
            min_idx = int(idx_local.min().item())
            if min_idx < 0 or max_idx >= encoder_outputs.size(0):
                raise RuntimeError(
                    f"[{name}] local index out of range: min={min_idx}, max={max_idx}, "
                    f"B={encoder_outputs.size(0)}, start={start}, end={end}, "
                    f"kept={idx_in.numel()}"
                )
            z = encoder_outputs.index_select(0, idx_local)
            return z, idx_local

        for key, cb in items:
            from utils import CORE_ELEMENTS
            if str(key) not in CORE_ELEMENTS:
                continue
            kstr = "all" if key is None else str(key)

            # --- mask/indices を取得し index tensor に統一（まずはグローバルとして整形） ---
            raw = None if mask_dict is None else mask_dict.get(kstr, None)
            if raw is None:
                # このキーに該当するデータが無い場合はスキップ
                continue

            # bool-mask / list / tensor を 1D LongTensor へ（グローバル想定）
            # 既存のユーティリティを利用。total_length は不要、ここでは型変換のみ。
            idx_global = self._as_index_tensor(raw, None, device)  # 注意: 第二引数は使わない実装にしておく

            # --- グローバル → ローカル（このチャンクに入るものだけ抽出） ---
            z, idx_local = _select_by_global_idx(
                encoder_outputs, idx_global, chunk_start, chunk_end, name=f"key={kstr}"
            )
            Ni = z.size(0)
            if Ni == 0:
                # このチャンクに該当サンプルなし
                continue

            # --- コードブック取り出し＆形状正規化 ---
            cb_t = self._unwrap_codebook_entry(cb)  # nn.Parameter / tensor 両対応で tensor を返す想定
            cb_t = self._squeeze_01(cb_t)  # [1,K,D] → [K,D] にするユーティリティ
            assert cb_t.dim() == 2, f"codebook for key={key} must be [K,D] or [1,K,D]; got {tuple(cb_t.shape)}"
            K, Dk = cb_t.shape
            assert Dk == D, f"latent D={D} != codebook D={Dk} for key={key}"

            # --- 最近傍コード探索（cosine か L2） ---
            if use_cosine:
                z_n = F.normalize(z, p=2, dim=-1)
                cb_n = F.normalize(cb_t, p=2, dim=-1)
                sim = z_n @ cb_n.t()  # [Ni, K]
                embed_ind = torch.argmax(sim, dim=-1)
            else:
                dist = torch.cdist(z, cb_t, p=2).pow(2)  # [Ni, K]
                embed_ind = torch.argmin(dist, dim=-1)

            e = cb_t.index_select(0, embed_ind)  # [Ni, D]

            # ==============================
            # commitment loss 計算　＋重み付け
            # ==============================
            # --- VQ-VAE 損失（EMA 更新は別で）---
            commit_part = F.mse_loss(z, e.detach(), reduction="mean")  # β‖z - sg(e)‖²
            codebk_part = F.mse_loss(e, z.detach(), reduction="mean")  # ‖sg(z) - e‖²

            # Ni で重み付け
            total_latent += Ni
            commit_num = commit_num + commit_part * Ni
            codebk_num = codebk_num + codebk_part * Ni

            # ==============================
            # repel loss 計算　＋重み付け
            # ==============================
            ret = self.compute_contrastive_loss(z, 0, logger, codebook[str(key)], key)
            repel_value = ret[0]
            cb_repel_value = ret[3]
            print(f"{key} : commit {commit_part:.5f}, repel {repel_value:.5f}, cb_repel {cb_repel_value:.5f}")
            repel_num = repel_num + repel_value * Ni
            total_cb_count += K
            cb_repel_num = cb_repel_num + cb_repel_value * K
            # ==============================
            # 記録
            # ==============================
            logger.info(f"{key} : commit {commit_part}, lat_repel {repel_value}, cb_repel {cb_repel_value}")

        if total_latent == 0:
            zero = encoder_outputs.new_zeros(())
            return zero, zero, zero, zero

        # ==============================
        # commitment loss 平均の計算
        # ==============================
        commit_loss = beta * (commit_num / total_latent)
        codebook_loss = (codebk_num / total_latent)
        # ==============================
        # repel loss 平均の計算
        # ==============================
        repel_loss = repel_num / total_latent
        cb_repel_loss = (cb_repel_num / total_cb_count)

        return commit_loss, codebook_loss, repel_loss, cb_repel_loss

    #              data, features, mask_dict, logger, chunk_i, epoch, mode
    def forward(self, x, feature, mask_dict=None, logger=None, chunk_i=None, epoch=0, mode=None):
        only_one = x.ndim == 2
        x = x.to("cuda")
        if only_one:
            x = rearrange(x, 'b d -> b 1 d')
        shape, device, heads, is_multiheaded, codebook_size = (
            x.shape, x.device, self.heads, self.heads > 1, self.codebook_size)
        need_transpose = not self.channel_last and not self.accept_image_fmap
        if self.accept_image_fmap:
            x = rearrange(x, 'b c h w -> b (h w) c')
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        if is_multiheaded:
            x = rearrange(x, 'b n (h d) -> h b n d', h=heads)
        # -------------------------------------------
        # _codebook (run kmeans, sil score, and EMA)
        # -------------------------------------------
        if mode == "init_kmeans_final":
            self._codebook(x, feature, mask_dict, logger, chunk_i, epoch, mode)
            return 0
        else:
            quantize_dict, embed_ind_dict, embed = self._codebook(x, feature, mask_dict, logger, chunk_i, epoch, mode)
        # # -------------------------------
        # # repel loss calculation
        # # -------------------------------
        # ret = self.orthogonal_loss_fn(mask_dict, embed_ind_dict, self._codebook.embed, init_feat, x, quantize_dict, logger, epoch, chunk_i)
        #
        # # Be permissive about key names
        # mid_repel_loss = ret.get("mid_repel_loss") or ret.get("repel_loss") or 0.0
        # cb_repel_loss = ret.get("cb_repel_loss") or 0.0
        # sil = ret.get("sil", [])
        # contrib = ret.get("contrib", None)

        # -------------------------------
        # commit loss calculation
        # -------------------------------
        # encoder_outputs, mask_dict, codebook
        commit_loss, codebook_loss, repel_loss, cb_repel_loss = self.commitment_loss(x, mask_dict, self._codebook.embed, logger, chunk_i)
        # ---------------------------------------------
        # only repel losses at the first several steps
        # ---------------------------------------------
        def _as_scalar_tensor(val, ref_tensor):
            import torch

            def zlike():
                # keeps graph if ref requires grad
                return ref_tensor.sum() * 0.0

            if isinstance(val, torch.Tensor):
                return val if val.ndim == 0 else val.mean()

            if isinstance(val, (list, tuple)):
                if not val:
                    return zlike()
                elems = [_as_scalar_tensor(v, ref_tensor) for v in val]
                return torch.stack(elems).mean()

            if isinstance(val, dict):
                if not val:
                    return zlike()
                elems = [_as_scalar_tensor(v, ref_tensor) for v in val.values()]
                return torch.stack(elems).mean()

            if isinstance(val, (float, int)):
                return torch.tensor(val, device=ref_tensor.device, dtype=ref_tensor.dtype)

            return zlike()

        ref = x if torch.is_tensor(x) else next(self.parameters()).detach().new_tensor(0.)

        repel_loss = _as_scalar_tensor(repel_loss, ref)
        cb_repel_loss = _as_scalar_tensor(cb_repel_loss, ref)
        commit_loss = _as_scalar_tensor(commit_loss, ref)
        codebook_loss = _as_scalar_tensor(codebook_loss, ref)
        # Example: warmup 5 epochs, then gentle exponential decay
        # repel_loss = repel_loss * alpha
        beta_commit = 1  # try 0.25–0.5 if you see codebook collapse
        gamma_cb = 0.05  # codebook (EMA) regularizer
        delta_mid = 0.05  # repel (midpoints)
        delta_cb = 0.0003  # codebook-codebook repel
        # commit_loss 2.4284323444589972e-05  >> 2e-02
        # codebook_loss 9.713729377835989e-05
        # repel_loss 0.32586005330085754
        # cb_repel_loss 0.9953231811523438
        print(f"repel_loss {repel_loss}")
        print(f"cb_repel_loss {cb_repel_loss}")
        print(f"commit_loss {commit_loss}")
        print(f"codebook_loss {codebook_loss}")

        warmup = 5
        alpha = float(torch.exp(torch.tensor(-(epoch - warmup) / 50.0)))
        # if epoch < warmup:
        loss = (delta_mid * repel_loss) + (beta_commit * commit_loss) + (gamma_cb * codebook_loss)
        # else:
        #     # half-life ~ 50 epochs
        #     repel_loss = alpha * repel_loss
        #     loss = (delta_mid * repel_loss) + (delta_cb * cb_repel_loss) \
        #            + (beta_commit * commit_loss) + (gamma_cb * codebook_loss)            #
            # commit_loss 0.0
            # cb_loss 0.0
            # sil_loss []
            # repel_loss 0.24993233382701874
            # cb_repel_loss 0.995575487613678
            # loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss
        return (loss, embed, commit_loss, codebook_loss, [], repel_loss, cb_repel_loss)
