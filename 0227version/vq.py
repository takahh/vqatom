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
        # print("\n[Used codebook size per element]")
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
        N = z.shape[0]

        # ---- 0. Subsample for pdist to avoid O(N^2) explosion ----
        max_pdist_points = 4096  # or 2048 / 8192, as you like

        if N > max_pdist_points:
            # ランダムに max_pdist_points 個だけ選んで repulsive term を計算
            perm = torch.randperm(N, device=device)
            idx = perm[:max_pdist_points]
            z_for_pdist = z[idx]
            # if logger is not None:
            #     logger.info(
            #         f"[VQ_REPEL] z subsampled for pdist: N={N} -> {max_pdist_points}"
            #     )
        else:
            z_for_pdist = z

        # ここから先は z_for_pdist を使う
        pdist_z = torch.pdist(z_for_pdist, p=2)  # [M*(M-1)/2], 1D, M <= max_pdist_points

        z = z.squeeze()
        # ---- 1) 距離の一次統計（1Dで扱う）----
        # z: [B, D]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # print(f"z {z.shape}")
        if z.shape[0] == 1:
            # print(f"latent count is only 1. Not calculating losses.")
            return 0, 0, 0, 0, 0

        # pdist_z = torch.pdist(z, p=2)  # [B*(B-1)/2], 1D

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
                # print(f"{key} {vals}")
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
            import torch
            import torch.utils.checkpoint as cp

            # checkpoint と streaming backward は排他
            assert not (use_checkpoint and stream_backward), \
                "checkpoint と streaming backward は同時に使えません。"

            B, D = z.shape
            device, dtype = z.device, z.dtype

            # -------------------------------
            # 0. INPUT GRAD CHECK
            # -------------------------------
            if self.training:
                assert z.requires_grad, "INPUT ERROR: z.requires_grad=False（上流で detach されている可能性）"

            # -------------------------------
            # 1. パラメータを Tensor 化
            # -------------------------------
            band = float(abs(high - low))
            sigma_val = max(1e-4, 0.20 * band)
            sharp_val = float(sharp) if sharp is not None else 10.0

            low_t = torch.as_tensor(low, device=device, dtype=dtype)
            high_t = torch.as_tensor(high, device=device, dtype=dtype)
            center_t = torch.as_tensor(center, device=device, dtype=dtype)
            sigma_t = torch.as_tensor(sigma_val, device=device, dtype=dtype)

            # -------------------------------
            # ブロック設定
            # -------------------------------
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

            # -------------------------------
            # 2. block_loss (ここで assert 挿入)
            # -------------------------------
            def block_loss(zi, zj, low_t, high_t, center_t, sigma_t):
                import torch

                # 必ず requires_grad をチェック
                if self.training:
                    assert zi.requires_grad, "zi lost grad!"
                    assert zj.requires_grad, "zj lost grad!"

                if zi.numel() == 0 or zj.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

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

                num = (w * bell).sum()
                den = w.sum().clamp_min(eps)
                out = num / den
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                # ここでも require_grad チェック
                if self.training:
                    assert out.requires_grad, "OUTPUT of block_loss lost grad!"

                return out

            # checkpoint wrapper
            def block_loss_ckpt(zi, zj, low_t, high_t, center_t, sigma_t):
                return block_loss(zi, zj, low_t, high_t, center_t, sigma_t)

            # -------------------------------
            # 3. 全ブロック合計
            # -------------------------------
            total = z.new_zeros(())
            i = 0
            while i < B:
                bi = min(row_block, B - i)
                zi = z[i:i + bi]

                j = i
                while j < B:
                    bj = min(col_block, B - j)
                    zj = z[j:j + bj]

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

            # -------------------------------
            # 4. FINAL OUTPUT requires_grad check
            # -------------------------------
            if self.training:
                assert out.requires_grad, "FINAL OUTPUT lost grad somewhere in repel_loss!"

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
        self.samples_latent_in_kmeans = args.samples_latent_in_kmeans
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

    def _get_absK_from_cb_dict(self, key) -> int | None:
        """
        Robust K lookup for cb_dict.
        Accepts key as original (int/str/tuple/etc).
        Returns int(K) or None if not found / invalid.
        """
        d = getattr(self, "cb_dict", None)
        if not isinstance(d, dict):
            return None

        # candidates: raw, str(raw), int(str) if digit
        cands = [key, str(key)]
        try:
            if isinstance(key, str) and key.isdigit():
                cands.append(int(key))
        except Exception:
            pass

        for k in cands:
            if k in d:
                try:
                    v = int(d[k])
                    return v if v > 0 else None
                except Exception:
                    return None
        return None

    import torch
    import torch.nn as nn

    # ---------------------------
    # 1) Safe key: add collision guard + create flag
    # ---------------------------
    def _get_or_create_safe_key(
            self,
            skey: str,
            K_e=None,
            D=None,
            device=None,
            *,
            create: bool = True,
    ) -> str:
        """
        Map original key -> safe_key.
        Optionally create self.embed[safe_key] Parameter when missing.
        Use create=False to NEVER create (so you can skip missing codebooks safely).
        """
        skey = str(skey)

        if skey in self.key_to_safe:
            safe = self.key_to_safe[skey]
        else:
            safe = self._safe_key(skey)

            # ---- collision guard ----
            if safe in self.safe_to_key and self.safe_to_key[safe] != skey:
                raise RuntimeError(
                    f"SAFE KEY COLLISION: safe='{safe}' maps to both "
                    f"'{self.safe_to_key[safe]}' and '{skey}'"
                )

            self.key_to_safe[skey] = safe
            self.safe_to_key[safe] = skey

        # ---- create on demand ----
        if create and (safe not in self.embed) and (K_e is not None) and (D is not None) and (device is not None):
            init = torch.randn(K_e, D, device=device) * 0.01
            self.embed[safe] = nn.Parameter(init, requires_grad=True)

        return safe

    # ---------------------------
    # 2) Better padding: use real data points for extra centers
    # ---------------------------
    def _pad_to_K_sample_points(means_1kd, counts_1k, K_req: int, *, data_nd: torch.Tensor):
        """
        means_1kd: [1, K_run, D]
        counts_1k: [1, K_run]
        data_nd:   [N, D]  (masked latents)  <-- used for padding
        """
        H, K_run, D = means_1kd.shape
        assert H == 1, f"H must be 1, got {H}"

        means_kd = means_1kd[0]  # [K_run, D]
        counts_k = counts_1k[0]  # [K_run]

        if K_req <= K_run:
            return means_kd[:K_req].contiguous(), counts_k[:K_req].contiguous()

        # pad by sampling actual points (much healthier than mu+tiny_noise)
        N = int(data_nd.shape[0])
        K_pad = int(K_req - K_run)

        # sample with replacement
        idx = torch.randint(0, N, (K_pad,), device=data_nd.device)
        pad_centers = data_nd.index_select(0, idx).to(device=means_kd.device, dtype=means_kd.dtype)

        out_means = torch.empty((K_req, D), device=means_kd.device, dtype=means_kd.dtype)
        out_counts = torch.zeros((K_req,), device=counts_k.device, dtype=counts_k.dtype)

        out_means[:K_run].copy_(means_kd)
        out_counts[:K_run].copy_(counts_k)

        out_means[K_run:].copy_(pad_centers)
        # padded counts remain zero
        return out_means, out_counts

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

    @torch.jit.ignore
    def init_embed_(self, data, mask_dict=None, use_cosine_sim: bool = False):
        """
        Initialize per-element codebooks using absolute K from self.cb_dict.
        data: [1, N, D]
        """
        assert mask_dict is not None, "mask_dict is required"
        print("++++++++++++++++ RUNNING init_embed (ABS K, NO FINAL CAP) +++++++++++++")

        device = data.device
        D = data.shape[-1]

        def get_idx(md, k):
            return md.get(k, md.get(str(k), md.get(str(k))))

        def get_absK(d, k):
            v = d.get(k, d.get(str(k), d.get(str(k))))
            if v is None:
                return None
            return int(v)

        def _stats(x_nd):
            if x_nd.numel() == 0:
                return None
            mu = x_nd.mean(dim=0)
            with torch.no_grad():
                dev = (x_nd - mu).pow(2).sum(dim=1).sqrt()
                sigma = torch.quantile(dev, 0.5)
            return mu, sigma

        def _pad_to_K(means_1kd, counts_1k, K_req: int, data_stats=None):
            H, K_run, Dd = means_1kd.shape
            assert H == 1 and Dd == D
            means_kd = means_1kd[0]
            counts_k = counts_1k[0]

            if K_req <= K_run:
                return means_kd[:K_req].contiguous(), counts_k[:K_req].contiguous()

            K_pad = K_req - K_run
            out_means = torch.empty((K_req, D), device=means_kd.device, dtype=means_kd.dtype)
            out_counts = torch.zeros((K_req,), device=counts_k.device, dtype=counts_k.dtype)

            out_means[:K_run].copy_(means_kd)
            out_counts[:K_run].copy_(counts_k)

            if data_stats is not None:
                mu, sigma = data_stats
                noise = torch.randn((K_pad, D), device=means_kd.device, dtype=means_kd.dtype) * (
                    0.01 * sigma + 1e-6
                )
                out_means[K_run:] = mu + noise
            else:
                out_means[K_run:] = 0

            return out_means, out_counts

        for raw_key in sorted(mask_dict.keys(), key=lambda x: str(x)):
            skey = str(raw_key)
            idx = get_idx(mask_dict, skey)
            if idx is None:
                continue
            masked = data[0][idx]  # (N_i, D)
            N_i = masked.shape[0]
            if N_i == 0:
                continue

            K_req = get_absK(self.cb_dict, skey)
            if K_req is None or K_req <= 0:
                K_req = 1

            K_run = min(K_req, N_i)

            means_1kd, counts_1k, used_per_head, used_per_label = kmeans(
                masked.unsqueeze(0).to(device),
                num_clusters=K_run,
                use_cosine_sim=use_cosine_sim,
            )

            means_kd, counts_k = _pad_to_K(means_1kd, counts_1k, K_req, data_stats=_stats(masked))

            # embed
            safe = self._get_or_create_safe_key(skey, K_req, D, device=device)
            if self.embed[safe].shape != (K_req, D):
                self.embed[safe] = nn.Parameter(
                    means_kd.detach().to(device=device, dtype=means_kd.dtype),
                    requires_grad=True,
                )
            else:
                self.embed[safe].data.copy_(means_kd)

            # cluster_size / embed_avg を必ず float32 で再構築（バッファ名は元キー）
            buf_name_cs = f"cluster_size_{skey}"
            buf_name_ea = f"embed_avg_{skey}"

            K_req = int(get_absK(self.cb_dict, skey))
            N_i = int(N_i)
            K_run = min(K_req, N_i)

            # ここで「counts_k と means_kd が K_req になってる」ことを保証
            assert counts_k.shape[0] == K_req, (skey, counts_k.shape, K_req)
            assert means_kd.shape[0] == K_req, (skey, means_kd.shape, K_req)

            cs = counts_k.to(device=device, dtype=torch.float32)
            ea = means_kd.detach().to(device=device, dtype=torch.float32) * cs.view(-1, 1)
            if hasattr(self, buf_name_cs):
                delattr(self, buf_name_cs)
            if hasattr(self, buf_name_ea):
                delattr(self, buf_name_ea)

            self.register_buffer(buf_name_cs, cs)
            self.register_buffer(buf_name_ea, ea)

            nz = int((counts_k > 0).sum().item())
            print(f"[init_embed_] Z={skey} N={N_i} K_req={K_req} K_run={K_run} K_used={nz}/{K_req}")

    # ------------------------------------------------------------------
    # 補助関数群（normalize mask, index 変換など）
    # ------------------------------------------------------------------
    def _normalize_mask_dict(self, mask_dict, logger, device=None):
        import numpy as np

        logger.info(f"fnorm 0")
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

    # ------------------------------------------------------------------
    # Forward: ここで EMA update を dtype 安全 & safe-key 化
    # ------------------------------------------------------------------
    def _get_absK_from_cb_dict(self, key_any, default=1) -> int:
        """
        Lookup absolute K for a given key from self.cb_dict.
        Accepts str/int keys, returns int >= 1.
        """
        cb = getattr(self, "cb_dict", None)
        if cb is None:
            return int(default)

        # normalize: try exact, str, int (if possible)
        candidates = []
        candidates.append(key_any)
        candidates.append(str(key_any))

        # if key looks like int, also try int form
        try:
            candidates.append(int(key_any))
        except Exception:
            pass

        for k in candidates:
            if k in cb:
                v = cb[k]
                if v is None:
                    break
                try:
                    v = int(v)
                except Exception:
                    v = int(float(v))
                return max(1, v)

        return int(default)

    def _get_code_for_key_no_create(self, key):
        """
        Return (code_tensor_or_None, safe_bool)

        - safe=True  : key exists and returns a Tensor/Parameter [K, D]
        - safe=False : key not found (or codebook not initialized)
        - NEVER creates a new codebook entry.
        """
        import torch
        import torch.nn as nn

        cb_src = getattr(self, "_codebook", None)
        if cb_src is None:
            return None, False

        # normalize key candidates
        key_s = str(key)
        candidates = [key, key_s]

        t = None

        # ---- dict / ModuleDict / ParameterDict ----
        if isinstance(cb_src, (dict, nn.ModuleDict, nn.ParameterDict)):
            for k in candidates:
                if k in cb_src:
                    t = cb_src[k]
                    break

            if t is None:
                # try relaxed match: str(k) equality
                if isinstance(cb_src, dict):
                    for k2, v2 in cb_src.items():
                        if str(k2) == key_s:
                            t = v2
                            break

            if t is None:
                return None, False

            # unwrap Embedding/Parameter
            if isinstance(t, nn.Embedding):
                t = t.weight
            elif isinstance(t, nn.Parameter):
                t = t

            if not torch.is_tensor(t):
                return None, False

            return t, True

        # ---- single Tensor / Parameter / Embedding ----
        if isinstance(cb_src, nn.Embedding):
            return cb_src.weight, True
        if isinstance(cb_src, nn.Parameter):
            return cb_src, True
        if torch.is_tensor(cb_src):
            return cb_src, True

        return None, False

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, feature=None, mask_dict=None, logger=None, chunk_i=None, epoch=None, mode=None):
        import os, time, torch

        key_list_to_dump = {
            "6_0_3_1_1_2_6_2_1_1_11_0",
            "7_0_3_0_0_2_0_1_0_0_3_0",
            "16_-1_4_0_0_1_0_0_0_0_16_0",
            "16_1_4_0_1_3_6_0_5_0_0_0",
        }

        # --------------------------------------------------------------
        # 0) Global latent offset bookkeeping (global id -> local chunk)
        # --------------------------------------------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0
        if chunk_i is not None and chunk_i == 0:
            self.latent_size_sum = 0

        # --------------------------------------------------------------
        # 1) Input reshape (expect [1,B,D])
        # --------------------------------------------------------------
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1,B,D]
        elif x.ndim >= 4:
            x = x.view(x.shape[0], -1, x.shape[-1])

        if x.ndim != 3:
            raise RuntimeError(f"x must end up 3D [1,B,D], got shape={tuple(x.shape)}")

        flatten = x  # [1,B,D]
        B, D = int(flatten.shape[1]), int(flatten.shape[2])

        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        self.quantize_dict = {}
        self.embed_ind_dict = {}

        # normalize mask_dict to global LongTensor indices
        mask_dict = self._normalize_mask_dict(mask_dict, logger, device=flatten.device) if mask_dict is not None else None
        if logger:
            print(f"[CODEBOOK] mode={mode}")

        def _lookup_K(key_any):
            import time
            t0 = time.time()
            logger.info(f"[HB] lookup_K enter key={key_any!r}")

            logger.info("[HB] before get_absK(key_any)")
            K_e = self._get_absK_from_cb_dict(key_any)
            logger.info(f"[HB] after get_absK(key_any) -> {K_e}")

            if K_e is None:
                logger.info("[HB] before get_absK(str)")
                K_e = self._get_absK_from_cb_dict(str(key_any))
                logger.info(f"[HB] after get_absK(str) -> {K_e}")

            if K_e is None:
                K_e = int(self.codebook_size)
                if logger:
                    logger.warning(f"[K_FALLBACK] key={str(key_any)} -> K_e={K_e} (not found in cb_dict)")

            K_e = int(K_e)
            logger.info(f"[HB] lookup_K exit K_e={K_e} dt={time.time() - t0:.3f}s")
            return K_e

        # ==============================================================
        # 2) init_kmeans_final phase
        #   IMPORTANT: do NOT create missing codebooks here
        # ==============================================================
        if mode == "init_kmeans_final":
            print(f"[HB] mode check passed? mode={mode!r}", flush=True)
            if not hasattr(self, "_kmeans_dump") or (chunk_i is not None and chunk_i == 0):
                self._kmeans_dump = {}

            if mask_dict is not None:
                for key, idx_global in mask_dict.items():
                    if idx_global is None or idx_global.numel() == 0:
                        continue

                    gmask = (idx_global >= global_start) & (idx_global < global_end)
                    if not gmask.any():
                        continue

                    idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                    if idx_local.numel() == 0:
                        continue

                    masked_latents = flatten[0].index_select(0, idx_local)  # [Ni, D]
                    if masked_latents.numel() == 0:
                        continue

                    skey = str(key)
                    K_e = _lookup_K(key)

                    # ---- get code WITHOUT creating (prevents random new codebooks) ----
                    code, safe = self._get_code_for_key_no_create(skey)
                    if code is None:
                        if logger:
                            logger.warning(f"[NO_CODEBOOK] key={skey} missing embed; skip assign/SS")
                        continue

                    # optional: codebook health stats
                    if logger is not None:
                        with torch.no_grad():
                            c = code.detach()
                            logger.info(
                                f"[CB_STATS] key={skey} K={c.shape[0]} D={c.shape[1]} "
                                f"mean={c.mean().item():.6f} std={c.std().item():.6f} "
                                f"row_std_mean={c.std(dim=1).mean().item():.6f}"
                            )

                    with torch.no_grad():
                        dist = torch.cdist(masked_latents, code, p=2).pow(2)  # [Ni, K]
                        idx_code = dist.argmin(dim=-1)  # [Ni]
                    quantize = code.index_select(0, idx_code)  # [Ni, D]

                    # store cpu for debug/dump (OK in init mode)
                    self.quantize_dict[skey] = quantize.detach().to("cpu", dtype=torch.float16)
                    self.embed_ind_dict[skey] = idx_code.detach().to("cpu", dtype=torch.int32)

                    # ---- Silhouette (use masked_latents / idx_code) ----
                    if logger is not None:
                        try:
                            X = masked_latents.detach().float().cpu()
                            labels = idx_code.detach().cpu().long()
                            u, cts = labels.unique(return_counts=True)

                            logger.info(
                                f"[SS_IN] key={skey} N={X.shape[0]} D={X.shape[1]} "
                                f"labels_unique={u.numel()} "
                                f"mincnt={int(cts.min()) if cts.numel() else -1} "
                                f"maxcnt={int(cts.max()) if cts.numel() else -1} "
                                f"x_mean={float(X.mean())} x_std={float(X.std())} "
                                f"nan={bool(torch.isnan(X).any())} inf={bool(torch.isinf(X).any())}"
                            )

                            if u.numel() >= 2 and X.shape[0] >= 3 and int(cts.min()) >= 2:
                                nmax = int(getattr(self, "samples_latent_in_kmeans", 0) or 0)
                                n = min(nmax, X.shape[0]) if nmax > 0 else min(4096, X.shape[0])

                                perm = torch.randperm(X.shape[0])[:n]
                                xs = X.index_select(0, perm)
                                ys = labels.index_select(0, perm)

                                sil = self.silhouette_score_torch(
                                    xs, ys,
                                    device="cuda" if torch.cuda.is_available() else "cpu"
                                )
                                logger.info(f"SS: {skey} {sil:.4f}, size {X.shape[0]}, K_e {K_e}")
                            else:
                                logger.info(
                                    f"[SS_SKIP] key={skey} N={X.shape[0]} uniq={u.numel()} "
                                    f"mincnt={int(cts.min()) if cts.numel() else -1} K_e={K_e}"
                                )
                        except Exception as e:
                            logger.warning(f"[SS_FAIL] key={skey} err={repr(e)}")

                    # ---- dump preparation ----
                    do_dump = (epoch is not None) and (epoch % 10 == 0 or epoch == 1)
                    if do_dump and (skey in key_list_to_dump):
                        lat_cpu = masked_latents.detach().to("cpu", dtype=torch.float16)
                        ctr_cpu = code.detach().to("cpu", dtype=torch.float16)
                        asg_cpu = idx_code.detach().to("cpu")

                        entry = self._kmeans_dump.get(skey)
                        if entry is None:
                            entry = {"latents": [], "centers": None, "assign": []}
                            self._kmeans_dump[skey] = entry

                        entry["latents"].append(lat_cpu)
                        entry["assign"].append(asg_cpu)
                        if entry["centers"] is None:
                            entry["centers"] = ctr_cpu

                    del masked_latents, code, idx_code, quantize

            self.latent_size_sum = global_end

            do_dump = (epoch is not None) and (epoch % 10 == 0 or epoch == 1)
            if do_dump:
                out = {}
                for k, v in self._kmeans_dump.items():
                    if k not in key_list_to_dump:
                        continue
                    out[k] = {
                        "latents": torch.cat(v["latents"], dim=0) if len(v["latents"]) else None,
                        "centers": v["centers"],
                        "assign": torch.cat(v["assign"], dim=0) if len(v["assign"]) else None,
                    }

                stamp = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs("dumps", exist_ok=True)
                path = os.path.join("dumps", f"init_kmeans_final_ep{epoch}_chunk{chunk_i}_{stamp}.pt")
                torch.save(out, path)
                self._kmeans_dump = {}

            torch.cuda.empty_cache()
            return 0

        # ==============================================================
        # 3) train / test / eval phase
        # ==============================================================
        logger.info(f"stop -1")
        if feature is None:
            raise RuntimeError("feature is required in train/test mode")
        logger.info(f"stop 0")
        feat_flat = feature if torch.is_tensor(feature) else torch.cat(feature, dim=0)
        feat_flat = feat_flat.contiguous().to(flatten.device)

        assert feat_flat.ndim == 2 and feat_flat.size(1) == 78, f"feat_flat shape={tuple(feat_flat.shape)}"
        assert feat_flat.size(0) == B, f"feat_flat N={feat_flat.size(0)} != B={B}"

        logger.info(f"stop 1")
        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue

                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not gmask.any():
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                if idx_local.numel() == 0:
                    continue

                masked_latents = flatten[0].index_select(0, idx_local)  # [Ni, D]
                if masked_latents.numel() == 0:
                    continue

                skey = str(key)
                K_e = _lookup_K(key)

                # create is OK in train/eval (but ideally everything exists already)
                safe = self._get_or_create_safe_key(skey, K_e=K_e, D=D, device=flatten.device, create=True)
                code_param = self.embed[safe]
                code = code_param.squeeze(0) if code_param.ndim == 3 else code_param

                with torch.no_grad():
                    dist = torch.cdist(masked_latents, code, p=2).pow(2)
                    idx_code = dist.argmin(dim=-1)
                quantize = code.index_select(0, idx_code)

                self.quantize_dict[skey] = quantize
                self.embed_ind_dict[skey] = idx_code.to(torch.int32)

                # ----- EMA update (train and early epochs only) -----
                if self.training and epoch is not None and epoch < 30:
                    if code_param.ndim == 3:
                        K_e, D_e = code_param.shape[1], code_param.shape[2]
                    else:
                        K_e, D_e = code_param.shape
                    device = code_param.device
                    self.cb_dict[skey] = int(K_e)

                    buf_name_cs = f"cluster_size_{skey}"
                    buf_name_ea = f"embed_avg_{skey}"

                    # cluster_size buffer
                    if hasattr(self, buf_name_cs):
                        cs = getattr(self, buf_name_cs)
                        if cs.dtype != torch.float32 or cs.shape[0] != K_e:
                            cs = cs.float()
                            if cs.shape[0] != K_e:
                                cs = torch.zeros(K_e, device=device, dtype=torch.float32)
                            setattr(self, buf_name_cs, cs)
                    else:
                        cs = torch.zeros(K_e, device=device, dtype=torch.float32)
                        self.register_buffer(buf_name_cs, cs)

                    # embed_avg buffer
                    if hasattr(self, buf_name_ea):
                        ea = getattr(self, buf_name_ea)
                        if ea.dtype != torch.float32 or ea.shape != (K_e, D_e):
                            ea = ea.float()
                            if ea.shape != (K_e, D_e):
                                ea = torch.zeros(K_e, D_e, device=device, dtype=torch.float32)
                            setattr(self, buf_name_ea, ea)
                    else:
                        ea = torch.zeros(K_e, D_e, device=device, dtype=torch.float32)
                        self.register_buffer(buf_name_ea, ea)

                    with torch.no_grad():
                        idx_code_long = idx_code.to(device=device, dtype=torch.long)

                        batch_counts = torch.zeros_like(cs, dtype=torch.float32)
                        batch_counts.index_add_(0, idx_code_long, torch.ones_like(idx_code_long, dtype=torch.float32))

                        batch_embed_sum = torch.zeros_like(ea, dtype=torch.float32)
                        batch_embed_sum.index_add_(0, idx_code_long,
                                                   masked_latents.to(device=device, dtype=torch.float32))

                        decay = self.decay
                        one_m = 1.0 - decay

                        cs.mul_(decay).add_(batch_counts * one_m)
                        ea.mul_(decay).add_(batch_embed_sum * one_m)

                        means = ea / (cs.unsqueeze(-1) + self.eps)

                        if code_param.ndim == 3:
                            code_param.data.copy_(means.unsqueeze(0))
                        else:
                            code_param.data.copy_(means)

                del masked_latents, code, idx_code, quantize

        # ---- SAFE: initialize with original (prevents uninitialized holes) ----
        quantize_full = flatten[0].clone()  # [B, D]

        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                skey = str(key)
                if skey not in self.quantize_dict:
                    continue

                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not gmask.any():
                    continue

                idx_in_chunk = idx_global[gmask]
                idx_local = (idx_in_chunk - global_start).to(device=quantize_full.device, dtype=torch.long)

                qk = self.quantize_dict[skey].to(device=quantize_full.device, dtype=quantize_full.dtype)
                if idx_local.numel() > 0:
                    quantize_full.index_copy_(0, idx_local, qk)

        quantize_st = flatten[0] + (quantize_full - flatten[0]).detach()
        quantize_st = quantize_st.unsqueeze(0)

        self.latent_size_sum = global_end
        torch.cuda.empty_cache()

        logger.info(f"stop 2")
        return quantize_st, self.embed_ind_dict, self.embed

import os
import time
import base64
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack, repeat

# Assumed existing in your codebase:
# - default
# - get_args
# - EuclideanCodebook (or CosineSimCodebook)
# - ContrastiveLoss
# - silhouette_score_torch (as self.silhouette_score_torch, already implemented elsewhere)


class VectorQuantize(nn.Module):
    """
    Per-key (feature-conditioned) codebooks stored in nn.ParameterDict.

    Key points:
      - Uses collision-free safe keys via base64 urlsafe encoding (keeps '-1' etc. distinct).
      - init_kmeans_final mode never creates missing codebooks.
      - train/eval mode can create missing codebooks (configurable).
      - Keeps global index bookkeeping across chunks (latent_size_sum).
    """

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
        commitment_weight=1,
        codebook_weight=1,
        lamb_sil=0.00001,
        lamb_cb=0.01,
        lamb_div=0.01,
        lamb_equiv_atom=1,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        sample_codebook_temp=0.0,
        sync_codebook=False,
        use_cosine=False,
        target_radius=3.0,
        radius_weight=1e-3,
        tau_init=0.5,
        tau_min=0.05,
        tau_max=2.0,
        tau_ema=0.9,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.separate_codebook_per_head = bool(separate_codebook_per_head)

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = int(codebook_dim) * int(heads)
        requires_projection = (codebook_input_dim != int(dim))
        # self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = float(eps)
        self.commitment_weight = float(commitment_weight)
        self.codebook_weight = float(codebook_weight)

        has_codebook_orthogonal_loss = (margin_weight > 0)

        self.margin_weight = float(margin_weight)
        self.spread_weight = float(spread_weight)
        self.lamb_div_ele = float(lamb_div_ele)
        self.lamb_div_bonds = float(lamb_div_bonds)
        self.lamb_div_aroma = float(lamb_div_aroma)
        self.lamb_div_ringy = float(lamb_div_ringy)
        self.lamb_div_h_num = float(lamb_div_h_num)
        self.lamb_div_equidist = float(lamb_div_equidist)
        self.lamb_div_elec_state = float(lamb_div_elec_state)
        self.lamb_div_charge = float(lamb_div_charge)
        self.lamb_equiv_atom = float(lamb_equiv_atom)
        self.lamb_sil = float(lamb_sil)
        self.lamb_cb = float(lamb_cb)
        self.lamb_div = float(lamb_div)
        self.pair_weight = float(pair_weight)
        self.orthogonal_reg_active_codes_only = bool(orthogonal_reg_active_codes_only)
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_max_norm = None  # e.g., 4.0
        codebook_class = EuclideanCodebook  # if use_cosine_sim else EuclideanCodebook
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
        self.epoch_at_mode_shift = getattr(args, "epoch_at_mode_shift", None)
        self.samples_latent_in_kmeans = getattr(args, "samples_latent_in_kmeans", 0)

        self.codebook_size = int(codebook_size)
        self.accept_image_fmap = bool(accept_image_fmap)
        self.channel_last = bool(channel_last)

        self.compute_contrastive_loss = ContrastiveLoss(dim, 136)

        self.use_cosine = bool(use_cosine)
        self.target_radius = float(target_radius)
        self.radius_weight = float(radius_weight)

        self.register_buffer("_tau", torch.tensor(float(tau_init)))
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.tau_ema = float(tau_ema)
        self.codebook_max_norm = codebook_max_norm

        # Per-key codebooks
        self.embed = nn.ParameterDict()  # safe_key -> [K, D] Parameter
        self.key_to_safe: Dict[str, str] = {}
        self.safe_to_key: Dict[str, str] = {}

        # Global latent offset bookkeeping (updated in forward)
        self.latent_size_sum = 0

        # cb_dict comes from your global CBDICT usage (as in your earlier code)
        self.cb_dict = getattr(self._codebook, "cb_dict", None) or getattr(args, "cb_dict", None) or globals().get("CBDICT", None)

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook
        return rearrange(codebook, "1 ... -> ...")

    # -------------------------------------------------------------------------
    # Safe key mapping (collision-free, preserves '-1' etc.)
    # -------------------------------------------------------------------------
    def _safe_key(self, skey: str) -> str:
        raw = str(skey).encode("utf-8")
        b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return "k_" + b64  # ParameterDict-safe and reversible

    def _unsafe_key(self, safe: str) -> str:
        assert safe.startswith("k_")
        b = safe[2:]
        pad = "=" * ((4 - (len(b) % 4)) % 4)
        return base64.urlsafe_b64decode(b + pad).decode("utf-8")

    # -------------------------------------------------------------------------
    # Utility: normalize various codebook containers -> [K,D] Tensor
    # -------------------------------------------------------------------------
    def _cb_to_tensor(self, cb: Any) -> torch.Tensor:
        if isinstance(cb, torch.Tensor):
            return cb
        if isinstance(cb, nn.Parameter):
            return cb

        if isinstance(cb, nn.Module):
            if hasattr(cb, "weight") and isinstance(cb.weight, torch.Tensor):
                return cb.weight
            if hasattr(cb, "embed"):
                e = cb.embed
                if isinstance(e, nn.Embedding):
                    return e.weight
                if isinstance(e, (torch.Tensor, nn.Parameter)):
                    return e
            for attr in ("codebook", "codes", "embedding"):
                if hasattr(cb, attr):
                    t = getattr(cb, attr)
                    if isinstance(t, (torch.Tensor, nn.Parameter)):
                        return t
            for _, p in cb.named_parameters(recurse=True):
                if isinstance(p, torch.Tensor) and p.ndim == 2:
                    return p
            for _, b in cb.named_buffers(recurse=True):
                if isinstance(b, torch.Tensor) and b.ndim == 2:
                    return b
            raise TypeError(f"_cb_to_tensor: unsupported nn.Module type {type(cb)} (no 2D param/buffer)")

        if isinstance(cb, nn.ParameterDict):
            tensors = []
            for k in sorted(cb.keys()):
                v = cb[k]
                if not isinstance(v, (torch.Tensor, nn.Parameter)):
                    raise TypeError(f"_cb_to_tensor: unsupported leaf type {type(v)} in ParameterDict for key {k!r}")
                tensors.append(v)
            if len(tensors) == 0:
                raise ValueError("_cb_to_tensor: empty ParameterDict")
            return torch.cat(tensors, dim=0)

        if isinstance(cb, dict):
            tensors = []
            for k in sorted(cb.keys()):
                tensors.append(self._cb_to_tensor(cb[k]))
            if len(tensors) == 0:
                raise ValueError("_cb_to_tensor: empty dict")
            return torch.cat(tensors, dim=0)

        raise TypeError(f"_cb_to_tensor: unsupported type {type(cb)}")

    # -------------------------------------------------------------------------
    # Indices -> codes helper (unchanged logic)
    # -------------------------------------------------------------------------
    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        codebook = self.codebook
        is_multiheaded = (codebook.ndim > 2)

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack([indices], "b * h")
        indices = rearrange(indices, "b n h -> b h n")
        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        (codes,) = unpack(codes, ps, "b * d")
        return codes

    # -------------------------------------------------------------------------
    # Misc losses/utilities (kept as-is, cleaned imports)
    # -------------------------------------------------------------------------
    def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, temperature=1.0, margin=0.1):
        device = embeddings.device
        batch_size = embeddings.size(0)

        if embed_ind.dim() == 1:
            embed_ind = embed_ind.unsqueeze(1)  # (N,1)

        logits = embed_ind.expand(-1, num_clusters)
        cluster_assignments = F.softmax(logits / temperature, dim=-1)

        hard_assignments = torch.zeros_like(cluster_assignments).scatter_(
            1, cluster_assignments.argmax(dim=1, keepdim=True), 1.0
        )

        cluster_sums = hard_assignments.T @ embeddings  # (K,D)
        cluster_sizes = hard_assignments.sum(dim=0, keepdim=True).T.clamp(min=1.0)  # (K,1)
        centroids = cluster_sums / cluster_sizes

        assigned_clusters = cluster_assignments.argmax(dim=1)
        assigned_centroids = centroids[assigned_clusters]
        a = torch.norm(embeddings - assigned_centroids, dim=1)

        mask = torch.ones((batch_size, num_clusters), device=device)
        mask.scatter_(1, assigned_clusters.unsqueeze(1), 0)

        all_distances = torch.cdist(embeddings, centroids)
        masked_distances = all_distances * mask + (1 - mask) * 1e6
        b, _ = torch.min(masked_distances, dim=1)

        max_dist = torch.max(a, b) + 1e-6
        silhouette = (b - a) / max_dist - margin
        loss = 1 - torch.mean(torch.tanh(silhouette))
        return loss

    def fast_find_equivalence_groups(self, latents):
        from collections import defaultdict
        hash_map = defaultdict(list)
        for idx, vector in enumerate(latents):
            key = tuple(vector.tolist())
            hash_map[key].append(idx)
        return [group for group in hash_map.values() if len(group) > 1]

    def orthogonal_loss_fn(self, embed_ind_dict, codebook, init_feat, latents, quantized_dict, logger, epoch, chunk=0):
        codebook = codebook.to("cuda")
        init_feat = init_feat.to("cuda")
        latents = latents.to("cuda")
        quantized_dict = quantized_dict.to("cuda")

        latent_len_sum = 0
        two_repel_loss_weighted_sum = 0
        cb_loss_weighted_sum = 0

        for key in embed_ind_dict.keys():
            latents_for_sil = torch.squeeze(latents)
            latents_size = latents_for_sil.shape[0]
            latent_len_sum += latents_size

            two_repel_loss, div_nega_loss, repel_loss_from_2, cb_loss, repel_loss_mid_high = (
                self.compute_contrastive_loss(latents_for_sil, chunk, logger, codebook[str(key)])
            )

            two_repel_loss_weighted_sum += latents_size * two_repel_loss
            cb_loss_weighted_sum += latents_size * cb_loss

        two_repel_loss_avg = two_repel_loss_weighted_sum / max(1, latent_len_sum)
        cb_loss_avg = cb_loss_weighted_sum / max(1, latent_len_sum)
        return (two_repel_loss_avg, cb_loss_avg)

    @staticmethod
    def pairwise_sq_dists(x, y):
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1).unsqueeze(0)
        xy = x @ y.t()
        d2 = x2 + y2 - 2 * xy
        return torch.clamp(d2, min=0.0)

    def _latent_radius_loss(self, z):
        norms = torch.linalg.norm(z, dim=-1)
        excess = F.relu(norms - self.target_radius)
        return (excess * excess).mean()

    def _center_batch(self, z):
        return z - z.mean(dim=0, keepdim=True)

    def _as_index_tensor(self, idx_raw, N, device):
        if idx_raw is None:
            return torch.arange(N, device=device)
        if torch.is_tensor(idx_raw):
            if idx_raw.dtype == torch.bool:
                return torch.nonzero(idx_raw, as_tuple=False).squeeze(-1).to(device)
            return idx_raw.to(device).long().view(-1)
        return torch.as_tensor(idx_raw, device=device).long().view(-1)

    def _unwrap_codebook_entry(self, cb):
        if isinstance(cb, (torch.Tensor, nn.Parameter)):
            return cb
        if isinstance(cb, (dict, nn.ParameterDict)):
            if "embed" in cb:
                return self._unwrap_codebook_entry(cb["embed"])
        raise TypeError(
            f"Unsupported codebook entry type: {type(cb)}. "
            "Expected Tensor/Parameter or dict/ParameterDict with 'embed'."
        )

    def _squeeze_01(self, x):
        return x.squeeze(0) if x.dim() == 3 and x.size(0) == 1 else x

    def _normalize_mask_dict(self, mask_dict):
        if mask_dict is None:
            return None
        norm = dict(mask_dict)
        for k, v in list(mask_dict.items()):
            if isinstance(k, str) and k.isdigit():
                norm.setdefault(int(k), v)
            if isinstance(k, int):
                norm.setdefault(str(k), v)
            if hasattr(k, "item") and callable(k.item):
                try:
                    ki = int(k.item())
                    norm.setdefault(ki, v)
                    norm.setdefault(str(ki), v)
                except Exception:
                    pass
        return norm

    # -------------------------------------------------------------------------
    # K lookup from CBDICT
    # -------------------------------------------------------------------------
    def _get_absK_from_cb_dict(self, key_any, default=1) -> int:
        cb = getattr(self, "cb_dict", None)
        if cb is None:
            return int(default)

        candidates = [key_any, str(key_any)]
        try:
            candidates.append(int(key_any))
        except Exception:
            pass

        for k in candidates:
            if k in cb:
                v = cb[k]
                if v is None:
                    break
                try:
                    v = int(v)
                except Exception:
                    v = int(float(v))
                return max(1, v)

        return int(default)

    # -------------------------------------------------------------------------
    # Safe-key mapping + creation
    # -------------------------------------------------------------------------
    def _get_or_create_safe_key(self, skey: str, K_e=None, D=None, device=None, *, create: bool = True) -> str:
        skey = str(skey)

        if skey in self.key_to_safe:
            safe = self.key_to_safe[skey]
        else:
            safe = self._safe_key(skey)
            if safe in self.safe_to_key and self.safe_to_key[safe] != skey:
                # With base64 encoding this should not happen unless corrupted state
                raise RuntimeError(f"SAFE KEY COLLISION (unexpected): safe={safe} {self.safe_to_key[safe]} != {skey}")
            self.key_to_safe[skey] = safe
            self.safe_to_key[safe] = skey

        if create and (safe not in self.embed):
            if (K_e is None) or (D is None) or (device is None):
                raise RuntimeError(f"Need K_e/D/device to create codebook for {skey}")
            init = torch.randn(int(K_e), int(D), device=device) * 0.01
            self.embed[safe] = nn.Parameter(init, requires_grad=True)

        return safe

    def _get_code_for_key_no_create(self, key) -> Tuple[Optional[torch.Tensor], bool]:
        skey = str(key)

        safe = self.key_to_safe.get(skey, None)
        if safe is not None and safe in self.embed:
            return self.embed[safe], True

        safe2 = self._safe_key(skey)
        if safe2 in self.embed:
            self.key_to_safe[skey] = safe2
            self.safe_to_key.setdefault(safe2, skey)
            return self.embed[safe2], True

        return None, False

    # -------------------------------------------------------------------------
    # Commitment loss (your weighted per-key loop, cleaned + base64 safe keys)
    # -------------------------------------------------------------------------
    def commitment_loss(
        self,
        encoder_outputs,   # [B,D]
        mask_dict,
        codebook,
        logger=None,
        chunk_start=None,
        beta=0.25,
        temperature=None,  # unused
        use_cosine=False,
    ):
        encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
        device = encoder_outputs.device
        B, D = encoder_outputs.shape
        dtype = encoder_outputs.dtype

        if chunk_start is None:
            chunk_start = 0
        chunk_end = int(chunk_start) + int(B)

        if mask_dict is None or len(mask_dict) == 0:
            zero = encoder_outputs.sum() * 0.0
            if logger is not None:
                logger.info(f"[VQ_COMMIT] mask_dict is empty -> skip in chunk [{chunk_start},{chunk_end})")
            return zero, zero, zero, zero

        total_latent = 0
        total_cb_count = 0

        commit_num = encoder_outputs.new_zeros(())
        codebk_num = encoder_outputs.new_zeros(())
        repel_num = encoder_outputs.new_zeros(())
        cb_repel_num = encoder_outputs.new_zeros(())

        def _get_cb_param_for_key(k_any):
            safe = self._get_or_create_safe_key(str(k_any), create=False)
            if safe in self.embed:
                return self.embed[safe]
            return None

        for k, g_idx in mask_dict.items():
            if isinstance(g_idx, (list, tuple)):
                if len(g_idx) == 0:
                    continue
                g_idx = torch.as_tensor(g_idx, device=device)
            elif not torch.is_tensor(g_idx):
                try:
                    g_idx = torch.as_tensor(g_idx, device=device)
                except Exception as e:
                    raise TypeError(f"mask_dict[{k}] has unsupported type {type(g_idx)}") from e
            else:
                g_idx = g_idx.to(device)

            if g_idx.numel() == 0:
                continue

            if g_idx.dtype == torch.bool:
                g_idx = torch.nonzero(g_idx, as_tuple=False).view(-1)

            g_idx = g_idx.long()
            if g_idx.numel() == 0:
                continue

            in_chunk = (g_idx >= chunk_start) & (g_idx < chunk_end)
            if not torch.any(in_chunk):
                continue

            local_idx = (g_idx[in_chunk] - chunk_start).long()
            if local_idx.numel() == 0:
                continue

            z = encoder_outputs[local_idx]
            N_i = int(z.shape[0])
            if N_i == 0:
                continue

            cb_param = _get_cb_param_for_key(k)
            if cb_param is None:
                continue

            cb = cb_param
            K_e, Dk = cb.shape
            if int(Dk) != int(D):
                raise RuntimeError(f"latent D={D} != codebook D={Dk} for key={k}")

            if use_cosine:
                z_norm = F.normalize(z, dim=-1)
                cb_norm = F.normalize(cb, dim=-1)
                sim = torch.matmul(z_norm, cb_norm.t())
                nn_idx = torch.argmax(sim, dim=-1)
            else:
                z2 = (z ** 2).sum(dim=-1, keepdim=True)
                e2 = (cb ** 2).sum(dim=-1).unsqueeze(0)
                dist2 = z2 + e2 - 2.0 * torch.matmul(z, cb.t())
                nn_idx = torch.argmin(dist2, dim=-1)

            e_star = cb[nn_idx]

            commit_part = F.mse_loss(z, e_star.detach(), reduction="mean")
            codebk_part = F.mse_loss(e_star, z.detach(), reduction="mean")

            total_latent += N_i
            commit_num = commit_num + commit_part * N_i
            codebk_num = codebk_num + codebk_part * N_i

            if isinstance(codebook, (dict, nn.ParameterDict)) and (k in codebook):
                cb_for_contrast = codebook[k]
            else:
                cb_for_contrast = cb

            ret = self.compute_contrastive_loss(z, 0, logger, cb_for_contrast, k)
            repel_val = ret[0]
            cb_repel_val = ret[3]

            repel_num = repel_num + repel_val * N_i
            total_cb_count += int(K_e)
            cb_repel_num = cb_repel_num + cb_repel_val * int(K_e)

        if total_latent == 0:
            zero = encoder_outputs.sum() * 0.0
            if logger is not None:
                logger.info(f"[VQ_EMPTY] total_latent=0 in chunk [{chunk_start},{chunk_end})")
            return zero, zero, zero, zero

        commit_loss = float(beta) * (commit_num / float(total_latent))
        codebook_loss = codebk_num / float(total_latent)
        repel_loss = repel_num / float(total_latent)

        if total_cb_count > 0:
            cb_repel_loss = cb_repel_num / float(total_cb_count)
        else:
            cb_repel_loss = cb_repel_num * 0.0

        if logger is not None:
            logger.info(
                f"[VQ_COMMIT] chunk_start={chunk_start}, B={B}, "
                f"total_latent={int(total_latent)}, commit_loss={commit_loss.item():.6f}"
            )

        return commit_loss, codebook_loss, repel_loss, cb_repel_loss

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # VQ forward (COPY-PASTE)
    # - 返り値は必ず: (total_loss, (commit_loss, cb_loss, repel_loss, cb_repel_loss))
    # - ただし “変更後の処理” (= quantize / embed_ind / embed 更新) は内部で維持
    # - mask_dict の global/offset 健全性チェックを軽量に復活（debug_index で強化）
    # -------------------------------------------------------------------------


    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, feature=None, mask_dict=None, logger=None, chunk_i=None, epoch=None, mode=None):
        import os, time, math
        import torch
        import numpy as np

        # -----------------------------
        # helpers
        # -----------------------------
        def _as_index_tensor(idx, device):
            if idx is None:
                return None
            if torch.is_tensor(idx):
                return idx.to(device=device, non_blocking=True).reshape(-1).long()
            if isinstance(idx, np.ndarray):
                return torch.from_numpy(idx).to(device=device, non_blocking=True).reshape(-1).long()
            if isinstance(idx, (list, tuple)):
                if len(idx) == 0:
                    return torch.empty(0, device=device, dtype=torch.long)
                # list of lists -> flatten
                if len(idx) > 0 and isinstance(idx[0], (list, tuple, np.ndarray)):
                    flat = []
                    for sub in idx:
                        flat.extend(list(sub))
                    idx = flat
                return torch.tensor(idx, device=device, dtype=torch.long).reshape(-1)
            return torch.tensor([int(idx)], device=device, dtype=torch.long)

        def _loss0(val, device, dtype):
            """loss を 0-dim Tensor に揃える（grad を壊さない）"""
            if torch.is_tensor(val):
                return val.to(device=device, dtype=dtype).reshape(())
            if val is None:
                return torch.zeros((), device=device, dtype=dtype)
            if isinstance(val, (float, int)):
                return torch.tensor(float(val), device=device, dtype=dtype)
            raise TypeError(f"loss must be Tensor/float/int/None, got {type(val)}")

        def _lookup_K(key_any):
            K_e = self._get_absK_from_cb_dict(key_any)
            if K_e is None:
                K_e = self._get_absK_from_cb_dict(str(key_any))
            if K_e is None:
                K_e = int(getattr(self, "codebook_size", 1))
                if logger is not None:
                    logger.warning(f"[K_FALLBACK] key={str(key_any)} -> K_e={K_e} (not found in cb_dict)")
            return int(K_e)

        def _check_mask_global(mask_dict, global_start, global_end, logger, chunk_i, mode, max_report=5):
            """
            mask_dict は「グローバル index の集合」でよい。
            このchunk外の index が含まれていても問題ない（forward側で gmask で切る）。
            ただし負の index はバグなので落とす。
            """
            import torch

            if mask_dict is None:
                return

            neg = 0
            oob = 0
            checked = 0
            examples = []

            for k, idx in mask_dict.items():
                if idx is None:
                    continue
                if not torch.is_tensor(idx):
                    idx = torch.as_tensor(idx)

                idx = idx.reshape(-1)
                if idx.numel() == 0:
                    continue

                # 重すぎ回避：先頭だけ
                if idx.numel() > 256:
                    idx = idx[:256]

                checked += int(idx.numel())

                neg_mask = (idx < 0)
                if neg_mask.any():
                    neg += int(neg_mask.sum().item())
                    if len(examples) < max_report:
                        examples.append((str(k), idx[neg_mask][:5].tolist()))

                oob_mask = (idx < global_start) | (idx >= global_end)
                if oob_mask.any():
                    oob += int(oob_mask.sum().item())
                    if len(examples) < max_report:
                        examples.append((str(k), idx[oob_mask][:5].tolist()))

                if checked >= 256:
                    break

            if neg > 0:
                msg = (f"[MASKCHK] NEGATIVE indices found! checked={checked} neg={neg} "
                       f"range=[{global_start},{global_end}) chunk_i={chunk_i} mode={mode} ex={examples}")
                raise RuntimeError(msg)

            # chunk外が混ざるのは普通にあり得るので warn のみ
            if oob > 0 and logger is not None:
                logger.warning(
                    f"[MASKCHK] checked={checked} oob={oob} range=[{global_start},{global_end}) "
                    f"chunk_i={chunk_i} mode={mode} (OK if mask_dict is global; forward will gmask-filter). "
                    f"ex={examples}"
                )

        # -----------------------------
        # 0) global offset bookkeeping
        # -----------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0

        # 旧挙動に寄せる：eval/test/init は毎回 0 始まり（必要なら外してOK）
        if mode in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = 0
        elif chunk_i is not None and int(chunk_i) == 0:
            self.latent_size_sum = 0

        # -----------------------------
        # 1) reshape -> flatten [1,B,D]
        # -----------------------------
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1,B,D]
        elif x.ndim >= 4:
            x = x.view(x.shape[0], -1, x.shape[-1])

        if x.ndim != 3:
            raise RuntimeError(f"x must be 3D [1,B,D], got shape={tuple(x.shape)}")

        flatten = x
        B = int(flatten.shape[1])
        D = int(flatten.shape[2])
        device = flatten.device
        dtype = flatten.dtype

        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        # -----------------------------
        # 2) normalize mask_dict (global LongTensor)
        # -----------------------------
        if mask_dict is not None:
            # あなたの normalize が (device, logger) を受けられるならそれを使う
            try:
                mask_dict = self._normalize_mask_dict(mask_dict, logger, device=device)
            except TypeError:
                mask_dict = self._normalize_mask_dict(mask_dict)

            # 最終保険: 必ず cuda long 1D に揃える
            md2 = {}
            for k, v in mask_dict.items():
                vv = _as_index_tensor(v, device=device)
                md2[k] = vv
            mask_dict = md2

        if logger is not None:
            logger.info(f"[CODEBOOK] mode={mode}")

        # 軽量グローバル健全性チェック
        _check_mask_global(mask_dict, global_start, global_end, logger, chunk_i, mode)

        # 強いチェック（重い）：debug_index が立ってたら oob を強制的に落とす
        if getattr(self, "debug_index", False) and mask_dict is not None and B > 0:
            # chunk0 のとき global_start は 0 のはず、という設計なら強制
            if chunk_i is not None and int(chunk_i) == 0 and mode not in ("eval", "test", "init_kmeans_final"):
                assert global_start == 0, f"[INDEX_MISMATCH] chunk head but global_start={global_start}"

            n_checked = 0
            n_oob = 0
            for k, gi in mask_dict.items():
                if gi is None or gi.numel() == 0:
                    continue
                s = gi
                if s.numel() > 256:
                    s = s[:256]
                in_chunk = (s >= global_start) & (s < global_end)
                n_oob += int((~in_chunk).sum().item())
                n_checked += int(s.numel())
                if n_checked >= 1024:
                    break
            assert n_oob == 0, (
                f"[INDEX_MISMATCH] mask_dict contains out-of-chunk indices. "
                f"range=[{global_start},{global_end}), checked={n_checked}, oob={n_oob}"
            )

        # -----------------------------
        # 3) init_kmeans_final: “作らない”で割当/SS/dump（返り値は loss 形式に合わせる）
        # -----------------------------
        if mode == "init_kmeans_final":
            key_list_to_dump = {
                "6_0_3_1_1_2_6_2_1_1_11_0",
                "7_0_3_0_0_2_0_1_0_0_3_0",
                "16_-1_4_0_0_1_0_0_0_0_16_0",
                "16_1_4_0_1_3_6_0_5_0_0_0",
            }

            if (not hasattr(self, "_kmeans_dump")) or (chunk_i is not None and int(chunk_i) == 0):
                self._kmeans_dump = {}

            self.quantize_dict = {}
            self.embed_ind_dict = {}

            if mask_dict is not None:
                for key, idx_global in mask_dict.items():
                    if idx_global is None or idx_global.numel() == 0:
                        continue
                    gmask = (idx_global >= global_start) & (idx_global < global_end)
                    if not gmask.any():
                        continue
                    idx_local = (idx_global[gmask] - global_start).to(device=device, dtype=torch.long)
                    if idx_local.numel() == 0:
                        continue

                    masked_latents = flatten[0].index_select(0, idx_local)  # [Ni,D]
                    if masked_latents.numel() == 0:
                        continue

                    skey = str(key)
                    K_e = _lookup_K(key)

                    # ここは “作らない”
                    code, found = self._get_code_for_key_no_create(skey)
                    if code is None or (not found):
                        if logger is not None:
                            logger.warning(f"[NO_CODEBOOK] key={skey} missing embed; skip assign/SS")
                        continue

                    with torch.no_grad():
                        dist = torch.cdist(masked_latents, code, p=2).pow(2)
                        idx_code = dist.argmin(dim=-1)
                    quantize = code.index_select(0, idx_code)

                    # debug 保存（CPU）
                    self.quantize_dict[skey] = quantize.detach().to("cpu", dtype=torch.float16)
                    self.embed_ind_dict[skey] = idx_code.detach().to("cpu", dtype=torch.int32)

                    # silhouette は必要ならここで（あなたの既存関数を呼ぶ）
                    if logger is not None and hasattr(self, "silhouette_score_torch"):
                        try:
                            X = masked_latents.detach().float().cpu()
                            labels = idx_code.detach().cpu().long()
                            u, cts = labels.unique(return_counts=True)
                            if u.numel() >= 2 and X.shape[0] >= 3 and int(cts.min()) >= 2:
                                nmax = int(getattr(self, "samples_latent_in_kmeans", 0) or 0)
                                n = min(nmax, X.shape[0]) if nmax > 0 else min(4096, X.shape[0])
                                perm = torch.randperm(X.shape[0])[:n]
                                xs = X.index_select(0, perm)
                                ys = labels.index_select(0, perm)
                                sil = self.silhouette_score_torch(xs, ys,
                                                                  device="cuda" if torch.cuda.is_available() else "cpu")
                                logger.info(f"SS: {skey} {sil:.4f}, size {X.shape[0]}, K_e {K_e}")
                            else:
                                logger.info(
                                    f"[SS_SKIP] key={skey} N={X.shape[0]} uniq={u.numel()} mincnt={int(cts.min()) if cts.numel() else -1} K_e={K_e}")
                        except Exception as e:
                            logger.warning(f"[SS_FAIL] key={skey} err={repr(e)}")

                    # dump
                    do_dump = (epoch is not None) and ((int(epoch) % 10 == 0) or (int(epoch) == 1))
                    if do_dump and (skey in key_list_to_dump):
                        entry = self._kmeans_dump.get(skey)
                        if entry is None:
                            entry = {"latents": [], "centers": None, "assign": []}
                            self._kmeans_dump[skey] = entry
                        entry["latents"].append(masked_latents.detach().to("cpu", dtype=torch.float16))
                        entry["assign"].append(idx_code.detach().to("cpu"))
                        if entry["centers"] is None:
                            entry["centers"] = code.detach().to("cpu", dtype=torch.float16)

                    del masked_latents, code, idx_code, quantize

            self.latent_size_sum = global_end

            do_dump = (epoch is not None) and ((int(epoch) % 10 == 0) or (int(epoch) == 1))
            if do_dump and len(getattr(self, "_kmeans_dump", {})) > 0:
                out = {}
                for k, v in self._kmeans_dump.items():
                    if k not in key_list_to_dump:
                        continue
                    out[k] = {
                        "latents": torch.cat(v["latents"], dim=0) if len(v["latents"]) else None,
                        "centers": v["centers"],
                        "assign": torch.cat(v["assign"], dim=0) if len(v["assign"]) else None,
                    }
                stamp = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs("dumps", exist_ok=True)
                path = os.path.join("dumps", f"init_kmeans_final_ep{epoch}_chunk{chunk_i}_{stamp}.pt")
                torch.save(out, path)
                self._kmeans_dump = {}

            torch.cuda.empty_cache()

            # 返り値は “loss 形式” に統一
            z = torch.zeros((), device=device, dtype=dtype)
            return z, (z, z, z, z)

        # -----------------------------
        # 4) train/eval: “変更後の処理” = quantize を作る + embed_ind を保存
        # -----------------------------
        if feature is None:
            raise RuntimeError("feature is required in train/eval mode")

        # feature はチェックだけ（必要ならここで feature 由来の key 生成等も可能）
        feat_flat = feature if torch.is_tensor(feature) else torch.cat(feature, dim=0)
        feat_flat = feat_flat.contiguous().to(device)
        if not (feat_flat.ndim == 2 and feat_flat.size(1) == 78):
            raise RuntimeError(f"feat_flat shape={tuple(feat_flat.shape)} (expected [B,78])")
        if feat_flat.size(0) != B:
            raise RuntimeError(f"feat_flat N={feat_flat.size(0)} != B={B}")

        self.quantize_dict = {}
        self.embed_ind_dict = {}

        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue

                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not gmask.any():
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=device, dtype=torch.long)
                if idx_local.numel() == 0:
                    continue

                masked_latents = flatten[0].index_select(0, idx_local)  # [Ni,D]
                if masked_latents.numel() == 0:
                    continue

                skey = str(key)
                K_e = _lookup_K(key)

                # train/eval は create OK（ただし理想は全部事前生成）
                safe = self._get_or_create_safe_key(skey, K_e=K_e, D=D, device=device, create=True)
                code_param = self.embed[safe]
                code = code_param.squeeze(0) if code_param.ndim == 3 else code_param  # [K,D]

                with torch.no_grad():
                    dist = torch.cdist(masked_latents, code, p=2).pow(2)
                    idx_code = dist.argmin(dim=-1)

                quantize = code.index_select(0, idx_code)  # [Ni,D]
                self.quantize_dict[skey] = quantize
                self.embed_ind_dict[skey] = idx_code.to(torch.int32)

                del masked_latents, code, idx_code, quantize

        # flatten[0] の一部を quantize に差し替える（“変更後の処理” の結果）
        quantize_full = flatten[0].clone()  # [B,D]
        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                skey = str(key)
                if skey not in self.quantize_dict:
                    continue
                if idx_global is None or idx_global.numel() == 0:
                    continue
                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not gmask.any():
                    continue
                idx_local = (idx_global[gmask] - global_start).to(device=device, dtype=torch.long)
                if idx_local.numel() == 0:
                    continue
                qk = self.quantize_dict[skey].to(device=device, dtype=quantize_full.dtype)
                quantize_full.index_copy_(0, idx_local, qk)

        # straight-through（返さないが、必要ならここで downstream 用に内部保持しても良い）
        quantize_st = flatten[0] + (quantize_full - flatten[0]).detach()
        quantize_st = quantize_st.unsqueeze(0)  # [1,B,D]
        self._last_quantize_st = quantize_st  # optional

        # -----------------------------
        # 5) loss 計算（loss だけ返すための “旧インタフェース”）
        #    - もし commitment_loss があるなら旧版の 4-tuple をそのまま使う
        #    - 無い/使わないなら 0 を返す（normalize 側が落ちないように）
        # -----------------------------
        commit_loss = None
        cb_loss = None
        repel_loss = None
        cb_repel_loss = None

        if hasattr(self, "commitment_loss") and mask_dict is not None and B > 0:
            # 期待: (commit, cb, rep, cb_rep)
            out = self.commitment_loss(
                encoder_outputs=flatten[0],  # [B,D]
                mask_dict=mask_dict,  # global indices
                codebook=getattr(self, "_codebook", None) or getattr(self, "embed", None),
                logger=logger,
                chunk_start=global_start,
                beta=getattr(self, "beta", 0.25),
                temperature=getattr(self, "temperature", None),
                use_cosine=getattr(self, "use_cosine", False),
            )
            if not (isinstance(out, (tuple, list)) and len(out) == 4):
                raise TypeError(f"commitment_loss must return 4-tuple, got {type(out)}: {out}")
            commit_loss, cb_loss, repel_loss, cb_repel_loss = out

        commit_loss = _loss0(commit_loss, device=device, dtype=dtype)
        cb_loss = _loss0(cb_loss, device=device, dtype=dtype)
        repel_loss = _loss0(repel_loss, device=device, dtype=dtype)
        cb_repel_loss = _loss0(cb_repel_loss, device=device, dtype=dtype)

        total_loss = commit_loss + cb_loss + repel_loss + cb_repel_loss

        # -----------------------------
        # 6) global offset update
        # -----------------------------
        if mode not in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = global_end
        torch.cuda.empty_cache()

        # 返り値は “loss 形式” だけ
        return total_loss, (commit_loss, cb_loss, repel_loss, cb_repel_loss)
