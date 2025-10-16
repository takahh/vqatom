import torch.distributed as distributed
from einops import rearrange, repeat, pack, unpack
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

def noop(x):
    return x

def batched_bincount(indices, minlength):
    H, N = indices.shape
    bins = torch.zeros(H, minlength, device=indices.device, dtype=torch.long)
    for h in range(H):
        bins[h].scatter_add_(0, indices[h], torch.ones_like(indices[h]))
    return bins

def kmeans(samples, num_clusters, use_cosine_sim=False, all_reduce_fn=noop, eps=1e-12):
    # samples: [H, N, D]
    H, N, D = samples.shape
    device, dtype = samples.device, samples.dtype

    # 1) Cap K to available unique samples (robust for small N cases)
    #    (H=1 in your logs; for H>1 you can compute per-H if needed)
    # uniq = torch.unique(samples.reshape(-1, D), dim=0).shape[0]
    num_clusters = int(min(num_clusters, N))
    if num_clusters <= 0:
        raise ValueError("No samples to cluster.")

    # 2) Prepare means
    means = torch.zeros((H, num_clusters, D), device=device, dtype=dtype)
    means[:, 0] = samples[:, torch.randint(0, N, (1,), device=device)]

    # Optional: proper cosine distance (nonnegative)
    if use_cosine_sim:
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)

    with torch.no_grad():
        # ---- K-Means++ init ----
        for k in range(1, num_clusters):
            if use_cosine_sim:
                # cosine distance in [0, 2]
                d = 1.0 - (samples @ means[:, :k].transpose(1, 2))
            else:
                d = torch.cdist(samples, means[:, :k], p=2)         # [H, N, k]

            min_d = d.min(dim=-1).values                            # [H, N]
            min_d = torch.clamp(min_d, min=0)                       # no negatives
            row_sum = min_d.sum(dim=-1, keepdim=True)               # [H, 1]

            # Safe probs: if sum==0 (all points exactly at chosen centers), fallback to uniform
            safe_probs = torch.where(
                row_sum > 0,
                min_d / (row_sum + eps),
                torch.full_like(min_d, 1.0 / N)
            )

            # multinomial requires nonnegative and finite
            safe_probs = torch.nan_to_num(safe_probs, nan=0.0, posinf=0.0, neginf=0.0)
            # Guarantee rows sum to 1
            safe_probs = safe_probs / (safe_probs.sum(dim=-1, keepdim=True) + eps)

            next_idx = torch.multinomial(safe_probs, 1).squeeze(-1) # [H]
            means[:, k] = samples[torch.arange(H, device=device), next_idx]

        # ---- Lloyd steps ----
        for _ in range(30):
            if use_cosine_sim:
                dists = samples @ means.transpose(1, 2)             # similarity
            else:
                dists = -torch.cdist(samples, means, p=2)           # negative distance

            buckets = torch.argmax(dists, dim=-1)                   # [H, N]
            bins = batched_bincount(buckets, minlength=num_clusters)
            all_reduce_fn(bins)

            zero_mask = bins == 0
            bins_safe = bins.masked_fill(zero_mask, 1)

            new_means = torch.zeros_like(means)
            new_means.scatter_add_(1, buckets.unsqueeze(-1).expand(-1, -1, D), samples)
            new_means = new_means / bins_safe.unsqueeze(-1)

            all_reduce_fn(new_means)
            if use_cosine_sim:
                new_means = torch.nn.functional.normalize(new_means, p=2, dim=-1)

            means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


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

    def forward(self, z, chunk, logger, codebook):
        import torch
        import torch.nn.functional as F

        device = z.device
        eps = 1e-8

        # ---- 1) 距離の一次統計（1Dで扱う）----
        # z: [B, D]
        pdist_z = torch.pdist(z, p=2)  # [B*(B-1)/2], 1D

        # （巨大時）サンプルを間引き
        sample = pdist_z
        if sample.numel() > 1_000_000:
            sample = self.sample_cap(sample, max_n=200_000)

        # 分位点などは学習に使う重みの境界値なので勾配不要
        with torch.no_grad():
            dynamic_threshold = torch.quantile(sample, 0.10)  # tensor
            lower_q, upper_q = 0.95, 0.99
            lower_thresh = torch.quantile(sample, lower_q)  # tensor
            upper_thresh = torch.quantile(sample, upper_q)  # tensor
            center = (lower_thresh + upper_thresh) / 2  # tensor

        # ---- ログ（負荷・転送を抑えて）----
        if chunk % 32 == 0:
            with torch.no_grad():
                s = sample
                if s.numel() > 100_000:
                    idx = torch.randperm(s.numel(), device=s.device)[:100_000]
                    s = s.index_select(0, idx)
                s_cpu = s.detach().to('cpu', dtype=torch.float32).flatten()
                hist = torch.histc(s_cpu, bins=10, min=0.0, max=15.0)
                vals = hist.tolist()
                logger.info(vals)
                print(vals)

        def latent_repel_mid_chunked(
                z,
                low,
                high,
                center,
                sigma=3.0,
                sharp=20.0,
                row_block=4096,
                col_block=4096,
                detach_weight=True,
                use_checkpoint=True,  # ← 推奨: True にすると各ブロックの中間を保持せず再計算
                stream_backward=False,  # ← True にすると各ブロックで即 backward（要: 呼び出し元での管理）
                eps=1e-8,
        ):
            """
            z: [B, D], requires_grad=True
            low/high/center: しきい値（no_grad Tensor 推奨）
            - use_checkpoint=True: 各ブロックの中間を保持せず、backward時に再計算（メモリ節約）
            - stream_backward=True: 各ブロックで縮尺済み loss を即 backward（forward内では使わないこと！）
            """
            import torch
            import torch.utils.checkpoint as cp

            assert not (use_checkpoint and stream_backward), "checkpoint と streaming backward は同時に使わないでください。"

            B, D = z.shape
            device = z.device

            # 方式Bで必要になる：総ブロック数（上三角の「ブロック間」だけ数える）
            def count_blocks(B, rb, cb):
                cnt = 0
                i = 0
                while i < B:
                    bi = min(rb, B - i)
                    j = i + rb
                    while j < B:
                        bj = min(cb, B - j)
                        cnt += 1
                        j += cb
                    i += rb
                return max(cnt, 1)

            n_blocks_total = count_blocks(B, row_block, col_block)

            # ブロック計算本体（Checkpoint で包めるよう関数化）
            def block_loss(zi, zj):
                # zi: [bi, D], zj: [bj, D]
                d = torch.cdist(zi, zj, p=2)  # [bi, bj]

                # 重み（Sigmoid）: 勾配を通さないなら detach で安全側に
                w = torch.sigmoid(sharp * (d - low)) * torch.sigmoid(sharp * (high - d))
                if detach_weight:
                    w = w.detach()

                # 中央付近を釣鐘で強調（ここは勾配を z に通す）
                bell = torch.exp(-(d - center) ** 2 / (2 * (sigma ** 2)))

                num = (w * bell).sum()
                den = w.sum().clamp_min(eps)
                return num / den  # scalar Tensor

            total = z.new_zeros(())

            # メインループ
            i = 0
            while i < B:
                bi = min(row_block, B - i)
                zi = z[i: i + bi]  # [bi, D]
                j = i + row_block
                while j < B:
                    bj = min(col_block, B - j)
                    zj = z[j: j + bj]  # [bj, D]

                    if stream_backward:
                        # 方式B: forward 内で使わないで。学習ループ側でこの関数を呼び、
                        # ここで loss_block / n_blocks_total を backward してメモリを即時解放する。
                        # ≒ 各ブロックのグラフが forward 中に溜まらない。
                        loss_block = block_loss(zi, zj) / n_blocks_total
                        # !!! forward 内で backward はアンチパターンなので、
                        # このモードは forward では使わず「損失計算関数」として学習ループ側で呼び出して下さい。
                        loss_block.backward()
                        # ログ用に合算（勾配には関与しない）
                        total = total + loss_block.detach()
                    else:
                        if use_checkpoint:
                            # 方式A: Checkpointing で中間を保持しない
                            # 注意: 入力は Tensor で、関数は Tensor を返す必要がある
                            # zi, zj をそのまま渡す
                            lb = cp.checkpoint(block_loss, zi, zj)
                        else:
                            # 通常合算（メモリは溜まる）
                            lb = block_loss(zi, zj)
                        total = total + lb

                    # 明示参照解除（Python 参照を早めに消す）
                    del zj
                    j += col_block

                del zi
                i += row_block

            if stream_backward:
                # 方式B: ここでは張っているグラフはない。total はログ用の数値（Tensor）として返す。
                return total  # detached の合算

            # 方式A / 通常: グラフ付きのスカラー Tensor を返す（上位で一括 backward）
            # 平均に正規化（スケール安定化）
            return total / n_blocks_total

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
            sigma=3.0, sharp=20.0,
            row_block=4096, col_block=4096,
            detach_weight=True,
            use_checkpoint=True,  # 推奨: True
            stream_backward=False,  # forward 内では False
        )
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
        self.cb_dict = {1: 47, 3: 10, 5: 43, 6: 4360, 7: 1760, 8: 1530, 9: 730, 11: 50, 14: 22, 15: 100, 16: 530,
                        17: 500, 19: 19, 34: 27, 35: 190, 53: 85}
        # key 1, code torch.Size([47, 16]), embed_k torch.Size([19, 16])
        assert not (
                    use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        element_keys = [1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 34, 35, 53]
        for elem in element_keys:
            self.register_buffer(f"cluster_size_{elem}", torch.zeros(self.cb_dict[elem]))
            self.register_buffer(f"embed_avg_{elem}", torch.zeros(self.cb_dict[elem], dim))
        self.learnable_codebook = learnable_codebook
        self.embed = nn.ParameterDict()
        # self.embed_avg = nn.ParameterDict()
        for key in element_keys:
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
    def init_embed_(self, data, mask_dict=None):
        """
        Initialize per-element codebooks using K-means on masked latents.

        Guarantees for each element `key`:
          - self.cb_dict[str(key)] == actual K found by k-means
          - self.embed[str(key)].shape == (K, D)
          - getattr(self, f"cluster_size_{key}").shape == (K,)
          - Copies centroids and counts with proper dtypes/devices
        """
        print("++++++++++++++++ RUNNING init_embed !!! ++++++++++++++++++++++++++++++")
        assert mask_dict is not None, "mask_dict is required"
        cluster_sizes_all = []

        # Deterministic order
        for key in sorted(mask_dict.keys()):
            # Normalize key usage (ParameterDict/Dict keys often stored as str)
            skey = str(key)

            # Target cb size from allocation ratio
            try:
                ratio = self.cb_dict[skey]
            except KeyError:
                ratio = self.cb_dict[key]
            cbsize = int(self.codebook_size * ratio / 10000)

            idx = mask_dict[key]  # indices for this element
            masked = data[0][idx]  # (Ni, D)
            # print(f"[key={key}] cbsize={cbsize}, masked.shape={tuple(masked.shape)}")

            # Run k-means (expects [H, N, D]); here H=1
            embed_k, cluster_size_k = kmeans(masked.unsqueeze(0), cbsize)

            # Normalize k-means outputs to (K, D) and (K,)
            if embed_k.dim() == 3:  # (1, K, D) -> (K, D)
                embed_k = embed_k[0].contiguous()
            if cluster_size_k.dim() == 2:  # (1, K)    -> (K,)
                cluster_size_k = cluster_size_k[0].contiguous()

            K_found = int(embed_k.shape[0])
            D_curr = int(self.embed[skey].shape[-1])
            # print(f"[key={key}] K_found={K_found}, D={D_curr}")

            # Record the actual K used for this element
            self.cb_dict[skey] = K_found

            # ---------- Ensure codebook parameter has shape [K_found, D] ----------
            code = self.embed[skey]
            if code.shape[0] != K_found:
                new_code = torch.empty(
                    K_found, D_curr, device=code.device, dtype=code.dtype
                )
                nn.init.normal_(new_code, mean=0.0, std=0.01)
                self.embed[skey] = nn.Parameter(new_code, requires_grad=True)
                code = self.embed[skey]  # refresh reference

                # update "embed_avg_{key}" shape
                self._buffers[f"embed_avg_{key}"] = torch.zeros(K_found, D_curr)

            # ---------- Ensure cluster_size buffer has shape [K_found] ------------
            cs_name = f"cluster_size_{key}"
            cs_buf = getattr(self, cs_name)
            if cs_buf.numel() != K_found:
                new_cs = torch.zeros(
                    K_found, device=cs_buf.device, dtype=cs_buf.dtype
                )
                # Re-register the buffer with the new shape
                self.register_buffer(cs_name, new_cs)
                cs_buf = getattr(self, cs_name)
            else:
                cs_buf.zero_()

            # (Optional) If you keep EMA stats per element, resize them here too.
            # Example (guarded):
            # for buf in (f"embed_avg_{key}", f"ema_cluster_size_{key}"):
            #     if hasattr(self, buf):
            #         old = getattr(self, buf)
            #         if old.dim() == 2 and old.shape[0] != K_found:  # (K, D)
            #             new = torch.zeros(K_found, D_curr, device=old.device, dtype=old.dtype)
            #             self.register_buffer(buf, new)
            #         elif old.dim() == 1 and old.numel() != K_found: # (K,)
            #             new = torch.zeros(K_found, device=old.device, dtype=old.dtype)
            #             self.register_buffer(buf, new)

            # ----------------------- Copy centroids & counts -----------------------
            with torch.no_grad():
                # Centroids
                if code.dim() == 3:  # (1, K, D) – unlikely here, but keep safety
                    code.data.copy_(embed_k.unsqueeze(0).to(code.dtype))
                else:  # (K, D)
                    code.data.copy_(embed_k.to(code.dtype))

                # Counts
                assert cs_buf.shape[0] == cluster_size_k.shape[0], \
                    f"cluster_size buffer {cs_buf.shape} vs kmeans {cluster_size_k.shape} for key={key}"
                cs_buf.add_(cluster_size_k.to(cs_buf.dtype))

            # For optional logging
            cluster_sizes_all.append(cluster_size_k)

            # print(f"[key={key}] codebook={tuple(code.shape)}, cluster_size_buf={tuple(cs_buf.shape)}")

        # Mark initialized
        self.initted.data.copy_(torch.tensor([True], device=self.initted.device))
        # Return last embed_k for compatibility (or return None / dict if you prefer)
        return embed_k

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

    @torch.no_grad()
    def silhouette_score_torch(X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Vectorized silhouette score using GPU.
        X: [N, D] tensor (cuda)
        labels: [N] int tensor (cuda)
        """
        X = X.squeeze()
        labels = labels.squeeze()
        N = X.size(0)
        dists = torch.cdist(X, X, p=2)  # [N, N], GPU-accelerated

        # same-cluster mask
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=X.device)
        same = same & ~eye

        # a_i: mean intra-cluster distance
        same_sum = same.sum(dim=1)
        same_sum[same_sum == 0] = 1  # avoid div/0
        a_i = (dists * same).sum(dim=1) / same_sum

        # b_i: min mean inter-cluster distance
        unique_labels = labels.unique()
        K = unique_labels.size(0)
        b_i = torch.full((N,), float('inf'), device=X.device)
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            cluster_mean = (dists[:, mask].sum(dim=1) / mask.sum()).float()
            b_i = torch.minimum(b_i, torch.where(mask, torch.inf, cluster_mean))

        # silhouette values
        sil = (b_i - a_i) / torch.maximum(a_i, b_i)
        sil[torch.isnan(sil)] = 0
        return sil.mean().item()

    # # old, slow
    # def silhouette_score_torch(self, X: torch.Tensor, labels: torch.Tensor):
    #     """
    #     X: [N, D] float tensor on GPU
    #     labels: [N] int tensor on GPU
    #     Returns: scalar silhouette score (float)
    #     """
    #     print(f"running sil score calculation")
    #     # pairwise distance matrix (N x N)
    #     # If N is large, consider chunking!
    #     X = X.squeeze()
    #     labels = labels.squeeze()
    #     dists = torch.cdist(X, X, p=2)  # GPU accelerated
    #
    #     N = X.shape[0]
    #     sil_samples = torch.empty(N, device=X.device)
    #
    #     unique_labels = labels.unique()
    #     for i in range(N):
    #         same_mask = (labels == labels[i])
    #         same_mask[i] = False  # exclude self
    #
    #         # mean intra-cluster distance a(i)
    #         if same_mask.any():
    #             a_i = dists[i][same_mask].mean()
    #         else:
    #             a_i = torch.tensor(0.0, device=X.device)
    #
    #         # mean distance to all other clusters
    #         b_i_vals = []
    #         for lab in unique_labels:
    #             if lab == labels[i]:
    #                 continue
    #             mask = (labels == lab)
    #             if mask.any():
    #                 b_i_vals.append(dists[i][mask].mean())
    #         b_i = torch.stack(b_i_vals).min() if b_i_vals else torch.tensor(0.0, device=X.device)
    #
    #         denom = torch.max(a_i, b_i)
    #         sil_samples[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    #
    #     return sil_samples.mean().item()
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, mask_dict=None, logger=None, chunk_i=None, epoch=None, mode=None):
        """Forward pass with per-element quantization and EMA update."""

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

        # ------------------------------------------------------------------
        # 2. per-element quantization loop
        # ------------------------------------------------------------------
        for key in mask_dict.keys():
            # -------------------- select latents for this element --------------------
            if mode == "init_kmeans_final":
                masked_latents = flatten[0][mask_dict[key]]  # global pass
            else:
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

            # -------------------- silhouette (init only, on CPU) --------------------
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
                        if logger: logger.info(msg)
                except Exception as e:
                    if logger: logger.warning(f"Silhouette failed for {key}: {e}")

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

        for key in mask_dict.keys():
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


    def orthogonal_loss_fn(self, embed_ind_dict, codebook, init_feat, latents, quantized_dict, logger, epoch, chunk=0):
        # embed_ind_dict.to("cuda")
        codebook.to("cuda")
        init_feat.to("cuda")
        latents.to("cuda")
        quantized_dict.to("cuda")
        latent_len_sum = 0
        two_repel_loss_weighted_sum = 0
        cb_loss_weighted_sum = 0
        for key in embed_ind_dict.keys():
            import torch
            latents_for_sil = torch.squeeze(latents)
            latents_size = latents_for_sil.shape[0]
            latent_len_sum += latents_size
            two_repel_loss, div_nega_loss, repel_loss_from_2, cb_loss, repel_loss_mid_high = (
                self.compute_contrastive_loss(latents_for_sil, chunk, logger, codebook[str(key)]))

            two_repel_loss_weighted_sum += latents_size * two_repel_loss
            cb_loss_weighted_sum += latents_size * cb_loss
        two_repel_loss_avg = two_repel_loss_weighted_sum / latent_len_sum
        cb_loss_avg = cb_loss_weighted_sum / latent_len_sum

        return (two_repel_loss_avg, cb_loss_avg)

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

    def commitment_loss(
        self,
        encoder_outputs,  # [B, D]
        codebook,         # [K, D] or [1, K, D]
        beta=0.25,
        temperature=None,       # if not None, overrides EMA
        use_cosine=False
    ):
        codebook = codebook.squeeze(0) if codebook.dim() == 3 else codebook
        assert encoder_outputs.dim() == 2 and codebook.dim() == 2
        B, D = encoder_outputs.shape
        K, Dk = codebook.shape
        assert D == Dk, f"latent D={D} != codebook D={Dk}"

        # # Optional codebook norm clamp to avoid run-away norms in codes
        # if (self.codebook_max_norm is not None) and self.training:
        #     with torch.no_grad():
        #         norms = torch.linalg.norm(codebook, dim=-1, keepdim=True)
        #         scale = torch.clamp(self.codebook_max_norm / (norms + 1e-8), max=1.0)
        #         codebook.mul_(scale)

        # Keep latents roughly centered to reduce drift
        z = self._center_batch(encoder_outputs)

        if use_cosine:
            # Cosine (unit-norm) path: very stable; distances in [-1, 1]
            z_n = F.normalize(z, dim=-1)
            cb_n = F.normalize(codebook, dim=-1)

            # Use cosine “distance” as 1 - cos sim
            sim = z_n @ cb_n.t()                     # [B,K]
            logits = sim / 0.07                      # fixed temperature in cosine land (tweakable)
            soft_assign = F.softmax(logits, dim=-1)  # [B,K]
            indices = logits.argmax(dim=-1)          # [B]
            quantized_hard = codebook[indices]       # decode with raw codebook (keeps magnitude info)
            quantized_soft = soft_assign @ codebook

        else:
            # Euclidean path with squared distances for stable gradients
            d2 = self.pairwise_sq_dists(z, codebook)     # [B,K]
            indices = d2.argmin(dim=-1)
            quantized_hard = codebook[indices]

            # Temperature handling: decouple from raw (growing) distance scale via EMA + clamp
            with torch.no_grad():
                med = torch.sqrt(d2.detach()).median()
                tau_obs = float(med + 1e-6)
                # EMA update
                tau_new = self.tau_ema * float(self._tau.item()) + (1.0 - self.tau_ema) * tau_obs
                tau_new = float(min(max(tau_new, self.tau_min), self.tau_max))
                self._tau.fill_(tau_new)

            tau_eff = float(self._tau.item()) if (temperature is None) else float(temperature)
            tau_eff = min(max(tau_eff, self.tau_min), self.tau_max)

            # Use true Euclidean in the softmax logits (better locality than d2)
            distances = torch.sqrt(torch.clamp(d2, min=1e-12))
            logits = -distances / tau_eff
            soft_assign = F.softmax(logits, dim=-1)
            quantized_soft = soft_assign @ codebook

        # Straight-through: forward hard, backward soft
        quantized = quantized_hard + (quantized_soft - quantized_soft.detach())

        # Codebook update pulls code vectors toward (detached) latents
        codebook_loss = F.mse_loss(quantized, z.detach())

        # Commitment pulls latents toward chosen code (detached)
        # latent_loss = beta * F.mse_loss(z, quantized.detach())
        latent_loss = beta * (z - quantized.detach()).abs().mean()  # L1 距離（線形）

        # Radius regularizer to prevent latent norm explosion (Euclidean path only—or keep for both)
        radius_loss = self._latent_radius_loss(z) * self.radius_weight

        # Total (return separated pieces so you can log them)
        total_latent_loss = latent_loss + radius_loss

        if self.training and torch.rand(()) < 0.02:
            with torch.no_grad():
                maxp = soft_assign.max(dim=-1).values.mean().item()
                if self.use_cosine:
                    print(f"[COS] K={K} max prob ~{maxp:.3f}")
                else:
                    print(f"[EUC] K={K} tau={tau_eff:.3f} max prob ~{maxp:.3f}")

        return total_latent_loss, codebook_loss

    def forward(self, x, init_feat, mask_dict=None, logger=None, chunk_i=None, epoch=0, mode=None):
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
            self._codebook(x, mask_dict, logger, chunk_i, epoch, mode)
            return 0
        else:
            quantize_dict, embed_ind_dict, embed = self._codebook(x, mask_dict, logger, chunk_i, epoch, mode)
        # -------------------------------
        # repel loss calculation
        # -------------------------------
        mid_repel_loss, cb_repel_loss \
            = self.orthogonal_loss_fn(embed_ind_dict, self._codebook.embed, init_feat, x, quantize_dict, logger, epoch, chunk_i)
        # -------------------------------
        # repel loss calculation
        # -------------------------------
        commit_loss, codebook_loss = self.commitment_loss(x.squeeze(), self._codebook.embed)
        # ---------------------------------------------
        # only repel losses at the first several steps
        # ---------------------------------------------
        alpha = 1 / ((epoch + 1) ** 2)
        repel_loss = mid_repel_loss
        repel_loss *= alpha
        if epoch < 3:
            loss = repel_loss
        elif epoch >= 3:
            print(f"repel {repel_loss}") # or some decaying schedule
            beta = 0.0001
            loss = beta * (commit_loss) + repel_loss
        return (loss, embed, commit_loss, commit_loss, [], repel_loss, cb_repel_loss)