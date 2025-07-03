import torch.distributed as distributed
from einops import rearrange, repeat, pack, unpack
from torch import nn, einsum
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


def kmeans(
        samples,
        num_clusters,
        num_iters=30,
        use_cosine_sim=False,
        all_reduce_fn=noop
):
    """　データ全体に kmeans をかける様に変えたのでOOMの可能性がある"""
    # [H, N, D]
    num_codebooks, num_samples, dim = samples.shape
    dtype, device = samples.dtype, samples.device

    # Sanity check
    assert dim > 1, f"Embedding dimension must be >1, but got {dim}"
    print(f"Starting K-Means: samples.shape = {samples.shape}")

    # Initialize means: [H, K, D]
    means = torch.zeros((num_codebooks, num_clusters, dim), device=device, dtype=dtype)

    # Pick first centroid at random per head
    rand_idx = torch.randint(0, num_samples, (num_codebooks, 1), device=device)  # [H, 1]
    first_centroid = torch.gather(samples, 1, rand_idx.unsqueeze(-1).expand(-1, -1, dim))  # [H, 1, D]
    means[:, 0, :] = first_centroid.squeeze(1)

    # def compute_chunked_dists(samples, means, chunk_size=5000):
    #     """
    #     samples: [H, N, D]
    #     means:   [H, K, D]
    #     returns: [H, N, K]
    #     """
    #     H, N, D = samples.shape
    #     H2, K, D2 = means.shape
    #     assert D == D2 and H == H2, "Shape mismatch"
    #     samples_sq = samples.pow(2).sum(dim=-1, keepdim=True)  # [H, N, 1]
    #     dists_chunks = []
    #     for start in range(0, K, chunk_size):
    #         end = min(start + chunk_size, K)
    #         means_chunk = means[:, start:end, :]  # [H, chunk, D]
    #         means_sq = means_chunk.pow(2).sum(dim=-1).unsqueeze(1)  # [H, 1, chunk]
    #         dot = torch.matmul(samples, means_chunk.transpose(-1, -2))  # [H, N, chunk]
    #         dists = samples_sq + means_sq - 2 * dot  # [H, N, chunk]
    #         dists_chunks.append(dists)
    #     return torch.cat(dists_chunks, dim=-1)  # [H, N, K]print("kmeans++ sampling start")
    #
    # Initialize min_dists to large values
    min_dists = torch.full((num_codebooks, num_samples), float('inf'), device=device)

    # --------------------------------------
    # k-means++ (initialization to
    # --------------------------------------
    chunk_size = 300
    for k in range(1, num_clusters):
        if k % 1000 == 0:
            mem = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"{k}/{num_clusters}, mem: {mem:.2f} GB")
        with torch.cuda.amp.autocast(enabled=False):
            for start in range(0, k, chunk_size):  # chunk over already chosen centroids
                end = min(start + chunk_size, k)
                means_chunk = means[:, start:end, :]  # [H, chunk, D]
                means_sq = means_chunk.pow(2).sum(dim=-1).unsqueeze(1)  # [H, 1, chunk]
                samples_sq = samples.pow(2).sum(dim=-1, keepdim=True)  # [H, N, 1]
                dot = torch.matmul(samples, means_chunk.transpose(-1, -2))  # [H, N, chunk]
                dists = samples_sq + means_sq - 2 * dot  # [H, N, chunk]
                min_chunk_dists, _ = dists.min(dim=-1)  # [H, N]
                min_dists = torch.minimum(min_dists, min_chunk_dists)  # update inplace

        sum_min_dists = min_dists.sum(dim=-1, keepdim=True) + 1e-6  # [H, 1]
        probs = min_dists / sum_min_dists  # [H, N]
        probs = torch.nan_to_num(probs, nan=1.0 / num_samples)
        probs = torch.clamp(probs, 0.0, 1.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Warning: fallback to uniform sampling at step {k}")
            probs = torch.full_like(probs, 1.0 / num_samples)

        next_idx = torch.multinomial(probs, 1)  # [H, 1]
        next_centroid = torch.gather(samples, 1, next_idx.unsqueeze(-1).expand(-1, -1, dim))  # [H, 1, D]
        means[:, k, :] = next_centroid.squeeze(1)

        torch.cuda.empty_cache()

    # ----------------------------------------------
    # k-means (relates latent vectors to clusters)
    # ----------------------------------------------
    print("kmeans++ sampling done. Running Lloyd iterations")
    # Lloyd iterations (K-Means)
    for i in range(num_iters):
        # Initialize:
        buckets = torch.empty((num_codebooks, num_samples), dtype=torch.long, device=samples.device)
        min_dists = torch.full((num_codebooks, num_samples), float('inf'), device=samples.device)

        if use_cosine_sim:
            samples_norm = F.normalize(samples, dim=-1)
            means_norm = F.normalize(means, dim=-1)
            dists = torch.matmul(samples_norm, means_norm.transpose(-1, -2))  # [H, N, K]
            buckets = dists.argmax(dim=-1)  # [H, N]
        else:
            samples_sq = samples.pow(2).sum(dim=-1, keepdim=True)  # [H, N, 1]
            for start in range(0, num_clusters, chunk_size):
                end = min(start + chunk_size, num_clusters)
                means_chunk = means[:, start:end, :]  # [H, chunk, D]
                means_sq = means_chunk.pow(2).sum(dim=-1).unsqueeze(1)  # [H, 1, chunk]
                dot = torch.matmul(samples, means_chunk.transpose(-1, -2))  # [H, N, chunk]
                dists = samples_sq + means_sq - 2 * dot  # [H, N, chunk]
                dists = dists  # lower = better

                # Find current min distances and update buckets
                new_min_dists, new_buckets = dists.min(dim=-1)  # [H, N]
                update_mask = new_min_dists < min_dists
                min_dists = torch.where(update_mask, new_min_dists, min_dists)
                buckets = torch.where(update_mask, new_buckets + start, buckets)

        # Compute new centroids
        bins = batched_bincount(buckets, minlength=num_clusters)  # [H, K]
        all_reduce_fn(bins)
        zero_mask = bins == 0  # [H, K]
        bins_safe = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros_like(means)  # [H, K, D]
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_safe, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = F.normalize(new_means, dim=-1)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


def batched_embedding(indices, embed):
    """
    Computes the differentiable embedding lookup using soft cluster assignments.
    """

    embed = embed.squeeze(0)  # Remove batch dimension if present (1, 10, 64) → (10, 64)
    indices = indices.view(-1, 1)  # Ensure shape (128, 1)

    # **Use `indices` Directly Instead of Recomputing**
    soft_weights = F.one_hot(indices.squeeze(-1).long(), num_classes=embed.shape[0]).float()
    soft_weights = soft_weights + (indices - indices.detach())  # **STE trick to keep gradients**

    # **Use Matmul for Differentiable Soft Embedding Lookup**
    quantized = torch.matmul(soft_weights, embed)  # Shape: (128, 64)

    return quantized

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

    def forward(self, z, chunk, epoch):
        eps = 1e-6
        latent_similarity_matrix = torch.mm(z, z.T)

        # if chunk == 0:
        print(f"simi_matrix max {latent_similarity_matrix.max()}, simi_matrix mean {latent_similarity_matrix.mean()}, min {latent_similarity_matrix.min()}")

        if chunk % 200 == 0:
            hist = torch.histc(latent_similarity_matrix.cpu().to(torch.float32), bins=10, min=0.0, max=15.0)
            print(hist)

        def calc_repel_loss(v, simi_matrix):
            # simi_matrix = torch.clamp(simi_matrix, -1 + eps, 1 - eps)
            # s_min, s_max = simi_matrix.min(), simi_matrix.max()
            # s_range = (s_max - s_min).clamp(min=eps)
            # simi_matrix = (simi_matrix - s_min) / s_range
            # if chunk == 0:
            #     print(f"simi_matrix max {simi_matrix.max()}, mean {simi_matrix.mean()}, min {simi_matrix.min()}")
            #     logger.info(f"simi_matrix max {simi_matrix.max()}, mean {simi_matrix.mean()}, min {simi_matrix.min()}")
            # hist = torch.histc(simi_matrix.cpu(), bins=10, min=0.0, max=1.0)
            identity = torch.eye(v.size(0), device=v.device, dtype=simi_matrix.dtype)
            # repel_loss = (torch.log(torch.cosh(simi_matrix - identity)) * (1 - identity)).mean()
            repel_loss = ((simi_matrix - identity) ** 2).mean()
            return repel_loss, simi_matrix

        def asymmetric_gaussian_loss(sim_matrix, mu=9.0, sigma_left=0.2, sigma_right=0.5):
            dtype = sim_matrix.dtype  # 保持されているdtypeを取得（float32 or float16）

            diff = sim_matrix - mu
            left_mask = diff < 0
            right_mask = ~left_mask

            # sigma をテンソルとして dtype を合わせる
            sigma_left = torch.tensor(sigma_left, dtype=dtype, device=sim_matrix.device)
            sigma_right = torch.tensor(sigma_right, dtype=dtype, device=sim_matrix.device)

            loss = torch.zeros_like(sim_matrix)

            loss[left_mask] = 1 - torch.exp(- (diff[left_mask] ** 2) / (2 * sigma_left ** 2)).to(dtype)
            loss[right_mask] = 1 - torch.exp(- (diff[right_mask] ** 2) / (2 * sigma_right ** 2)).to(dtype)

            return loss.mean(), sim_matrix

        def adaptive_bell_repel_loss(simi_matrix, mu=9.5, sigma=0.5):
            identity = torch.eye(simi_matrix.size(0), device=simi_matrix.device)
            simi_matrix = simi_matrix - identity
            # Gaussian bump centered at high similarity
            loss_matrix = torch.exp(-((simi_matrix - mu) ** 2) / (2 * sigma ** 2))
            return loss_matrix.mean(), simi_matrix

        def attract_high_sim(simi_matrix, threshold=7.5):
            identity = torch.eye(simi_matrix.size(0), device=simi_matrix.device)
            simi_matrix = simi_matrix * (1 - identity)  # zero diagonal
            target = 1.0 - simi_matrix / 10.0  # your attraction term
            loss_matrix = torch.where(simi_matrix > threshold, target, torch.zeros_like(simi_matrix))
            return loss_matrix.mean()

        def inverse_similarity_loss(simi_matrix):
            identity = torch.eye(simi_matrix.size(0), device=simi_matrix.device)
            simi_matrix = simi_matrix * (1 - identity)  # zero diagonal
            eps = 1e-6  # to prevent divide-by-zero
            loss = 1.0 / (simi_matrix + eps) ** 2
            return loss.mean(),  simi_matrix

        latent_repel_loss, sim_mat = inverse_similarity_loss(latent_similarity_matrix)
        attract_loss = attract_high_sim(sim_mat)
        attract_weight = 0.5  # 0.005

        final_loss = latent_repel_loss + attract_weight * attract_loss
        neg_loss = 1

        return final_loss, neg_loss, latent_repel_loss, latent_repel_loss


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

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        assert not (
                    use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        self.embed = nn.Parameter(embed, requires_grad=True)
    def reset_kmeans(self):
        self.initted.data.copy_(torch.Tensor([False]))

    @torch.jit.ignore
    def init_embed_(self, data, logger):
        # if self.initted[0] != 0:
        #     # print("return!!!!!")
        #     return
        print(f"++++++++++++++++ RUNNING int_embed !!! ++++++++++++++++++++++++++++++")
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
        )
        with torch.no_grad():
            self.embed.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size = torch.zeros(cluster_size.shape, device=cluster_size.device)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        return embed

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

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    import torch

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, logger=None, chunk_i=None, epoch=None, mode=None):
        x = x.float()
        needs_codebook_dim = x.ndim < 4
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        flatten = x.view(x.shape[0], -1, x.shape[-1])
        # if self.training and chunk_i % 320 == 0:  # mine
        # if chunk_i == 0:  # mine
        #     self.init_embed_(flatten, logger)  # ❌ Ensure this function does NOT detach tensors
        if mode == "init_kmeans_final":
            self.init_embed_(flatten, logger)  # ❌ Ensure this function does NOT detach tensors
            return 0
        embed = self.embed  # ✅ DO NOT detach embed
        init_cb = self.embed.clone().contiguous()  # ❌ No `.detach()`
        dist = (flatten.unsqueeze(2) - embed.unsqueeze(1)).pow(2).sum(dim=-1)  # Shape: (1, 128, 10)
        dist = -dist  # Negative similarity
        embed_ind_soft = F.softmax(dist, dim=-1)
        embed_ind_hard_idx = dist.argmax(dim=-1)
        embed_ind_hard = F.one_hot(embed_ind_hard_idx, num_classes=self.embed.shape[1]).float()
        embed_ind_one_hot = embed_ind_hard + (embed_ind_soft - embed_ind_soft.detach())
        embed_ind = torch.matmul(embed_ind_one_hot, torch.arange(embed_ind_one_hot.shape[-1], dtype=torch.float32,
                                                                 device=embed_ind_one_hot.device).unsqueeze(1))
        used_codebook_indices = torch.unique(embed_ind_hard_idx)
        used_codebook = self.embed[:, used_codebook_indices, :]
        embed_ind = embed_ind.view(1, -1, 1)
        quantize = batched_embedding(embed_ind, self.embed)  # ✅ Ensures gradients flow
        embed_ind = (embed_ind.round() - embed_ind).detach() + embed_ind
        device = flatten.device
        if self.training:
            distances = torch.randn(1, flatten.shape[1], self.codebook_size)  # Distance to each codebook vector
            temperature = 0.1  # Softmax temperature
            # Soft assignment instead of one-hot (fixes gradient flow)
            embed_probs = F.softmax(-distances / temperature, dim=-1)  # Softmax-based assignments
            embed_onehot = embed_probs  # Fully differentiable soft assignment
            embed_onehot = embed_onehot.squeeze(2) if embed_onehot.dim() == 4 else embed_onehot
            embed_onehot = embed_onehot.to(device)

        embed_ind = embed_ind.to(device)
        quantize_unique = torch.unique(quantize, dim=0)
        num_unique = quantize_unique.shape[0]

        if self.training:
            # embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            # with torch.no_grad():
            #     self.embed_avg = torch.lerp(self.embed_avg, embed_sum, 1 - self.decay)
            # cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            # embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            # self.embed.data.copy_(embed_normalized)
            # self.expire_codes_(x)
            # del distances, embed_probs, embed_onehot, embed_sum, cluster_size, embed_normalized
            del distances, embed_probs, embed_onehot
        torch.cuda.empty_cache()  # Frees unused GPU memory
        return quantize, embed_ind, dist, self.embed, flatten, init_cb, num_unique, used_codebook


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
            commitment_weight=0.25,  # using
            codebook_weight=0.01,  # using
            lamb_sil=0.00001,           # using
            lamb_cb=0.01,           # using
            lamb_div=0.01,           # using
            lamb_equiv_atom=1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False
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
            sample_codebook_temp=sample_codebook_temp
        )

        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.compute_contrastive_loss = ContrastiveLoss(dim, 136)

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


    def orthogonal_loss_fn(self, embed_ind, codebook, init_feat, latents, quantized, logger, epoch, chunk=0):
        embed_ind.to("cuda")
        codebook.to("cuda")
        init_feat.to("cuda")
        latents.to("cuda")
        quantized.to("cuda")
        # dist_matrix = torch.squeeze(torch.cdist(codebook, codebook, p=2) + 1e-6)  # Avoid zero distances
        # mask = ~torch.eye(dist_matrix.size(0), dtype=bool, device=dist_matrix.device)
        # dist_matrix_no_diag = dist_matrix[mask].view(dist_matrix.size(0), -1)

        def spread_loss(z, eps=1e-6):
            # z: [N, D] latent vectors
            var = torch.var(z, dim=0) + eps  # variance along each dimension
            inv_var = 1.0 / var
            return inv_var.mean()

        embed_ind_for_sil = torch.squeeze(embed_ind)
        latents_for_sil = torch.squeeze(latents)
        # sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, codebook.shape[-2])
        # final_loss, neg_loss, latent_repel_loss, cb_repel_loss
        two_repel_loss, div_nega_loss, repel_loss, cb_repel_loss = (
            self.compute_contrastive_loss(latents_for_sil, chunk, epoch))
        # spread_loss = spread_loss(latents_for_sil)
        if chunk == 0:
            logger.info(f"lat repel: {repel_loss}, spread: {spread_loss}")
        return (repel_loss, embed_ind, repel_loss, repel_loss, div_nega_loss, two_repel_loss, repel_loss)


    def commitment_loss(self, encoder_outputs, codebook, temperature=0.1):
        distances = torch.cdist(encoder_outputs, codebook)
        soft_assignments = F.softmax(-distances / temperature, dim=-1)
        quantized = torch.einsum('bn,nk->bk', soft_assignments, codebook)
        latent_loss = F.mse_loss(encoder_outputs.detach(), quantized, reduction='mean')
        codebook_loss = F.mse_loss(encoder_outputs, quantized.detach(), reduction='mean')
        return latent_loss, codebook_loss


    def forward(self, x, init_feat, logger, chunk_i=None, epoch=0, mode=None):
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
        if mode == "init_kmeans_final":
            self._codebook(x, logger, chunk_i, epoch, mode)
            return 0
        else:
            quantize, embed_ind, dist, embed, latents, init_cb, num_unique, used_cb = self._codebook(x, logger, chunk_i, epoch, mode)
        quantize = quantize.squeeze(0)
        x_tmp = x.squeeze(1).unsqueeze(0)
        quantize = x_tmp + (quantize - x_tmp)
        codebook = self._codebook.embed
        spread_loss, embed_ind, sil_loss, two_repel_loss, div_nega_loss, repel_loss, cb_repel_loss \
            = self.orthogonal_loss_fn(embed_ind, codebook, init_feat, x, quantize, logger, epoch, chunk_i)
        if len(embed_ind.shape) == 3:
            embed_ind = embed_ind[0]
        if embed_ind.ndim == 2:
            embed_ind = rearrange(embed_ind, 'b 1 -> b')
        elif embed_ind.ndim != 1:
            raise ValueError(f"Unexpected shape for embed_ind: {embed_ind.shape}")
        commit_loss, codebook_loss = self.commitment_loss(x.squeeze(), quantize.squeeze())
        # ---------------------------------------------
        # only repel losses at the first several steps
        # ---------------------------------------------
        if epoch > 10:
            loss = (self.commitment_weight * commit_loss + self.commitment_weight * codebook_loss + repel_loss)
        else:
            # loss = repel_loss + self.spread_weight * spread_loss
            loss = repel_loss
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        if only_one:
            if len(quantize.shape) == 3:
                # this line is executed
                quantize = rearrange(quantize, '1 b d -> b d')
            if len(embed_ind.shape) == 2:
                embed_ind = rearrange(embed_ind, 'b 1 -> b')
        return (quantize, embed_ind, loss, dist, embed, commit_loss, latents, div_nega_loss, x, commit_loss, sil_loss,
                num_unique, repel_loss, cb_repel_loss)