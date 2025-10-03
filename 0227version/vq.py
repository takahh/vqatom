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


def kmeans(
        samples,
        num_clusters,
        use_cosine_sim=False,
        all_reduce_fn=noop
):
    num_iters = 30
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    # num_iters = 30

    # K-Means++ initialization
    means = torch.zeros((num_codebooks, num_clusters, dim), device=device, dtype=dtype)

    # Randomly select the first centroid
    means[:, 0] = samples[:, torch.randint(0, samples.shape[1], (1,))]
    samples = samples.to("cuda")
    means = means.to("cuda")
    with torch.no_grad():
        for k in range(1, num_clusters):
            # Compute full distance to all means (H, N, num_clusters)
            if use_cosine_sim:
                all_dists = 1 - (samples @ rearrange(means, 'h n d -> h d n'))  # (H, N, num_clusters)
            else:
                all_dists = torch.cdist(samples, means, p=2)  # (H, N, num_clusters)

            # Mask out distances beyond k
            masked_dists = all_dists[:, :, :k]  # Slice fixed shape
            min_dists = masked_dists.min(dim=-1).values
            probs = min_dists / min_dists.sum(dim=-1, keepdim=True)
            next_centroid_idx = torch.multinomial(probs, 1)
            means[:, k] = samples[:, next_centroid_idx.squeeze(-1)]

        # Iterative optimization
        for _ in range(num_iters):
            if use_cosine_sim:
                dists = samples @ rearrange(means, 'h n d -> h d n')
            else:
                dists = -torch.cdist(samples, means, p=2)

            buckets = torch.argmax(dists, dim=-1)
            bins = batched_bincount(buckets, minlength=num_clusters)
            all_reduce_fn(bins)

            zero_mask = bins == 0
            bins_min_clamped = bins.masked_fill(zero_mask, 1)

            new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

            new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
            new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
            all_reduce_fn(new_means)

            if use_cosine_sim:
                new_means = l2norm(new_means)

            means = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                means,
                new_means
            )
            buckets_flat = buckets.flatten()  # [H * N]
            del dists, buckets, bins_min_clamped, new_means, zero_mask

    return means, bins  # [H, K, D], [H, K]


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

    def forward(self, z, chunk, logger, codebook):
        import torch
        latent_dist_matrix = torch.cdist(z, z, p=2)
        sample = latent_dist_matrix.flatten()
        if sample.numel() > 1_000_000:
            sample = sample[torch.randperm(sample.numel())[:1_000_000]]
        dynamic_threshold = torch.quantile(sample, 0.1).item()

        if chunk % 32 == 0:
            import torch
            hist = torch.histc(latent_dist_matrix.cpu().to(torch.float32), bins=10, min=0.0, max=15.0)
            logger.info(hist.cpu().tolist())
            print(hist.cpu().tolist())

        import torch.nn.functional as F
        def repel_codebooks(codebook, sigma=1.0):
            dmat = torch.cdist(codebook, codebook, p=2)  # [1, K, K]
            K = codebook.size(1)
            mask = ~torch.eye(K, dtype=torch.bool, device=codebook.device)  # [K, K]
            mask = mask.unsqueeze(0)  # [1, K, K] to match repel shape
            repel = torch.exp(-dmat.pow(2) / (2 * sigma ** 2))  # [1, K, K]
            return repel[mask].mean()

        def calc_attract_loss(z, cb, temperature=1.0):
            """
            z         : [B, D] - latent vectors
            codebook  : [K, D] - codebook embeddings
            temperature : scaling for soft assignment sharpness

            Returns:
                attract_loss : scalar
            """
            # Step 1: Compute soft assignment weights
            distances = torch.cdist(z, cb, p=2)  # [B, K]
            soft_assign = F.softmax(-distances / temperature, dim=-1)  # [B, K]

            # Step 2: Compute weighted centroids (cluster means)
            soft_assign_T = soft_assign.transpose(0, 1)  # [K, B]
            weighted_sum = soft_assign_T @ z  # [K, D]
            cluster_mass = soft_assign_T.sum(dim=1, keepdim=True) + 1e-8
            cluster_mean = weighted_sum / cluster_mass  # [K, D]

            # Step 3: Distance between each z and its assigned centroids
            z_expand = z.unsqueeze(1)  # [B, 1, D]
            means_expand = cluster_mean.unsqueeze(0)  # [1, K, D]
            l2_dist = ((z_expand - means_expand) ** 2).sum(dim=-1)  # [B, K]

            # Step 4: Weighted attraction loss
            attract_loss = (soft_assign * l2_dist).sum(dim=-1).mean()  # scalar

            return attract_loss

        # def calc_attractive_loss(dmat, threshold=1):
        #     attract_mask = dmat < threshold
        #     attract_term = (dmat[attract_mask] ** 2).mean()
        #     return attract_term

        # Compute lower and upper quantiles for middle 40%
        lower_q = 0.95
        upper_q = 0.99
        lower_thresh = torch.quantile(sample, lower_q)
        upper_thresh = torch.quantile(sample, upper_q)
        center = (lower_thresh + upper_thresh) / 2

        def soft_middle_weight(dmat, low, high, sharpness=20.0):
            """
            Smooth weights near 1 for distances in [low, high], tapering outside.
            """
            w_low = torch.sigmoid(sharpness * (dmat - low))
            w_high = torch.sigmoid(sharpness * (high - dmat))
            return w_low * w_high  # shape same as dmat

        def soft_middle_weight_half(dmat, low, sharpness=20.0):
            """
            Smooth weights near 1 for distances in [low, high], tapering outside.
            """
            w_low = torch.sigmoid(sharpness * (dmat - low))
            return w_low  # shape same as dmat

        def calc_repel_loss_mid(dmat, low, high, center=2.0, sigma=3.0):
            weights = soft_middle_weight(dmat, low, high)  # soft mask in middle range
            bell = torch.exp(-(dmat - center) ** 2 / (2 * sigma ** 2))  # bell curve repel
            weighted_bell = weights * bell
            # Normalize by sum of weights to keep scale consistent
            return weighted_bell.sum() / (weights.sum() + 1e-8)

        def repel_from_zero(dmat, margin=1.0):
            margin = margin.detach()  # keep value, avoid weird gradient behavior
            loss = torch.relu(margin - dmat).mean()
            return loss

        def calc_repel_loss(dmat, center=2.0, sigma=3.0):
            bell = torch.exp(-(dmat - center) ** 2 / (2 * sigma ** 2))
            return bell.mean()

        latent_repel_loss_mid = calc_repel_loss_mid(latent_dist_matrix, lower_thresh, upper_thresh, center)
        repel_from_2 = calc_repel_loss(latent_dist_matrix)
        cb_loss = repel_codebooks(codebook)
        latent_repel_loss = repel_from_zero(latent_dist_matrix, lower_thresh) + cb_loss
        # latent_repel_loss = calc_repel_loss(latent_dist_matrix,) + repel_from_zero(latent_dist_matrix)

        attract_weight = 1  # or your preferred weight
        repel_weight = 1  # 0.005
        # final_loss = repel_weight * latent_repel_loss + attract_weight * attract_loss
        final_loss = repel_weight * latent_repel_loss_mid
        # latent_repel_loss += cb_loss
        # final_loss = repel_weight * latent_repel_loss
        # print(f"attract loss {attract_loss}, latent_repel_loss {latent_repel_loss}, ")
        neg_loss = 1
        # 最後の　latent_repel_loss　を使用中
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
    def init_embed_(self, data, mask_dict=None):
        cb_dict = {
            6: 4360, 7: 1760, 8: 1530, 9: 730, 17: 500, 16: 530, 35: 190,
            15: 100, 53: 85, 11: 50, 1: 47, 14: 22, 34: 27, 5: 43, 19: 19, 3: 10
        }
        print(f"++++++++++++++++ RUNNING init_embed !!! ++++++++++++++++++++++++++++++")
        embeds = []            # List to store embeddings
        cluster_sizes = []     # List to store cluster sizes
        for key in mask_dict.keys():
            print(f"data.shape: {data.shape}") # [1, 502283, 16]
            cbsize = int(self.codebook_size * cb_dict[key] / 10000)
            print(f"cbsize for key {key}: {cbsize}")
            masked_data = data[0][mask_dict[key]]
            masked_data = masked_data.unsqueeze(0)
            embed, cluster_size = kmeans(masked_data, cbsize)
            embeds.append(embed)
            cluster_sizes.append(cluster_size)
        # Combine all embeddings into a single tensor
        big_embed = torch.cat(embeds, dim=0)
        total_cluster_size = torch.cat(cluster_sizes, dim=0)
        with torch.no_grad():
            self.embed.copy_(big_embed)
        self.embed_avg.data.copy_(big_embed.clone())
        self.cluster_size = torch.zeros_like(total_cluster_size, device=total_cluster_size.device)
        self.cluster_size.data.copy_(total_cluster_size)
        self.initted.data.copy_(torch.tensor([True]))
        return big_embed


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
    def silhouette_score_torch(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Efficient silhouette score for large N using vectorized PyTorch ops.

        X: [N, D] float tensor on GPU
        labels: [N] int tensor on GPU
        Returns: scalar silhouette score (float)
        """
        X = X.squeeze()
        labels = labels.squeeze()

        N = X.shape[0]
        dists = torch.cdist(X, X, p=2)  # [N, N]

        unique_labels = labels.unique()
        K = unique_labels.size(0)

        # Create [N, N] mask for each cluster
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        eye = torch.eye(N, dtype=torch.bool, device=X.device)
        same_cluster_mask = label_eq & ~eye  # exclude self

        a_i = torch.zeros(N, device=X.device)
        for i in range(N):
            mask = same_cluster_mask[i]
            if mask.any():
                a_i[i] = dists[i, mask].mean()

        # b_i: minimum mean distance to points in other clusters
        b_i = torch.full((N,), float('inf'), device=X.device)
        for label in unique_labels:
            cluster_mask = labels == label  # [N]
            if cluster_mask.sum() == 0:
                continue
            cluster_dists = dists[:, cluster_mask]  # [N, N_label]
            cluster_mean = cluster_dists.mean(dim=1)  # [N]
            is_same_label = (labels == label).float()
            b_i = torch.where(is_same_label.bool(), b_i, torch.min(b_i, cluster_mean))

        sil = (b_i - a_i) / torch.maximum(a_i, b_i)
        sil[torch.isnan(sil)] = 0.0
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
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, mask_dict=None, logger=None, chunk_i=None, epoch=None, mode=None):
        x = x.float()
        if x.ndim < 4:
            x = rearrange(x, '... -> 1 ...')  # shape: (1, B, D)
        flatten = x.view(x.shape[0], -1, x.shape[-1])  # (1, B, D)

        if mode == "init_kmeans_final":
            # if mode == "init_kmeans_final" and epoch < 5:
            self.init_embed_(flatten, mask_dict)
            print(f"init_embed is done")
        embed = self.embed  # (1, K, D)  K: codebook size
        dist = torch.cdist(flatten.squeeze(0), embed.squeeze(0), p=2).pow(2).unsqueeze(0)  # (1, B, K) B: batch size
        # min_dists_sq, min_indices = torch.min(dist, dim=-1)  # (1, B)

        # hist = torch.histc(min_dists_sq.cpu().to(torch.float32), bins=10, min=0.0, max=15.0)
        # logger.info(hist.cpu().tolist())

        # dist = -dist  # negative distance = similarity
        # embed_ind_soft = F.softmax(dist, dim=-1)  # (1, B, K)
        # indices = torch.arange(embed.shape[1], dtype=torch.float32, device=embed.device)  # (K,)

        # # For monitoring codebook usage: use hard assignment
        # embed_ind_hard = embed_ind_soft.argmax(dim=-1).squeeze(0)  # (B,)
        # used_codebook_indices = torch.unique(embed_ind_hard)

        min_dists_sq, embed_ind_hard = torch.min(dist, dim=-1)  # (1, B)
        used_codebook_indices = torch.unique(embed_ind_hard.squeeze(0))

        # [1, B, K] -> hard indices: [B]
        # embed_ind_hard = embed_ind_soft.argmax(dim=-1).squeeze(0)  # (B,)

        # One-hot encode hard assignments
        embed_ind_hard_onehot = F.one_hot(embed_ind_hard, num_classes=self.embed.shape[1]).float()  # (B, K)

        # Re-insert batch dim to match shape: [1, B, K]
        # embed_ind_hard_onehot = embed_ind_hard_onehot.unsqueeze(0)
        embed_ind_hard_onehot = embed_ind_hard_onehot.squeeze(0)  # from (1, B, K) → (B, K)

        # Straight-through estimator: combine hard and soft
        # embed_ind_onehot = embed_ind_hard_onehot + (embed_ind_soft - embed_ind_soft.detach())

        # Soft quantized vector (with gradient)
        # print(f"embed_ind_hard_onehot {embed_ind_hard_onehot.shape}, self.embed {self.embed.shape}")
        # quantize = torch.einsum('nbk,nkd->nbd', embed_ind_hard_onehot, self.embed.squeeze())  # (1, B, D)
        quantize = torch.einsum('bk,kd->bd', embed_ind_hard_onehot, self.embed.squeeze(0))

        quantize_unique = torch.unique(quantize, dim=1)
        num_unique = quantize_unique.shape[1]
        embed_ind = embed_ind_hard  # If you want to explicitly name it

        # ------------
        # sil score
        # ------------
        from sklearn.metrics import silhouette_score
        from sklearn.utils import resample
        import numpy as np

        if mode == "init_kmeans_final":
            # Save full arrays
            np.savez(f"./naked_embed_{epoch}.npz", embed=embed.cpu().detach().numpy())
            np.savez(f"./naked_latent_{epoch}.npz", latent=x.cpu().detach().numpy())

            # Sample 1000 points for silhouette score calculation
            x_np = x.cpu().squeeze().detach().numpy()
            labels_np = embed_ind.cpu().squeeze().detach().numpy()

            x_sample, labels_sample = resample(
                x_np, labels_np, n_samples=self.samples_latent_in_kmeans, random_state=42
            )
            # x_sample and labels_sample are NumPy arrays after resample
            x = torch.from_numpy(x_sample).float().to('cuda')  # move to GPU if needed
            labels = torch.from_numpy(labels_sample).long().to('cuda')

            # sil_score = silhouette_score(x_sample, labels_sample)
            sil_score = self.silhouette_score_torch(x.squeeze(), labels.squeeze())
            print(f"Silhouette Score (subsample): {sil_score:.4f}")
            logger.info(f"Silhouette Score (subsample): {sil_score:.4f}")

            logger.info(
                f"-- epoch {epoch}: used_codebook_indices.shape {used_codebook_indices.shape} -----------------")
            print(
                f"-- epoch {epoch}: used_codebook_indices.shape {used_codebook_indices.shape} -----------------")
            return 0

        if self.training and epoch < 30:
            temperature = 0.1
            distances = torch.randn(1, flatten.shape[1], self.codebook_size, device=flatten.device)
            embed_probs = F.softmax(-distances / temperature, dim=-1)  # (1, B, K)
            embed_onehot = embed_probs  # [1, B, K]

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            with torch.no_grad():
                self.embed_avg = torch.lerp(self.embed_avg, embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
            cluster_size = cluster_size * self.cluster_size.sum()
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')

            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        torch.cuda.empty_cache()
        return quantize, embed_ind, dist, self.embed, flatten, self.embed.clone(), num_unique, self.embed[:,
                                                                                               used_codebook_indices, :]


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
        sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, codebook.shape[-2])
        # final_loss, neg_loss, latent_repel_loss_mid, cb_loss, latent_repel_loss
        two_repel_loss, div_nega_loss, repel_loss_from_2, cb_loss, repel_loss_mid_high = (
            self.compute_contrastive_loss(latents_for_sil, chunk, logger, codebook))
        # spread_loss = spread_loss(latents_for_sil)
        # if chunk == 0:
        #     logger.info(f"lat repel: {repel_loss}, spread: {spread_loss}")
        return (repel_loss_from_2, embed_ind, sil_loss, repel_loss_from_2, div_nega_loss, two_repel_loss, cb_loss, repel_loss_mid_high)

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
        if mode == "init_kmeans_final":
            self._codebook(x, mask_dict, logger, chunk_i, epoch, mode)
            return 0
        else:
            #     ( x, logger=None, chunk_i=None, epoch=None, mode=None):
            quantize, embed_ind, dist, embed, latents, init_cb, num_unique, used_cb = self._codebook(x, mask_dict, logger, chunk_i, epoch, mode)
        quantize = quantize.squeeze(0)
        x_tmp = x.squeeze(1).unsqueeze(0)
        quantize = x_tmp + (quantize - x_tmp)
        codebook = self._codebook.embed
        # repel_loss_from_2, embed_ind, sil_loss, repel_loss_from_2, div_nega_loss, two_repel_loss, cb_loss, repel_loss_mid_high)
        repel_loss_from_2, embed_ind, sil_loss, repel_loss_from_2, div_nega_loss, mid_repel_loss, cb_repel_loss, two_repel_loss \
            = self.orthogonal_loss_fn(embed_ind, codebook, init_feat, x, quantize, logger, epoch, chunk_i)
        if len(embed_ind.shape) == 3:
            embed_ind = embed_ind[0]
        if embed_ind.ndim == 2:
            embed_ind = embed_ind.flatten()
        elif embed_ind.ndim != 1:
            raise ValueError(f"Unexpected shape for embed_ind: {embed_ind.shape}")

        # ------- change if needed ---------
        EPOCH_TO_SHIFT = 2
        # ----------------------------------

        # # ------------DELETE THIS SOON !!!!!!!!! ----------------------
        # commit_loss, codebook_loss = self.commitment_loss(x.squeeze(), codebook)
        # # ------------DELETE THIS SOON !!!!!!!!! ----------------------
        # if epoch >= EPOCH_TO_SHIFT:
        commit_loss, codebook_loss = self.commitment_loss(x.squeeze(), codebook)
        # else:
        #     commit_loss = torch.tensor(1)
        # ---------------------------------------------
        # only repel losses at the first several steps
        # ---------------------------------------------
        args = get_args()
        alpha = 1 / ((epoch + 1) ** 2)
        repel_loss = mid_repel_loss
        repel_loss *= alpha
        if epoch < 3:
            # logger.info("~~~~~~~ using repel loss ~~~~~~~~~~~")
            loss = repel_loss
        elif epoch >= 3:
            # self._codebook.embed.requires_grad_(False)
            print(f"repel {repel_loss}") # or some decaying schedule
            # logger.info("~~~~~~~ using commit loss ~~~~~~~~~~~")
            beta = 0.0001
            loss = beta * (commit_loss) + repel_loss

            # loss = 0.1 * commit_loss + 0.1 * codebook_loss + two_repel_loss
            # print(f"commit loss {self.commitment_weight * commit_loss}")
        # else:
        #     # loss = (self.commitment_weight * commit_loss + self.commitment_weight * codebook_loss)
        # else:
        #     # loss = repel_loss + self.spread_weight * spread_loss
        #     print(f"commit loss {self.commitment_weight * commit_loss} two repel {two_repel_loss}")
        #     loss = two_repel_loss
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        if only_one:
            if len(quantize.shape) == 3:
                # this line is executed
                quantize = rearrange(quantize, '1 b d -> b d')
            if len(embed_ind.shape) == 2:
                embed_ind = rearrange(embed_ind, 'b 1 -> b')
        # delete this soon ------------------------------------
        # div_nega_loss = commit_loss
        # sil_loss = commit_loss
        # repel_loss = commit_loss
        # cb_repel_loss = commit_loss
        # delete this soon ------------------------------------

        return (quantize, embed_ind, loss, dist, embed, commit_loss, latents, div_nega_loss, x, commit_loss, sil_loss,
                num_unique, repel_loss, cb_repel_loss)