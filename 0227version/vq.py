from distutils.fancy_getopt import neg_alias_re

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from numba.misc.help.inspector import commit
# from scipy.version import commit_count
from statsmodels.stats.dist_dependence_measures import distance_covariance
from torch import nn, einsum
from torch.ao.quantization import quantize
from torch.amp import autocast
from torch.onnx.symbolic_opset9 import pairwise_distance, one_hot
from einops import rearrange, repeat
from torch.distributions import MultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal

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


def get_ind(logits):

    # # Add Gumbel noise for stochastic sampling
    # gumbels = -torch.empty_like(logits).exponential_().log()
    # noisy_logits = (logits + gumbels) / temperature
    #
    # # Get the indices of the maximum values
    # indices = noisy_logits.argmax(dim=dim)
    #
    # # Convert indices to one-hot encoding (maintains differentiability upstream)
    # one_hot = torch.nn.functional.one_hot(indices, num_classes=logits.size(dim)).float()
    # if temperature < 1e-5:
    temperature = 1e-7

    # print("------ in gamble sample 1 -------")
    # print(f"one_hot.requires_grad: {logits.requires_grad}")
    # print(f"one_hot.grad_fn: {logits.grad_fn}")
    # print(logits)
    # one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=dim)

    def deterministic_hard(logits):
        # Directly compute the argmax of logits (without softmax or temperature)
        hard_out = torch.argmax(logits, dim=-1, keepdim=True)

        # Create a one-hot tensor
        one_hot_out = torch.zeros_like(logits).scatter_(-1, hard_out, 1.0)

        # Straight-through estimator: pretend the one-hot output is continuous for backprop
        return one_hot_out - logits.detach() + logits

    one_hot = deterministic_hard(logits)

    # print("------ in gamble sample 1 -------")
    # print(f"one_hot.requires_grad: {one_hot.requires_grad}")
    # print(f"one_hot.grad_fn: {one_hot.grad_fn}")
    #
    # print(one_hot)
    return one_hot

#
# def gumbel_sample(logits, dim=-1, temperature=1.0):
#     print("------ in gamble sample -1 -------")
#     if temperature == 0:
#         return logits.argmax(dim=dim)  # Deterministic sampling when temperature is 0
#
#     print("------ in gamble sample 0 -------")
#     print(f"embed_ind.requires_grad: {logits.requires_grad}")
#     print(f"embed_ind.grad_fn: {logits.grad_fn}")
#     # Sample Gumbel noise
#     gumbels = -torch.empty_like(logits).exponential_().log()
#
#     # Add noise to logits and scale by temperature
#     noisy_logits = (logits + gumbels) / temperature
#
#     print("------ in gamble sample 1 -------")
#     print(f"embed_ind.requires_grad: {logits.requires_grad}")
#     print(f"embed_ind.grad_fn: {logits.grad_fn}")
#
#     # Return the indices of the maximum values
#     return noisy_logits.argmax(dim=dim)


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


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    # print(f"{local_samples} local_samples")
    # print(f"{num} num")

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, '... -> 1 ...')


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


from torch.distributions import MultivariateNormal

import torch
from torch.distributions import MultivariateNormal


def gmm(
        samples,
        cluster_size=500,  # Fixed number of clusters
        num_iters=30,
        sample_fn=None,  # Optional sampling function
        all_reduce_fn=lambda x: x  # No-op by default
):
    """
    Optimized Gaussian Mixture Model (GMM) with a fixed number of clusters.

    Vectorized over clusters using batched distributions and einsum to compute
    the M-step without Python loops. Log probabilities are computed in chunks
    to reduce memory usage.

    Args:
        samples: Tensor of shape [num_codebooks, num_samples, dim].
        cluster_size: Fixed number of clusters (default=1500).
        num_iters: Number of EM iterations.
        sample_fn: Optional sampling function.
        all_reduce_fn: Optional reduction function for distributed training.

    Returns:
        means: Tensor of cluster means.
        bins: Tensor containing cluster assignments for each sample.
    """
    num_codebooks, num_samples, dim = samples.shape
    num_clusters = cluster_size

    # Initialize means with k-means++ logic per codebook
    means = torch.empty(num_codebooks, num_clusters, dim, dtype=samples.dtype, device=samples.device)
    for h in range(num_codebooks):
        means[h, 0] = samples[h, torch.randint(0, num_samples, (1,))]
        for k in range(1, num_clusters):
            # Compute squared distances between each sample and the current centers.
            dists = torch.cdist(samples[h].unsqueeze(0), means[h, :k].unsqueeze(0), p=2).squeeze(0) ** 2
            min_dists, _ = torch.min(dists, dim=1)
            prob = min_dists / (min_dists.sum() + 1e-9)
            chosen_idx = torch.multinomial(prob, 1)
            means[h, k] = samples[h, chosen_idx]

    # Initialize covariances and weights
    covariances = torch.eye(dim, device=samples.device).expand(num_codebooks, num_clusters, dim, dim).clone()
    weights = torch.full((num_codebooks, num_clusters), 1.0 / num_clusters, device=samples.device, dtype=samples.dtype)

    epsilon = 1e-6  # Regularization for numerical stability
    chunk_size = 10  # Adjust this value based on your memory constraints

    for iter in range(num_iters):
        # Add regularization to ensure positive-definiteness.
        cov_eps = covariances + torch.eye(dim, device=samples.device).unsqueeze(0).unsqueeze(0) * epsilon

        # Compute log probabilities in chunks to avoid OOM
        log_probs_chunks = []
        for start in range(0, num_clusters, chunk_size):
            end = min(start + chunk_size, num_clusters)
            mvn_chunk = MultivariateNormal(
                loc=means[:, start:end, :],
                covariance_matrix=cov_eps[:, start:end, :, :]
            )
            # samples.unsqueeze(2) has shape [C, N, 1, D]
            # mvn_chunk.log_prob broadcasts over the cluster dimension: result [C, N, chunk_size]
            log_probs_chunk = mvn_chunk.log_prob(samples.unsqueeze(2))
            log_probs_chunks.append(log_probs_chunk)
        log_probs = torch.cat(log_probs_chunks, dim=2)  # [C, N, K]

        # Incorporate cluster weights and compute responsibilities
        log_probs = log_probs + weights.log().unsqueeze(1)
        responsibilities = torch.softmax(log_probs, dim=-1)

        # M-step: Update means
        resp_sums = responsibilities.sum(dim=1, keepdim=True)  # [C, 1, K]
        weighted_samples = responsibilities.unsqueeze(-1) * samples.unsqueeze(2)  # [C, N, K, D]
        means = weighted_samples.sum(dim=1) / (resp_sums.squeeze(1).unsqueeze(-1) + 1e-9)

        # M-step: Update covariances via vectorized einsum.
        diff = samples.unsqueeze(2) - means.unsqueeze(1)  # [C, N, K, D]
        covariances = torch.einsum('cnk,cnkd,cnke->ckde', responsibilities, diff, diff)
        covariances = covariances / (resp_sums.squeeze(1).unsqueeze(-1).unsqueeze(-1) + 1e-9)

    # Final cluster assignments based on highest responsibility
    bins = torch.argmax(responsibilities, dim=-1)
    return means, bins


import torch
from einops import rearrange, repeat


def kmeans(
        samples,
        num_clusters,
        num_iters=100,
        use_cosine_sim=False,
        all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    num_iters = 30

    # K-Means++ initialization
    means = torch.zeros((num_codebooks, num_clusters, dim), device=device, dtype=dtype)

    # Randomly select the first centroid
    means[:, 0] = samples[:, torch.randint(0, samples.shape[1], (1,))]
    samples = samples.to("cuda")
    means = means.to("cuda")

    for k in range(1, num_clusters):

        if use_cosine_sim:
            dists = 1 - (samples @ rearrange(means[:, :k], 'h n d -> h d n'))
        else:
            dists = torch.cdist(samples, means[:, :k], p=2)

        min_dists = dists.min(dim=-1).values  # Minimum distance to existing centroids
        probs = min_dists / min_dists.sum(dim=-1, keepdim=True)  # Probabilities proportional to distance
        next_centroid_idx = torch.multinomial(probs, 1)  # Sample next centroid based on probabilities
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

    return means, bins


def mini_batch_kmeans(
        samples,
        num_clusters,
        batch_size=256,
        num_iters=100,
        logger=None,
        use_cosine_sim=False,
        all_reduce_fn=noop
):

    # Get basic dimensions and move tensors to GPU
    num_codebooks, num_samples, dim = samples.shape[0], samples.shape[1], samples.shape[-1]

    samples = samples.to('cuda')
    dtype, device = samples.dtype, samples.device

    # K-Means++ initialization
    means = torch.zeros((num_codebooks, num_clusters, dim), device=device, dtype=dtype)
    first_idx = torch.randint(0, num_samples, (1,), device=device)
    means[:, 0] = samples[:, first_idx].squeeze(1)
    for k in range(1, num_clusters):
        if use_cosine_sim:
            dists = 1 - (samples @ rearrange(means[:, :k], 'h n d -> h d n'))
        else:
            dists = torch.cdist(samples, means[:, :k], p=2) ** 2  # ** Use squared Euclidean distance **

        # More efficient way to compute min distances without calling .values
        min_dists, _ = torch.min(dists, dim=-1)

        # Normalize to avoid overflow issues in probabilities
        min_dists = min_dists + 1e-10  # Small epsilon to prevent division errors
        probs = min_dists / min_dists.sum(dim=-1, keepdim=True)

        # Faster centroid selection
        next_centroid_idx = torch.multinomial(probs, 1, replacement=False)

        means[:, k] = samples[torch.arange(num_codebooks, device=device), next_centroid_idx.squeeze(-1)]

    # Initialize counts to track the number of points assigned to each centroid
    counts = torch.zeros((num_codebooks, num_clusters), device=device, dtype=torch.int64)

    # Iterative optimization with mini-batches
    means_squared = (means ** 2).sum(dim=-1, keepdim=True)  # Precompute squared means for fast distance calculations

    for _ in range(num_iters):
        # Precompute squared batch for fast distance computation
        batch_indices = torch.randint(0, num_samples, (batch_size,), device=device)
        batch = samples[:, batch_indices]
        batch_squared = (batch ** 2).sum(dim=-1, keepdim=True)  # Precompute squared batch

        if use_cosine_sim:
            dists = batch @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -2 * torch.matmul(batch, means.transpose(1, 2)) + means_squared.transpose(1, 2) + batch_squared

        assignments = torch.argmax(dists, dim=-1)  # Assign clusters

        batch_sums = torch.zeros_like(means)
        batch_sums.scatter_add_(1, repeat(assignments, 'h b -> h b d', d=dim), batch)
        batch_counts = batched_bincount(assignments, minlength=num_clusters)

        old_counts = counts.clone().to(dtype)
        counts = counts + batch_counts

        counts_float = counts.to(dtype)
        update_mask = (batch_counts > 0).unsqueeze(-1)

        if update_mask.any():
            new_means = (rearrange(old_counts, 'h n -> h n 1') * means + batch_sums) / rearrange(counts_float,
                                                                                                 'h n -> h n 1')
            means = torch.where(update_mask, new_means, means)

        all_reduce_fn(counts)
        all_reduce_fn(means)

        if use_cosine_sim:
            means = l2norm(means)

    return means, counts


# def batched_embedding(indices, embed):
#     indices = indices.squeeze(1)  # Remove extra dimension if present
#     dim = embed.shape[-1]  # Get embedding dimension
#
#     # Ensure indices is 2D before repeating
#     indices = indices.reshape(indices.shape[0], -1)  # (h, n)
#
#     # Ensure indices are int64
#     indices = indices.long()
#
#     # Correct shape before repeating
#     indices = repeat(indices, 'h n -> h n d', d=dim)  # Ensure proper shape
#
#     return torch.gather(embed, 1, indices)
# def batched_embedding(indices, embed):
#     print(f"indices shape {indices.shape}")
#     print(f"embed.shape {embed.shape}")
#     embed = torch.squeeze(embed)
#     indices = indices.squeeze(1)
#     indices = indices.long()
#     # Convert indices to one-hot encoding
#     one_hot = F.one_hot(indices, num_classes=embed.shape[0]).float()  # (batch, num_clusters)
#     quantized = torch.matmul(one_hot, embed)  # (batch, embedding_dim)
#
#     return quantized
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



def feat_elem_divergence_loss(embed_ind, atom_types, num_codebooks=1500, temperature=0.02):

    def soft_one_hot(indices, num_classes, temperature=0.1):  # Increased default temperature for stability
        class_indices = torch.arange(num_classes, device=indices.device).float()
        indices = indices.float() / (indices.max() + 1e-6)
        logits = -(indices.unsqueeze(-1) - class_indices) ** 2 / temperature
        soft_assignments = torch.softmax(logits, dim=-1)
        return soft_assignments

    # embed ind を確率に変更
    embed_one_hot = soft_one_hot(embed_ind, num_classes=num_codebooks)
    unique_atom_numbers = torch.unique(atom_types, sorted=True)
    atom_types_mapped = torch.searchsorted(unique_atom_numbers.contiguous(), atom_types.contiguous())
    atom_type_one_hot = torch.nn.functional.one_hot(atom_types_mapped, num_classes=len(unique_atom_numbers)).float().detach()
    soft_assignments = torch.softmax(embed_one_hot / temperature, dim=-1)
    co_occurrence = torch.einsum("ni,nj->ij", [soft_assignments, atom_type_one_hot])
    co_occurrence_normalized = co_occurrence / (co_occurrence.sum(dim=1, keepdim=True) + 1e-6)
    row_entropy = -torch.sum(co_occurrence_normalized * torch.log(co_occurrence_normalized + 1e-6), dim=1)
    sparsity_loss = row_entropy.mean()
    return sparsity_loss


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, latent_dim, atom_feat_dim, margin=0.5, temperature=0.1, init_sigmoid_base=0.5):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(margin))
        self.temperature = temperature
        self.sigmoid_base = nn.Parameter(torch.tensor(init_sigmoid_base))
        self.layer_norm_z = nn.LayerNorm(latent_dim)
        self.layer_norm_atom = nn.LayerNorm(latent_dim)

    def forward(self, z, atom_types, epoch, logger):
        eps = 1e-6

        # # Add stronger noise early in training to break symmetry
        # if epoch < 5:
        #     z = z + 0.1 * torch.randn_like(z)

        # Normalize z to control magnitude and prevent similarity collapse
        z = F.normalize(z, p=2, dim=1, eps=eps)

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(z, z.T)
        similarity_matrix = torch.clamp(similarity_matrix, -1 + eps, 1 - eps)

        # Normalize atom types for cosine similarity
        atom_types_fp32 = atom_types.float()
        atom_types_norm = F.normalize(atom_types_fp32, p=2, dim=1, eps=eps)
        type_similarity_matrix = torch.mm(atom_types_norm, atom_types_norm.T)
        type_similarity_matrix = torch.clamp(type_similarity_matrix, -1 + eps, 1 - eps)

        # Normalize similarity matrices to [0, 1]
        s_min, s_max = similarity_matrix.min(), similarity_matrix.max()
        s_range = (s_max - s_min).clamp(min=eps)
        similarity_matrix = (similarity_matrix - s_min) / s_range

        t_min, t_max = type_similarity_matrix.min(), type_similarity_matrix.max()
        t_range = (t_max - t_min).clamp(min=eps)
        type_similarity_matrix = (type_similarity_matrix - t_min) / t_range

        # Repel loss to prevent collapse
        identity = torch.eye(z.size(0), device=z.device, dtype=similarity_matrix.dtype)
        repel_loss = ((similarity_matrix - identity) ** 2).mean()

        # Contrastive loss: positive & negative based on type similarity
        pos_loss = torch.mean((1 - similarity_matrix) * type_similarity_matrix)
        neg_mask = F.relu(type_similarity_matrix - 0.8)
        neg_loss = torch.mean(F.relu(similarity_matrix - 0.9) * neg_mask)
        # contrastive_loss = pos_loss + neg_loss + eps
        contrastive_loss = 100 * neg_loss

        # Logging
        logger.info(
            f"nega loss: {neg_loss.item():.4f}, pos loss: {pos_loss.item():.4f}, repel: {repel_loss.item():.4f}")
        # print("similarity_matrix:\n", similarity_matrix[:5, :5].detach().cpu())
        # print("type_similarity_matrix:\n", type_similarity_matrix[:5, :5].detach().cpu())
        # print("z std:", z.std().item(), "mean norm:", z.norm(dim=1).mean().item())
        #
        # May07 23-16-43: nega loss: 0.0035, pos loss: 0.0522, repel: 0.7744
        # May07 23-16-44: nega loss: 0.0033, pos loss: 0.0508, repel: 0.7709
        # Final loss with stronger repel term early on
        repel_weight = 0.5
        # repel_weight = 0.5 if epoch < 10 else 0.1
        final_loss = contrastive_loss + repel_weight * repel_loss

        return final_loss, neg_loss


import torch.nn.functional as F

def print_non_empty_cluster_count(embed_ind, embeddings, num_clusters, target_non_empty_clusters=500, min_cluster_size=5):
    # Count the size of each cluster
    cluster_sizes = [(k, (embed_ind == k).sum().item()) for k in range(num_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size (descending)
    # Identify non-empty clusters
    non_empty_clusters = [k for k, size in cluster_sizes if size > 0]
    current_non_empty_count = len(non_empty_clusters)
    print(f"Increasing clusters: {current_non_empty_count}, target {target_non_empty_clusters}...")


def increase_non_empty_clusters(embed_ind, embeddings, num_clusters, target_non_empty_clusters=500, min_cluster_size=5):
    # Count the size of each cluster
    cluster_sizes = [(k, (embed_ind == k).sum().item()) for k in range(num_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size (descending)

    # Identify non-empty clusters
    non_empty_clusters = [k for k, size in cluster_sizes if size > 0]
    current_non_empty_count = len(non_empty_clusters)

    # クラスタ数が設定値以下の場合だけ実行
    if current_non_empty_count < target_non_empty_clusters:
        print(f"Increasing clusters from {current_non_empty_count} to {target_non_empty_clusters}...")
        new_embed_ind = embed_ind.clone()

        # Determine how many clusters need to be added
        clusters_to_add = target_non_empty_clusters - current_non_empty_count

        # Start splitting the largest clusters
        for k, size in cluster_sizes:
            if clusters_to_add == 0:
                break
            if size > min_cluster_size:  # Only split clusters with enough points
                cluster_mask = (embed_ind == k)
                cluster_points = embeddings[cluster_mask]

                # Split cluster into two
                midpoint = cluster_points.mean(dim=0)
                distances = torch.norm(cluster_points - midpoint, dim=1)
                split_mask = distances > distances.median()  # Split into two groups

                # Create a new cluster for the split points
                new_cluster_id = embed_ind.max().item() + 1  # Ensure unique cluster ID
                new_embed_ind[cluster_mask] = torch.where(
                    split_mask,
                    torch.tensor(new_cluster_id, device=embed_ind.device, dtype=embed_ind.dtype),
                    k
                )
                clusters_to_add -= 1
                # print(f"Created new cluster {new_cluster_id} with size {(new_embed_ind == new_cluster_id).sum().item()}")

        # print(f"Final non-empty clusters: {target_non_empty_clusters - clusters_to_add}")
        return new_embed_ind
    else:
        # print(f"No need to increase clusters; current count is {current_non_empty_count}.")
        return embed_ind


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
        # print(f"Using EuclidCodebook {num_codebooks}")
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        # print(f"init_fn: {init_fn}")
        embed = init_fn(num_codebooks, codebook_size, dim)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
                    use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        # use_ddp = True
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        # if learnable_codebook:
        self.embed = nn.Parameter(embed, requires_grad=True)
        # else:
        #     self.register_buffer('embed', embed)

        # print("self.embed.grad_fn in Eu init")  # Must be True
        # print(self.embed.grad_fn)  # Must be True

    def reset_kmeans(self):
        self.initted.data.copy_(torch.Tensor([False]))


    @torch.jit.ignore
    def init_embed_for_plot(self, data, logger):
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            # use_cosine_sim=True,
            # sample_fn=self.sample_fn,
            # all_reduce_fn=self.kmeans_all_reduce_fn
        )
        return embed

    @torch.jit.ignore
    def init_embed_(self, data, logger):
        if self.initted:
            return
        print(f"++++++++++++++++ RUNNING int_embed !!! ++++++++++++++++++++++++++++++")
        # samples,
        # num_clusters,
        # num_iters=100,
        # use_cosine_sim=False,
        # all_reduce_fn=noop
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            # use_cosine_sim=True,
            # sample_fn=self.sample_fn,
            # all_reduce_fn=self.kmeans_all_reduce_fn
        )
        # self.embed.data.copy_(embed)
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
    import torch.nn.functional as F
    from einops import rearrange

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, logger=None, epoch=None):
        x = x.float()
        needs_codebook_dim = x.ndim < 4
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        flatten = x.view(x.shape[0], -1, x.shape[-1])  # Keeps gradient connection
        # Initialize codebook vectors (Ensure it does not detach)
        if self.training and epoch == 1:  # mine
            self.init_embed_(flatten, logger)  # ❌ Ensure this function does NOT detach tensors
        embed = self.embed  # ✅ DO NOT detach embed
        init_cb = self.embed.clone().contiguous()  # ❌ No `.detach()`
        # Compute Distance Without Breaking Gradients
        dist = (flatten.unsqueeze(2) - embed.unsqueeze(1)).pow(2).sum(dim=-1)  # Shape: (1, 128, 10)
        dist = -dist  # Negative similarity
        # Compute soft assignment
        embed_ind_soft = F.softmax(dist, dim=-1)
        # Convert to hard assignments
        embed_ind_hard_idx = dist.argmax(dim=-1)
        # Access embeddings correctly with shape (1, 1000, 64)
        embed_ind_hard = F.one_hot(embed_ind_hard_idx, num_classes=self.embed.shape[1]).float()
        # Apply STE trick
        embed_ind_one_hot = embed_ind_hard + (embed_ind_soft - embed_ind_soft.detach())
        # **Compute Soft Indices (Weighted Sum)**
        embed_ind = torch.matmul(embed_ind_one_hot, torch.arange(embed_ind_one_hot.shape[-1], dtype=torch.float32,
                                                                 device=embed_ind_one_hot.device).unsqueeze(1))
        # **Fix Shape for batched_embedding()**
        embed_ind = embed_ind.view(1, -1, 1)
        quantize = batched_embedding(embed_ind, self.embed)  # ✅ Ensures gradients flow
        embed_ind = (embed_ind.round() - embed_ind).detach() + embed_ind

        if self.training:  # mine
            distances = torch.randn(1, flatten.shape[1], self.codebook_size)  # Distance to each codebook vector
            temperature = 0.1  # Softmax temperature
            # Soft assignment instead of one-hot (fixes gradient flow)
            embed_probs = F.softmax(-distances / temperature, dim=-1)  # Softmax-based assignments
            embed_onehot = embed_probs  # Fully differentiable soft assignment
            embed_onehot = embed_onehot.squeeze(2) if embed_onehot.dim() == 4 else embed_onehot
            device = flatten.device
            embed_ind = embed_ind.to(device)

            quantize_unique = torch.unique(quantize, dim=0)
            num_unique = quantize_unique.shape[0]
            print(f"Number of unique cb vectors: {num_unique}")

            embed_onehot = embed_onehot.to(device)
            # Compute the sum of assigned embeddings
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            with torch.no_grad():
                self.embed_avg = torch.lerp(self.embed_avg, embed_sum, 1 - self.decay)
            # Compute normalized cluster sizes
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            # Normalize the codebook embeddings
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            # Expire unused codes (optional step)
            self.expire_codes_(x)
            del distances, embed_probs, embed_onehot, embed_sum, cluster_size, embed_normalized
            torch.cuda.empty_cache()  # Frees unused GPU memory

        #  ORIGINAL VQGRAPH version
        # if self.training:
        #     cluster_size = embed_onehot.sum(dim=1)
        #     # print(cluster_size.shape)
        #     self.all_reduce_fn(cluster_size)
        #     self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)
        #
        #     embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
        #     self.all_reduce_fn(embed_sum.contiguous())
        #     self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)
        #
        #     cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
        #     # print(cluster_size.shape)
        #     embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        #     self.embed.data.copy_(embed_normalized)
        #     self.expire_codes_(x)

        return quantize, embed_ind, dist, self.embed, flatten, init_cb, num_unique


class CosineSimCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            num_codebooks=1,
            kmeans_init=False,
            kmeans_iters=10,
            sync_kmeans=True,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=0,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        self.initted = False

    @torch.jit.ignore
    # @torch.jit.unused
    def init_embed_(self, data):
        if self.initted:
            return
        # embed, cluster_size = mini_batch_kmeans(
        #     data,
        #     self.codebook_size,
        #     256,
        #     self.kmeans_iters,
        #     use_cosine_sim=True,
        # )
        #         #
        # def mini_batch_kmeans(
        #         samples,
        #         num_clusters,
        #         batch_size=256,
        #         num_iters=100,
        #         use_cosine_sim=False,
        #         all_reduce_fn=noop
        # ):
        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn
        )
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        # this line means init_embed is run only once
        self.initted.data.copy_(torch.Tensor([True]))


    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

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


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)
        # -----------------------
        # initialize codebook
        # optimization of codebook done here
        # -----------------------
        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)
        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind, dist, self.embed, flatten

    # main class


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
            commitment_weight=0.01,  # using
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

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook
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

    # def compute_contrastive_loss(quantized, atom_types):
    #     """
    #     Compute contrastive loss efficiently while keeping gradients for backpropagation.
    #     """
    #     # Compute pairwise distances (keeps requires_grad)
    #     pairwise_distances = torch.cdist(quantized, quantized, p=2)  # Shape: (N, N)
    #
    #     # Enable memory efficiency using automatic mixed precision
    #     with torch.autocast(device_type="cuda", dtype=torch.float16):
    #         # Create mask for positive pairs (same atom type)
    #         same_type_mask = (atom_types.unsqueeze(0) == atom_types.unsqueeze(1)).to(pairwise_distances.dtype)
    #
    #         # Ensure only positive pairs contribute to loss
    #         num_pairs = same_type_mask.sum() + 1e-6  # Keeps tensor learnable
    #
    #         # Compute loss while preserving gradients
    #         positive_loss = (same_type_mask * pairwise_distances ** 2).sum() / num_pairs
    #
    #     return positive_loss  # Loss remains differentiable
    import torch
    import torch.nn.functional as F
    import torch
    import torch.nn.functional as F

    def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, temperature=1.0, margin=0.1):
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Get soft cluster assignments with temperature control
        if embed_ind.dim() == 1:
            embed_ind = embed_ind.unsqueeze(1)  # (N, 1)

        logits = embed_ind.expand(-1, num_clusters)  # (N, K)
        cluster_assignments = F.softmax(logits / temperature, dim=-1)  # (N, K)

        # Get hard assignments for centroid calculation
        hard_assignments = torch.zeros_like(cluster_assignments).scatter_(
            1, cluster_assignments.argmax(dim=1, keepdim=True), 1.0
        )
        # Compute cluster centroids using hard assignments for stability
        cluster_sums = hard_assignments.T @ embeddings  # (K, D)
        cluster_sizes = hard_assignments.sum(dim=0, keepdim=True).T  # (K, 1)
        cluster_sizes = cluster_sizes.clamp(min=1.0)  # Avoid division by very small numbers
        centroids = cluster_sums / cluster_sizes  # (K, D)

        # Compute distances to assigned cluster (a)
        assigned_clusters = cluster_assignments.argmax(dim=1)  # (N,)
        assigned_centroids = centroids[assigned_clusters]  # (N, D)
        a = torch.norm(embeddings - assigned_centroids, dim=1)  # (N,)

        # Compute nearest different cluster distance (b)
        # Create a mask to ignore the assigned cluster
        mask = torch.ones((batch_size, num_clusters), device=device)
        mask.scatter_(1, assigned_clusters.unsqueeze(1), 0)

        # Calculate distances to all centroids
        all_distances = torch.cdist(embeddings, centroids)  # (N, K)
        masked_distances = all_distances * mask + (1 - mask) * 1e6  # Set assigned cluster distance high

        # Get the nearest different cluster
        b, _ = torch.min(masked_distances, dim=1)  # (N,)

        # Calculate silhouette score with a margin to encourage separation
        max_dist = torch.max(a, b) + 1e-6
        silhouette = (b - a) / max_dist - margin
        # Apply a smoothing function to make the loss more gradient-friendly
        loss = 1 - torch.mean(torch.tanh(silhouette))

        return loss

    def fast_find_equivalence_groups(self, latents):

        from collections import defaultdict
        """
        Finds equivalence groups where two vectors belong to the same group if they are identical.

        Uses hashing for fast lookup instead of O(N^2) comparisons.

        Args:
            latents (torch.Tensor): Tensor of shape (N, D), where N is the number of vectors and D is the embedding dimension.

        Returns:
            equivalence_groups (list of list): A list of index groups where each sublist contains indices of identical vectors.
        """
        hash_map = defaultdict(list)  # Dictionary to store groups by unique vector

        # Convert tensor rows into hashable tuples and store indices
        for idx, vector in enumerate(latents):
            key = tuple(vector.tolist())  # Convert tensor to a hashable tuple
            hash_map[key].append(idx)

        # Extract only groups with more than one element
        equivalence_groups = [group for group in hash_map.values() if len(group) > 1]

        return equivalence_groups



        # embed_ind, codebook, init_feat, latents, quantize, logger
    def orthogonal_loss_fn(self, embed_ind, codebook, init_feat, latents, quantized, logger, min_distance=0.5, epoch=0):
        # Normalize embeddings (optional: remove if not necessary)
        embed_ind.to("cuda")
        codebook.to("cuda")
        init_feat.to("cuda")
        latents.to("cuda")
        quantized.to("cuda")
        # latents_norm = torch.norm(latents, dim=1, keepdim=True) + 1e-6
        # latents = latents / latents_norm

        # # Pairwise distances
        dist_matrix = torch.squeeze(torch.cdist(codebook, codebook, p=2) + 1e-6)  # Avoid zero distances
        #
        # # Remove diagonal
        mask = ~torch.eye(dist_matrix.size(0), dtype=bool, device=dist_matrix.device)
        dist_matrix_no_diag = dist_matrix[mask].view(dist_matrix.size(0), -1)
        #
        # # Debug: Log distance statistics
        # # print(f"Min: {dist_matrix_no_diag.min().item()}, Max: {dist_matrix_no_diag.max().item()}, Mean: {dist_matrix_no_diag.mean().item()}")
        #
        # Margin loss: Encourage distances >= min_distance
        smooth_penalty = torch.nn.functional.relu(min_distance - dist_matrix_no_diag)
        margin_loss = torch.mean(smooth_penalty)  # Use mean for better gradient scaling

        # Spread loss: Encourage diversity
        spread_loss = torch.var(codebook)

        # Pair distance loss: Regularize distances
        pair_distance_loss = torch.mean(torch.log(dist_matrix_no_diag))

        # sil loss
        # def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, target_non_empty_clusters=500):

        embed_ind_for_sil = torch.squeeze(embed_ind)
        latents_for_sil = torch.squeeze(latents)

        sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, codebook.shape[-2])
        equivalent_gtroup_list = self.fast_find_equivalence_groups(latents_for_sil)
        # print(equivalent_gtroup_list[:10])
                                                        # cluster_indices, embed_ind, equivalence_groups, logger
        # equivalent_atom_loss = self.vq_codebook_regularization_loss(embed_ind, equivalent_gtroup_list, logger)
        # embed_ind, sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, t.shape[-2], t.shape[-2])
        # atom_type_div_loss = torch.tensor(1)
        # bond_num_div_loss = torch.tensor(1)
        # charge_div_loss = torch.tensor(1)
        # elec_state_div_loss = torch.tensor(1)
        # aroma_div_loss = torch.tensor(1)
        # ringy_div_loss = torch.tensor(1)
        feat_div_loss, div_nega_loss = self.compute_contrastive_loss(latents_for_sil, init_feat, epoch, logger)

        # Should not be None
        # equidist_cb_loss = compute_duplicate_nearest_codebook_loss(latents, codebook)

        # atom_type_div_loss = compute_contrastive_loss(quantized, init_feat[:, 0])
        # bond_num_div_loss = compute_contrastive_loss(quantized, init_feat[:, 1])
        # charge_div_loss = compute_contrastive_loss(quantized, init_feat[:, 2])
        # elec_state_div_loss = compute_contrastive_loss(quantized, init_feat[:, 3])
        # aroma_div_loss = compute_contrastive_loss(quantized, init_feat[:, 4])
        # ringy_div_loss = compute_contrastive_loss(quantized, init_feat[:, 5])
        # h_num_div_loss = compute_contrastive_loss(quantized, init_feat[:, 6])
        # div_loss_list = [atom_type_div_loss, bond_num_div_loss, charge_div_loss, elec_state_div_loss, aroma_div_loss, ringy_div_loss, h_num_div_loss]
        # print(f"sil_loss {sil_loss}")
        # print(f"equivalent_atom_loss {equivalent_atom_loss}")
        # print(f"atom_type_div_loss {atom_type_div_loss}")
        return (spread_loss, embed_ind, sil_loss, feat_div_loss, div_nega_loss)


    def commitment_loss(self, encoder_outputs, codebook, temperature=0.1):
        # Compute distances between encoder outputs and codebook vectors
        distances = torch.cdist(encoder_outputs, codebook)

        # Soft assignment with temperature
        soft_assignments = F.softmax(-distances / temperature, dim=-1)

        # Straight-through estimator for discrete selection
        quantized = torch.einsum('bn,nk->bk', soft_assignments, codebook)

        # print(f"0 quantized Gradients: {quantized}")
        # print(f"0 encoder_outputs Gradients: {encoder_outputs}")

        # Encourage encoder outputs to be close to selected codebook vectors
        latent_loss = F.mse_loss(encoder_outputs.detach(), quantized, reduction='mean')
        # Encourage codebook vectors to be close to encoder outputs
        codebook_loss = F.mse_loss(encoder_outputs, quantized.detach(), reduction='mean')

        # print(f"1 quantized Gradients: {quantized}")
        # print(f"1 encoder_outputs Gradients: {encoder_outputs}")
        # print(f"commitment_loss: {commitment_loss}")
        # Entropy regularization to prevent codebook collapse 罰だから、プラスであってほしいい
        # entropy_loss = -torch.mean(
        #     torch.sum(soft_assignments * torch.log(soft_assignments + 1e-8), dim=-1)
        # )
        # print(f"entropy_loss: {entropy_loss}")
        # entropy_loss = torch.mean(
        #     torch.sum(soft_assignments * torch.log(soft_assignments + 1e-8), dim=-1)
        # )
        # Combine losses with tunable weights
        # commitment_loss = latent_loss + codebook_loss
        print(f"latent_loss: {latent_loss}, codebook_loss: {codebook_loss}")
        """
        commitment_loss: 0.001366406329907477
        entropy_loss: 8.031081199645996"""
        # total_loss = commitment_loss + 0.01 * entropy_loss
        # total_loss = commitment_loss + 0.00001 * entropy_loss

        return latent_loss, codebook_loss

    import torch
    import torch.nn.functional as F
    from einops import rearrange

    def forward(self, x, init_feat, logger, epoch=None):
        only_one = x.ndim == 2
        x = x.to("cuda")

        if only_one:
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size = (
            x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        )
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # print("Input x shape:", x.shape)

        if is_multiheaded:
            x = rearrange(x, 'b n (h d) -> h b n d', h=heads)

        # Debug before codebook step
        # print(f"Before codebook: x shape = {x.shape}")

        # print(f"x: {x}")
        # print(f"init_feat: {init_feat}")
        quantize, embed_ind, dist, embed, latents, init_cb, num_unique = self._codebook(x, logger, epoch)
        # print(f"After codebook: embed_ind shape = {embed_ind.shape}, unique IDs = {torch.unique(embed_ind)}")
        """
        quantize: データの数だけ存在する、離散化されたクラスタ中心
        embed_ind: データの数だけ存在する、所属するクラスタID
        embed: クラスタ中心ベクトル
        latents: データの数だけ存在する、潜在変数ベクトル
        init_cb: 
        """

        quantize = quantize.squeeze(0)
        x_tmp = x.squeeze(1).unsqueeze(0)

        if self.training:
            quantize = x_tmp + (quantize - x_tmp)

        quantize_unique = torch.unique(quantize, dim=0)
        num_unique = quantize_unique.shape[0]
        # print(f"Number of unique cb vectors: {num_unique}")
        # commit_loss = F.mse_loss(torch.squeeze(quantize), torch.squeeze(x), reduction='mean') \
        #               / torch.tensor(quantize.shape[1], dtype=torch.float, device=quantize.device)

        # if self.orthogonal_reg_active_codes_only:
        #     unique_code_ids = torch.unique(embed_ind)
        #     print(f"Unique codebook IDs used: {unique_code_ids}")
        #     codebook = torch.squeeze(self._codebook.embed)[unique_code_ids]

        codebook = self._codebook.embed

        # print(f"embed_ind 0: {embed_ind}")
        spread_loss, embed_ind, sil_loss, feat_div_loss, div_nega_loss = self.orthogonal_loss_fn(embed_ind, codebook, init_feat, x, quantize,
                                                                   logger, epoch)
        # print(f"embed_ind: {embed_ind}")
        if len(embed_ind.shape) == 3:
            embed_ind = embed_ind[0]
        if embed_ind.ndim == 2:
            embed_ind = rearrange(embed_ind, 'b 1 -> b')
        elif embed_ind.ndim != 1:
            raise ValueError(f"Unexpected shape for embed_ind: {embed_ind.shape}")
        """
        /vqatom/0227version/vq.py:1457: UserWarning: Using a target size (torch.Size([6290, 1, 64])) that is different to the input size (torch.Size([1, 6290, 64])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
          commit_loss = F.mse_loss(quantize.detach().squeeze(1), x.squeeze(0))
        /vqatom/0227version/vq.py:1458: UserWarning: Using a target size (torch.Size([6290, 1, 64])) that is different to the input size (torch.Size([1, 6290, 64])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
          codebook_loss = F.mse_loss(quantize.squeeze(1), x.detach().squeeze(0))"""
        commit_loss, codebook_loss = self.commitment_loss(x.squeeze(), quantize.squeeze())
        """
        # feat_div_loss: 0.0001748909562593326 * 100
        commit_loss: 0.00524178147315979     * 0.01  """
        # loss = self.commitment_weight * commit_loss + self.lamb_cb * codebook_loss
        print(f"commit loss {self.commitment_weight * commit_loss}, div nega {self.lamb_div * div_nega_loss}, sil loss {self.lamb_sil * sil_loss}")
        # if epoch < 5:
        #     print(f"epoch is less than 10")
        #     loss = (self.lamb_div * feat_div_loss)
        #     # loss = (self.commitment_weight * commit_loss)
        # else:
        #     print(f"epoch is more than 10 !!")
        # loss = self.lamb_div * feat_div_loss
        #
        #     commitment_weight=0.01,  # using
        #     lamb_div=0.01,           # using
        # commit loss 7.9770e-07, div nega 2.502e-05, sil loss 4.6171e-06
        loss = (self.commitment_weight * commit_loss + self.lamb_div * feat_div_loss)
        #
        # loss = (self.commitment_weight * commit_loss + self.lamb_div * feat_div_loss
        #         + self.lamb_cb * codebook_loss + self.lamb_sil * sil_loss)

        # quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        # if self.accept_image_fmap:
        #     print("fmap ===========")
        #     quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
        #     embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)
        # commit_loss = torch.tensor(1)
        if only_one:
            if len(quantize.shape) == 3:
                # this line is executed
                quantize = rearrange(quantize, '1 b d -> b d')
            if len(embed_ind.shape) == 2:
                embed_ind = rearrange(embed_ind, 'b 1 -> b')

        """
        (quantize, emb_ind, loss, dist, embed, commit_loss, latents, spread_loss, detached_quantize,
         x, init_cb, sil_loss, commit_loss) = quantize_output"""
        return quantize, embed_ind, loss, dist, embed, commit_loss, latents, div_nega_loss, x, commit_loss, sil_loss, num_unique