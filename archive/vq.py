import torch
import torch.distributed as distributed
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
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

import torch
from einops import rearrange, repeat

import torch
from einops import rearrange
import torch

import torch
from einops import rearrange, repeat



def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)

import torch
import time

def compute_squared_distances(samples, means):
    """
    Computes squared Euclidean distances efficiently.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    samples_sq = torch.sum(samples**2, dim=-1, keepdim=True)  # (num_codebooks, num_samples, 1)
    means_sq = torch.sum(means**2, dim=-1, keepdim=True).transpose(-2, -1)  # (num_codebooks, 1, num_clusters)
    cross_term = 2 * torch.matmul(samples, means.transpose(-2, -1))  # (num_codebooks, num_samples, num_clusters)

    dists = samples_sq + means_sq - cross_term  # (num_codebooks, num_samples, num_clusters)
    return dists

def soft_kmeans(samples, num_iters=100):
    num_codebooks, num_samples, dim = samples.shape
    device = samples.device
    args = get_args()
    num_clusters = args.codebook_size

    # Initialize centroids
    means = torch.randn(num_codebooks, num_clusters, dim, device=device, requires_grad=True)
    # ^^^^^^^^^^^^^ samples torch.Size([1, 10000, 64])
    # ^^^^^^^^^^^^^ means torch.Size([1, 100, 64])
    # ^^^^^^^^^^^^^ means to return torch.Size([1, 100, 64])
    for _ in range(num_iters):
        # Initialize accumulators for batch-wise mean updates
        accumulate_means = torch.zeros_like(means, device=device)  # Accumulate new centroids
        cluster_sizes = torch.zeros(num_codebooks, num_clusters, device=device)  # Track cluster sizes

        # Compute squared Euclidean distances
        # dists = torch.sum((samples.unsqueeze(2) - means.unsqueeze(1)) ** 2, dim=-1)
        dists = compute_squared_distances(samples, means)

        # Soft assignment using Softmax
        cluster_assignments = torch.nn.functional.softmax(-dists, dim=-1)  # [1, 100, 10000]

        # Compute weighted sum for new centroids
        batch_means = cluster_assignments.transpose(-1, -2) @ samples  # [num_codebooks, num_clusters, dim]
        accumulate_means += batch_means.detach()  # Detach to prevent graph growth
        cluster_sizes += cluster_assignments.sum(dim=1).detach()

        del dists, cluster_assignments, batch_means
        torch.cuda.empty_cache()  # Free GPU memory

        # Normalize centroids using accumulated counts
        means = accumulate_means / (cluster_sizes.unsqueeze(-1) + 1e-8)  # Avoid division by zero

    return means, cluster_sizes




def kmeans(
        samples,
        num_clusters,
        num_iters=20,
        use_cosine_sim=False,
        sample_fn=batched_sample_vectors,
        all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

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


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def mini_batch_kmeans(
        samples,
        num_clusters,
        batch_size=256,
        num_iters=100,
        logger=None,
        use_cosine_sim=False,
        all_reduce_fn=noop
):
    import time
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


# def batched_embedding(indices, embeds):
#     indices = indices.long()
#     batch, dim = indices.shape[1], embeds.shape[-1]
#     indices = repeat(indices, 'h b n -> h b n d', d=dim)
#     embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
#     return embeds.gather(2, indices)


def batched_embedding_old(indices, embeds):
    indices = torch.nn.functional.softmax(indices, dim=-1)  # Convert indices to soft assignments
    print("indices.shape")
    print(indices.shape)
    print("embeds.shape")
    print(embeds.shape)
    return indices @ embeds  # Weighted sum instead of discrete lookup

def batched_embedding(indices, embeds):
    def batched_embedding(indices, embed):
        """
        Fetch embeddings based on soft or hard indices.
        """
        print(f"indices shape {indices.shape}")  # Debugging
        print(f"embed.shape {embed.shape}")  # Debugging

        indices = indices.long()  # ✅ Convert to int64 for `.gather()`

        # Ensure correct shape for `.gather()`
        if indices.dim() == 2:  # (batch, 1) → (batch,)
            indices = indices.squeeze(1)

        print(f"indices after reshape: {indices.shape}")

        quantized = embed.gather(1, indices.unsqueeze(-1).expand(-1, -1, embed.shape[-1]))  # (batch, embed_dim)

        print(f"quantized.shape: {quantized.shape}")  # Expected: (batch, embed_dim)

        return quantized


def cluster_penalty_loss(feats, quantized, cluster_assignments): # init_feat, quantized, embed_ind
    # init_feat torch.Size([132, 7]), quantized torch.Size([132, 64]), embed_ind torch.Size([132, 1000])
    # --------------------------------------------------------------
    # distance matrix
    # --------------------------------------------------------------
    quantized = quantized.squeeze(0)  # Remove batch dim
    quantized = (quantized - quantized.detach()).detach() + quantized

    dist_matrix = torch.norm(quantized[:, None, :] - quantized[None, :, :], dim=-1)
    dist_matrix = dist_matrix / (dist_matrix.max() + 1e-6)  # Normalize distances

    # ------------------------------------
    # same ID mask
    # ------------------------------------
    num_classes = int(cluster_assignments.max().item()) + 1  # Number of unique clusters
    cluster_embedding = torch.nn.functional.one_hot(cluster_assignments.long().squeeze(), num_classes).float()
    same_id_mask = torch.mm(cluster_embedding, cluster_embedding.T)
    same_id_mask = same_id_mask - torch.eye(same_id_mask.shape[0], device=same_id_mask.device)
    same_id_mask = same_id_mask.clamp(min=0)  # Remove negative values

    # --------------------------------------------------------------
    # different feat mask
    # --------------------------------------------------------------

    # feat_dist_matrix = torch.cdist(feats.float(), feats.float(), p=1)
    # diff_feat_mask = 1 - (feat_dist_matrix == 0).float()
    # diff_feat_mask = (feats.unsqueeze(1) - feats.unsqueeze(0)).abs().sum(dim=-1) > 0
    # diff_feat_mask = diff_feat_mask.float()
    diff_feat_mask = torch.norm(feats.unsqueeze(1) - feats.unsqueeze(0), p=2, dim=-1) > 1e-3
    diff_feat_mask = diff_feat_mask.float()

    # --------------------------------------------------------------
    # Calculate penalty
    # --------------------------------------------------------------
    # Gaussian-based penalty function (or alternative)
    diff_feat_same_cluster_dist = dist_matrix * (diff_feat_mask * same_id_mask).detach()
    # non_zero_values = diff_feat_same_cluster_dist[diff_feat_same_cluster_dist != 0]
    # penalty = torch.logsumexp(-diff_feat_same_cluster_dist, dim=-1).mean()/100
    # penalty = (diff_feat_same_cluster_dist + 1e-6).log().mean()/10000
    # penalty = 10 * torch.logsumexp(-diff_feat_same_cluster_dist / 10, dim=-1).mean()
    penalty = (diff_feat_same_cluster_dist + 1e-6).log().mean()/1000000

    print(f"penalty {penalty}")
    return penalty


# this is the old temporary snippet
def compute_contrastive_loss(z, atom_types, name=None, margin=1.0, threshold=0.5, num_atom_types=100):
    """
    Contrastive loss to separate different atom types using embeddings.
    """
    # One-hot encode atom types
    # atom_types = torch.nn.functional.one_hot(atom_types.long(), num_atom_types).float()
    # print(f"🚨 atom_types: {atom_types}")
    # print(f"🚨 num_atom_types: {num_atom_types}")
    z = z.to("cuda")
    atom_types = atom_types.to("cuda")

    try:
        # print(f"Min atom_types: {atom_types.min()}, Max atom_types: {atom_types.max()}")
        atom_types = torch.nn.functional.one_hot(atom_types.long(), num_atom_types).float()
    except Exception as e:
        print("Error in one_hot:", e)
        print("Atom types values:", atom_types)
        raise

    # Compute pairwise distances for the z vectors
    with torch.no_grad():
        pairwise_distances = torch.cdist(z, z, p=2)

    pairwise_distances = torch.cdist(z, z, p=2)  # Pairwise Euclidean distances

    # Normalize the atom_types vectors
    atom_types = atom_types / (torch.norm(atom_types, dim=1, keepdim=True) + 1e-8)

    # Compute pairwise similarity for the atom_types
    pairwise_similarities = torch.mm(atom_types, atom_types.T)  # Cosine similarity

    same_type_mask = (pairwise_similarities >= 1).float()

    # Compute positive and negative losses
    positive_loss = same_type_mask * pairwise_distances ** 2

    # Compute negative loss (push different types apart)
    negative_loss = torch.exp(- (1.0 - same_type_mask) * pairwise_distances ** 2)
    # negative_loss = (1.0 - same_type_mask) * torch.clamp(margin - pairwise_distances, min=0.0) ** 2
    # print("same_type_mask shape:", same_type_mask.shape)
    # print("pairwise_distances shape:", pairwise_distances.shape)
    # print("Min index in mask:",
    #       torch.nonzero(same_type_mask).min().item() if same_type_mask.sum() > 0 else "No nonzero indices")
    # print("Max index in mask:",
    #       torch.nonzero(same_type_mask).max().item() if same_type_mask.sum() > 0 else "No nonzero indices")
    import gc
    torch.cuda.empty_cache()
    del pairwise_distances, pairwise_similarities, same_type_mask
    gc.collect()
    # Combine and return mean loss
    return (positive_loss + negative_loss).mean() / 10000

#
# def compute_contrastive_loss(z, atom_types, name, margin=0.2, threshold=0.5, num_atom_types=20):
#     """
#     Contrastive loss to separate different atom types using embeddings.
#     """
#     # One-hot encode atom types
#     z = z.to("cuda")
#     atom_types = atom_types.to("cuda")
#     try:
#         # print(f"Min atom_types: {atom_types.min()}, Max atom_types: {atom_types.max()}")
#         atom_types = torch.nn.functional.one_hot(atom_types.long(), num_atom_types + 1).float()
#     except Exception as e:
#         print("Error in one_hot:", e)
#         print("Atom types values:", atom_types)
#         raise
#     # Compute pairwise distances for the z vectors
#     pairwise_distances = torch.cdist(z, z, p=2)
#     # print(f"pairwise_distances.max() {pairwise_distances.max()}")
#     pairwise_distances = pairwise_distances / (pairwise_distances.max() + 1e-6)  # Normalize to [0,1]
#
#     # Normalize the atom_types vectors
#     atom_types = atom_types / (torch.norm(atom_types, dim=1, keepdim=True) + 1e-8)
#
#     # Compute pairwise similarity for the atom_types
#     pairwise_similarities = torch.mm(atom_types, atom_types.T)  # Cosine similarity
#     """
#     atom_types
#     tensor([ 1., 20.,  1.,  ...,  1., 10.,  1.], device='cuda:0')
#     pairwise_distances.mean() 0.0001491462899139151
#     pairwise_distances.min() 0.0
#     pairwise_distances.max() 0.00027295752079226077
#     pairwise_similarities.mean() 0.58056640625
#     pairwise_similarities.min() 0.0
#     negative 0.4193979799747467, positive 1.426142492988447e-08
# """
#     # Create the mask for "same type" based on similarity threshold
#     same_type_mask = (pairwise_similarities >= threshold).float()  # 1 if similarity >= threshold, else 0
#
#     # Compute positive loss (pull same types together)
#     positive_loss = same_type_mask * pairwise_distances ** 2
#
#     # Compute negative loss (push different types apart)
#     # negative_loss = (1.0 - same_type_mask) * torch.relu(margin - pairwise_distances) ** 2
#     negative_loss = torch.exp(- (1.0 - same_type_mask) * pairwise_distances ** 2)
#     # if name == 'atom':
#     #     print(f"negative {negative_loss.mean()}, positive {positive_loss.mean()}")
#     # Combine and return mean loss
#     return (positive_loss + negative_loss).mean() / 10000
#     # return negative_loss.mean()


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
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)


    def reset_kmeans(self):
        self.initted.data.copy_(torch.Tensor([False]))


    @torch.jit.ignore
    def init_embed_(self, data, logger=None):
        #
        # samples,
        # num_clusters,
        # batch_size=256,
        # num_iters=100,
        # logger=None,
        # embed, cluster_size = mini_batch_kmeans(
        #     data,
        #     self.codebook_size,
        #     256,
        #     self.kmeans_iters,
        #     logger
        #     # use_cosine_sim=True,
        #     # sample_fn=self.sample_fn,
        #     # all_reduce_fn=self.kmeans_all_reduce_fn
        # )
        args = get_args()
        embed, cluster_size = soft_kmeans(
            data,
            args.codebook_size
        )
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size = torch.zeros(cluster_size.shape, device=cluster_size.device)
        self.cluster_size.data.copy_(cluster_size)

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

    import torch
    import time
    import torch.nn.functional as F
    from einops import rearrange, einsum

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, logger=None):
        x = x.float()

        needs_codebook_dim = x.ndim < 4
        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        flatten = rearrange(x, 'h ... d -> h (...) d').clone()

        # Initialize codebook vectors
        self.init_embed_(flatten, logger)
        embed = self.embed
        init_cb = self.embed.detach().clone().contiguous()

        # Normalize Without Breaking Gradient Flow
        flatten = flatten / (torch.norm(flatten, dim=-1, keepdim=True) + 1e-8)
        embed = embed / (torch.norm(embed, dim=-1, keepdim=True) + 1e-8)

        # Compute Distance
        dist = (flatten.unsqueeze(2) - embed.unsqueeze(1)).pow(2).sum(dim=-1)  # Shape: (1, 128, 10)
        dist = -dist  # Negative similarity

        # Gumbel-Softmax for Soft Cluster Assignments
        tau = 1.0
        embed_ind_one_hot = F.gumbel_softmax(dist.view(dist.shape[0] * dist.shape[1], -1), tau=tau, hard=False)

        # **STE Trick to Maintain Differentiability**
        soft_indices = embed_ind_one_hot @ torch.arange(embed_ind_one_hot.shape[-1], dtype=torch.float32,
                                                        device=embed_ind_one_hot.device).unsqueeze(1)
        embed_ind = soft_indices + (embed_ind_one_hot - embed_ind_one_hot.detach()) @ torch.arange(
            embed_ind_one_hot.shape[-1], dtype=torch.float32, device=embed_ind_one_hot.device).unsqueeze(1)

        # Fix Shape for batched_embedding()
        embed_ind = embed_ind.view(1, -1, 1)
        quantize = batched_embedding(embed_ind, self.embed)

        # **Ensure Gradients Are Retained**
        quantize.retain_grad()
        x.retain_grad()
        flatten.retain_grad()

        return quantize, embed_ind, dist, self.embed, flatten, init_cb

    @torch.amp.autocast('cuda', enabled=False)
    def forward_old(self, x, logger=None):
        needs_codebook_dim = x.ndim < 4
        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')
        flatten = rearrange(x, 'h ... d -> h (...) d')

        # ----------------------------------------------------
        # set the initial codebook vectors by k-means
        # ----------------------------------------------------
        self.init_embed_(flatten, logger)
        embed = self.embed
        # init_cb = self.embed.detach().clone().contiguous()
        dist = -torch.cdist(flatten, self.embed, p=2)

        # ----------------------------------------------------
        # get codebook ID assigned
        # ----------------------------------------------------
        embed_ind = get_ind(dist)
        print(f"embed {embed.shape}, embed_ind {embed_ind.shape}")
        indices = torch.argmax(embed_ind, dim=-1, keepdim=True)  # Non-differentiable forward pass
        embed_ind = indices + (embed_ind - embed_ind.detach())  # Straight-through trick
        indices = embed_ind[:, :, 0]  # Keep the float tensor
        proxy_indices = indices.long()  # Convert to integer for forward pass
        embed_ind = proxy_indices + (indices - indices.detach())

        # Validate values
        if embed_ind.min() < 0:
            raise ValueError("embed_ind contains negative values.")
        if embed_ind.max() >= self.codebook_size:
            raise ValueError(
                f"embed_ind contains out-of-range values: max={embed_ind.max()}, codebook_size={self.codebook_size}")
        # embed_ind は, サンプルの所属するクラスタID、embed は kmeans からの centroid vectors
        # embed_ind torch.Size([1, 2977, 1000]) embed torch.Size([1, 1000, 64])
        #
        embed_ind = embed_ind.unsqueeze(0)
        quantize = batched_embedding(embed_ind, self.embed)
        # quantize = embed_ind @ self.embed  # Weighted sum of embeddings

        flatten.retain_grad()

        return quantize, embed_ind, dist, self.embed, flatten, embed


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

    @torch.jit.ignore
    # @torch.jit.unused
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = mini_batch_kmeans(
            data,
            self.codebook_size,
            256,
            self.kmeans_iters,
            use_cosine_sim=True,
        )
        #         #
        # def mini_batch_kmeans(
        #         samples,
        #         num_clusters,
        #         batch_size=256,
        #         num_iters=100,
        #         use_cosine_sim=False,
        #         all_reduce_fn=noop
        # ):
        # embed, cluster_size = kmeans(
        #     data,
        #     self.codebook_size,
        #     self.kmeans_iters,
        #     use_cosine_sim=True,
        #     sample_fn=self.sample_fn,
        #     all_reduce_fn=self.kmeans_all_reduce_fn
        # )
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
            commitment_weight=0.003,
            margin_weight=0.8,
            spread_weight=0.2,
            pair_weight=0.01,
            lamb_div_ele=2,
            lamb_div_bonds=1,
            lamb_div_aroma=1,
            lamb_div_ringy=1,
            lamb_div_h_num=1,
            lamb_div_elec_state=1,
            lamb_div_charge=1,
            lamb_sil=0.01,
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
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
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

    def fast_silhouette_loss(self, embeddings, embed_ind, num_clusters, target_non_empty_clusters=500):
        # Preprocess clusters to ensure the desired number of non-empty clusters
        # print_non_empty_cluster_count(embed_ind, embeddings, num_clusters, target_non_empty_clusters)
        # embed_ind = increase_non_empty_clusters(embed_ind, embeddings, num_clusters, target_non_empty_clusters)
        embed_ind.data.copy_(embed_ind)
        # Compute pairwise distances for all points
        pairwise_distances = torch.cdist(embeddings, embeddings)  # Shape: (N, N)
        inter_cluster_distances = []
        # Iterate over clusters
        for k in range(num_clusters):
            cluster_mask = (embed_ind == k)
            cluster_indices = cluster_mask.nonzero(as_tuple=True)[0]

            if cluster_indices.numel() == 0:
                continue  # Skip empty clusters
            # Compute inter-cluster distances
            other_mask = ~cluster_mask
            if other_mask.sum() > 0:
                other_distances = pairwise_distances[cluster_indices][:, other_mask]
                inter_cluster_distances.append(other_distances.mean())
            else:
                inter_cluster_distances.append(torch.tensor(float('inf'), device=embeddings.device))

        # Stack inter-cluster distances into a tensor
        b = torch.stack(inter_cluster_distances, dim=0) if inter_cluster_distances else torch.tensor([],
                                                                                                     device=embeddings.device)

        # Compute inter-cluster loss
        epsilon = 1e-6  # Small value to avoid division by zero
        b_normalized = b / (b.max() + epsilon)  # Normalize distances
        loss = -torch.mean(torch.log(b_normalized + epsilon))  # Maximize inter-cluster distances

        return embed_ind, loss



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


    def vq_codebook_regularization_loss(self, embed_ind, equivalence_groups, logger):
        """
        VQ Codebook Regularization Loss to ensure equivalent atoms (e.g., carbons in benzene)
        are assigned to the same discrete codebook entry while being memory efficient.
        """
        embed_ind = embed_ind.to('cuda')
        args = get_args()
        loss = torch.tensor(0.0, device=embed_ind.device, requires_grad=True)  # Keep it learnable
        num_groups = len(equivalence_groups)

        # Ensure correct shape to avoid unnecessary squeeze/unsqueeze calls
        embed_ind = embed_ind.view(-1)  # Flatten instead of squeeze()

        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Helps with debugging but may slow performance
        #
        # logger.info(f"Number of equivalence groups: {len(equivalence_groups)}")
        # logger.info(f"embed_ind.shape: {embed_ind.shape}")

        for group in equivalence_groups:
            if len(group) < 2:
                continue  # Skip if the group is too small

            group_tensor = torch.tensor(group, device=embed_ind.device)  # Move indices to the correct device
            # logger.info(f"group_tensor {group_tensor}" )
            if torch.max(group_tensor) >= embed_ind.shape[0]:
                # logger.warning(f"Skipping group {group} - Index out of bounds!")
                continue  # Skip this group

            # Extract cluster indices for equivalent atoms
            equivalent_cluster_indices = torch.index_select(embed_ind, 0, group_tensor)
            # logger.info(f"equivalent_cluster_indices {equivalent_cluster_indices}")
            # Ensure indices are within codebook bounds
            max_index = torch.max(equivalent_cluster_indices).item()
            if max_index >= args.codebook_size:
                # logger.warning(f"Skipping group {group} - Index {max_index} exceeds codebook size {args.codebook_size}")
                continue

            # Compute pairwise agreement loss efficiently
            equivalent_cluster_indices = equivalent_cluster_indices.unsqueeze(1).float()
            pairwise_diffs = torch.cdist(equivalent_cluster_indices, equivalent_cluster_indices, p=2)
            # logger.info(f"pairwise_diffs {pairwise_diffs}")
            # Update loss in a memory-friendly way
            loss = loss + torch.mean(pairwise_diffs) / (num_groups + 1e-6)  # Normalize to prevent large loss values

        return loss

    def simple_loss_function(self, quantized):
        quantized = quantized.squeeze(0)  # Remove batch dim if needed
        # dist_matrix = torch.cdist(quantized, quantized, p=2)  # Pairwise Euclidean distance
        loss = quantized.mean()  # Simple mean distance loss
        return loss


        # embed_ind, codebook, init_feat, latents, quantize, logger
    def orthogonal_loss_fn(self, embed_ind, t, init_feat, latents, quantized, logger, min_distance=0.5):
        # Normalize embeddings (optional: remove if not necessary)
        embed_ind.to("cuda")
        t.to("cuda")
        init_feat.to("cuda")
        latents.to("cuda")
        # quantized.to("cuda")
        t_norm = torch.norm(t, dim=1, keepdim=True) + 1e-6
        t = t / t_norm

        latents_norm = torch.norm(latents, dim=1, keepdim=True) + 1e-6
        latents = latents / latents_norm

        # # Pairwise distances
        # dist_matrix = torch.squeeze(torch.cdist(t, t, p=2) + 1e-6)  # Avoid zero distances
        #
        # # Remove diagonal
        # mask = ~torch.eye(dist_matrix.size(0), dtype=bool, device=dist_matrix.device)
        # dist_matrix_no_diag = dist_matrix[mask].view(dist_matrix.size(0), -1)

        # Debug: Log distance statistics
        # print(f"Min: {dist_matrix_no_diag.min().item()}, Max: {dist_matrix_no_diag.max().item()}, Mean: {dist_matrix_no_diag.mean().item()}")

        # Margin loss: Encourage distances >= min_distance
        # smooth_penalty = torch.nn.functional.relu(min_distance - dist_matrix_no_diag)
        # margin_loss = torch.mean(smooth_penalty)  # Use mean for better gradient scaling
        #
        # # Spread loss: Encourage diversity
        # spread_loss = torch.var(t)
        #
        # # Pair distance loss: Regularize distances
        # pair_distance_loss = torch.mean(torch.log(dist_matrix_no_diag))

        # sil loss
        embed_ind_for_sil = torch.squeeze(embed_ind)
        latents_for_sil = torch.squeeze(latents)
        equivalent_gtroup_list = self.fast_find_equivalence_groups(latents_for_sil)
                                                        # cluster_indices, embed_ind, equivalence_groups, logger
        equivalent_atom_loss = self.vq_codebook_regularization_loss(embed_ind, equivalent_gtroup_list, logger)

        embed_ind, sil_loss = self.fast_silhouette_loss(latents_for_sil, embed_ind_for_sil, t.shape[-2], t.shape[-2])
        # sil_loss = None
        # bond_num_div_loss = torch.tensor(1)
        # charge_div_loss = torch.tensor(1)
        # elec_state_div_loss = torch.tensor(1)
        # aroma_div_loss = torch.tensor(1)
        # ringy_div_loss = torch.tensor(1)
        # h_num_div_loss = torch.tensor(1)
        bond_num_div_loss = compute_contrastive_loss(quantized, init_feat[:, 1], "bond")
        charge_div_loss = compute_contrastive_loss(quantized, init_feat[:, 2], "charge")
        elec_state_div_loss = compute_contrastive_loss(quantized, init_feat[:, 3], "elec")
        aroma_div_loss = compute_contrastive_loss(quantized, init_feat[:, 4], "aroma")
        ringy_div_loss = compute_contrastive_loss(quantized, init_feat[:, 5], "ringy")
        h_num_div_loss = compute_contrastive_loss(quantized, init_feat[:, 6], "h_num")
        atom_type_div_loss = compute_contrastive_loss(quantized, init_feat[:, 0], "atom")
        # atom_type_div_loss = cluster_penalty_loss(init_feat, quantized, embed_ind)
        # atom_type_div_loss = self.simple_loss_function(quantized)

        return (1, 1, 1, atom_type_div_loss, bond_num_div_loss, aroma_div_loss,
                ringy_div_loss, h_num_div_loss, sil_loss, embed_ind, charge_div_loss, elec_state_div_loss, equivalent_atom_loss)

    import torch
    import time
    from einops import rearrange

    def forward(self, x, init_feat, logger, mask=None):
        only_one = x.ndim == 2
        x = x.to("cuda")

        if only_one:
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=heads)
         # quantize, embed_ind, dist, self.embed = self._codebook(x, logger)
        quantize, embed_ind, dist, embed, latents, init_cb = self._codebook(x, logger)
        quantize = quantize.squeeze(0)
        x_tmp = x.squeeze(1).unsqueeze(0)
        quantize = (quantize - quantize.detach()).detach() + quantize

        #
        # if self.training:
        #     quantize = x_tmp + (quantize - x_tmp)

        # loss = torch.zeros(1, device=device, requires_grad=True)
        loss = torch.tensor(0., device=device)  # ✅ Keeps loss in computation graph
        raw_commit_loss = torch.zeros(1, device=device)  # ✅ Ensure it’s part of computation
        detached_quantize = torch.zeros(1, device=device)  # ✅ Track gradients properly

        codebook = self._codebook.embed

        if self.orthogonal_reg_active_codes_only:
            unique_code_ids = torch.unique(embed_ind)
            codebook = torch.squeeze(codebook)[unique_code_ids]

        num_codes = codebook.shape[0]
        if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
            rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
            codebook = codebook[rand_ids]

        (margin_loss, spread_loss, pair_distance_loss, div_ele_loss, bond_num_div_loss, aroma_div_loss, ringy_div_loss,
         h_num_div_loss, silh_loss, embed_ind, charge_div_loss, elec_state_div_loss, equiv_atom_loss) = \
            self.orthogonal_loss_fn(embed_ind, codebook, init_feat, latents, quantize, logger)
        # embed_ind = embed_ind.reshape(embed_ind.shape[-1], 1)
        if embed_ind.ndim == 2:
            embed_ind = rearrange(embed_ind, 'b 1 -> b')
        elif embed_ind.ndim != 1:
            raise ValueError(f"Unexpected shape for embed_ind: {embed_ind.shape}")
        # print(f"div_ele_loss {div_ele_loss}")
        # loss += self.lamb_div_ele * div_ele_loss  # ✅ Keeps all loss contributions
        loss = (loss + self.lamb_div_ele * div_ele_loss
                + self.lamb_div_bonds * bond_num_div_loss + self.lamb_div_aroma * aroma_div_loss
                + self.lamb_div_charge * charge_div_loss + self.lamb_div_elec_state * elec_state_div_loss
                + self.lamb_div_ringy * ringy_div_loss + self.lamb_div_h_num * h_num_div_loss
                + self.lamb_equiv_atom * equiv_atom_loss)

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)
        if only_one:
            quantize = rearrange(quantize, 'b d -> b d')  # Keep batch dimension dynamic
            if len(embed_ind.shape) == 2:
                embed_ind = rearrange(embed_ind, 'b 1 -> b')

        return (quantize, embed_ind, loss, dist, embed, raw_commit_loss, latents, margin_loss, spread_loss,
                pair_distance_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss, aroma_div_loss,
                ringy_div_loss, h_num_div_loss, silh_loss, charge_div_loss, elec_state_div_loss, equiv_atom_loss)

