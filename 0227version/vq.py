import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# Utility functions
def l2norm(t, dim=-1, eps=1e-8):
    return F.normalize(t, dim=dim, eps=eps)

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def batched_bincount(indices, minlength):
    H, N = indices.shape
    bins = torch.zeros(H, minlength, device=indices.device, dtype=torch.long)
    for h in range(H):
        bins[h].scatter_add_(0, indices[h], torch.ones_like(indices[h]))
    return bins

# KMeans++ and iterative clustering

def kmeans(samples, num_clusters, use_cosine_sim=False, all_reduce_fn=lambda x: x):
    H, N, D = samples.shape
    means = torch.zeros(H, num_clusters, D, device=samples.device)
    means[:, 0] = samples[:, torch.randint(0, N, (1,))]

    for k in range(1, num_clusters):
        dists = 1 - (samples @ rearrange(means[:, :k], 'h k d -> h d k')) if use_cosine_sim else torch.cdist(samples, means[:, :k])
        probs = dists.min(dim=-1).values
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(probs, 1).squeeze()
        means[:, k] = samples[torch.arange(H), next_idx]

    for _ in range(30):
        dists = samples @ rearrange(means, 'h k d -> h d k') if use_cosine_sim else -torch.cdist(samples, means)
        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, num_clusters)
        bins_clamped = bins.masked_fill(bins == 0, 1)

        new_means = torch.zeros_like(means)
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=D), samples)
        new_means /= rearrange(bins_clamped, 'h k -> h k 1')

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(bins == 0, 'h k -> h k 1'), means, new_means)
    return means, bins

# Codebook class
class EuclideanCodebook(nn.Module):
    def __init__(self, dim, codebook_size, decay=0.1, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.register_buffer('embed', torch.empty(1, codebook_size, dim).uniform_(-1, 1))
        self.register_buffer('cluster_size', torch.zeros(1, codebook_size))
        self.register_buffer('embed_avg', self.embed.clone())

    def forward(self, x):
        x = rearrange(x, 'b n d -> 1 (b n) d')
        dist = torch.cdist(x, self.embed).pow(2)
        indices = dist.argmin(dim=-1)
        embed_onehot = F.one_hot(indices, self.codebook_size).float()
        quantized = torch.einsum('bnk,nkd->nbd', embed_onehot, self.embed)

        if self.training:
            cluster_size = embed_onehot.sum(dim=1)
            embed_sum = torch.einsum('bnk,nbd->nkd', embed_onehot, x)

            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * cluster_size
            self.embed_avg = self.decay * self.embed_avg + (1 - self.decay) * embed_sum
            normalized_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
            self.embed.copy_(self.embed_avg / normalized_size.unsqueeze(-1))

        return quantized.squeeze(0), indices.squeeze(0), dist.squeeze(0)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z):
        dist = torch.cdist(z, z)
        dynamic_threshold = torch.quantile(dist.flatten(), 0.1).item()
        repel_loss = torch.exp(-(dist - dynamic_threshold).pow(2) / (2 * 3**2)).mean()
        return repel_loss

# VectorQuantize wrapper
class VectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight=1.0, codebook_weight=0.1):
        super().__init__()
        self.codebook = EuclideanCodebook(dim, codebook_size)
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.contrastive = ContrastiveLoss()

    def forward(self, x):
        quantized, indices, dist = self.codebook(x)
        commit_loss = F.mse_loss(x, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x.detach())
        repel_loss = self.contrastive(x.squeeze(0))
        loss = self.commitment_weight * commit_loss + self.codebook_weight * codebook_loss + 0.1 * repel_loss
        return quantized, indices, loss
