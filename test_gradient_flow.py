import torch
from vq import soft_kmeans

def check_gradient_flow():
    num_codebooks = 1
    num_samples = 10000
    dim = 64
    num_clusters = 100
    batch_size = 256
    num_iters = 10

    # Create synthetic data with gradients enabled
    samples = torch.randn(num_codebooks, num_samples, dim, device="cuda", requires_grad=True)

    # Run mini-batch soft K-Means
    means, cluster_sizes = soft_kmeans(samples, num_clusters, batch_size, num_iters)

    # Create a dummy loss (sum of means) to force gradient propagation
    loss = means.sum()
    loss.backward()

    # Check gradient flow
    print(f"Gradient mean of samples: {samples.grad.abs().mean().item()}")

check_gradient_flow()
