import torch
from vq import EuclideanCodebook

def check_gradient_flow_euclidean():
    num_codebooks = 1
    num_samples = 10000
    dim = 64
    num_clusters = 100
    batch_size = 256

    # Create input tensor with gradients enabled
    x = torch.randn(num_codebooks, num_samples, dim, device="cuda", requires_grad=True)

    # Initialize EuclideanCodebook
    model = EuclideanCodebook(dim, num_clusters, num_codebooks=num_codebooks).cuda()

    # Run forward pass
    quantize, embed_ind, dist, embed, flatten, embed_final = model(x)

    # Compute a dummy loss to force gradient propagation
    loss = quantize.sum()
    loss.backward()

    # Check if x retains gradients
    print(f"Gradient mean of x: {x.grad.abs().mean().item()}")

check_gradient_flow_euclidean()
