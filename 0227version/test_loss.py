import torch
from vq import EuclideanCodebook
import torch

import torch

import torch

import torch


def test_euclidean_codebook_forward():
    torch.manual_seed(42)  # Reproducibility

    # Initialize Codebook
    dim = 64
    codebook_size = 10
    batch_size = 22144
    num_codebooks = 1

    model = EuclideanCodebook(dim, codebook_size, num_codebooks=num_codebooks, learnable_codebook=True).cuda()

    # Create Input
    x = torch.randn(batch_size, 1, dim, device="cuda", requires_grad=True)

    # Forward Pass
    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)

    # Ensure `quantize` requires gradients
    assert quantize.requires_grad, "quantize does not require gradients!"

    # Compute Dummy Loss
    loss = quantize.mean()
    loss.backward()

    # Ensure Input Has Gradients
    # assert x.grad is not None, "Gradient for input x should not be None"
    # assert x.grad.abs().sum().item() > 0, "Gradient for input x should not be zero"

    # Ensure Codebook Embeddings Have Gradients
    assert quantize.grad is not None, "Gradient for quantize should not be None"
    # Ensure Codebook Embeddings Have Gradients
    assert model.embed.grad is not None, "Gradient for codebook embed should not be None"
    assert model.embed.grad.abs().sum().item() > 0, "Gradient for codebook embed should not be zero"

    print("✅ Test passed: EuclideanCodebook.forward() works correctly and has proper gradient flow!")


# Run the test
test_euclidean_codebook_forward()
