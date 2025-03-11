import torch
from vq import EuclideanCodebook
import torch

import torch

import torch


def test_euclidean_codebook_forward():
    torch.manual_seed(42)  # For reproducibility

    # **Initialize Codebook**
    dim = 64
    codebook_size = 10
    batch_size = 22144
    num_codebooks = 1

    model = EuclideanCodebook(dim, codebook_size, num_codebooks=num_codebooks).cuda()

    # **Create Input**
    x = torch.randn(batch_size, 1, dim, device="cuda", requires_grad=True)

    # **Forward Pass**
    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)
    quantize = torch.squeeze(quantize)
    x = torch.squeeze(x)
    # **Fix Output Shape Check**
    expected_shape = (1, *x.shape)
    assert quantize.shape == expected_shape, f"Quantized output shape mismatch: expected {expected_shape}, got {quantize.shape}"

    assert embed_ind.shape[1] == batch_size, "Embedding index shape mismatch"
    assert dist.shape == (num_codebooks, batch_size, codebook_size), "Distance matrix shape mismatch"

    assert embed_ind.min() >= 0, "embed_ind contains negative values"
    assert embed_ind.max() < codebook_size, "embed_ind contains out-of-range values"

    # **Gradient Flow Check**
    loss = quantize.mean()  # Dummy loss function
    loss.backward()

    assert x.grad is not None, "Gradient for input x should not be None"
    assert x.grad.abs().sum().item() > 0, "Gradient for input x should not be zero"

    assert model.embed.grad is not None, "Gradient for codebook embed should not be None"
    assert model.embed.grad.abs().sum().item() > 0, "Gradient for codebook embed should not be zero"

    print("✅ Test passed: EuclideanCodebook.forward() works correctly and has proper gradient flow!")


# Run the test
test_euclidean_codebook_forward()
