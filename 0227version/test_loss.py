import torch
from vq import EuclideanCodebook
import torch

import torch

import torch

import torch

def test_gradient_flow():
    torch.manual_seed(42)

    dim = 64
    codebook_size = 10
    batch_size = 128  # Smaller batch for testing

    model = EuclideanCodebook(dim, codebook_size, learnable_codebook=True).cuda()

    x = torch.randn(batch_size, 1, dim, device="cuda", requires_grad=True)

    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)

    print("Requires Gradients:")
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"flatten.requires_grad: {flatten.requires_grad}")
    print(f"dist.requires_grad: {dist.requires_grad}")
    print(f"embed.requires_grad: {embed.requires_grad}")
    print(f"quantize.requires_grad: {quantize.requires_grad}")

    loss = quantize.mean()
    loss.backward()

    print("\nGradients After Backpropagation:")
    print(f"x.grad: {x.grad}")
    print(f"quantize.grad: {quantize.grad}")
    print(f"embed.grad: {model.embed.grad}")

test_gradient_flow()
