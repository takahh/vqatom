import torch
from vq import EuclideanCodebook
import torch

import torch

import torch

import torch


# **Test Function**
def test_gradient_flow():
    torch.manual_seed(42)

    dim = 64
    codebook_size = 10
    batch_size = 128

    model = EuclideanCodebook(dim, codebook_size, learnable_codebook=True).cuda()
    x = torch.randn(1, batch_size, dim, device="cuda", requires_grad=True)

    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)

    print("\n✅ Requires Gradients:")
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"flatten.requires_grad: {flatten.requires_grad}")
    print(f"dist.requires_grad: {dist.requires_grad}")
    print(f"embed.requires_grad: {embed.requires_grad}")
    print(f"quantize.requires_grad: {quantize.requires_grad}")

    # **Check x.grad after backprop**
    loss = quantize.mean()
    loss.backward()

    print("\n📢 Gradients After Backpropagation:")
    print(f"x.grad is None: {x.grad is None}")
    if x.grad is not None:
        print(f"x.grad.sum(): {x.grad.sum()}")

    print(f"flatten.grad is None: {flatten.grad is None}")
    if flatten.grad is not None:
        print(f"flatten.grad.sum(): {flatten.grad.sum()}")


test_gradient_flow()
