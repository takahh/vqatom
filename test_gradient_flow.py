import torch
from vq import VectorQuantize

x = torch.randn(10, 64, device="cuda", requires_grad=True)  # Example input
init_feat = torch.randn(10, 64, device="cuda", requires_grad=True)  # Example feature
vq = VectorQuantize(dim=64, codebook_size=1000).cuda()  # Instantiate the module
quantize, _, loss, *_ = vq(x, init_feat, logger=None)

# Compute loss and backpropagate
loss.backward()
print(f"x.grad: {x.grad}")  # If None, gradient flow is broken

