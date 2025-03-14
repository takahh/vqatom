import torch
from archive.vq import EuclideanCodebook

# Initialize VectorQuantize module
vq = EuclideanCodebook(dim=64, codebook_size=1000).cuda()

# Create input tensor with requires_grad=True
x = torch.randn(1, 1000, 64, device="cuda", requires_grad=True)

# Forward pass
quantize, embed_ind, loss, *_ = vq(x)
print(f"quantize.requires_grad: {quantize.requires_grad}")
print(f"quantize.grad_fn: {quantize.grad_fn}")

# Compute a fake loss
fake_loss = quantize.mean()  # Simple loss function to test gradients

# Backward pass
fake_loss.backward()

# Check if x has gradients
print(f"x.grad: {x.grad}")  # Should NOT be None if gradient flow is correct
