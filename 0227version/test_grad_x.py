import torch
from vq import EuclideanCodebook

def test_gradient_flow():
    # Simulated input
    x = torch.randn(1, 128, 64, device="cuda", requires_grad=True)  # Ensure x requires gradients

    dim = 64
    codebook_size = 10
    batch_size = 128  # Smaller batch for testing

    # Call forward function
    model = EuclideanCodebook(dim, codebook_size, learnable_codebook=True).cuda()
    # Replace with your actual model instance
    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)
    flatten.retain_grad()  # Retain gradients for debugging

    # Backpropagate
    loss = quantize.sum()
    loss.backward()

    # Debug prints
    print(f"x.grad is None: {x.grad is None}")
    if x.grad is not None:
        print(f"x.grad.sum(): {x.grad.sum()}")

    print(f"flatten.grad is None: {flatten.grad is None}")
    if flatten.grad is not None:
        print(f"flatten.grad.sum(): {flatten.grad.sum()}")


# Run the test
test_gradient_flow()
