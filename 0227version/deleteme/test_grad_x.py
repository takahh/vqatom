import torch
from vq import EuclideanCodebook


def test_gradient_flow():
    # Simulated input
    x = torch.randn(1, 128, 64, device="cuda", requires_grad=True)

    dim = 64
    codebook_size = 10

    # Call forward function
    model = EuclideanCodebook(dim, codebook_size, learnable_codebook=True).cuda()
    quantize, embed_ind, dist, embed, flatten, init_cb = model(x)

    # Backpropagate through dist
    dist.sum().backward()  # 🔥 Test backpropagation before embedding lookup

    # Debug prints
    print(f"x.grad is None: {x.grad is None}")
    if x.grad is not None:
        print(f"x.grad.sum(): {x.grad.sum()}")

    print(f"flatten.grad is None: {flatten.grad is None}")
    if flatten.grad is not None:
        print(f"flatten.grad.sum(): {flatten.grad.sum()}")

# Run the test
test_gradient_flow()

