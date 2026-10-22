import torch
import torch.nn.functional as F
from vq import batched_embedding

def test_batched_embedding_gradient_flow():
    # **Define Dummy Inputs**
    batch_size = 128
    num_clusters = 10
    embedding_dim = 64

    # **Simulated Cluster Indices (Just Random Integers)**
    indices = torch.randint(0, num_clusters, (batch_size, 1), dtype=torch.long, requires_grad=False)

    # **Simulated Embeddings (Cluster Centroids)**
    embed = torch.randn(1, num_clusters, embedding_dim, requires_grad=True)  # Learnable embeddings

    # **Run Forward Pass**
    quantized = batched_embedding(indices, embed)

    # **Define Fake Loss Function (Sum for Simplicity)**
    loss = quantized.sum()

    # **Backward Pass**
    loss.backward()

    # **Check Gradient Flow**
    print(f"quantized.requires_grad: {quantized.requires_grad}")  # Should be True
    print(f"embed.grad is None: {embed.grad is None}")  # Should be False
    print(f"embed.grad.sum(): {embed.grad.sum()}")  # Should not be zero

    # **Assertions**
    assert quantized.requires_grad, "❌ Gradient is not flowing to quantized"
    assert embed.grad is not None, "❌ Gradient is not flowing to embed"
    assert embed.grad.abs().sum() > 0, "❌ Gradients are zero, meaning no backpropagation"

    print("✅ Gradient flow test passed!")

# Run the test
test_batched_embedding_gradient_flow()
