import torch

def test_compute_contrastive_loss():
    torch.manual_seed(42)  # For reproducibility

    # Create dummy latent vectors (z) and atom types
    batch_size = 10
    latent_dim = 16
    num_atom_types = 5

    # Random latent vectors with requires_grad=True for gradient tracking
    z = torch.randn(batch_size, latent_dim, device="cuda", requires_grad=True)

    # Random atom types as integer labels
    atom_types = torch.randint(0, num_atom_types, (batch_size,), device="cuda")

    # Compute loss
    loss = compute_contrastive_loss(z, atom_types, num_atom_types=num_atom_types)

    # **Basic Functionality Checks**
    assert loss.dim() == 0, "Loss should be a scalar value."
    assert loss.item() >= 0, "Loss should be non-negative."

    # **Gradient Flow Check**
    loss.backward()
    assert z.grad is not None, "Gradient for z should not be None."
    assert z.grad.abs().sum().item() > 0, "Gradient for z should not be zero."

    print("✅ Test passed: compute_contrastive_loss works correctly and has proper gradient flow!")

# Run the test
test_compute_contrastive_loss()
