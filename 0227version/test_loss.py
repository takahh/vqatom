import torch
import pytest
from your_module import compute_contrastive_loss  # Replace 'your_module' with the actual module name


def test_compute_contrastive_loss():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test parameters
    num_samples = 5
    latent_dim = 3
    num_atom_types = 4

    # Create synthetic latent vectors (z)
    z = torch.randn(num_samples, latent_dim)

    # Case 1: All atoms are of the same type
    atom_types_same = torch.zeros(num_samples, dtype=torch.long)
    loss_same = compute_contrastive_loss(z, atom_types_same, num_atom_types=num_atom_types)

    assert isinstance(loss_same, torch.Tensor), "Loss should be a tensor."
    assert loss_same.dim() == 0, "Loss should be a scalar tensor."
    assert loss_same.item() >= 0, "Loss should be non-negative."

    # Case 2: All atoms are different types
    atom_types_diff = torch.arange(num_samples) % num_atom_types
    loss_diff = compute_contrastive_loss(z, atom_types_diff, num_atom_types=num_atom_types)

    assert loss_diff.item() >= 0, "Loss should be non-negative."

    # Case 3: Random atom types
    atom_types_rand = torch.randint(0, num_atom_types, (num_samples,))
    loss_rand = compute_contrastive_loss(z, atom_types_rand, num_atom_types=num_atom_types)

    assert loss_rand.item() >= 0, "Loss should be non-negative."

    # Case 4: Check loss behavior (same type should have higher positive loss)
    assert loss_same.item() > loss_diff.item(), "Loss should be higher when all types are the same."

    print("All tests passed successfully!")


if __name__ == "__main__":
    pytest.main([__file__])
