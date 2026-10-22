import torch
import unittest
from models import EquivariantThreeHopGINE
from train_teacher import get_args

class TestYourModel(unittest.TestCase):
    def setUp(self):
        # Setup a minimal instance of your model
        args = get_args()
        self.model = EquivariantThreeHopGINE(in_feats=64, hidden_feats=64,
                                        out_feats=64, args=args)

        self.model.train()  # Ensure training mode

    def test_forward_gradient_flow(self):
        x = torch.randn(1, 128, 64, requires_grad=True, device="cuda")  # Example input
        feature = torch.randn(100, 7, device="cuda")  # Random tensor for feature

        # Ensure model parameters require gradients
        self.assertTrue(self.model.vq._codebook.embed.requires_grad, "self.embed should require gradients before forward()")

        # Check grad_fn is None initially (expected for parameters)
        self.assertIsNone(self.model.vq._codebook.embed.grad_fn, "self.embed.grad_fn should be None initially")

        # Forward pass
        quantize, embed_ind, dist, embed, flatten, init_cb = self.model.forward(x, feature, 1)

        # Ensure output requires gradients
        self.assertTrue(quantize.requires_grad, "Quantized output should require gradients")
        self.assertTrue(dist.requires_grad, "Distance tensor should require gradients")
        self.assertTrue(flatten.requires_grad, "Flattened input should require gradients")

        # Backward pass
        loss = quantize.sum()  # Sample loss function
        loss.backward()

        # Ensure gradients were computed
        self.assertIsNotNone(self.model.vq._codebook.embed.grad, "Gradients should flow to self.embed")
        self.assertNotEqual(self.model.vq._codebook.embed.grad.abs().sum().item(), 0, "Gradient sum should be non-zero")

if __name__ == '__main__':
    unittest.main()
