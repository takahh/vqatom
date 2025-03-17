import torch
import matplotlib.pyplot as plt

# Create soft_weights values ranging from 0 to 1
soft_weights = torch.linspace(0, 1, 100)

# Gaussian mask around 0.5 with variance 0.01
dupe_mask = torch.exp(-((soft_weights - 0.5) ** 4) / 0.001)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(soft_weights.numpy(), dupe_mask.numpy(), label="Gaussian Mask", color="blue")
plt.title("Gaussian Mask Function Around 0.5")
plt.xlabel("Soft Weight Value")
plt.ylabel("Mask Value")
plt.legend()
plt.grid(True)
plt.show()
