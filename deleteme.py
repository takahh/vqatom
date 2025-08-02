import torch
import matplotlib.pyplot as plt

def soft_middle_weight(dmat, low, high, sharpness=20.0):
    w_low = torch.sigmoid(sharpness * (dmat - low))
    w_high = torch.sigmoid(sharpness * (high - dmat))
    return w_low * w_high

# Generate a range of distances
dists = torch.linspace(0, 20, steps=1000)
low = 3.0
high = 7.0
sharpness = 20.0

# Compute weights
weights = soft_middle_weight(dists, low, high, sharpness).numpy()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(dists.numpy(), weights, label=f"low={low}, high={high}, sharpness={sharpness}")
plt.axvline(low, color='gray', linestyle='--', label='low threshold')
plt.axvline(high, color='gray', linestyle='--', label='high threshold')
plt.title("Soft Middle Weight Function")
plt.xlabel("Pairwise Distance")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
