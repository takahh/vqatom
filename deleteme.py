import torch
import matplotlib.pyplot as plt

s = torch.linspace(0, 1, steps=500)
mu = 0.5
sigmas = [0.05, 0.1, 0.2, 0.5, 1.0]

plt.figure(figsize=(8, 5))
for sigma in sigmas:
    bell = torch.exp(-((s - mu) ** 2) / (2 * sigma ** 2))
    plt.plot(s.numpy(), bell.numpy(), label=f'sigma={sigma}')

plt.title("Effect of Sigma on Bell-Shaped Repel Loss")
plt.xlabel("Similarity")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
