import numpy as np
import matplotlib.pyplot as plt

# Distance values from 0 to 10
d_values = np.linspace(0, 10, 500)


# Repel function
def repel_strength(d, sigma):
    return np.exp(-d ** 2 / (2 * sigma ** 2))


# Sigma values to try
sigmas = [3, 4, 5, 6]

# Plot
plt.figure(figsize=(8, 6))
for sigma in sigmas:
    plt.plot(d_values, repel_strength(d_values, sigma), label=f'sigma={sigma}')

plt.title("Repel Strength vs Distance for Different Sigma Values")
plt.xlabel("Distance")
plt.ylabel("Repel Strength (exp(-d² / (2σ²)))")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
