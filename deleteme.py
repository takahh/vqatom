import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 3
center = 0.0

# Create a range of distances
dmat = np.linspace(-10, 10, 1000)

# Compute the bell-shaped curve
bell = np.exp(-(dmat - center) ** 2 / (2 * sigma ** 2))

# Plotting
plt.plot(dmat, bell)
plt.title("Bell-shaped Curve (σ = 3)")
plt.xlabel("Distance")
plt.ylabel("Value")
plt.grid(True)
plt.show()
