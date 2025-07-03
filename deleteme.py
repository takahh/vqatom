import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 4.0, 200)
for s in [0.1, 0.3, 0.5, 1.0]:
    y = np.exp(-(x**2) / (2 * s**2))
    plt.plot(x, y, label=f'sigma={s}')
plt.title("Repel loss curve vs. distance")
plt.xlabel("Distance")
plt.ylabel("Loss")
plt.legend()
plt.show()
