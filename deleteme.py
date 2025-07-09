import numpy as np
import matplotlib.pyplot as plt
path = "/Users/taka/Downloads/log"
sil_list = []
with open(path, "r") as f:
    for lines in f.readlines():
        if "Silh" in lines:
            sil_list.append(float(lines.split(" ")[-1].strip()))


# Plot
plt.figure(figsize=(8, 6))
plt.plot(sil_list)

plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
