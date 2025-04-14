import matplotlib.pyplot as plt
import numpy as np


# Load the .pth file
epochs = []
train_losses = []
with open("/Users/taka/Downloads/log", 'r') as file:
    for lines in file.readlines():
        print(lines)
        if "epoch" in lines:
            ele = lines.split()
            epochs.append(int(ele[3].replace(":", "")))
            train_losses.append(float(ele[7].replace(",", "")))

plt.figure()
plt.plot(epochs, train_losses)
plt.show()
