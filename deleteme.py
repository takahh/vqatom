import numpy as np


# Load the .pth file
model_data = np.load("/Users/taka/Documents/vqatom_results/1500_64/init_codebook_34.npz")

print(model_data["arr_0"].shape)
