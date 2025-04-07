import matplotlib.pyplot as plt
import numpy as np
best_epoch_dict = {}

def plot(cbsize, dim):
    # Data extraction
    epochs = []
    best_epoch = 0
    best_loss = float(100.0)

    # Parse the log filepath = "/Users/taka/Downloads/0331_1500/"
    with open(f'/Users/taka/Documents/vqatom_results/{str(cbsize)}_{str(dim)}/outputs/log', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "epoch" not in line:
                continue
            # Extract epoch
            epoch_match = line.split()[3].replace(":", "")
            epochs.append(int(epoch_match))
            # Extract losses from both lines
            line1_parts = line.split(' ')
            if best_loss > float(line1_parts[7].strip()):
                best_loss = float(line1_parts[7].strip())
                best_epoch = int(epoch_match)
    best_epoch_dict[f"{cbsize}_{dim}"] = best_epoch
    print(best_epoch_dict)


for cb in [1000, 1500, 2000]:
    for dim in [64, 128, 256]:
        plot(cb, dim)
