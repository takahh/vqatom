import matplotlib.pyplot as plt
import numpy as np


def plot(type):
    # Data extraction
    epochs = []
    feat_div_loss_train = []
    feat_div_loss_test = []
    cb_loss_train = []
    cb_loss_test = []
    sil_loss_train = []
    sil_loss_test = []
    train_loss = []
    test_loss = []

    # Parse the log filepath = "/Users/taka/Downloads/0331_1500/"
    with open('/Users/taka/Documents/vqatom_results/1000_64/outputs/log', 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            # Extract epoch
            epoch_match = lines[i].split('epoch ')[1].split(':')[0]
            epochs.append(int(epoch_match))

            # Extract losses from both lines
            line1_parts = lines[i].split(' ')
            line2_parts = lines[i + 1].split(' ')
            line3_parts = lines[i + 2].split(' ')
            print(line1_parts)
            print(line2_parts)
            print(line3_parts)

            # Extract specific losses
            train_loss.append(float(line1_parts[5].replace(',', '')))
            test_loss.append(float(line1_parts[7].strip()))

            # Feature divergence loss
            feat_div_loss_train.append(float(line2_parts[7].replace(',', '')))
            feat_div_loss_test.append(float(line3_parts[7].replace(',', '')))

            # CB loss
            cb_loss_train.append(float(line2_parts[17].split(',')[0]))
            cb_loss_test.append(float(line3_parts[17].split(',')[0]))
            # Silhouette losses (from two consecutive lines)
            sil_loss_train.append(float(line2_parts[-1].replace(',', '')))
            sil_loss_test.append(float(line3_parts[-1].replace(',', '')))

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot different loss metrics
    if type == 0:
        plt.plot(epochs, train_loss, label='Train Loss', marker='o')
        plt.plot(epochs, test_loss, label='Test Loss', marker='s')
        plt.title('Loss Across Epochs', fontsize=16)
    elif type == 1:
        plt.plot(epochs, feat_div_loss_train, label='Feature Div Loss Train', marker='^')
        plt.plot(epochs, feat_div_loss_test, label='Feature Div Loss Train', marker='_')
        plt.title('feat_div_loss Across Epochs', fontsize=16)
    elif type == 2:
        plt.plot(epochs, cb_loss_train, label='CB Loss Train', marker='v')
        plt.plot(epochs, cb_loss_test, label='CB Loss Test', marker='x')
        plt.title('CB loss Across Epochs', fontsize=16)
    else:
        plt.plot(epochs, sil_loss_train, label='Silhouette Loss Train', marker='d')
        plt.plot(epochs, sil_loss_test, label='Silhouette Loss Test', marker='p')
        plt.title('Sil loss Across Epochs', fontsize=16)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Using log scale due to small values
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig('training_metrics_plot.png')
    plt.close()

    print("Plot has been saved as 'training_metrics_plot.png'")

for type in range(0, 4):
    plot(type)
