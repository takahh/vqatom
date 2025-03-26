import matplotlib.pyplot as plt
import numpy as np

# Data extraction
epochs = []
feat_div_loss = []
cb_loss = []
sil_loss1 = []
sil_loss2 = []
train_loss = []
test_loss = []

# Parse the log file
with open('/Users/taka/Downloads/0326_cb1000/outputs/log', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        # Extract epoch
        epoch_match = lines[i].split('epoch ')[1].split(':')[0]
        epochs.append(int(epoch_match))

        # Extract losses from both lines
        line1_parts = lines[i].split(', ')
        line2_parts = lines[i + 1].split(', ')

        # Extract specific losses
        train_loss.append(float(line1_parts[0].split('loss ')[1]))
        test_loss.append(float(line1_parts[1].split('test_loss ')[1]))

        # Feature divergence loss
        feat_div_loss.append(float(line2_parts[0].split('feat_div loss:  ')[1]))

        # CB loss
        cb_loss.append(float(line2_parts[2].split('cb_loss:  ')[1]))

        # Silhouette losses (from two consecutive lines)
        sil_loss1.append(float(line2_parts[3].split('sil_loss:  ')[1]))
        sil_loss2.append(float(lines[i + 1].split('sil_loss:  ')[1].strip()))

# Create the plot
plt.figure(figsize=(12, 8))

# Plot different loss metrics
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, test_loss, label='Test Loss', marker='s')
plt.plot(epochs, feat_div_loss, label='Feature Div Loss', marker='^')
plt.plot(epochs, cb_loss, label='CB Loss', marker='v')
plt.plot(epochs, sil_loss1, label='Silhouette Loss 1', marker='d')
plt.plot(epochs, sil_loss2, label='Silhouette Loss 2', marker='p')

plt.title('Training Metrics Across Epochs', fontsize=16)
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