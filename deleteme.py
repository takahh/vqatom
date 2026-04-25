import json
import glob
import os
import re
import matplotlib.pyplot as plt

# folder containing epoch_001.json, epoch_002.json, ...
json_dir = "/Users/taka/Downloads/vq_json/"

files = glob.glob(os.path.join(json_dir, "epoch_*.json"))

def get_epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1

files = sorted(files, key=get_epoch_from_name)

epochs = []

train_auc = []
valid_auc = []
final_auc = []

loss_total = []
loss_y = []
loss_base = []
loss_delta = []

for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    ep = d.get("epoch", get_epoch_from_name(fp))
    epochs.append(ep)

    train_metrics = d.get("train_metrics", {})
    valid_metrics = d.get("valid_metrics", {})
    final_metrics = d.get("final_eval_metrics", {})

    train_stat = d.get("train_stat", {})

    train_auc.append(train_metrics.get("auroc"))
    valid_auc.append(valid_metrics.get("auroc"))
    final_auc.append(final_metrics.get("auroc"))

    loss_total.append(train_stat.get("loss"))
    loss_y.append(train_stat.get("loss_y"))
    loss_base.append(train_stat.get("loss_base"))
    loss_delta.append(train_stat.get("loss_delta"))

# -------------------
# Plot AUC
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_auc, marker="o", label="train auc")
plt.plot(epochs, valid_auc, marker="o", label="valid auc")
plt.plot(epochs, final_auc, marker="o", label="final auc")
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.title("AUROC vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------
# Plot Loss
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_total, marker="o", label="loss total")
plt.plot(epochs, loss_y, marker="o", label="loss y")
plt.plot(epochs, loss_base, marker="o", label="loss base")
plt.plot(epochs, loss_delta, marker="o", label="loss delta")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()