import json
import glob
import os
import re
import math
import matplotlib.pyplot as plt

json_dir = "/Users/taka/Downloads/"
# json_dir = "/Users/taka/Documents/dti_results/smiles_no_pretrained/seed0"
# json_dir = "/Users/taka/Downloads"
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


def safe_float(x):
    if x is None:
        return float("-inf")
    try:
        x = float(x)
        if math.isnan(x):
            return float("-inf")
        return x
    except:
        return float("-inf")


# -------------------
# Best epoch by VALID
# -------------------
best_i = max(range(len(epochs)), key=lambda i: safe_float(valid_auc[i]))

print("===== BEST BY VALID AUROC =====")
print(f"best epoch      : {epochs[best_i]}")
print(f"valid AUROC     : {valid_auc[best_i]:.6f}")
print(f"final AUROC     : {final_auc[best_i]:.6f}")
print(f"train AUROC     : {train_auc[best_i]:.6f}")

# debug only
best_final_i = max(range(len(epochs)), key=lambda i: safe_float(final_auc[i]))

print("\n===== BEST FINAL AUROC (debug only) =====")
print(f"epoch           : {epochs[best_final_i]}")
print(f"final AUROC     : {final_auc[best_final_i]:.6f}")
print(f"valid AUROC     : {valid_auc[best_final_i]:.6f}")
print(f"train AUROC     : {train_auc[best_final_i]:.6f}")


# -------------------
# Plot AUC
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_auc, marker="o", label="train auc")
plt.plot(epochs, valid_auc, marker="o", label="valid auc")
plt.plot(epochs, final_auc, marker="o", label="final auc")

plt.axvline(
    epochs[best_i],
    linestyle="--",
    label=f"best valid ep={epochs[best_i]}"
)

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
# -------------------
# EF Plot (Final Eval) + Save
# -------------------

final_ef1 = []
final_ef5 = []
final_ef10 = []

for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    final_metrics = d.get("final_eval_metrics", {})

    final_ef1.append(final_metrics.get("ef1"))
    final_ef5.append(final_metrics.get("ef5"))
    final_ef10.append(final_metrics.get("ef10"))


plt.figure(figsize=(8, 5))

plt.plot(epochs, final_ef1, marker="o", label="Final EF1")
plt.plot(epochs, final_ef5, marker="o", label="Final EF5")
plt.plot(epochs, final_ef10, marker="o", label="Final EF10")

plt.axvline(
    epochs[best_i],
    linestyle="--",
    label=f"best valid ep={epochs[best_i]}"
)

plt.xlabel("Epoch")
plt.ylabel("Enrichment Factor")
plt.title("Final Eval EF vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(json_dir, "final_eval_ef_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"saved to: {save_path}")

plt.show()