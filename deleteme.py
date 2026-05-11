import json
import glob
import os
import re
import math
import matplotlib.pyplot as plt

json_dir = "/Users/taka/Downloads/"
files = glob.glob(os.path.join(json_dir, "epoch_*.json"))


def get_epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def safe_float(x):
    if x is None:
        return float("-inf")
    try:
        x = float(x)
        if math.isnan(x):
            return float("-inf")
        return x
    except Exception:
        return float("-inf")


files = sorted(files, key=get_epoch_from_name)

epochs = []

train_auc = []
valid_auc = []
final_auc = []

loss_total = []
loss_cls = []
loss_reg = []
loss_contact = []

final_ef1 = []
final_ef5 = []
final_ef10 = []

valid_rmse = []
final_rmse = []
valid_spearman = []
final_spearman = []


for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    ep = d.get("epoch", get_epoch_from_name(fp))
    epochs.append(ep)

    train_metrics = d.get("train_metrics", {})
    valid_metrics = d.get("valid_metrics", {})
    final_metrics = d.get("final_eval_metrics", {}) or {}

    valid_reg_metrics = d.get("valid_reg_metrics", {}) or {}
    final_reg_metrics = d.get("final_eval_reg_metrics", {}) or {}

    train_stat = d.get("train_stat", {})

    train_auc.append(train_metrics.get("auroc"))
    valid_auc.append(valid_metrics.get("auroc"))
    final_auc.append(final_metrics.get("auroc"))

    # new keys + old-key fallback
    loss_total.append(train_stat.get("loss"))
    loss_cls.append(train_stat.get("loss_cls", train_stat.get("loss_y")))
    loss_reg.append(train_stat.get("loss_reg", train_stat.get("loss_delta")))
    loss_contact.append(train_stat.get("loss_contact"))

    final_ef1.append(final_metrics.get("ef1"))
    final_ef5.append(final_metrics.get("ef5"))
    final_ef10.append(final_metrics.get("ef10"))

    valid_rmse.append(valid_reg_metrics.get("rmse"))
    final_rmse.append(final_reg_metrics.get("rmse"))
    valid_spearman.append(valid_reg_metrics.get("spearman"))
    final_spearman.append(final_reg_metrics.get("spearman"))


if not epochs:
    raise RuntimeError(f"No epoch_*.json files found in: {json_dir}")


# -------------------
# Best epoch by VALID AUROC
# -------------------
best_i = max(range(len(epochs)), key=lambda i: safe_float(valid_auc[i]))

print("===== BEST BY VALID AUROC =====")
print(f"best epoch      : {epochs[best_i]}")
print(f"valid AUROC     : {safe_float(valid_auc[best_i]):.6f}")
print(f"final AUROC     : {safe_float(final_auc[best_i]):.6f}")
print(f"train AUROC     : {safe_float(train_auc[best_i]):.6f}")
print(f"loss total      : {safe_float(loss_total[best_i]):.6f}")
print(f"loss cls        : {safe_float(loss_cls[best_i]):.6f}")
print(f"loss reg        : {safe_float(loss_reg[best_i]):.6f}")
print(f"loss contact    : {safe_float(loss_contact[best_i]):.6f}")

best_final_i = max(range(len(epochs)), key=lambda i: safe_float(final_auc[i]))

print("\n===== BEST FINAL AUROC debug only =====")
print(f"epoch           : {epochs[best_final_i]}")
print(f"final AUROC     : {safe_float(final_auc[best_final_i]):.6f}")
print(f"valid AUROC     : {safe_float(valid_auc[best_final_i]):.6f}")
print(f"train AUROC     : {safe_float(train_auc[best_final_i]):.6f}")


# -------------------
# Plot AUROC
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_auc, marker="o", label="train AUROC")
plt.plot(epochs, valid_auc, marker="o", label="valid AUROC")
plt.plot(epochs, final_auc, marker="o", label="final AUROC")

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

save_path = os.path.join(json_dir, "auc_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"saved to: {save_path}")

plt.show()


# -------------------
# Plot Train Losses
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_total, marker="o", label="loss total")
plt.plot(epochs, loss_cls, marker="o", label="loss cls")
plt.plot(epochs, loss_reg, marker="o", label="loss reg")
plt.plot(epochs, loss_contact, marker="o", label="loss contact")

plt.axvline(
    epochs[best_i],
    linestyle="--",
    label=f"best valid ep={epochs[best_i]}"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(json_dir, "train_loss_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"saved to: {save_path}")

plt.show()


# -------------------
# Plot Final EF
# -------------------
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


# -------------------
# Plot Regression RMSE
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, valid_rmse, marker="o", label="valid RMSE")
plt.plot(epochs, final_rmse, marker="o", label="final RMSE")

plt.axvline(
    epochs[best_i],
    linestyle="--",
    label=f"best valid ep={epochs[best_i]}"
)

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Regression RMSE vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(json_dir, "regression_rmse_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"saved to: {save_path}")

plt.show()


# -------------------
# Plot Regression Spearman
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, valid_spearman, marker="o", label="valid Spearman")
plt.plot(epochs, final_spearman, marker="o", label="final Spearman")

plt.axvline(
    epochs[best_i],
    linestyle="--",
    label=f"best valid ep={epochs[best_i]}"
)

plt.xlabel("Epoch")
plt.ylabel("Spearman")
plt.title("Regression Spearman vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(json_dir, "regression_spearman_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"saved to: {save_path}")

plt.show()