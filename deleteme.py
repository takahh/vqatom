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


def none_to_nan(x):
    if x is None:
        return float("nan")
    try:
        x = float(x)
        if math.isnan(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def plot_metric(filename, title, ylabel, series, best_epoch=None, hline0=False):
    plt.figure(figsize=(8, 5))

    for label, values in series:
        plt.plot(epochs, values, marker="o", label=label)

    if hline0:
        plt.axhline(0.0, linestyle=":", linewidth=1)

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best valid ep={best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(json_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved to: {save_path}")
    plt.show()


files = sorted(files, key=get_epoch_from_name)

epochs = []

train_auc, valid_auc, final_auc = [], [], []
train_ap, valid_ap, final_ap = [], [], []
train_f1, valid_f1, final_f1 = [], [], []

loss_total, loss_cls, loss_reg, loss_contact = [], [], [], []

contact_gap, contact_pos_mean, contact_neg_mean = [], [], []

final_ef1, final_ef5, final_ef10 = [], [], []
valid_ef1, valid_ef5, valid_ef10 = [], [], []

train_rmse, valid_rmse, final_rmse = [], [], []
train_mae, valid_mae, final_mae = [], [], []
train_spearman, valid_spearman, final_spearman = [], [], []
train_pearson, valid_pearson, final_pearson = [], [], []

train_recall_top1, valid_recall_top1, final_recall_top1 = [], [], []
train_hit_top1, valid_hit_top1, final_hit_top1 = [], [], []
train_ef_top1, valid_ef_top1, final_ef_top1 = [], [], []

train_top1_pos, valid_top1_pos, final_top1_pos = [], [], []
train_top1_k, valid_top1_k, final_top1_k = [], [], []


for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    ep = d.get("epoch", get_epoch_from_name(fp))
    epochs.append(ep)

    # current JSON format
    train_metrics = d.get("train", {}) or {}
    valid_metrics = d.get("valid", {}) or {}
    final_metrics = d.get("final", {}) or {}
    train_stat = d.get("train_loss", {}) or {}

    # classification metrics
    train_auc.append(none_to_nan(train_metrics.get("auroc")))
    valid_auc.append(none_to_nan(valid_metrics.get("auroc")))
    final_auc.append(none_to_nan(final_metrics.get("auroc")))

    train_ap.append(none_to_nan(train_metrics.get("ap")))
    valid_ap.append(none_to_nan(valid_metrics.get("ap")))
    final_ap.append(none_to_nan(final_metrics.get("ap")))

    train_f1.append(none_to_nan(train_metrics.get("f1")))
    valid_f1.append(none_to_nan(valid_metrics.get("f1")))
    final_f1.append(none_to_nan(final_metrics.get("f1")))

    valid_ef1.append(none_to_nan(valid_metrics.get("ef1")))
    valid_ef5.append(none_to_nan(valid_metrics.get("ef5")))
    valid_ef10.append(none_to_nan(valid_metrics.get("ef10")))

    final_ef1.append(none_to_nan(final_metrics.get("ef1")))
    final_ef5.append(none_to_nan(final_metrics.get("ef5")))
    final_ef10.append(none_to_nan(final_metrics.get("ef10")))

    # losses
    loss_total.append(none_to_nan(train_stat.get("loss")))
    loss_cls.append(none_to_nan(train_stat.get("loss_cls")))
    loss_reg.append(none_to_nan(train_stat.get("loss_reg")))
    loss_contact.append(none_to_nan(train_stat.get("loss_contact")))

    # contact guide stats
    contact_gap.append(none_to_nan(train_stat.get("contact_gap")))
    contact_pos_mean.append(none_to_nan(train_stat.get("contact_pos_mean")))
    contact_neg_mean.append(none_to_nan(train_stat.get("contact_neg_mean")))

    # regression metrics
    train_rmse.append(none_to_nan(train_metrics.get("rmse")))
    valid_rmse.append(none_to_nan(valid_metrics.get("rmse")))
    final_rmse.append(none_to_nan(final_metrics.get("rmse")))

    train_mae.append(none_to_nan(train_metrics.get("mae")))
    valid_mae.append(none_to_nan(valid_metrics.get("mae")))
    final_mae.append(none_to_nan(final_metrics.get("mae")))

    train_spearman.append(none_to_nan(train_metrics.get("spearman")))
    valid_spearman.append(none_to_nan(valid_metrics.get("spearman")))
    final_spearman.append(none_to_nan(final_metrics.get("spearman")))

    train_pearson.append(none_to_nan(train_metrics.get("pearson")))
    valid_pearson.append(none_to_nan(valid_metrics.get("pearson")))
    final_pearson.append(none_to_nan(final_metrics.get("pearson")))

    train_recall_top1.append(none_to_nan(train_metrics.get("recall_top1")))
    valid_recall_top1.append(none_to_nan(valid_metrics.get("recall_top1")))
    final_recall_top1.append(none_to_nan(final_metrics.get("recall_top1")))

    train_hit_top1.append(none_to_nan(train_metrics.get("hit_rate_top1")))
    valid_hit_top1.append(none_to_nan(valid_metrics.get("hit_rate_top1")))
    final_hit_top1.append(none_to_nan(final_metrics.get("hit_rate_top1")))

    train_ef_top1.append(none_to_nan(train_metrics.get("ef_top1")))
    valid_ef_top1.append(none_to_nan(valid_metrics.get("ef_top1")))
    final_ef_top1.append(none_to_nan(final_metrics.get("ef_top1")))

    train_top1_pos.append(none_to_nan(train_metrics.get("top1_pos")))
    valid_top1_pos.append(none_to_nan(valid_metrics.get("top1_pos")))
    final_top1_pos.append(none_to_nan(final_metrics.get("top1_pos")))

    train_top1_k.append(none_to_nan(train_metrics.get("top1_k")))
    valid_top1_k.append(none_to_nan(valid_metrics.get("top1_k")))
    final_top1_k.append(none_to_nan(final_metrics.get("top1_k")))


if not epochs:
    raise RuntimeError(f"No epoch_*.json files found in: {json_dir}")


best_i = max(range(len(epochs)), key=lambda i: safe_float(valid_auc[i]))
best_epoch = epochs[best_i]

best_final_i = max(range(len(epochs)), key=lambda i: safe_float(final_auc[i]))


print("===== BEST BY VALID AUROC =====")
print(f"best epoch       : {epochs[best_i]}")
print(f"valid AUROC      : {safe_float(valid_auc[best_i]):.6f}")
print(f"final AUROC      : {safe_float(final_auc[best_i]):.6f}")
print(f"train AUROC      : {safe_float(train_auc[best_i]):.6f}")
print(f"valid RMSE       : {safe_float(valid_rmse[best_i]):.6f}")
print(f"final RMSE       : {safe_float(final_rmse[best_i]):.6f}")
print(f"valid Spearman   : {safe_float(valid_spearman[best_i]):.6f}")
print(f"final Spearman   : {safe_float(final_spearman[best_i]):.6f}")
print(f"valid Pearson    : {safe_float(valid_pearson[best_i]):.6f}")
print(f"final Pearson    : {safe_float(final_pearson[best_i]):.6f}")
print(f"valid R@1%       : {safe_float(valid_recall_top1[best_i]):.6f}")
print(f"final R@1%       : {safe_float(final_recall_top1[best_i]):.6f}")
print(f"valid Hit@1%     : {safe_float(valid_hit_top1[best_i]):.6f}")
print(f"final Hit@1%     : {safe_float(final_hit_top1[best_i]):.6f}")
print(f"valid EF@1%      : {safe_float(valid_ef_top1[best_i]):.6f}")
print(f"final EF@1%      : {safe_float(final_ef_top1[best_i]):.6f}")
print(f"loss total       : {safe_float(loss_total[best_i]):.6f}")
print(f"loss cls         : {safe_float(loss_cls[best_i]):.6f}")
print(f"loss reg         : {safe_float(loss_reg[best_i]):.6f}")
print(f"loss contact     : {safe_float(loss_contact[best_i]):.6f}")
print(f"contact gap      : {safe_float(contact_gap[best_i]):.6f}")

print("\n===== BEST FINAL AUROC debug only =====")
print(f"epoch            : {epochs[best_final_i]}")
print(f"final AUROC      : {safe_float(final_auc[best_final_i]):.6f}")
print(f"valid AUROC      : {safe_float(valid_auc[best_final_i]):.6f}")
print(f"train AUROC      : {safe_float(train_auc[best_final_i]):.6f}")


# classification
plot_metric(
    "auc_plot.png",
    "AUROC vs Epoch",
    "AUROC",
    [
        ("train AUROC", train_auc),
        ("valid AUROC", valid_auc),
        ("final AUROC", final_auc),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "ap_plot.png",
    "Average Precision vs Epoch",
    "AP",
    [
        ("train AP", train_ap),
        ("valid AP", valid_ap),
        ("final AP", final_ap),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "f1_plot.png",
    "F1 vs Epoch",
    "F1",
    [
        ("train F1", train_f1),
        ("valid F1", valid_f1),
        ("final F1", final_f1),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "valid_final_ef_plot.png",
    "Classification EF vs Epoch",
    "Enrichment Factor",
    [
        ("valid EF1", valid_ef1),
        ("valid EF5", valid_ef5),
        ("valid EF10", valid_ef10),
        ("final EF1", final_ef1),
        ("final EF5", final_ef5),
        ("final EF10", final_ef10),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "final_eval_ef_plot.png",
    "Final Eval EF vs Epoch",
    "Enrichment Factor",
    [
        ("final EF1", final_ef1),
        ("final EF5", final_ef5),
        ("final EF10", final_ef10),
    ],
    best_epoch=best_epoch,
)


# losses
plot_metric(
    "train_loss_plot.png",
    "Train Loss vs Epoch",
    "Loss",
    [
        ("loss total", loss_total),
        ("loss cls", loss_cls),
        ("loss reg", loss_reg),
        ("loss contact", loss_contact),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "contact_gap_plot.png",
    "Contact Pair Separation vs Epoch",
    "z-scored pair score",
    [
        ("contact gap pos-neg", contact_gap),
        ("contact pos mean", contact_pos_mean),
        ("contact neg mean", contact_neg_mean),
    ],
    best_epoch=best_epoch,
    hline0=True,
)


# regression core
plot_metric(
    "regression_rmse_plot.png",
    "Regression RMSE vs Epoch",
    "RMSE",
    [
        ("train RMSE", train_rmse),
        ("valid RMSE", valid_rmse),
        ("final RMSE", final_rmse),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_mae_plot.png",
    "Regression MAE vs Epoch",
    "MAE",
    [
        ("train MAE", train_mae),
        ("valid MAE", valid_mae),
        ("final MAE", final_mae),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_spearman_plot.png",
    "Regression Spearman vs Epoch",
    "Spearman",
    [
        ("train Spearman", train_spearman),
        ("valid Spearman", valid_spearman),
        ("final Spearman", final_spearman),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_pearson_plot.png",
    "Regression Pearson vs Epoch",
    "Pearson",
    [
        ("train Pearson", train_pearson),
        ("valid Pearson", valid_pearson),
        ("final Pearson", final_pearson),
    ],
    best_epoch=best_epoch,
)


# regression ranking / virtual screening
plot_metric(
    "regression_recall_top1_plot.png",
    "Regression Recall@Top1% vs Epoch",
    "Recall@Top1%",
    [
        ("train R@1%", train_recall_top1),
        ("valid R@1%", valid_recall_top1),
        ("final R@1%", final_recall_top1),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_hit_top1_plot.png",
    "Regression Hit-rate@Top1% vs Epoch",
    "Hit-rate@Top1%",
    [
        ("train Hit@1%", train_hit_top1),
        ("valid Hit@1%", valid_hit_top1),
        ("final Hit@1%", final_hit_top1),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_ef_top1_plot.png",
    "Regression EF@Top1% vs Epoch",
    "EF@Top1%",
    [
        ("train EF@1%", train_ef_top1),
        ("valid EF@1%", valid_ef_top1),
        ("final EF@1%", final_ef_top1),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_top1_hits_plot.png",
    "Top1% Positive Hits vs Epoch",
    "Count",
    [
        ("train top1_pos", train_top1_pos),
        ("valid top1_pos", valid_top1_pos),
        ("final top1_pos", final_top1_pos),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "regression_top1_k_plot.png",
    "Top1% K vs Epoch",
    "K",
    [
        ("train top1_k", train_top1_k),
        ("valid top1_k", valid_top1_k),
        ("final top1_k", final_top1_k),
    ],
    best_epoch=best_epoch,
)