import json
import glob
import os
import re
import math
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
json_dir = "/Users/taka/Downloads/vqatom_pre"
files = glob.glob(os.path.join(json_dir, "*.json"))
files = glob.glob(os.path.join(json_dir, "epoch_*.json"))


# =========================================================
# Utils
# =========================================================
def get_epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x):
            return float("-inf")
        return x
    except Exception:
        return float("-inf")


def none_to_nan(x):
    try:
        x = float(x)
        if math.isnan(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def get_dict(d, *names):
    for name in names:
        v = d.get(name)
        if isinstance(v, dict):
            return v
    return {}


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


# =========================================================
# Load files
# =========================================================
files = sorted(files, key=get_epoch_from_name)

epochs = []

# classification
train_auc, valid_auc, final_auc = [], [], []
train_ap, valid_ap, final_ap = [], [], []
train_f1, valid_f1, final_f1 = [], [], []

# losses
loss_total, loss_cls, loss_reg, loss_contact = [], [], [], []
contact_gap, contact_pos_mean, contact_neg_mean = [], [], []

# EF
train_ef1, train_ef5, train_ef10 = [], [], []
valid_ef1, valid_ef5, valid_ef10 = [], [], []
final_ef1, final_ef5, final_ef10 = [], [], []

# old positive recall-like metrics
train_r1, train_r5, train_r10 = [], [], []
valid_r1, valid_r5, valid_r10 = [], [], []
final_r1, final_r5, final_r10 = [], [], []

# regression
train_rmse, valid_rmse, final_rmse = [], [], []
train_mae, valid_mae, final_mae = [], [], []
train_spearman, valid_spearman, final_spearman = [], [], []
train_pearson, valid_pearson, final_pearson = [], [], []

# prediction distribution
train_pred_mean, valid_pred_mean, final_pred_mean = [], [], []
train_pred_std, valid_pred_std, final_pred_std = [], [], []
train_pred_pos_rate, valid_pred_pos_rate, final_pred_pos_rate = [], [], []

# new group ranking metrics
train_top1_exact, valid_top1_exact, final_top1_exact = [], [], []

train_best1, valid_best1, final_best1 = [], [], []
train_best5, valid_best5, final_best5 = [], [], []
train_best10, valid_best10, final_best10 = [], [], []

train_ndcg1, valid_ndcg1, final_ndcg1 = [], [], []
train_ndcg5, valid_ndcg5, final_ndcg5 = [], [], []
train_ndcg10, valid_ndcg10, final_ndcg10 = [], [], []

train_n_groups, valid_n_groups, final_n_groups = [], [], []


# =========================================================
# Parse JSON
# =========================================================
for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    ep = d.get("epoch", get_epoch_from_name(fp))
    epochs.append(ep)

    train_stat = get_dict(d, "train_stat", "train_loss")

    train_metrics = get_dict(d, "train_metrics", "train")
    valid_metrics = get_dict(d, "valid_metrics", "valid")
    final_metrics = get_dict(d, "final_eval_metrics", "final_eval", "final")

    train_reg_metrics = get_dict(d, "train_reg_metrics")
    valid_reg_metrics = get_dict(d, "valid_reg_metrics")
    final_reg_metrics = get_dict(d, "final_eval_reg_metrics", "final_reg_metrics")

    train_rank = get_dict(d, "train_rank_metrics")
    valid_rank = get_dict(d, "valid_rank_metrics")
    final_rank = get_dict(d, "final_eval_rank_metrics")

    # -------------------------
    # Classification
    # -------------------------
    train_auc.append(none_to_nan(train_metrics.get("auroc")))
    valid_auc.append(none_to_nan(valid_metrics.get("auroc")))
    final_auc.append(none_to_nan(final_metrics.get("auroc")))

    train_ap.append(none_to_nan(train_metrics.get("ap")))
    valid_ap.append(none_to_nan(valid_metrics.get("ap")))
    final_ap.append(none_to_nan(final_metrics.get("ap")))

    train_f1.append(none_to_nan(train_metrics.get("f1")))
    valid_f1.append(none_to_nan(valid_metrics.get("f1")))
    final_f1.append(none_to_nan(final_metrics.get("f1")))

    # -------------------------
    # EF
    # -------------------------
    train_ef1.append(none_to_nan(train_metrics.get("ef1")))
    train_ef5.append(none_to_nan(train_metrics.get("ef5")))
    train_ef10.append(none_to_nan(train_metrics.get("ef10")))

    valid_ef1.append(none_to_nan(valid_metrics.get("ef1")))
    valid_ef5.append(none_to_nan(valid_metrics.get("ef5")))
    valid_ef10.append(none_to_nan(valid_metrics.get("ef10")))

    final_ef1.append(none_to_nan(final_metrics.get("ef1")))
    final_ef5.append(none_to_nan(final_metrics.get("ef5")))
    final_ef10.append(none_to_nan(final_metrics.get("ef10")))

    # -------------------------
    # Old r@top%
    # -------------------------
    train_r1.append(none_to_nan(train_metrics.get("r1")))
    train_r5.append(none_to_nan(train_metrics.get("r5")))
    train_r10.append(none_to_nan(train_metrics.get("r10")))

    valid_r1.append(none_to_nan(valid_metrics.get("r1")))
    valid_r5.append(none_to_nan(valid_metrics.get("r5")))
    valid_r10.append(none_to_nan(valid_metrics.get("r10")))

    final_r1.append(none_to_nan(final_metrics.get("r1")))
    final_r5.append(none_to_nan(final_metrics.get("r5")))
    final_r10.append(none_to_nan(final_metrics.get("r10")))

    # -------------------------
    # Prediction distribution
    # -------------------------
    train_pred_mean.append(none_to_nan(train_metrics.get("pred_mean")))
    valid_pred_mean.append(none_to_nan(valid_metrics.get("pred_mean")))
    final_pred_mean.append(none_to_nan(final_metrics.get("pred_mean")))

    train_pred_std.append(none_to_nan(train_metrics.get("pred_std")))
    valid_pred_std.append(none_to_nan(valid_metrics.get("pred_std")))
    final_pred_std.append(none_to_nan(final_metrics.get("pred_std")))

    train_pred_pos_rate.append(none_to_nan(train_metrics.get("pred_pos_rate@thr")))
    valid_pred_pos_rate.append(none_to_nan(valid_metrics.get("pred_pos_rate@thr")))
    final_pred_pos_rate.append(none_to_nan(final_metrics.get("pred_pos_rate@thr")))

    # -------------------------
    # Losses
    # -------------------------
    loss_total.append(none_to_nan(train_stat.get("loss")))
    loss_cls.append(none_to_nan(train_stat.get("loss_cls")))
    loss_reg.append(none_to_nan(train_stat.get("loss_reg")))
    loss_contact.append(none_to_nan(train_stat.get("loss_contact")))

    contact_gap.append(none_to_nan(train_stat.get("contact_gap")))
    contact_pos_mean.append(none_to_nan(train_stat.get("contact_pos_mean")))
    contact_neg_mean.append(none_to_nan(train_stat.get("contact_neg_mean")))

    # -------------------------
    # Regression
    # -------------------------
    train_rmse.append(none_to_nan(train_reg_metrics.get("rmse")))
    valid_rmse.append(none_to_nan(valid_reg_metrics.get("rmse")))
    final_rmse.append(none_to_nan(final_reg_metrics.get("rmse")))

    train_mae.append(none_to_nan(train_reg_metrics.get("mae")))
    valid_mae.append(none_to_nan(valid_reg_metrics.get("mae")))
    final_mae.append(none_to_nan(final_reg_metrics.get("mae")))

    train_spearman.append(none_to_nan(train_reg_metrics.get("spearman")))
    valid_spearman.append(none_to_nan(valid_reg_metrics.get("spearman")))
    final_spearman.append(none_to_nan(final_reg_metrics.get("spearman")))

    train_pearson.append(none_to_nan(train_reg_metrics.get("pearson")))
    valid_pearson.append(none_to_nan(valid_reg_metrics.get("pearson")))
    final_pearson.append(none_to_nan(final_reg_metrics.get("pearson")))

    # -------------------------
    # New group ranking metrics
    # -------------------------
    train_n_groups.append(none_to_nan(train_rank.get("n_groups")))
    valid_n_groups.append(none_to_nan(valid_rank.get("n_groups")))
    final_n_groups.append(none_to_nan(final_rank.get("n_groups")))

    train_top1_exact.append(none_to_nan(train_rank.get("top1_exact")))
    valid_top1_exact.append(none_to_nan(valid_rank.get("top1_exact")))
    final_top1_exact.append(none_to_nan(final_rank.get("top1_exact")))

    train_best1.append(none_to_nan(train_rank.get("best_in_top1pct")))
    valid_best1.append(none_to_nan(valid_rank.get("best_in_top1pct")))
    final_best1.append(none_to_nan(final_rank.get("best_in_top1pct")))

    train_best5.append(none_to_nan(train_rank.get("best_in_top5pct")))
    valid_best5.append(none_to_nan(valid_rank.get("best_in_top5pct")))
    final_best5.append(none_to_nan(final_rank.get("best_in_top5pct")))

    train_best10.append(none_to_nan(train_rank.get("best_in_top10pct")))
    valid_best10.append(none_to_nan(valid_rank.get("best_in_top10pct")))
    final_best10.append(none_to_nan(final_rank.get("best_in_top10pct")))

    train_ndcg1.append(none_to_nan(train_rank.get("ndcg1")))
    valid_ndcg1.append(none_to_nan(valid_rank.get("ndcg1")))
    final_ndcg1.append(none_to_nan(final_rank.get("ndcg1")))

    train_ndcg5.append(none_to_nan(train_rank.get("ndcg5")))
    valid_ndcg5.append(none_to_nan(valid_rank.get("ndcg5")))
    final_ndcg5.append(none_to_nan(final_rank.get("ndcg5")))

    train_ndcg10.append(none_to_nan(train_rank.get("ndcg10")))
    valid_ndcg10.append(none_to_nan(valid_rank.get("ndcg10")))
    final_ndcg10.append(none_to_nan(final_rank.get("ndcg10")))


if not epochs:
    raise RuntimeError(f"No epoch_*.json files found in: {json_dir}")


# =========================================================
# Best epoch
# =========================================================
best_i = max(range(len(epochs)), key=lambda i: safe_float(valid_auc[i]))
best_epoch = epochs[best_i]

best_final_i = max(range(len(epochs)), key=lambda i: safe_float(final_auc[i]))


# =========================================================
# Print summary
# =========================================================
print("===== BEST BY VALID AUROC =====")
print(f"best epoch       : {epochs[best_i]}")
print(f"valid AUROC      : {safe_float(valid_auc[best_i]):.6f}")
print(f"final AUROC      : {safe_float(final_auc[best_i]):.6f}")
print(f"train AUROC      : {safe_float(train_auc[best_i]):.6f}")

print(f"valid AP         : {safe_float(valid_ap[best_i]):.6f}")
print(f"final AP         : {safe_float(final_ap[best_i]):.6f}")

print(f"valid EF1        : {safe_float(valid_ef1[best_i]):.6f}")
print(f"final EF1        : {safe_float(final_ef1[best_i]):.6f}")

print(f"valid r1 old     : {safe_float(valid_r1[best_i]):.6f}")
print(f"final r1 old     : {safe_float(final_r1[best_i]):.6f}")

print(f"valid Top1Exact  : {safe_float(valid_top1_exact[best_i]):.6f}")
print(f"final Top1Exact  : {safe_float(final_top1_exact[best_i]):.6f}")

print(f"valid Best@1%    : {safe_float(valid_best1[best_i]):.6f}")
print(f"final Best@1%    : {safe_float(final_best1[best_i]):.6f}")
print(f"valid Best@5%    : {safe_float(valid_best5[best_i]):.6f}")
print(f"final Best@5%    : {safe_float(final_best5[best_i]):.6f}")
print(f"valid Best@10%   : {safe_float(valid_best10[best_i]):.6f}")
print(f"final Best@10%   : {safe_float(final_best10[best_i]):.6f}")

print(f"valid NDCG1      : {safe_float(valid_ndcg1[best_i]):.6f}")
print(f"final NDCG1      : {safe_float(final_ndcg1[best_i]):.6f}")
print(f"valid NDCG5      : {safe_float(valid_ndcg5[best_i]):.6f}")
print(f"final NDCG5      : {safe_float(final_ndcg5[best_i]):.6f}")
print(f"valid NDCG10     : {safe_float(valid_ndcg10[best_i]):.6f}")
print(f"final NDCG10     : {safe_float(final_ndcg10[best_i]):.6f}")

print(f"valid groups     : {safe_float(valid_n_groups[best_i]):.0f}")
print(f"final groups     : {safe_float(final_n_groups[best_i]):.0f}")

print(f"valid RMSE       : {safe_float(valid_rmse[best_i]):.6f}")
print(f"final RMSE       : {safe_float(final_rmse[best_i]):.6f}")
print(f"valid Spearman   : {safe_float(valid_spearman[best_i]):.6f}")
print(f"final Spearman   : {safe_float(final_spearman[best_i]):.6f}")
print(f"valid Pearson    : {safe_float(valid_pearson[best_i]):.6f}")
print(f"final Pearson    : {safe_float(final_pearson[best_i]):.6f}")

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
print(f"final Top1Exact  : {safe_float(final_top1_exact[best_final_i]):.6f}")
print(f"final Best@1%    : {safe_float(final_best1[best_final_i]):.6f}")
print(f"final NDCG10     : {safe_float(final_ndcg10[best_final_i]):.6f}")


# =========================================================
# Plots: classification
# =========================================================
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


# =========================================================
# Plots: EF / old r
# =========================================================
plot_metric(
    "ef_plot.png",
    "EF vs Epoch",
    "Enrichment Factor",
    [
        ("train EF1", train_ef1),
        ("valid EF1", valid_ef1),
        ("final EF1", final_ef1),
        ("valid EF5", valid_ef5),
        ("final EF5", final_ef5),
        ("valid EF10", valid_ef10),
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

plot_metric(
    "r_at_top_plot_old_positive_recall.png",
    "Old Positive Recall-like r@Top% vs Epoch",
    "r = recovered positives / all positives",
    [
        ("train r1", train_r1),
        ("valid r1", valid_r1),
        ("final r1", final_r1),
        ("valid r5", valid_r5),
        ("final r5", final_r5),
        ("valid r10", valid_r10),
        ("final r10", final_r10),
    ],
    best_epoch=best_epoch,
)


# =========================================================
# Plots: new true top/best-ligand ranking metrics
# =========================================================
plot_metric(
    "top1_exact_plot.png",
    "Top1 Exact Hit Rate vs Epoch",
    "Fraction of proteins where predicted #1 equals true #1",
    [
        ("train Top1Exact", train_top1_exact),
        ("valid Top1Exact", valid_top1_exact),
        ("final Top1Exact", final_top1_exact),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "best_ligand_recovery_at_top_percent_plot.png",
    "Best Ligand Recovery vs Epoch",
    "Fraction of proteins where true best ligand is in predicted top X%",
    [
        ("valid Best@1%", valid_best1),
        ("final Best@1%", final_best1),
        ("valid Best@5%", valid_best5),
        ("final Best@5%", final_best5),
        ("valid Best@10%", valid_best10),
        ("final Best@10%", final_best10),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "final_best_ligand_recovery_plot.png",
    "Final Eval Best Ligand Recovery vs Epoch",
    "Recovery",
    [
        ("final Best@1%", final_best1),
        ("final Best@5%", final_best5),
        ("final Best@10%", final_best10),
        ("final Top1Exact", final_top1_exact),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "ndcg_plot.png",
    "NDCG vs Epoch",
    "NDCG",
    [
        ("valid NDCG1", valid_ndcg1),
        ("final NDCG1", final_ndcg1),
        ("valid NDCG5", valid_ndcg5),
        ("final NDCG5", final_ndcg5),
        ("valid NDCG10", valid_ndcg10),
        ("final NDCG10", final_ndcg10),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "final_ndcg_plot.png",
    "Final Eval NDCG vs Epoch",
    "NDCG",
    [
        ("final NDCG1", final_ndcg1),
        ("final NDCG5", final_ndcg5),
        ("final NDCG10", final_ndcg10),
    ],
    best_epoch=best_epoch,
)


# =========================================================
# Plots: prediction distribution
# =========================================================
plot_metric(
    "pred_distribution_plot.png",
    "Prediction Distribution vs Epoch",
    "Value",
    [
        ("train pred mean", train_pred_mean),
        ("valid pred mean", valid_pred_mean),
        ("final pred mean", final_pred_mean),
        ("train pred std", train_pred_std),
        ("valid pred std", valid_pred_std),
        ("final pred std", final_pred_std),
    ],
    best_epoch=best_epoch,
)

plot_metric(
    "pred_pos_rate_plot.png",
    "Predicted Positive Rate @ Threshold vs Epoch",
    "Predicted positive rate",
    [
        ("train pred_pos_rate", train_pred_pos_rate),
        ("valid pred_pos_rate", valid_pred_pos_rate),
        ("final pred_pos_rate", final_pred_pos_rate),
    ],
    best_epoch=best_epoch,
)


# =========================================================
# Plots: losses
# =========================================================
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


# =========================================================
# Plots: regression
# =========================================================
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