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


files = sorted(files, key=get_epoch_from_name)

epochs = []

train_auc, valid_auc, final_auc = [], [], []
train_ap, valid_ap, final_ap = [], [], []
train_f1, valid_f1, final_f1 = [], [], []

loss_total, loss_cls, loss_reg, loss_contact = [], [], [], []
contact_gap, contact_pos_mean, contact_neg_mean = [], [], []

train_ef1, train_ef5, train_ef10 = [], [], []
valid_ef1, valid_ef5, valid_ef10 = [], [], []
final_ef1, final_ef5, final_ef10 = [], [], []

train_r1, train_r5, train_r10 = [], [], []
valid_r1, valid_r5, valid_r10 = [], [], []
final_r1, final_r5, final_r10 = [], [], []

train_rmse, valid_rmse, final_rmse = [], [], []
train_mae, valid_mae, final_mae = [], [], []
train_spearman, valid_spearman, final_spearman = [], [], []
train_pearson, valid_pearson, final_pearson = [], [], []

train_pred_mean, valid_pred_mean, final_pred_mean = [], [], []
train_pred_std, valid_pred_std, final_pred_std = [], [], []
train_pred_pos_rate, valid_pred_pos_rate, final_pred_pos_rate = [], [], []


for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    ep = d.get("epoch", get_epoch_from_name(fp))
    epochs.append(ep)

    # new format, with fallback to old names
    train_stat = get_dict(d, "train_stat", "train_loss")
    train_metrics = get_dict(d, "train_metrics", "train")
    valid_metrics = get_dict(d, "valid_metrics", "valid")
    final_metrics = get_dict(d, "final_eval_metrics", "final_eval", "final")

    train_reg_metrics = get_dict(d, "train_reg_metrics")
    valid_reg_metrics = get_dict(d, "valid_reg_metrics")
    final_reg_metrics = get_dict(d, "final_eval_reg_metrics", "final_reg_metrics")

    # classification
    train_auc.append(none_to_nan(train_metrics.get("auroc")))
    valid_auc.append(none_to_nan(valid_metrics.get("auroc")))
    final_auc.append(none_to_nan(final_metrics.get("auroc")))

    train_ap.append(none_to_nan(train_metrics.get("ap")))
    valid_ap.append(none_to_nan(valid_metrics.get("ap")))
    final_ap.append(none_to_nan(final_metrics.get("ap")))

    train_f1.append(none_to_nan(train_metrics.get("f1")))
    valid_f1.append(none_to_nan(valid_metrics.get("f1")))
    final_f1.append(none_to_nan(final_metrics.get("f1")))

    train_ef1.append(none_to_nan(train_metrics.get("ef1")))
    train_ef5.append(none_to_nan(train_metrics.get("ef5")))
    train_ef10.append(none_to_nan(train_metrics.get("ef10")))

    valid_ef1.append(none_to_nan(valid_metrics.get("ef1")))
    valid_ef5.append(none_to_nan(valid_metrics.get("ef5")))
    valid_ef10.append(none_to_nan(valid_metrics.get("ef10")))

    final_ef1.append(none_to_nan(final_metrics.get("ef1")))
    final_ef5.append(none_to_nan(final_metrics.get("ef5")))
    final_ef10.append(none_to_nan(final_metrics.get("ef10")))

    # r1/r5/r10 in your new log
    train_r1.append(none_to_nan(train_metrics.get("r1")))
    train_r5.append(none_to_nan(train_metrics.get("r5")))
    train_r10.append(none_to_nan(train_metrics.get("r10")))

    valid_r1.append(none_to_nan(valid_metrics.get("r1")))
    valid_r5.append(none_to_nan(valid_metrics.get("r5")))
    valid_r10.append(none_to_nan(valid_metrics.get("r10")))

    final_r1.append(none_to_nan(final_metrics.get("r1")))
    final_r5.append(none_to_nan(final_metrics.get("r5")))
    final_r10.append(none_to_nan(final_metrics.get("r10")))

    # prediction distribution
    train_pred_mean.append(none_to_nan(train_metrics.get("pred_mean")))
    valid_pred_mean.append(none_to_nan(valid_metrics.get("pred_mean")))
    final_pred_mean.append(none_to_nan(final_metrics.get("pred_mean")))

    train_pred_std.append(none_to_nan(train_metrics.get("pred_std")))
    valid_pred_std.append(none_to_nan(valid_metrics.get("pred_std")))
    final_pred_std.append(none_to_nan(final_metrics.get("pred_std")))

    train_pred_pos_rate.append(none_to_nan(train_metrics.get("pred_pos_rate@thr")))
    valid_pred_pos_rate.append(none_to_nan(valid_metrics.get("pred_pos_rate@thr")))
    final_pred_pos_rate.append(none_to_nan(final_metrics.get("pred_pos_rate@thr")))

    # losses
    loss_total.append(none_to_nan(train_stat.get("loss")))
    loss_cls.append(none_to_nan(train_stat.get("loss_cls")))
    loss_reg.append(none_to_nan(train_stat.get("loss_reg")))
    loss_contact.append(none_to_nan(train_stat.get("loss_contact")))

    contact_gap.append(none_to_nan(train_stat.get("contact_gap")))
    contact_pos_mean.append(none_to_nan(train_stat.get("contact_pos_mean")))
    contact_neg_mean.append(none_to_nan(train_stat.get("contact_neg_mean")))

    # regression
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
print(f"valid AP         : {safe_float(valid_ap[best_i]):.6f}")
print(f"final AP         : {safe_float(final_ap[best_i]):.6f}")
print(f"valid EF1        : {safe_float(valid_ef1[best_i]):.6f}")
print(f"final EF1        : {safe_float(final_ef1[best_i]):.6f}")
print(f"valid r1         : {safe_float(valid_r1[best_i]):.6f}")
print(f"final r1         : {safe_float(final_r1[best_i]):.6f}")
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
    "r_at_top_plot.png",
    "Recall-like r@Top% vs Epoch",
    "r",
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


# regression
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