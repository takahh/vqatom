import json
import glob
import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# settings
# =========================================================
base_dir = "/Users/taka/Documents/dti_results/"
# exp_names = ["smiles", "vq"]   # results/smiles, results/dti
exp_names = ["smiles_no_pretrained", "vq_no_pretrained"]   # results/smiles, results/dti
seed_pattern = "seed*"

save_summary_csv = os.path.join(base_dir, "summary_best_valid_per_seed.csv")
save_summary_xlsx = os.path.join(base_dir, "summary_best_valid.xlsx")
save_mean_csv = os.path.join(base_dir, "summary_mean_std.csv")


# =========================================================
# utils
# =========================================================
def get_epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def is_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and not (isinstance(x, float) and math.isnan(x))
    )


def load_epoch_rows(json_dir):
    files = glob.glob(os.path.join(json_dir, "epoch_*.json"))
    files = sorted(files, key=get_epoch_from_name)

    rows = []
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)

        ep = d.get("epoch", get_epoch_from_name(fp))

        row = {
            "epoch": ep,

            # train
            "train_auroc": safe_get(d, "train_metrics", "auroc"),
            "train_ap": safe_get(d, "train_metrics", "ap"),
            "train_f1": safe_get(d, "train_metrics", "f1"),
            "train_rmse": safe_get(d, "train_metrics", "rmse"),
            "train_sp": safe_get(d, "train_metrics", "sp"),

            # valid
            "valid_auroc": safe_get(d, "valid_metrics", "auroc"),
            "valid_ap": safe_get(d, "valid_metrics", "ap"),
            "valid_f1": safe_get(d, "valid_metrics", "f1"),
            "valid_rmse": safe_get(d, "valid_metrics", "rmse"),
            "valid_sp": safe_get(d, "valid_metrics", "sp"),

            # final
            "final_auroc": safe_get(d, "final_eval_metrics", "auroc"),
            "final_ap": safe_get(d, "final_eval_metrics", "ap"),
            "final_f1": safe_get(d, "final_eval_metrics", "f1"),
            "final_rmse": safe_get(d, "final_eval_metrics", "rmse"),
            "final_sp": safe_get(d, "final_eval_metrics", "sp"),
            "final_ef1": safe_get(d, "final_eval_metrics", "ef1"),
            "final_ef5": safe_get(d, "final_eval_metrics", "ef5"),
            "final_ef10": safe_get(d, "final_eval_metrics", "ef10"),

            # loss
            "loss": safe_get(d, "train_stat", "loss"),
            "loss_y": safe_get(d, "train_stat", "loss_y"),
            "loss_base": safe_get(d, "train_stat", "loss_base"),
            "loss_delta": safe_get(d, "train_stat", "loss_delta"),
        }
        rows.append(row)

    return rows


def choose_best_by_valid_auc(rows):
    candidates = [r for r in rows if is_number(r["valid_auroc"])]
    if not candidates:
        return None
    # valid AUC 最大、同点なら若い epoch
    candidates = sorted(candidates, key=lambda r: (-r["valid_auroc"], r["epoch"]))
    return candidates[0]


# =========================================================
# plotting
# =========================================================
def plot_auc(rows, out_path, title, best_epoch=None):
    epochs = [r["epoch"] for r in rows]
    train_auc = [r["train_auroc"] for r in rows]
    valid_auc = [r["valid_auroc"] for r in rows]
    final_auc = [r["final_auroc"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_auc, marker="o", label="train auc")
    plt.plot(epochs, valid_auc, marker="o", label="valid auc")
    plt.plot(epochs, final_auc, marker="o", label="final auc")

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best valid epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss(rows, out_path, title, best_epoch=None):
    epochs = [r["epoch"] for r in rows]
    loss_total = [r["loss"] for r in rows]
    loss_y = [r["loss_y"] for r in rows]
    loss_base = [r["loss_base"] for r in rows]
    loss_delta = [r["loss_delta"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_total, marker="o", label="loss total")
    plt.plot(epochs, loss_y, marker="o", label="loss y")
    plt.plot(epochs, loss_base, marker="o", label="loss base")
    plt.plot(epochs, loss_delta, marker="o", label="loss delta")

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best valid epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# main
# =========================================================
all_best_rows = []

for exp_name in exp_names:
    exp_dir = os.path.join(base_dir, exp_name)
    seed_dirs = sorted(glob.glob(os.path.join(exp_dir, seed_pattern)))

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        # json_dir = os.path.join(seed_dir, "json")
        json_dir = seed_dir

        if not os.path.isdir(json_dir):
            print(f"[skip] no json dir: {json_dir}")
            continue

        rows = load_epoch_rows(json_dir)
        if not rows:
            print(f"[skip] no epoch files: {json_dir}")
            continue

        best = choose_best_by_valid_auc(rows)
        if best is None:
            print(f"[warn] no valid_auroc found: {json_dir}")
            continue

        best_epoch = best["epoch"]

        # save plots
        auc_plot_path = os.path.join(seed_dir, "auc_curve.png")
        loss_plot_path = os.path.join(seed_dir, "loss_curve.png")

        plot_auc(
            rows,
            auc_plot_path,
            f"{exp_name}/{seed_name} AUROC vs Epoch",
            best_epoch=best_epoch,
        )
        plot_loss(
            rows,
            loss_plot_path,
            f"{exp_name}/{seed_name} Train Loss vs Epoch",
            best_epoch=best_epoch,
        )

        out_row = {
            "experiment": exp_name,
            "seed": seed_name,
            "best_epoch_by_valid_auc": best["epoch"],

            "train_auroc": best["train_auroc"],
            "valid_auroc": best["valid_auroc"],
            "final_auroc": best["final_auroc"],

            "train_ap": best["train_ap"],
            "valid_ap": best["valid_ap"],
            "final_ap": best["final_ap"],

            "train_f1": best["train_f1"],
            "valid_f1": best["valid_f1"],
            "final_f1": best["final_f1"],

            "train_rmse": best["train_rmse"],
            "valid_rmse": best["valid_rmse"],
            "final_rmse": best["final_rmse"],

            "train_sp": best["train_sp"],
            "valid_sp": best["valid_sp"],
            "final_sp": best["final_sp"],

            "final_ef1": best["final_ef1"],
            "final_ef5": best["final_ef5"],
            "final_ef10": best["final_ef10"],

            "loss": best["loss"],
            "loss_y": best["loss_y"],
            "loss_base": best["loss_base"],
            "loss_delta": best["loss_delta"],

            "json_dir": json_dir,
            "auc_plot": auc_plot_path,
            "loss_plot": loss_plot_path,
        }
        all_best_rows.append(out_row)

        final_auc_str = f"{best['final_auroc']:.4f}" if is_number(best["final_auroc"]) else "None"
        valid_auc_str = f"{best['valid_auroc']:.4f}" if is_number(best["valid_auroc"]) else "None"

        print(
            f"[done] {exp_name}/{seed_name} | "
            f"best epoch={best['epoch']} | "
            f"valid_auc={valid_auc_str} | "
            f"final_auc={final_auc_str}"
        )

# =========================================================
# save per-seed summary
# =========================================================
df = pd.DataFrame(all_best_rows)

if len(df) == 0:
    print("No results found.")
    raise SystemExit

cols_order = [
    "experiment", "seed", "best_epoch_by_valid_auc",
    "train_auroc", "valid_auroc", "final_auroc",
    "train_ap", "valid_ap", "final_ap",
    "train_f1", "valid_f1", "final_f1",
    "train_rmse", "valid_rmse", "final_rmse",
    "train_sp", "valid_sp", "final_sp",
    "final_ef1", "final_ef5", "final_ef10",
    "loss", "loss_y", "loss_base", "loss_delta",
    "json_dir", "auc_plot", "loss_plot",
]
df = df[cols_order]

df.to_csv(save_summary_csv, index=False)

# =========================================================
# mean/std over seeds
# =========================================================
metric_cols = [
    "best_epoch_by_valid_auc",
    "train_auroc", "valid_auroc", "final_auroc",
    "train_ap", "valid_ap", "final_ap",
    "train_f1", "valid_f1", "final_f1",
    "train_rmse", "valid_rmse", "final_rmse",
    "train_sp", "valid_sp", "final_sp",
    "final_ef1", "final_ef5", "final_ef10",
    "loss", "loss_y", "loss_base", "loss_delta",
]

grouped = df.groupby("experiment")[metric_cols].agg(["mean", "std"])
grouped.to_csv(save_mean_csv)

# =========================================================
# Excel
# =========================================================
with pd.ExcelWriter(save_summary_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="best_per_seed", index=False)
    grouped.to_excel(writer, sheet_name="mean_std")

print("\nSaved:")
print("  ", save_summary_csv)
print("  ", save_mean_csv)
print("  ", save_summary_xlsx)

# =========================================================
# print compact summary
# =========================================================
print("\n=== best epoch selected by valid AUROC ===")
show_cols = [
    "experiment", "seed", "best_epoch_by_valid_auc",
    "valid_auroc", "final_auroc", "final_ef1", "final_ef5", "final_ef10"
]
print(df[show_cols].to_string(index=False))

print("\n=== mean ± std over seeds ===")
for exp_name in df["experiment"].unique():
    sub = df[df["experiment"] == exp_name]

    def ms(col):
        x = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(x) == 0:
            return "NA"
        if len(x) == 1:
            return f"{x.mean():.4f} ± 0.0000"
        return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"

    print(
        f"{exp_name:>6} | "
        f"valid_auc {ms('valid_auroc')} | "
        f"final_auc {ms('final_auroc')} | "
        f"EF1 {ms('final_ef1')} | "
        f"EF5 {ms('final_ef5')} | "
        f"EF10 {ms('final_ef10')}"
    )