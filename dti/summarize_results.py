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
# exp_names = ["smiles", "vq"]
exp_names = ["smiles_no_pretrained", "vq_no_pretrained"]
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


# =========================================================
# EF correction
# =========================================================
def correct_ef(raw, pos_rate):
    if is_number(raw) and is_number(pos_rate) and pos_rate > 0:
        return raw / pos_rate
    return None


# =========================================================
# load json
# =========================================================
def load_epoch_rows(json_dir):
    files = glob.glob(os.path.join(json_dir, "epoch_*.json"))
    files = sorted(files, key=get_epoch_from_name)

    rows = []
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)

        ep = d.get("epoch", get_epoch_from_name(fp))

        # raw EF (hit rate)
        raw_ef1 = safe_get(d, "final_eval_metrics", "ef1")
        raw_ef5 = safe_get(d, "final_eval_metrics", "ef5")
        raw_ef10 = safe_get(d, "final_eval_metrics", "ef10")

        pos_rate = safe_get(d, "final_eval_metrics", "pos_rate")

        row = {
            "epoch": ep,

            # train
            "train_auroc": safe_get(d, "train_metrics", "auroc"),
            "train_ap": safe_get(d, "train_metrics", "ap"),

            # valid
            "valid_auroc": safe_get(d, "valid_metrics", "auroc"),

            # final
            "final_auroc": safe_get(d, "final_eval_metrics", "auroc"),
            "final_ap": safe_get(d, "final_eval_metrics", "ap"),

            # corrected EF
            "final_ef1": correct_ef(raw_ef1, pos_rate),
            "final_ef5": correct_ef(raw_ef5, pos_rate),
            "final_ef10": correct_ef(raw_ef10, pos_rate),

            # debug
            "final_hit_rate1": raw_ef1,
            "final_hit_rate5": raw_ef5,
            "final_hit_rate10": raw_ef10,
            "final_pos_rate": pos_rate,
        }

        rows.append(row)

    return rows


def choose_best_by_valid_auc(rows):
    candidates = [r for r in rows if is_number(r["valid_auroc"])]
    if not candidates:
        return None
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
        plt.axvline(best_epoch, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
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
        rows = load_epoch_rows(seed_dir)
        if not rows:
            continue

        best = choose_best_by_valid_auc(rows)
        if best is None:
            continue

        best_epoch = best["epoch"]

        plot_auc(
            rows,
            os.path.join(seed_dir, "auc_curve.png"),
            f"{exp_name}",
            best_epoch=best_epoch,
        )

        out_row = {
            "experiment": exp_name,
            "seed": os.path.basename(seed_dir),
            "best_epoch": best_epoch,

            "valid_auroc": best["valid_auroc"],
            "final_auroc": best["final_auroc"],

            "EF1": best["final_ef1"],
            "EF5": best["final_ef5"],
            "EF10": best["final_ef10"],
        }

        all_best_rows.append(out_row)

        print(
            f"[done] {exp_name} {os.path.basename(seed_dir)} "
            f"| AUC={best['final_auroc']:.4f} "
            f"| EF1={best['final_ef1']:.2f}"
        )


# =========================================================
# save
# =========================================================
df = pd.DataFrame(all_best_rows)
df.to_csv(save_summary_csv, index=False)

grouped = df.groupby("experiment")[["final_auroc", "EF1", "EF5", "EF10"]].agg(["mean", "std"])
grouped.to_csv(save_mean_csv)

print("\n=== mean ± std ===")
for exp_name in df["experiment"].unique():
    sub = df[df["experiment"] == exp_name]

    def ms(col):
        x = pd.to_numeric(sub[col], errors="coerce").dropna()
        return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"

    print(
        f"{exp_name} | "
        f"AUC {ms('final_auroc')} | "
        f"EF1 {ms('EF1')} | "
        f"EF5 {ms('EF5')} | "
        f"EF10 {ms('EF10')}"
    )