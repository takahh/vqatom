import json
import glob
import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# settings
# =========================================================
base_dir = "/Users/taka/Documents/dti_final_results/"

representations = ["smiles", "vq"]
exp_names = ["no_pretrain", "pretrained"]
seed_names = [f"seed{i}" for i in range(5)]

save_summary_csv = os.path.join(base_dir, "summary_best_valid_per_seed.csv")
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

        ef1 = safe_get(d, "final_eval_metrics", "ef1")
        ef5 = safe_get(d, "final_eval_metrics", "ef5")
        ef10 = safe_get(d, "final_eval_metrics", "ef10")
        pos_rate = safe_get(d, "final_eval_metrics", "pos_rate")

        hit1 = ef1 * pos_rate if is_number(ef1) and is_number(pos_rate) else None

        row = {
            "epoch": ep,
            "valid_auroc": safe_get(d, "valid_metrics", "auroc"),
            "final_auroc": safe_get(d, "final_eval_metrics", "auroc"),
            "final_ap": safe_get(d, "final_eval_metrics", "ap"),
            "EF1": ef1,
            "EF5": ef5,
            "EF10": ef10,
            "Hit@1%": hit1,
            "pos_rate": pos_rate,
        }

        rows.append(row)

    return rows


def choose_best_by_valid_auc(rows):
    candidates = [r for r in rows if is_number(r["valid_auroc"])]
    if not candidates:
        return None

    return sorted(candidates, key=lambda r: (-r["valid_auroc"], r["epoch"]))[0]


# =========================================================
# main collect
# =========================================================
all_best_rows = []

for rep in representations:
    for exp_name in exp_names:
        exp_dir = os.path.join(base_dir, rep, exp_name)

        for seed_name in seed_names:
            seed_dir = os.path.join(exp_dir, seed_name)

            if not os.path.isdir(seed_dir):
                continue

            rows = load_epoch_rows(seed_dir)
            if not rows:
                continue

            best = choose_best_by_valid_auc(rows)
            if best is None:
                continue

            all_best_rows.append({
                "representation": rep,
                "experiment": exp_name,
                "seed": seed_name,
                "AUC": best["final_auroc"],
                "AP": best["final_ap"],
                "EF1": best["EF1"],
                "EF5": best["EF5"],
                "EF10": best["EF10"],
                "Hit@1%": best["Hit@1%"],
            })

            print(f"[done] {rep} {exp_name} {seed_name} | AUC={best['final_auroc']:.4f}")


# =========================================================
# dataframe
# =========================================================
df = pd.DataFrame(all_best_rows)
df.to_csv(save_summary_csv, index=False)

metric_cols = ["AUC", "AP", "EF1", "EF5", "EF10", "Hit@1%"]
grouped = df.groupby(["representation", "experiment"])[metric_cols].agg(["mean", "std"])
grouped.to_csv(save_mean_csv)


# =========================================================
# print LaTeX table rows
# =========================================================
print("\n=== LaTeX table rows ===")

order = [
    ("vq", "pretrained"),
    ("smiles", "pretrained"),
    ("vq", "no_pretrain"),
    ("smiles", "no_pretrain"),
]

def ms(sub, col):
    x = pd.to_numeric(sub[col], errors="coerce").dropna()
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"

def hit_percent(sub):
    x = pd.to_numeric(sub["Hit@1%"], errors="coerce").dropna()
    return f"{100*x.mean():.1f}\\%"

for rep, exp_name in order:
    sub = df[(df["representation"] == rep) & (df["experiment"] == exp_name)]
    if sub.empty:
        continue

    pretrain_label = "Yes" if exp_name == "pretrained" else "No"
    rep_name = "VQ-Atom" if rep == "vq" else "SMILES"

    print(
        f"{rep_name} & {pretrain_label} & "
        f"{ms(sub, 'AUC')} & "
        f"{ms(sub, 'EF1')} & "
        f"{ms(sub, 'EF5')} & "
        f"{ms(sub, 'EF10')} & "
        f"{hit_percent(sub)} \\\\"
    )


# =========================================================
# variance plot (2-line labels)
# =========================================================
df = pd.read_csv(save_summary_csv)

df["setting"] = df["representation"] + "_" + df["experiment"]

df["setting_label"] = df["setting"].replace({
    "vq_pretrained": "VQ-Atom\n(pre)",
    "smiles_pretrained": "SMILES\n(pre)",
    "vq_no_pretrain": "VQ-Atom\n(no pre)",
    "smiles_no_pretrain": "SMILES\n(no pre)",
})

order_plot = [
    "VQ-Atom\n(pre)",
    "SMILES\n(pre)",
    "VQ-Atom\n(no pre)",
    "SMILES\n(no pre)",
]
plt.figure(figsize=(6,4))

sns.stripplot(
    data=df,
    x="setting_label",
    y="AUC",
    order=order_plot,
    jitter=True,
    size=6
)

# mean line追加
means = df.groupby("setting_label")["AUC"].mean().reindex(order_plot)

for i, m in enumerate(means):
    plt.hlines(m, i-0.2, i+0.2)

plt.title("AUROC across seeds")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "variance_auroc.png"), dpi=200)
plt.close()
plt.figure(figsize=(6,4))

sns.stripplot(
    data=df,
    x="setting_label",
    y="EF1",
    order=order_plot,
    jitter=0.15,
    size=6
)

# mean線
means = df.groupby("setting_label")["EF1"].mean().reindex(order_plot)

for i, m in enumerate(means):
    plt.hlines(m, i-0.2, i+0.2)

plt.title("EF1 across seeds")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "variance_ef1.png"), dpi=200)
plt.close()
# EF5
plt.figure(figsize=(6,4))
sns.stripplot(data=df, x="setting_label", y="EF5", order=order_plot, jitter=0.15, size=6)
means = df.groupby("setting_label")["EF5"].mean().reindex(order_plot)
for i, m in enumerate(means):
    plt.hlines(m, i-0.2, i+0.2)
plt.title("EF5 across seeds")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "variance_ef5.png"), dpi=200)
plt.close()

# EF10
plt.figure(figsize=(6,4))
sns.stripplot(data=df, x="setting_label", y="EF10", order=order_plot, jitter=0.15, size=6)
means = df.groupby("setting_label")["EF10"].mean().reindex(order_plot)
for i, m in enumerate(means):
    plt.hlines(m, i-0.2, i+0.2)
plt.title("EF10 across seeds")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "variance_ef10.png"), dpi=200)
plt.close()