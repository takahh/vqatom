import json, glob, os, re, math
import pandas as pd

BASE = "/Users/taka/Documents/VQ-Atom_DTI_results"

RUNS = {}

# models = ["smiles", "smiles_pre", "vq_pre", "continu_pre"]
models = ["smiles", "smiles_pre", "vq_pre", "vq", "continu", "continu_pre", "continu_pre_simple_cross", "vq_pre_simple_cross",
          "continu_pre_simple_cross", "cont_simple_cross", "vq_pre_simple_cross", "vq_simple_cross"]

for model in models:
    for seed in range(5):
        RUNS[f"{model}_seed_{seed}"] = os.path.join(BASE, f"{model}_seed_{seed}")

# for seed in range(5):
#     RUNS[f"vq_pre_seed_{seed}"] = os.path.join(BASE, f"vq_pre_seed_{seed}")


def epoch_from_name(path):
    m = re.search(r"epoch_(\d+).*\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def get_dict(d, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            return v
    return {}


def f(x):
    try:
        x = float(x)
        return x if not math.isnan(x) else float("nan")
    except Exception:
        return float("nan")


def load_run(name, path):
    rows = []

    files = sorted(
        glob.glob(os.path.join(path, "epoch_*.json")),
        key=epoch_from_name
    )

    if not files:
        print(f"[WARN] no epoch_*.json found: {path}")

    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            d = json.load(fh)

        valid = get_dict(d, "valid_metrics", "valid")
        final = get_dict(d, "final_eval_metrics", "final_eval", "final")

        valid_rank = get_dict(d, "valid_rank_metrics", "valid_rank")
        final_rank = get_dict(d, "final_eval_rank_metrics", "final_rank")

        rows.append({
            "run": name,
            "epoch": d.get("epoch", epoch_from_name(fp)),

            "valid_auc": f(valid.get("auroc", valid.get("auc"))),
            "final_auc": f(final.get("auroc", final.get("auc"))),

            "valid_ef1": f(valid.get("ef1")),
            "final_ef1": f(final.get("ef1")),

            "valid_ef5": f(valid.get("ef5")),
            "final_ef5": f(final.get("ef5")),

            "valid_ef10": f(valid.get("ef10")),
            "final_ef10": f(final.get("ef10")),

            "valid_top1": f(valid_rank.get("top1_exact")),
            "final_top1": f(final_rank.get("top1_exact")),

            "valid_best1": f(valid_rank.get("best_in_top1pct")),
            "final_best1": f(final_rank.get("best_in_top1pct")),

            "valid_best5": f(valid_rank.get("best_in_top5pct")),
            "final_best5": f(final_rank.get("best_in_top5pct")),

            "valid_best10": f(valid_rank.get("best_in_top10pct")),
            "final_best10": f(final_rank.get("best_in_top10pct")),

            "valid_ndcg10": f(valid_rank.get("ndcg10")),
            "final_ndcg10": f(final_rank.get("ndcg10")),
        })

    return pd.DataFrame(rows)


df = pd.concat(
    [load_run(name, path) for name, path in RUNS.items()],
    ignore_index=True
)

print("\n===== BEST BY VALID NDCG10 =====")

results = []

for run, g in df.groupby("run"):
    g2 = g.dropna(subset=["valid_ndcg10"])

    if g2.empty:
        print(f"\n--- {run} ---")
        print("valid_ndcg10 が見つかりません")
        continue

    idx = g2["valid_ndcg10"].idxmax()

    row = g.loc[idx].copy()
    row["best_epoch"] = int(row["epoch"])

    results.append(row)

    print(f"\n--- {run} ---")
    print(f"epoch          : {int(row['epoch'])}")
    print(f"valid NDCG10   : {row['valid_ndcg10']:.4f}")
    print(f"final NDCG10   : {row['final_ndcg10']:.4f}")
    print(f"final Best@1%  : {row['final_best1']:.4f}")
    print(f"final Best@5%  : {row['final_best5']:.4f}")
    print(f"final Best@10% : {row['final_best10']:.4f}")
    print(f"final Top1     : {row['final_top1']:.4f}")
    print(f"final EF1      : {row['final_ef1']:.4f}")
    print(f"final EF5      : {row['final_ef5']:.4f}")
    print(f"final EF10     : {row['final_ef10']:.4f}")
    print(f"final AUROC    : {row['final_auc']:.4f}")


summary = pd.DataFrame(results)

import matplotlib.pyplot as plt

# plot target
plot_methods = {
    "vq_pre": {"label": "VQ-Atom+Pretrain", "linestyle": "-"},
    "continu_pre": {"label": "Continuous+Pretrain", "linestyle": "--"},
}

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for method, style in plot_methods.items():
    for seed in range(5):
        run_name = f"{method}_seed_{seed}"
        g = df[df["run"] == run_name].copy()
        if g.empty:
            print(f"[WARN] missing run: {run_name}")
            continue

        g = g.sort_values("epoch")
        ax.plot(
            g["epoch"],
            g["valid_ndcg10"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            alpha=0.75,
            label=style["label"] if seed == 0 else None,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation NDCG@10")
ax.set_title("Convergence of VQ-Atom and continuous representations")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ndcg10_convergence_vq_vs_continuous.pdf", bbox_inches="tight")
plt.savefig("ndcg10_convergence_vq_vs_continuous.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n===== SIDE-BY-SIDE =====")
print(summary[
    [
        "run",
        "epoch",
        "valid_ndcg10",
        "final_ndcg10",
        "final_best1",
        "final_best5",
        "final_best10",
        "final_top1",
        "final_ef1",
        "final_ef5",
        "final_ef10",
        "final_auc",
    ]
].to_string(index=False))


print("\n===== MEAN ± STD BY METHOD =====")

summary["method"] = summary["run"].str.replace(r"_seed_\d+$", "", regex=True)

metrics = [
    "final_ndcg10",
    "final_best1",
    "final_best5",
    "final_best10",
    "final_top1",
    "final_ef1",
    "final_ef5",
    "final_ef10",
    "final_auc",
    "best_epoch",
]

for method, g in summary.groupby("method"):
    print(f"\n--- {method} ---")
    for m in metrics:
        print(f"{m:15s}: {g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}")

# =========================
# Figure: NDCG@10 ablation
# Simple Cross vs Full Cross
# =========================

import numpy as np
import matplotlib.pyplot as plt

def mean_metric(method, metric):
    g = summary[summary["method"] == method]
    if g.empty:
        print(f"[WARN] missing method: {method}")
        return np.nan
    return g[metric].mean()

methods = [
    ("Continuous", "cont_simple_cross", "continu"),
    ("Continuous+Pretrain", "continu_pre_simple_cross", "continu_pre"),
    ("VQ-Atom", "vq_simple_cross", "vq"),
    ("VQ-Atom+Pretrain", "vq_pre_simple_cross", "vq_pre"),
]

labels = [m[0] for m in methods]
simple_vals = [mean_metric(m[1], "final_ndcg10") for m in methods]
full_vals   = [mean_metric(m[2], "final_ndcg10") for m in methods]

x = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(6.6, 4.2))

ax.bar(x - width/2, simple_vals, width, label="Simple Cross")
ax.bar(x + width/2, full_vals, width, label="Full Cross")

ax.set_ylabel("Final NDCG@10")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0.25, 0.70)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

for i, (s, f) in enumerate(zip(simple_vals, full_vals)):
    if not np.isnan(s):
        ax.text(x[i] - width/2, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    if not np.isnan(f):
        ax.text(x[i] + width/2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("ndcg10_cross_ablation_bar.pdf", bbox_inches="tight")
plt.savefig("ndcg10_cross_ablation_bar.png", dpi=300, bbox_inches="tight")
plt.show()