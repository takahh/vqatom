import json, glob, os, re, math
import pandas as pd

BASE = "/Users/taka/Documents/VQ-Atom_DTI_results"

RUNS = {}

# models = ["smiles", "smiles_pre", "vq_pre", "continu_pre"]
models = ["smiles", "smiles_pre", "vq_pre", "vq", "continu", "continu_pre"]

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