import json, glob, os, re, math
import pandas as pd

BASE = "/Users/taka/Downloads/"

RUNS = {
    "continuous": os.path.join(BASE, "cont"),
    "vqatom_pretrained": os.path.join(BASE, "vqa_pr"),
    "vqatom": os.path.join(BASE, "vqa"),
}


def epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
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
    except:
        return float("nan")


def load_run(name, path):
    rows = []

    files = sorted(
        glob.glob(os.path.join(path, "epoch_*.json")),
        key=epoch_from_name
    )

    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)

        valid = get_dict(d, "valid_metrics", "valid")
        final = get_dict(d, "final_eval_metrics", "final_eval", "final")

        valid_rank = get_dict(d, "valid_rank_metrics")
        final_rank = get_dict(d, "final_eval_rank_metrics")

        rows.append({
            "run": name,
            "epoch": d.get("epoch", epoch_from_name(fp)),

            "valid_auc": f(valid.get("auroc")),
            "final_auc": f(final.get("auroc")),

            "valid_ef1": f(valid.get("ef1")),
            "final_ef1": f(final.get("ef1")),

            "valid_ef10": f(valid.get("ef10")),
            "final_ef10": f(final.get("ef10")),

            "valid_top1": f(valid_rank.get("top1_exact")),
            "final_top1": f(final_rank.get("top1_exact")),

            "valid_best1": f(valid_rank.get("best_in_top1pct")),
            "final_best1": f(final_rank.get("best_in_top1pct")),

            "valid_ndcg10": f(valid_rank.get("ndcg10")),
            "final_ndcg10": f(final_rank.get("ndcg10")),
        })

    return pd.DataFrame(rows)


df = pd.concat([
    load_run(name, path)
    for name, path in RUNS.items()
], ignore_index=True)


print("\n===== BEST BY VALID NDCG10 =====")

results = []

for run, g in df.groupby("run"):

    idx = g["valid_ndcg10"].idxmax()
    row = g.loc[idx]

    results.append(row)

    print(f"\n--- {run} ---")
    print(f"epoch         : {int(row['epoch'])}")
    print(f"valid NDCG10  : {row['valid_ndcg10']:.4f}")
    print(f"final NDCG10  : {row['final_ndcg10']:.4f}")
    print(f"final Best@1% : {row['final_best1']:.4f}")
    print(f"final Top1    : {row['final_top1']:.4f}")
    print(f"final EF1     : {row['final_ef1']:.4f}")
    print(f"final EF10    : {row['final_ef10']:.4f}")
    print(f"final AUROC   : {row['final_auc']:.4f}")


summary = pd.DataFrame(results)

print("\n===== SIDE-BY-SIDE =====")
print(summary[
    [
        "run",
        "epoch",
        "valid_ndcg10",
        "final_ndcg10",
        "final_best1",
        "final_top1",
        "final_ef1",
        "final_ef10",
        "final_auc",
    ]
].to_string(index=False))