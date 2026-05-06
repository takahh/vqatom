import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== paths =====
train_path = "/Users/taka/Downloads/train.csv"
valid_path = "/Users/taka/Downloads/valid.csv"
test_path  = "/Users/taka/Downloads/test.csv"

out_dir = "/Users/taka/Downloads/kiba_split_analysis"
os.makedirs(out_dir, exist_ok=True)

Y_THR = 12.1

def load_df(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"seq": "protein", "smiles": "ligand", "y": "score"})
    df["label"] = (df["score"] >= Y_THR).astype(int)
    return df

train = load_df(train_path)
valid = load_df(valid_path)
test  = load_df(test_path)


def plot_pos_rate_distribution(series, title, fname):
    plt.figure(figsize=(5, 4))
    plt.hist(series, bins=50, alpha=0.7)

    plt.axvline(series.median(), linestyle="--", label=f"median={series.median():.2f}")
    plt.axvline(series.mean(), linestyle="-", label=f"mean={series.mean():.2f}")

    plt.xlim(0, 1)  # ← これ追加

    plt.title(title)
    plt.xlabel("positive rate")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_pairs_distribution(series, title, fname, log=True):
    x = np.log1p(series) if log else series

    plt.figure(figsize=(5, 4))
    plt.hist(x, bins=50, alpha=0.7)
    plt.title(title + (" (log1p)" if log else ""))
    plt.xlabel("log1p(pairs per entity)" if log else "pairs per entity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def analyze(df, name="dataset"):
    print(f"\n===== {name} =====")

    n = len(df)
    pos_rate = df["label"].mean()
    n_prot = df["protein"].nunique()
    n_lig = df["ligand"].nunique()

    print(f"#pairs: {n}")
    print(f"positive rate: {pos_rate:.4f}")
    print(f"#unique proteins: {n_prot}")
    print(f"#unique ligands: {n_lig}")

    prot_summary = (
        df.groupby("protein")["label"]
        .agg(n_pairs="size", n_pos="sum", pos_rate="mean")
        .reset_index()
    )

    lig_summary = (
        df.groupby("ligand")["label"]
        .agg(n_pairs="size", n_pos="sum", pos_rate="mean")
        .reset_index()
    )

    print("\n[per protein]")
    print(f"pairs per protein: mean={prot_summary['n_pairs'].mean():.2f}, median={prot_summary['n_pairs'].median():.2f}")
    print(f"pos rate per protein: mean={prot_summary['pos_rate'].mean():.4f}, std={prot_summary['pos_rate'].std():.4f}")
    print(f"pos rate per protein: median={prot_summary['pos_rate'].median():.4f}, min={prot_summary['pos_rate'].min():.4f}, max={prot_summary['pos_rate'].max():.4f}")

    print("\n[per ligand]")
    print(f"pairs per ligand: mean={lig_summary['n_pairs'].mean():.2f}, median={lig_summary['n_pairs'].median():.2f}")
    print(f"pos rate per ligand: mean={lig_summary['pos_rate'].mean():.4f}, std={lig_summary['pos_rate'].std():.4f}")
    print(f"pos rate per ligand: median={lig_summary['pos_rate'].median():.4f}, min={lig_summary['pos_rate'].min():.4f}, max={lig_summary['pos_rate'].max():.4f}")

    # save csv
    prot_summary.to_csv(os.path.join(out_dir, f"{name}_per_protein.csv"), index=False)
    lig_summary.to_csv(os.path.join(out_dir, f"{name}_per_ligand.csv"), index=False)

    # save plots
    plot_pairs_distribution(
        prot_summary["n_pairs"],
        f"{name}: pairs per protein",
        os.path.join(out_dir, f"{name}_pairs_per_protein.png"),
        log=True,
    )
    plot_pairs_distribution(
        lig_summary["n_pairs"],
        f"{name}: pairs per ligand",
        os.path.join(out_dir, f"{name}_pairs_per_ligand.png"),
        log=True,
    )
    plot_pos_rate_distribution(
        prot_summary["pos_rate"],
        f"{name}: positive rate per protein",
        os.path.join(out_dir, f"{name}_pos_rate_per_protein.png"),
    )
    plot_pos_rate_distribution(
        lig_summary["pos_rate"],
        f"{name}: positive rate per ligand",
        os.path.join(out_dir, f"{name}_pos_rate_per_ligand.png"),
    )

    return {
        "split": name,
        "n_pairs": n,
        "pos_rate": pos_rate,
        "n_protein": n_prot,
        "n_ligand": n_lig,
        "prot_pairs_mean": prot_summary["n_pairs"].mean(),
        "prot_pairs_median": prot_summary["n_pairs"].median(),
        "lig_pairs_mean": lig_summary["n_pairs"].mean(),
        "lig_pairs_median": lig_summary["n_pairs"].median(),
        "prot_pos_rate_std": prot_summary["pos_rate"].std(),
        "lig_pos_rate_std": lig_summary["pos_rate"].std(),
    }


def overlap(a, b, name_a="A", name_b="B"):
    prot_overlap = len(set(a["protein"]) & set(b["protein"]))
    lig_overlap = len(set(a["ligand"]) & set(b["ligand"]))

    print(f"\n===== overlap {name_a} vs {name_b} =====")
    print(f"protein overlap: {prot_overlap}")
    print(f"ligand overlap: {lig_overlap}")

    return {
        "split_a": name_a,
        "split_b": name_b,
        "protein_overlap": prot_overlap,
        "ligand_overlap": lig_overlap,
    }


# ===== run =====
stats = [
    analyze(train, "train"),
    analyze(valid, "valid"),
    analyze(test, "test"),
]

pd.DataFrame(stats).to_csv(os.path.join(out_dir, "split_summary.csv"), index=False)

overlaps = [
    overlap(train, valid, "train", "valid"),
    overlap(train, test, "train", "test"),
    overlap(valid, test, "valid", "test"),
]

pd.DataFrame(overlaps).to_csv(os.path.join(out_dir, "split_overlap.csv"), index=False)

print(f"\nSaved outputs to: {out_dir}")