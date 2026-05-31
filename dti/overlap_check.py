import pandas as pd

base = "/Users/taka/Downloads"
cluster_path = "/Users/taka/Desktop/kiba_fast_out/seq_to_cluster_40id.csv"

train = pd.read_csv(f"{base}/train.csv")
valid = pd.read_csv(f"{base}/valid.csv")
test  = pd.read_csv(f"{base}/test.csv")
seqcl = pd.read_csv(cluster_path)

seq_to_cluster = dict(zip(seqcl["seq"], seqcl["protein_cluster"]))

for name, df in [("train", train), ("valid", valid), ("test", test)]:
    df["protein_cluster"] = df["seq"].map(seq_to_cluster)
    print(name, "rows", len(df), "missing_cluster", df["protein_cluster"].isna().sum())

def show_overlap(a_name, a, b_name, b):
    inter = set(a["protein_cluster"]) & set(b["protein_cluster"])
    inter.discard(float("nan"))
    print(f"{a_name}-{b_name} protein_cluster overlap:", len(inter))

print("\n=== 40% protein cluster overlap ===")
show_overlap("train", train, "valid", valid)
show_overlap("train", train, "test", test)
show_overlap("valid", valid, "test", test)

print("\n=== exact seq overlap ===")
print("train-valid:", len(set(train["seq"]) & set(valid["seq"])))
print("train-test :", len(set(train["seq"]) & set(test["seq"])))
print("valid-test :", len(set(valid["seq"]) & set(test["seq"])))

print("\n=== exact pair overlap ===")
def pairs(df):
    return set(zip(df["seq"], df["smiles"]))

print("train-valid:", len(pairs(train) & pairs(valid)))
print("train-test :", len(pairs(train) & pairs(test)))
print("valid-test :", len(pairs(valid) & pairs(test)))