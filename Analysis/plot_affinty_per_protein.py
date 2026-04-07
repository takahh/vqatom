import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/taka/Downloads/train.csv")

# proteinごとにgroup化
groups = df.groupby("seq")

# ligand数がある程度あるproteinに限定（重要）
groups = {k: v for k, v in groups if len(v) >= 5}

# ランダムに20個選ぶ
sample_keys = np.random.choice(list(groups.keys()), size=20, replace=False)

sampled = [groups[k] for k in sample_keys]
plt.figure(figsize=(12, 6))

for i, g in enumerate(sampled):
    y = g["y"].values
    x = np.ones_like(y) * i

    # 少し横にばらす（見やすくする）
    x = x + np.random.normal(0, 0.05, size=len(y))

    plt.scatter(x, y, alpha=0.6)

plt.xticks(range(len(sample_keys)), range(len(sample_keys)))
plt.xlabel("Protein (sampled)")
plt.ylabel("Affinity (y)")
plt.title("Affinity distribution per protein")
plt.show()