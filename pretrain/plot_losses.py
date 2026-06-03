import json
import pandas as pd
import matplotlib.pyplot as plt

logfile = "/Users/taka/Downloads/train_log (8).jsonl"

rows = []
with open(logfile) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except:
            pass

df = pd.DataFrame(rows)

print(df.columns.tolist())

# 自動判定
if "loss" in df.columns:
    loss_col = "loss"
elif "mlm_loss" in df.columns:
    loss_col = "mlm_loss"
else:
    raise ValueError("loss column not found")

print("using:", loss_col)

df = df[df["step"] >= 60000].copy()

df["loss_ma"] = df[loss_col].rolling(
    window=20,
    min_periods=1
).mean()

plt.figure(figsize=(12,6))

plt.plot(
    df["step"],
    df[loss_col],
    alpha=0.3,
    label="raw"
)

plt.plot(
    df["step"],
    df["loss_ma"],
    linewidth=2,
    label="MA20"
)

plt.xlabel("Step")
plt.ylabel(loss_col)
plt.title(f"{loss_col} after step 60000")
plt.grid(True)
plt.legend()
plt.show()