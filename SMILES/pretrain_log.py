import json
import matplotlib.pyplot as plt

log_path = "/Users/taka/Downloads/SMILES_pretrain/train_log.json"

steps = []
losses = []
valid_steps = []
valid_losses = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        # train log
        if rec.get("event") is None and "mlm_loss" in rec:
            steps.append(rec["step"])
            losses.append(rec["mlm_loss"])

        # valid log
        if rec.get("event") == "valid" and "mlm_loss" in rec:
            valid_steps.append(rec["step"])
            valid_losses.append(rec["mlm_loss"])

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, label="train MLM loss")
if valid_steps:
    plt.plot(valid_steps, valid_losses, marker="o", linestyle="-", label="valid MLM loss")
plt.xlabel("step")
plt.ylabel("MLM loss")
plt.title("SMILES MLM loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(steps, losses, label="train MLM loss")
if valid_steps:
    plt.plot(valid_steps, valid_losses, marker="o", linestyle="-", label="valid MLM loss")
plt.xlabel("step")
plt.ylabel("MLM loss")
plt.ylim(ymin=0, ymax=0.4)
plt.title("SMILES MLM loss")
plt.grid(True)
plt.title("SMILES MLM loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

