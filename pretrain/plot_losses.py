#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

PATH = Path("/Users/taka/Downloads/train_console.log")

# epochごとに集約
train_by_ep = defaultdict(list)
valid_ep = []
valid_loss = []

with PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "mlm_loss" not in d or "epoch" not in d:
            continue

        ep = d["epoch"]
        loss = d["mlm_loss"]

        if d.get("event") == "valid":
            valid_ep.append(ep)
            valid_loss.append(loss)
        elif "event" not in d:
            train_by_ep[ep].append(loss)

# ---- trainをepoch平均でなめらかに ----
train_ep = sorted(train_by_ep.keys())
train_loss_mean = [np.mean(train_by_ep[e]) for e in train_ep]

# ---- プロット ----
plt.figure()
plt.plot(train_ep, train_loss_mean, label="train (epoch mean)")
plt.plot(valid_ep, valid_loss, marker="o", linestyle="-", label="valid")
plt.xlabel("epoch")
plt.ylabel("mlm_loss")
plt.ylim(ymin=3.3, ymax=4)
plt.title("MLM Loss (train & valid)")
plt.legend()
plt.tight_layout()
plt.show()