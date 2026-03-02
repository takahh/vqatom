#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import matplotlib.pyplot as plt

LOG_PATH = Path("/Users/taka/Downloads/train_log.jsonl")

# 表示を滑らかにしたいなら。0ならスムージング無し。
SMOOTH_WINDOW = 200  # 例: 200ステップ移動平均

# 出力画像（不要なら None）
OUT_PNG = LOG_PATH.with_suffix(".loss.png")

def moving_average(x, w: int):
    if w <= 1 or len(x) < w:
        return x
    s = [0.0]
    for v in x:
        s.append(s[-1] + float(v))
    out = []
    half = w // 2
    # centered MA (端は短くなる)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out.append((s[hi] - s[lo]) / (hi - lo))
    return out

def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"not found: {LOG_PATH}")

    train_step, train_loss = [], []
    valid_step, valid_loss = [], []
    epoch_end_steps = []  # 目印が欲しければ

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            ev = r.get("event", None)

            if ev is None and "mlm_loss" in r and "step" in r:
                # train
                train_step.append(int(r["step"]))
                train_loss.append(float(r["mlm_loss"]))

            elif ev == "valid" and "mlm_loss" in r and "step" in r:
                valid_step.append(int(r["step"]))
                valid_loss.append(float(r["mlm_loss"]))

            elif ev == "epoch_end" and "step" in r:
                epoch_end_steps.append(int(r["step"]))

    # smoothing
    train_loss_s = moving_average(train_loss, SMOOTH_WINDOW)

    plt.figure()
    plt.plot(train_step, train_loss_s, label=f"train (MA{SMOOTH_WINDOW})" if SMOOTH_WINDOW > 1 else "train")
    plt.plot(valid_step, valid_loss, marker="o", linestyle="-", label="valid")

    # epoch end 縦線（邪魔なら消してOK）
    for s in epoch_end_steps:
        plt.axvline(s, linewidth=0.5, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("MLM Loss")
    plt.title("MLM Train / Valid Loss")
    plt.legend()
    plt.tight_layout()

    if OUT_PNG is not None:
        plt.savefig(OUT_PNG, dpi=200)
        print(f"saved: {OUT_PNG}")

    plt.show()

    # ついでに最終値も表示
    if train_loss:
        print(f"train last: step={train_step[-1]} loss={train_loss[-1]:.4f}")
    if valid_loss:
        print(f"valid last: step={valid_step[-1]} loss={valid_loss[-1]:.4f}")

if __name__ == "__main__":
    main()