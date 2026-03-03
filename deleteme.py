#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

LOG_PATH = Path("/Users/taka/Downloads/train_console.log")  # ←ここ変えて
OUT_PNG = LOG_PATH.with_suffix(".epoch_loss.png")

# train の集計で masked_tokens が小さすぎる行を捨てたい場合（例: 5000）
MIN_MASKED_TOKENS_TRAIN = 0  # 0ならフィルタ無し。例: 5000
# valid は通常まとまって大きいのでフィルタ不要（必要なら同様に追加可）

# 例:
# [ep 18/30] step 269150/448770 mlm_loss=3.7127 ... masked_tokens=13292
# [ep 18/30] VALID mlm_loss=3.6573 ... masked_tokens=210274

RE_EP = re.compile(r"\[ep\s+(\d+)/(\d+)\]")
RE_LOSS = re.compile(r"mlm_loss=([0-9]*\.[0-9]+|[0-9]+)")
RE_MASKED = re.compile(r"masked_tokens=(\d+)")
RE_IS_VALID = re.compile(r"\bVALID\b")

def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"not found: {LOG_PATH}")

    train_losses = defaultdict(list)   # ep -> [loss...]
    valid_loss = {}                    # ep -> loss (最後のVALIDを採用)
    valid_masked = {}                  # ep -> masked_tokens (参考)

    with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_ep = RE_EP.search(line)
            if not m_ep:
                continue
            ep = int(m_ep.group(1))

            m_loss = RE_LOSS.search(line)
            if not m_loss:
                continue
            loss = float(m_loss.group(1))

            m_masked = RE_MASKED.search(line)
            masked = int(m_masked.group(1)) if m_masked else None

            is_valid = bool(RE_IS_VALID.search(line))

            if is_valid:
                # VALID mlm_loss は valid としてそのまま1点
                valid_loss[ep] = loss
                if masked is not None:
                    valid_masked[ep] = masked
            else:
                # それ以外の mlm_loss は train
                if masked is not None and masked < MIN_MASKED_TOKENS_TRAIN:
                    continue
                train_losses[ep].append(loss)

    if not train_losses:
        raise RuntimeError("No train losses parsed. Check LOG_PATH and log format.")

    epochs = sorted(train_losses.keys())
    train_mean = []
    for ep in epochs:
        xs = train_losses[ep]
        train_mean.append(sum(xs) / len(xs))

    # valid はあるepochだけ
    x_valid = []
    y_valid = []
    for ep in epochs:
        if ep in valid_loss:
            x_valid.append(ep)
            y_valid.append(valid_loss[ep])

    plt.figure()
    plt.plot(epochs, train_mean, marker="o", label="train (epoch mean)")
    if x_valid:
        plt.plot(x_valid, y_valid, marker="o", label="valid (raw)")
    else:
        print("warning: no VALID lines parsed")

    plt.xlabel("Epoch")
    plt.ylabel("MLM Loss")
    plt.title("MLM Loss (Train=Epoch Mean, Valid=Raw)")
    plt.legend()
    plt.tight_layout()

    if OUT_PNG is not None:
        plt.savefig(OUT_PNG, dpi=200)
        print(f"saved: {OUT_PNG}")

    plt.show()

    last_ep = epochs[-1]
    print(f"train last epoch={last_ep} mean_loss={train_mean[-1]:.4f} (n={len(train_losses[last_ep])})")
    if x_valid:
        print(f"valid last epoch={x_valid[-1]} loss={y_valid[-1]:.4f}")
        vm = valid_masked.get(x_valid[-1])
        if vm is not None:
            print(f"valid last masked_tokens={vm}")

if __name__ == "__main__":
    main()