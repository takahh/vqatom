#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import matplotlib.pyplot as plt

LOG_PATH = "log_to_analyze"
EPOCH_MARKER = "[CODEBOOK] mode="


def parse_log_epoch_ss(path: str):
    """
    ログから (epoch, ss) を抽出
    epoch 境界は [CODEBOOK] mode=... 行
    """
    epoch_vals = defaultdict(list)
    epoch = -1

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()

            # --- epoch marker ---
            if EPOCH_MARKER in line_s:
                epoch += 1
                continue

            # --- Silhouette ---
            if "SS:" in line_s:
                if epoch < 0:
                    epoch = 0

                tail = line_s.split("SS:", 1)[1].strip()
                parts = tail.split()
                ss = float(parts[1].rstrip(","))

                # 非ゼロのみ（0は意味が違うので除外）
                if ss > 0:
                    epoch_vals[epoch].append(ss)

    return epoch_vals


def main():
    epoch_vals = parse_log_epoch_ss(LOG_PATH)

    epochs = sorted(epoch_vals.keys())
    data = [epoch_vals[ep] for ep in epochs]

    plt.figure(figsize=(10, 4))

    plt.boxplot(
        data,
        positions=epochs,
        widths=0.6,
        showfliers=True,     # 外れ値も表示
        patch_artist=True
    )

    plt.xlabel("Epoch")
    plt.ylabel("Silhouette Score (non-zero)")
    plt.title("Distribution of Silhouette Scores per Epoch")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
