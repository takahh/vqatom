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
            if "subsample" in line_s:
                if epoch < 0:
                    epoch = 0
                print("EEEE")
                tail = line_s.split()[6].strip(",")
                print(tail)
                ss = float(tail)
                print(ss)
                # 非ゼロのみ（0は意味が違うので除外）
                if ss > 0:
                    epoch_vals[epoch].append(ss)

    return epoch_vals


def main():
    epoch_vals = parse_log_epoch_ss(LOG_PATH)

    # epochごとに、値が1つ以上あるものだけ残す
    items = [(ep, vals) for ep, vals in sorted(epoch_vals.items())
             if len(vals) > 0]

    if not items:
        print("No non-zero silhouette scores found in log.")
        return

    epochs = [ep for ep, vals in items]
    data = [vals for ep, vals in items]

    # boxplot の positions は 1..len(epochs) にして、
    # x 軸ラベルに実際の epoch 番号を使う
    positions = list(range(1, len(epochs) + 1))

    plt.figure(figsize=(10, 4))

    plt.boxplot(
        data,
        positions=positions,
        widths=0.6,
        showfliers=True,      # 外れ値も表示
        patch_artist=True,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Silhouette Score (non-zero)")
    plt.title("Distribution of Silhouette Scores per Epoch")

    plt.xticks(positions, epochs)  # x 軸ラベルだけ epoch にする
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
