#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

LOG_PATH = "log_to_analyze"
OUT_CSV  = "ss_matrix.csv"

EPOCH_MARKER = "[CODEBOOK] mode="  # ← この行が来たら epoch を進める


def parse_log_epoch_ss(path: str):
    """
    (epoch, ss) を抽出する。
    - epoch 境界は [CODEBOOK] mode=... 行でカウントする
    - Silhouette 行はその時点の epoch に紐づける
    """
    records = []
    epoch = -1  # marker が来たら 0 になる

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()

            # --- epoch marker ---
            if EPOCH_MARKER in line_s:
                epoch += 1
                continue

            # --- Silhouette 行 ---
            if "Silhouette Score (subsample):" in line_s:
                if epoch < 0:
                    # marker が無いログでも動くフォールバック
                    epoch = 0

                tail = line_s.split("Silhouette Score (subsample):", 1)[1].strip()
                parts = tail.split()

                # parts[0] = key, parts[1] = ss,
                ss = float(parts[1].rstrip(","))
                records.append((epoch, ss))

    max_epoch = max((ep for ep, _ in records), default=0)
    return records, max_epoch


def compute_epoch_nonzero_mean(records, max_epoch):
    """
    各 epoch ごとに SS>0 の平均を計算（SS>0 が無ければ 0.0）
    """
    epoch_vals = defaultdict(list)
    for ep, ss in records:
        if ss > 0:
            epoch_vals[ep].append(ss)

    means = []
    for ep in range(max_epoch + 1):
        vals = epoch_vals.get(ep, [])
        means.append(sum(vals) / len(vals) if vals else 0.0)

    return means


def write_epoch_mean_csv(means, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("epoch,平均\n")
        for ep, m in enumerate(means):
            f.write(f"{ep},{m:.6f}\n")

    print(f"CSV saved → {path}")


def main():
    records, max_epoch = parse_log_epoch_ss(LOG_PATH)
    means = compute_epoch_nonzero_mean(records, max_epoch)
    write_epoch_mean_csv(means, OUT_CSV)


if __name__ == "__main__":
    main()
