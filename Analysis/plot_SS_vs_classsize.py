#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import matplotlib.pyplot as plt

LOG_PATH = "log_to_analyze"
EPOCH_MARKER = "[CODEBOOK] mode="


def parse_log_last_epoch_ss_size(path: str):
    """
    Parse log and return:
      last_epoch, list of (key, ss, sample_size)
    """
    epoch = -1
    epoch_records = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()

            # --- epoch marker ---
            if EPOCH_MARKER in line_s:
                epoch += 1
                continue

            # --- Silhouette line ---
            if "Silhouette Score (subsample):" in line_s:
                if epoch < 0:
                    epoch = 0

                tail = line_s.split("Silhouette Score (subsample):", 1)[1].strip()
                parts = tail.split()

                # expected:
                # key ss, sample size N, ...
                key = parts[0]
                ss = float(parts[1].rstrip(","))

                # find sample size
                sample_size = None
                for i, p in enumerate(parts):
                    if p == "size" and i + 1 < len(parts):
                        try:
                            sample_size = int(parts[i + 1].rstrip(","))
                        except ValueError:
                            pass

                # keep only valid entries
                if ss > 0 and sample_size is not None:
                    epoch_records[epoch].append((key, ss, sample_size))

    if not epoch_records:
        raise RuntimeError("No Silhouette Score records found.")

    last_epoch = max(epoch_records.keys())
    return last_epoch, epoch_records[last_epoch]


def main():
    last_epoch, records = parse_log_last_epoch_ss_size(LOG_PATH)

    keys = [r[0] for r in records]
    ss_vals = [r[1] for r in records]
    sizes = [r[2] for r in records]

    plt.figure(figsize=(6, 5))

    plt.scatter(
        sizes,
        ss_vals,
        alpha=0.6,
        s=20,
    )

    plt.xscale("log")  # almost always necessary
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Silhouette Score")
    plt.title(f"SS vs Sample Size (last epoch = {last_epoch})")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
