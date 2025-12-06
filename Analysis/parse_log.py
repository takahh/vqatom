#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict

def parse_log(path):
    """
    ログから (epoch, key, ss) を抽出して返す。
    """
    records = []
    epoch = -1
    seen_ss = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()

            # --- Silhouette 行 ---
            if "Silhouette Score (subsample):" in line_s:
                if epoch == -1:
                    epoch = 0

                tail = line_s.split("Silhouette Score (subsample):", 1)[1].strip()
                parts = tail.split()

                key = parts[0]
                ss = float(parts[1].rstrip(","))

                records.append((epoch, key, ss))
                seen_ss = True
                continue

            # --- エポック境界検出 ---
            if "hnum:" in line_s:
                if seen_ss:
                    epoch += 1
                    seen_ss = False

    return records


def make_matrix(records):
    """
    records: list of (epoch, key, ss)

    戻り値:
      matrix: { key: {epoch: ss} }
      max_epoch: int
    """
    matrix = defaultdict(dict)
    max_epoch = 0

    for ep, key, ss in records:
        matrix[key][ep] = ss
        if ep > max_epoch:
            max_epoch = ep

    return matrix, max_epoch


def write_matrix_csv(matrix, max_epoch, path):
    with open(path, "w") as f:
        # header
        header = ["key"] + [str(e) for e in range(max_epoch + 1)]
        f.write(",".join(header) + "\n")

        # each key
        for key in sorted(matrix.keys()):
            row = [key]
            for ep in range(max_epoch + 1):
                row.append(str(matrix[key].get(ep, "")))
            f.write(",".join(row) + "\n")

    print(f"matrix CSV saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--csv", default="ss_matrix.csv")
    args = parser.parse_args()

    records = parse_log(args.logfile)
    matrix, max_epoch = make_matrix(records)
    write_matrix_csv(matrix, max_epoch, args.csv)


if __name__ == "__main__":
    main()
