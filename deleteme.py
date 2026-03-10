#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np

PDBID = "1a4k"

CSV_PATH = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv"
DIST_PATH = f"/Users/taka/Downloads/dist_mats_npz_mean/{PDBID}.npz"


def get_sequence(pdbid):
    pdbid = pdbid.lower()

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["pdbid"].lower() == pdbid:
                return row["seq"].strip()

    return None


def main():

    # 距離読み込み
    arr = np.load(DIST_PATH, allow_pickle=True)
    d_mean = arr["d_mean"]

    # 配列取得
    seq = get_sequence(PDBID)

    print("PDBID:", PDBID)
    print("Sequence length:", len(seq))
    print("d_mean length :", len(d_mean))
    print()

    # 小さい順 index
    idx = np.argsort(d_mean)[:5]

    print("Top 5 closest residues:")
    print("-----------------------")

    for i in idx:
        aa = seq[i] if i < len(seq) else "?"
        print(
            f"res_idx={i:3d}  aa={aa}  d_mean={d_mean[i]:.3f}"
        )


if __name__ == "__main__":
    main()