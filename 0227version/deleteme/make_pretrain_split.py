#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing pretrain_ragged_batch*.pt")
    ap.add_argument("--pattern", default="pretrain_ragged_batch*.pt")
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None, help="Output split json path (default: data_dir/split.json)")
    ap.add_argument("--write_txt", action="store_true", help="Also write train_list.txt/valid_list.txt")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    paths = sorted(glob.glob(os.path.join(data_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(data_dir, args.pattern)}")

    rng = random.Random(args.seed)
    rng.shuffle(paths)

    n = len(paths)
    n_valid = max(1, int(round(n * args.valid_ratio)))
    valid = paths[:n_valid]
    train = paths[n_valid:]

    split = {
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "pattern": args.pattern,
        "n_total": n,
        "n_train": len(train),
        "n_valid": len(valid),
        # store relative paths to be portable across machines
        "train": [os.path.relpath(p, data_dir) for p in train],
        "valid": [os.path.relpath(p, data_dir) for p in valid],
    }

    out_json = args.out_json or os.path.join(data_dir, "split.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    if args.write_txt:
        with open(os.path.join(data_dir, "train_list.txt"), "w", encoding="utf-8") as f:
            for p in split["train"]:
                f.write(p + "\n")
        with open(os.path.join(data_dir, "valid_list.txt"), "w", encoding="utf-8") as f:
            for p in split["valid"]:
                f.write(p + "\n")

    print("[OK] wrote:", out_json)
    print(f" total={n} train={len(train)} valid={len(valid)} seed={args.seed} valid_ratio={args.valid_ratio}")

if __name__ == "__main__":
    main()