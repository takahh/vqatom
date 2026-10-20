#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
from collections import defaultdict


# --------------------------------------------------
# Parsers
# --------------------------------------------------

# A) "key 123" / "key: 123" / '"key": 123,' など
LINE_KEY_INT = re.compile(
    r'^\s*"?([^":,\s]+)"?\s*[: ,]\s*([0-9]+)\s*,?\s*$'
)

# B) Silhouette 行（freq = sample size）
LINE_SIL = re.compile(
    r"""
    Silhouette\ Score.*?:\s*
    (?P<key>[^\s]+)\s+
    (?P<ss>[0-9]*\.[0-9]+|[0-9]+)\s*,\s*
    sample\ size\s+(?P<n>[0-9]+)\s*,\s*
    K_e\s+(?P<ke>[0-9]+)
    """,
    re.VERBOSE,
)


def load_raw_counts(path: str) -> dict:
    """
    raw[key] = total freq
    - Silhouette 行: freq = sample size
    - key:int 行: freq = int
    - それ以外: 無視
    """
    raw = defaultdict(int)

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            m = LINE_SIL.search(s)
            if m:
                key = m.group("key")
                freq = int(m.group("n"))
                raw[key] += freq
                continue

            m = LINE_KEY_INT.match(s)
            if m:
                key = m.group(1)
                freq = int(m.group(2))
                raw[key] += freq
                continue

            # [VQ_COMMIT] 等は無視

    return dict(raw)


# --------------------------------------------------
# Codebook size proposal (あなたのロジックそのまま)
# --------------------------------------------------

def propose_cb_sizes(raw: dict) -> dict:
    proposed = {}
    for k, freq in raw.items():
        if freq <= 30:
            cb_size = 1
        elif freq < 1000:
            cb_size = max(1, math.ceil(freq / 100))
        else:
            cb_size = max(1, math.ceil(freq / 30))
        proposed[k] = cb_size
    return proposed


# --------------------------------------------------
# Output
# --------------------------------------------------

def write_cbdict_txt(proposed: dict, path: str):
    """
    CBDICT を Python dict 形式でファイルに書き出す
    （cb_size 降順）
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("CBDICT = {\n")
        for k, v in sorted(proposed.items(), key=lambda x: x[1], reverse=True):
            f.write(f"    '{k}': {v},\n")
        f.write("}\n")


def print_summary(raw: dict, proposed: dict):
    """
    統計情報のみ print
    """
    total_atoms = sum(raw.values())
    total_codes = sum(proposed.values())

    print(f"[SUMMARY]")
    print(f"  num_keys              = {len(raw)}")
    print(f"  total_atoms (freq)    = {total_atoms}")
    print(f"  total_codebook_size   = {total_codes}")

    if total_codes > 0:
        avg_freq = total_atoms / total_codes
        print(f"  avg freq per code     ≈ {avg_freq:.2f}")
    else:
        print(f"  avg freq per code     = 0.0")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="key_raw_data", help="input file")
    ap.add_argument("--output", default="cbdict.txt", help="output cbdict file")
    args = ap.parse_args()

    raw = load_raw_counts(args.input)
    proposed = propose_cb_sizes(raw)

    # ← ここだけ print
    print_summary(raw, proposed)

    # ← CBDICT はファイルのみ
    write_cbdict_txt(proposed, args.output)


if __name__ == "__main__":
    main()
