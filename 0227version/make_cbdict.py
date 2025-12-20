#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import OrderedDict


# ==================================================
# 1. 変更点：RAW_TEXT を使わず外部ファイルを読む
# ==================================================
import re

# A: "key 123" / "key: 123" / '"key": 123,' etc
LINE_KEY_INT = re.compile(
    r'^\s*"?([^":,\s]+)"?\s*[: ,]\s*([0-9]+)\s*,?\s*$'
)

# B: Silhouette 行（あなたの例に一致）
LINE_SIL = re.compile(
    r"""
    Silhouette\ Score.*?:\s*
    (?P<key>[^\s]+)\s+                              # ← key（空白以外。- を含んでもOK）
    (?P<ss>[0-9]*\.[0-9]+|[0-9]+)\s*,\s*            # ← ss（使わないがパースはする）
    sample\ size\s+(?P<n>[0-9]+)\s*,\s*
    K_e\s+(?P<ke>[0-9]+)
    """,
    re.VERBOSE,
)

def load_key_freq_pairs(path="key_raw_data", debug=False):
    """
    外部ファイルから (key, freq) を抽出。
    - Silhouette 行は (key, sample_size) として採用
    - "key:int" 形式も採用
    - それ以外はスキップ
    """
    pairs = []
    skipped = 0

    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            # 1) Silhouette 行（freq = sample size）
            m = LINE_SIL.search(s)
            if m:
                key = m.group("key")
                freq = int(m.group("n"))
                pairs.append((key, freq))
                if debug:
                    print(f"[OK:SIL] {lineno}: key={key} freq={freq}")
                continue

            # 2) "key:int" 行
            m = LINE_KEY_INT.match(s)
            if m:
                key = m.group(1)
                freq = int(m.group(2))
                pairs.append((key, freq))
                if debug:
                    print(f"[OK:RAW] {lineno}: key={key} freq={freq}")
                continue

            # 3) その他スキップ（VQ_COMMIT 等）
            skipped += 1
            if debug:
                print(f"[SKIP] {lineno}: {s}")

    if debug:
        print(f"[DONE] extracted={len(pairs)} skipped={skipped}")

    return pairs



# ==================================================
# 2. ここから下は「元スクリプトと同じ」
# ==================================================

def build_count_dict(pairs):
    """
    (key, freq) -> key: total_freq
    """
    d = {}
    for k, v in pairs:
        d[k] = d.get(k, 0) + v
    return d


def sort_counts(counts):
    """
    value 降順、key 昇順
    """
    return OrderedDict(
        sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def dump_as_python_dict(d, var_name="KEY_COUNTS"):
    lines = [f"{var_name} = {{"]
    for k, v in d.items():
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    return "\n".join(lines)


def main():
    # ---- RAW_TEXT の代わりにここで読む ----
    pairs = load_key_freq_pairs("key_raw_data")

    counts = build_count_dict(pairs)
    sorted_counts = sort_counts(counts)

    print(dump_as_python_dict(sorted_counts, var_name="CBDICT"))


if __name__ == "__main__":
    main()
