import re
import math
import sys
from collections import namedtuple

Entry = namedtuple("Entry", ["key", "ss", "n", "k_e"])

LINE_RE = re.compile(
    r"Silhouette Score \(subsample\):\s+"
    r"(?P<key>[0-9_\-]+)\s+"
    r"(?P<ss>[0-9.]+),\s*sample size\s+(?P<n>[0-9]+),\s*K_e\s+(?P<ke>[0-9]+)"
)

def parse_log(path):
    """
    ログファイルから Silhouette エントリを全部拾って返す。
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            key = m.group("key")
            ss = float(m.group("ss"))
            n = int(m.group("n"))
            ke = int(m.group("ke"))
            entries.append(Entry(key, ss, n, ke))
    return entries

def propose_ke(
    entries,
    ss_threshold=0.25,          # SS がこれ以下なら K_e を増やす対象
    target_n_per_cluster=30,    # sample_size / K_e ≃ 30 を目指す
):
    """
    SS が ss_threshold 以下のクラスについて、
    sample_size / K_e が target_n_per_cluster になるように K_e を「増やす」案を出す。

    戻り値: dict[key] = new_ke
    （増やさないクラスも元の K_e で含める）
    """
    result = {}
    for e in entries:
        # デフォルトは元の K_e
        new_ke = e.k_e

        if e.ss <= ss_threshold:
            # 目標 K_e = ceil(n / target_n_per_cluster)
            target_ke = math.ceil(e.n / target_n_per_cluster)
            # 「増やしたい」だけなので、今より小さければ据え置き
            if target_ke > e.k_e:
                new_ke = target_ke

        result[e.key] = new_ke

    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python adjust_ke.py path/to/log.txt", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    entries = parse_log(log_path)
    new_ke_dict = propose_ke(entries, ss_threshold=0.25, target_n_per_cluster=30)

    # そのまま Python の dict としてコピペしやすい形で出力
    print("CBDICT_KE = {")
    for e in entries:
        key = e.key
        old_ke = e.k_e
        new_ke = new_ke_dict[key]
        # コメントで元の値と SS, n も付けておく
        print(
            f"    '{key}': {new_ke},  "
            f"# old K_e={old_ke}, n={e.n}, SS={e.ss:.4f}"
        )
    print("}")

if __name__ == "__main__":
    main()
