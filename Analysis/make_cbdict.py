import re
import sys
from collections import namedtuple

Entry = namedtuple("Entry", ["key", "ss", "n", "k_e"])

LINE_RE = re.compile(
    r"Silhouette Score \(subsample\):\s+"
    r"(?P<key>[0-9_\-]+)\s+"
    r"(?P<ss>[0-9.]+),\s*sample size\s+(?P<n>[0-9]+),\s*K_e\s+(?P<ke>[0-9]+)"
)


def parse_log(path):
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


def calc_new_ke(
    entries,
    ss_threshold=0.22,
    n_min=5000,
    max_ke=512,
):
    """
    全 key について「新 K_e」を計算。
    ・(SS < threshold) & (N > n_min) → K_e 2倍（ただし max_ke でクリップ）
    ・それ以外は元の K_e を維持
    """
    result = {}
    for e in entries:
        if e.ss < ss_threshold and e.n > n_min:
            # 倍増
            k_new = min(e.k_e * 2, max_ke)
        else:
            # 据え置き
            k_new = e.k_e

        result[e.key] = {
            "ss": e.ss,
            "n": e.n,
            "k_old": e.k_e,
            "k_new": k_new,
            "changed": (k_new != e.k_e),
        }
    return result


def main(path):
    entries = parse_log(path)
    print(f"# parsed entries: {len(entries)}\n")

    result = calc_new_ke(entries)

    print("# === Proposed K_e changes (including unchanged ones) ===")
    for key, info in sorted(result.items(), key=lambda kv: kv[1]["ss"]):
        ss = info["ss"]
        n = info["n"]
        k_old = info["k_old"]
        k_new = info["k_new"]

        if info["changed"]:
            print(
                f"{key:>20} | SS={ss:.4f}, N={n:7d}, "
                f"{k_old:4d} → {k_new:4d} (2x)"
            )
        else:
            print(
                f"{key:>20} | SS={ss:.4f}, N={n:7d}, "
                f"{k_old:4d} → KEEP"
            )

    print("\n# === FULL K_e override dict (copy & paste) ===")
    print("CB_K_E_OVERRIDES = {")
    for key, info in sorted(result.items()):
        print(f"    '{key}': {info['k_new']},  # SS={info['ss']:.4f}, N={info['n']}, old={info['k_old']}")
    print("}")
    print("\n# === Done ===")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_silhouette_log_full.py <log_path>")
        sys.exit(1)
    main(sys.argv[1])
