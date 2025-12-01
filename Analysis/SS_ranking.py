import re
from dataclasses import dataclass
from typing import List

@dataclass
class Entry:
    key: str
    ss: float
    n: int
    k_e: int

# 「数字と _」が続く key（11個でも12個でもOK）にマッチ
LINE_RE = re.compile(
    r"Silhouette Score \(subsample\):\s+"
    r"(?P<key>[0-9_\-]+)\s+"
    r"(?P<ss>[0-9.]+),\s*sample size\s+(?P<n>[0-9]+),\s*K_e\s+(?P<ke>[0-9]+)"
)

def parse_log(path: str) -> List[Entry]:
    entries: List[Entry] = []
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

if __name__ == "__main__":
    entries = parse_log("ss_log.txt")

    print(f"Total entries: {len(entries)}")

    # 1) SS>0, K_e>1, n>=1000 に絞る
    valid = [e for e in entries if e.ss > 0.0 and e.k_e > 1 and e.n >= 1000]
    print(f"Valid entries (ss>0, K_e>1, n>=1000): {len(valid)}")

    if valid:
        avg_ss = sum(e.ss for e in valid) / len(valid)
        min_ss = min(e.ss for e in valid)
        max_ss = max(e.ss for e in valid)
        print(f"SS stats: avg={avg_ss:.4f}, min={min_ss:.4f}, max={max_ss:.4f}")

        # 2) SS の低い順トップ30
        print("\n=== Lowest SS (top 30, n>=1000, K_e>1) ===")
        for e in sorted(valid, key=lambda x: x.ss)[:30]:
            print(f"{e.ss:.4f}\tn={e.n:6d}\tK_e={e.k_e:4d}\t{e.key}")

        # 3) SS の高い順トップ30
        print("\n=== Highest SS (top 30, n>=1000, K_e>1) ===")
        for e in sorted(valid, key=lambda x: x.ss, reverse=True)[:30]:
            print(f"{e.ss:.4f}\tn={e.n:6d}\tK_e={e.k_e:4d}\t{e.key}")

    # 4) K_e=1 のものをざっと確認
    ke1 = [e for e in entries if e.k_e == 1]
    print(f"\nEntries with K_e=1: {len(ke1)}")
    if ke1:
        print("Example (first 20):")
        for e in ke1[:20]:
            print(f"{e.ss:.4f}\tn={e.n:6d}\tK_e={e.k_e:4d}\t{e.key}")
