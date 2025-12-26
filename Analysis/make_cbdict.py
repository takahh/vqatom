import re
import math
from collections import namedtuple

Entry = namedtuple("Entry", ["key", "n"])

RE_FREQ = re.compile(
    r":\s+(?P<key>[0-9_X\-]+):\s+(?P<n>[0-9]+),"
)


def parse_freq_log(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = RE_FREQ.search(line)
            if not m:
                continue
            entries.append(Entry(key=m.group("key"), n=int(m.group("n"))))
    return entries


def recommend_ke(n: int, max_k: int | None = None) -> int:
    """
    Guarantee n/Ke <= 100 (unless max_k clips it).
    """
    if n <= 100:
        ke = 1
    else:
        ke = math.ceil(n / 100)

    if max_k is not None:
        ke = min(ke, max_k)

    return ke


def main():

    log_path = "log.txt"
    out_path = "cbdict.txt"

    entries = parse_freq_log(log_path)

    # keep last occurrence
    uniq = {}
    for e in entries:
        uniq[e.key] = e.n
    entries = [Entry(k, n) for k, n in uniq.items()]

    # --------- stats ----------
    total_cb = 0
    ratios = []

    for e in entries:
        ke = recommend_ke(e.n, max_k=None)   # <-- hard guarantee
        total_cb += ke
        ratios.append(e.n / ke)

    min_ratio = min(ratios)
    mean_ratio = sum(ratios) / len(ratios)
    max_ratio = max(ratios)

    print(f"#keys                 = {len(entries)}")
    print(f"assigned CB count     = {total_cb}")
    print(f"min  n/Ke             = {min_ratio:.2f}")
    print(f"mean n/Ke             = {mean_ratio:.2f}")
    print(f"max  n/Ke             = {max_ratio:.2f}")

    # -------- write CBDICT --------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CBDICT = {\n")
        for e in sorted(entries, key=lambda x: x.key):
            ke = recommend_ke(e.n, max_k=None)
            f.write(f"    '{e.key}': {ke},   # {e.n}\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
