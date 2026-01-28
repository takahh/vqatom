import re
import math
from collections import namedtuple

Entry = namedtuple("Entry", ["key", "n"])

# Set B elements: H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}

# init_embed_ lines:
# [init_embed_] Z=16_-1_3_0_0_0 N=11 ...
RE_INIT = re.compile(r"\[init_embed_.*?\]\s+Z=(?P<key>[0-9_\-]+)\s+N=(?P<n>[0-9]+)")

# skip lines (e.g. "[Silhouette] skip key=...:" or other "skip key=...:")
RE_SKIP = re.compile(r"\bskip\s+key=(?P<key>[0-9_\-]+)\b")

def key_allowed(key: str) -> bool:
    """Return True if key's element Z is in the allowed whitelist."""
    # key format: "<Z>_<charge>_<hyb>_..."
    # Just parse the leading Z before the first underscore.
    try:
        z_str = key.split("_", 1)[0]
        z = int(z_str)
    except Exception:
        return False
    return z in ALLOWED_Z

def parse_freq_log(path: str):
    """
    Parse init_embed_ lines (key,N). Skip lines are optional (n=0) but we won't
    let them overwrite real counts later.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = RE_INIT.search(line)
            if m:
                entries.append(Entry(key=m.group("key"), n=int(m.group("n"))))
                continue

            m = RE_SKIP.search(line)
            if m:
                entries.append(Entry(key=m.group("key"), n=0))
                continue

    return entries

def recommend_ke(n: int, max_k: int | None = None) -> int:
    """
    Choose Ke so that n/Ke <= 30 (unless max_k clips it).
    """
    if n <= 0:
        ke = 1
    elif n <= 30:
        ke = 1
    else:
        ke = math.ceil(n / 30)

    if max_k is not None:
        ke = min(ke, max_k)

    return max(1, ke)


def main():
    log_path = "log_to_analyze"
    out_path = "cbdict.txt"

    raw_entries = parse_freq_log(log_path)

    # ---- merge by key; prefer real N over skip (n=0) ----
    freq = {}
    for e in raw_entries:
        # keep the maximum N seen for that key
        freq[e.key] = max(freq.get(e.key, 0), e.n)

    # ---- filter to Set B allowed elements ----
    entries = [Entry(k, n) for k, n in freq.items() if key_allowed(k)]

    # If nothing matched, avoid crashing on min()/mean()
    if not entries:
        print("No entries matched the allowed element whitelist (Set B).")
        return

    # --------- stats ----------
    total_cb = 0
    ratios = []

    for e in entries:
        ke = recommend_ke(e.n, max_k=None)
        total_cb += ke
        ratios.append(e.n / ke if ke > 0 else 0.0)

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
        for e in sorted(entries, key=lambda x: (int(x.key.split('_', 1)[0]), x.key)):
            ke = recommend_ke(e.n, max_k=None)
            f.write(f"    '{e.key}': {ke},   # N={e.n}\n")
        f.write("}\n")

if __name__ == "__main__":
    main()
