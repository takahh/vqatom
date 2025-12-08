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


def parse_log(path: str):
    """
    ログファイルから Silhouette エントリを全部拾って返す。
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            entries.append(
                Entry(
                    key=m.group("key"),
                    ss=float(m.group("ss")),
                    n=int(m.group("n")),
                    k_e=int(m.group("ke")),
                )
            )
    return entries


def recommend_ke(n: int, base: float = 10.0, k_max: int = 200) -> int:
    """
    N を入力として K_e を出す式に基づく推奨値。

        K_e ≒ sqrt(N / base)

    - 小さいクラスでは 1 にクリップ
    - 大きすぎるクラスでは k_max にクリップ
    """
    if n <= 0:
        return 1

    ke = round(math.sqrt(n / base))
    ke = max(1, ke)
    ke = min(k_max, ke)
    return ke


def propose_ke(
    entries,
    ss_threshold: float = 0.25,  # SS がこれ以下なら K_e を見直す
    base: float = 10.0,          # recommend_ke の分母
    k_max: int = 200,            # K_e の上限
):
    """
    SS が ss_threshold 以下のクラスについて、
    N に基づく推奨 K_e（recommend_ke）まで「増やす」案を出す。

    戻り値: dict[key] = new_ke
    （増やさないクラスも元の K_e で含める）
    """
    result = {}

    for e in entries:
        # デフォルトは元の K_e
        new_ke = e.k_e

        # N から計算した目標 K_e
        target_ke = recommend_ke(e.n, base=base, k_max=k_max)

        # 「増やす」方向だけにする
        if e.ss <= ss_threshold and target_ke > e.k_e:
            new_ke = target_ke

        result[e.key] = new_ke

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python adjust_ke.py path/to/log.txt", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    entries = parse_log(log_path)

    out_path = sys.argv[2]
    new_ke_dict = propose_ke(
        entries,
        ss_threshold=0.25,
        base=10.0,
        k_max=200,
    )

    # そのまま Python の dict としてコピペしやすい形で出力
    with open(out_path, "w+", encoding="utf-8") as f:
        f.writelines("CBDICT_KE = {")
        for e in entries:
            key = e.key
            old_ke = e.k_e
            new_ke = new_ke_dict[key]
            # コメントで元の値と SS, n も付けておく
            f.writelines(f"    '{key}': {new_ke},  "
                f"# old K_e={old_ke}, n={e.n}, SS={e.ss:.4f}\n")
        f.writelines("}")


if __name__ == "__main__":
    main()
