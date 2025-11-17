from collections import OrderedDict
from torch.distributed.fsdp._optim_utils import sorted_items

import re
from collections import OrderedDict

# ログをファイルに保存している前提
LOG_PATH = "key_raw_data"  # ← あなたのログファイル名に合わせて変えてください

# 1行のフォーマット:
# Nov16 19-19-59: key 6_0_3_1_1_2_6_2_1_1 -- 8996
line_re = re.compile(r"key\s+([0-9\-._]+)\s+--\s+(\d+)")

counts = {}
from typing import Dict, Tuple
import math

def build_cb_dict_from_counts(
    counts: Dict[str, int],
    total_k: int = 40000,
    alpha: float = 0.5,
    min_k: int = 1,
    max_k: int | None = None,
    ignore_zero_like: bool = True,
) -> Dict[str, int]:
    """
    counts: {key_str: atom_count_in_batch}
    total_k: CBDICT の合計クラスタ数（例: 40000）
    alpha:  0<alpha<=1. 0.5 なら sqrt スケーリング（ロングテール重視）
    min_k:  count>0 のクラスに割り当てる最小クラスタ数
    max_k:  各クラスの最大クラスタ数（制限したくなければ None）
    ignore_zero_like: "0_0_0_0..." などのパディング key を除外するか
    """
    # 0系 key を弾く場合のヘルパー
    def is_zero_like(key: str) -> bool:
        if not ignore_zero_like:
            return False
        # 完全ゼロかどうかを見る
        parts = key.split("_")
        return all(p == "0" for p in parts)

    # フィルタ済 key & count をリストに
    keys: list[str] = []
    ns: list[int] = []
    for k, n in counts.items():
        if n <= 0:
            continue
        if is_zero_like(k):
            continue
        keys.append(k)
        ns.append(int(n))

    if not keys:
        raise ValueError("No valid keys after filtering.")

    # 重み w_i = n_i ** alpha
    ws = [n ** alpha for n in ns]
    total_w = sum(ws)

    # ① 実数の割当 k_i_float
    k_float = [w / total_w * total_k for w in ws]

    # ② 整数化（min_k, max_k 適用） + 剰余（小数部分）を記録
    k_int = []
    remainders = []  # (小数部分, index) で後でソートして調整
    for i, kf in enumerate(k_float):
        base = int(math.floor(kf))  # とりあえず floor
        frac = kf - base

        # 一旦 floor + 余りで持っておいて後で min_k をかける方がきれいだが、
        # 分かりやすさのためここで round を採用（あとで調整するのでOK）
        ki = int(round(kf))

        if ki < min_k:
            ki = min_k

        if max_k is not None and ki > max_k:
            ki = max_k

        k_int.append(ki)
        remainders.append((frac, i))

    # ③ 合計を total_k に合わせる
    current_sum = sum(k_int)
    diff = current_sum - total_k

    if diff > 0:
        # 割当が多すぎるので diff 個減らす
        # 「小数部分が小さい順」に減らす（= round された寄与が小さいところから）
        remainders_sorted = sorted(remainders, key=lambda x: x[0])  # frac 昇順
        idx = 0
        while diff > 0 and idx < len(remainders_sorted):
            _, i = remainders_sorted[idx]
            if k_int[i] > min_k:
                k_int[i] -= 1
                diff -= 1
            else:
                # これ以上減らせないので次へ
                idx += 1
        # まだ diff>0 なら、適当に大きいところからさらに削る
        if diff > 0:
            # 大きいクラスから1ずつ削る
            order = sorted(range(len(k_int)), key=lambda i: k_int[i], reverse=True)
            j = 0
            while diff > 0 and j < len(order):
                i = order[j]
                if k_int[i] > min_k:
                    k_int[i] -= 1
                    diff -= 1
                else:
                    j += 1

    elif diff < 0:
        # 割当が少なすぎるので -diff 個増やす
        inc = -diff
        # 「小数部分が大きい順」に足す（=本来多めに割当すべきところ）
        remainders_sorted = sorted(remainders, key=lambda x: x[0], reverse=True)
        idx = 0
        while inc > 0 and idx < len(remainders_sorted):
            _, i = remainders_sorted[idx]
            if (max_k is None) or (k_int[i] < max_k):
                k_int[i] += 1
                inc -= 1
            else:
                idx += 1
        # まだ inc>0 なら小さいクラスから順番に +1
        if inc > 0:
            order = sorted(range(len(k_int)), key=lambda i: k_int[i])
            j = 0
            while inc > 0 and j < len(order):
                i = order[j]
                if (max_k is None) or (k_int[i] < max_k):
                    k_int[i] += 1
                    inc -= 1
                else:
                    j += 1

    # 最終チェック
    final_sum = sum(k_int)
    if final_sum != total_k:
        # 理論上ここには来ないはずだが、念のため
        raise RuntimeError(f"Allocation failed: sum={final_sum}, expected={total_k}")

    # ④ CBDICT にまとめる
    CBDICT = {k: ki for k, ki in zip(keys, k_int)}
    return CBDICT

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = line_re.search(line)
        if not m:
            continue
        key_str = m.group(1)         # 例: "6_0_3_1_1_2_6_2_1_1"
        cnt = int(m.group(2))        # 例: 8996

        # 同じキーが複数行に出てきた場合に備えて加算
        counts[key_str] = counts.get(key_str, 0) + cnt

print(f"total unique keys: {len(counts)}")

# 出現頻度の多い順にソート
sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# # 上位20個だけ確認
# for k, v in sorted_items[:20]:
#     print(k, v)


def is_padding_like(key: str) -> bool:
    # 完全ゼロキーだけ除外したい場合
    # return key == "0_0_0_0_0_0_0_0_0_0"

    # もし「先頭が0なら全部除外」にしたいならこちら:
    return key.startswith("0_")

# しきい値を決める
MIN_COUNT = 30  # ← 好きな値に。2000, 5000 などでもよい

CBDICT = OrderedDict()

for key, cnt in sorted_items:
    if cnt < MIN_COUNT:
        break  # 以降は全部小さいので打ち切り（sorted_items は降順）

    if is_padding_like(key):
        continue

    CBDICT[key] = cnt

print(f"CBDICT size: {len(CBDICT)}")

# 中身をざっと確認
# for k, v in list(CBDICT.items()):
#     print(k, v)

cbdict = build_cb_dict_from_counts(CBDICT)

for key, cnt in cbdict.items():
    print(f"'{key}': {cnt},")
