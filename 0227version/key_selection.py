import re
import numbers
import matplotlib.pyplot as plt
import numpy as np

def summarize_split_by_prefix(class_dict, prefix="6_0_3_1_1_3_6"):
    """
    class_dict: { key(str): indices(Tensor / list[int] / count(int)) }

    prefix に一致するキーを全部拾って、
    「prefix の後ろがどう分かれていて、各サブクラスに何個あるか」を表示する。
    """
    result = {}
    total = 0

    # "6_0_3_1_1_3_6" なら、"6_0_3_1_1_3_6" または
    # "6_0_3_1_1_3_6_XXX" の両方を対象にする。
    prefix_re = re.compile(rf"^{re.escape(prefix)}(_.*)?$")

    for k, idx in class_dict.items():
        if not prefix_re.match(k):
            continue

        # === 個数 n の取り方 ===
        if isinstance(idx, numbers.Integral):
            n = int(idx)
        else:
            try:
                n = len(idx)
            except TypeError:
                # boolean mask Tensor などのとき用（あれば）
                if hasattr(idx, "sum"):
                    try:
                        n = int(idx.sum().item())
                    except Exception:
                        n = 0
                else:
                    n = 0

        # prefix の「後ろの部分」を取り出す
        rest = k[len(prefix):]
        if rest.startswith("_"):
            rest = rest[1:]
        if rest == "":
            rest = "(base)"  # ぴったり prefix と一致したクラス

        result[rest] = result.get(rest, 0) + n
        total += n

    # 表示
    print(f"=== prefix: {prefix} ===")
    print(f"total atoms: {total}")
    for rest, n in sorted(result.items(), key=lambda x: -x[1]):
        if rest != "(base)":
            print(f"{prefix}_{rest} : {n}")
        else:
            print(f"{prefix} : {n}")


def parse_key_counts_from_log(path):
    """
    ログファイルから
    'Silhouette Score (subsample): <KEY> <score>, sample size N, K_e 1'
    の行だけを拾って {key: N} の dict を返す。
    同じ key が複数行ある場合は sample size を合計する。
    """
    key_dict = {}

    # 例:
    # Dec02 03-56-46: Silhouette Score (subsample): 6_0_4_0_1_3_7_2_2_0_9_0 0.0000, sample size 22, K_e 1
    pattern = re.compile(
        r"Silhouette Score \(subsample\):\s+(\S+).*?sample size\s+(\d+)",
        re.ASCII,
    )

    with open(path) as f:
        for line in f:
            line = line.strip()
            m = pattern.search(line)
            if not m:
                # "Save failed for key ..." みたいな行はスキップ
                continue

            key = m.group(1)
            count = int(m.group(2))

            # 同じ key が複数回出たら合算しておく
            key_dict[key] = key_dict.get(key, 0) + count

    return key_dict


def plot_histogram_loglog(class_dict, prefix=None, bins=200):
    """
    Plot a log–log histogram of class counts.
    If prefix is given, only plot keys matching that prefix.
    """
    counts = []

    if prefix is not None:
        prefix_re = re.compile(rf"^{re.escape(prefix)}(_.*)?$")
    else:
        prefix_re = None

    for k, idx in class_dict.items():
        if prefix_re is not None and not prefix_re.match(k):
            continue

        # ---- determine count n ----
        if isinstance(idx, numbers.Integral):
            n = int(idx)
        else:
            try:
                n = len(idx)
            except TypeError:
                if hasattr(idx, "sum"):
                    try:
                        n = int(idx.sum().item())
                    except Exception:
                        n = 0
                else:
                    n = 0

        if n > 0:
            counts.append(n)

    if len(counts) == 0:
        print("No data matched for histogram.")
        return

    counts = np.array(counts)

    # ---- plot (log–log) ----
    plt.figure(figsize=(7, 5))
    plt.hist(counts, bins=bins, edgecolor="black")
    plt.xscale("log")
    plt.yscale("log")

    plt.title("Histogram of class counts (log–log)")
    plt.xlabel("Class count (log scale)")
    plt.ylabel("Frequency (log scale)")

    plt.tight_layout()
    plt.show()


# ========= 実行部分 =========
# ログファイルから key_dict を作る
key_dict = parse_key_counts_from_log("./key_raw_data")

# 中身をざっと確認したければ
for k, v in key_dict.items():
    print(f"{k}: {v},")

# 任意の prefix でサマリ
# summarize_split_by_prefix(key_dict, prefix="6_0_3_1_1_3_6")

# ロングテール込みの log–log ヒストグラム
plot_histogram_loglog(key_dict)
