def summarize_split_by_prefix(class_dict, prefix="6_0_3_1_1_3_6"):
    """
    class_dict: { key(str): indices(Tensor / list[int] / count(int)) }

    prefix に一致するキーを全部拾って、
    「prefix の後ろがどう分かれていて、各サブクラスに何個あるか」を表示する。
    """
    import re
    import numbers

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
            # すでに「個数」として入っている場合（今回これ）
            n = int(idx)
        else:
            # list / tuple / Tensor など
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

key_dict = {}
with open("./key_raw_data") as f:
    for lines in f:
        ele = lines.split()
        key_dict[ele[3]] = int(ele[5])


for k, v in key_dict.items():
    if int(v) > 30:
        print(f"{k}: {v},")

# summarize_split_by_prefix(key_dict, prefix="6_0_3_1_1_3_6")


