import math
import ast

CBDICT_PATH = "cbdict.txt"
OUT_PATH = "wdict.txt"   # or .py if you prefer


def load_cbdict(path):
    """
    Reads a file like:
      CBDICT = { 'k': 3, ... }
    and returns the dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # drop leading `CBDICT =`
    if "=" in text:
        text = text.split("=", 1)[1].strip()

    return ast.literal_eval(text)


def compute_weights_from_ke(cbdict, alpha=-0.5, normalize=True):
    """
    alpha = -0.5 -> w = 1/sqrt(K)
    alpha =  0.0 -> uniform
    alpha = +0.5 -> sqrt(K)
    """
    weights = {}
    raw_list = []

    for key, ke in cbdict.items():
        ke = max(int(ke), 1)  # safety
        w = ke ** alpha
        weights[key] = w
        raw_list.append(w)

    if normalize and raw_list:
        mean_w = sum(raw_list) / len(raw_list)
        for k in weights:
            weights[k] /= mean_w

    return weights


def main():
    cbdict = load_cbdict(CBDICT_PATH)

    weights = compute_weights_from_ke(
        cbdict,
        alpha=0.2,   # <-- change here if needed
        normalize=True
    )

    # ---- WRITE AS PYTHON ASSIGNMENT ----
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("WDICT = {\n")
        for k, v in weights.items():
            f.write(f"    '{k}': {v:.8f},\n")
        f.write("}\n")

    print(f"Wrote WDICT with {len(weights)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
