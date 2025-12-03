import re
import pandas as pd
with open("ss_log") as f:
    contents = f.read()
LOG_TEXT = contents

lines = [ln.strip() for ln in LOG_TEXT.splitlines() if ln.strip()]

# 先頭の "value=0  count=..." を拾う
first = lines[0]
m0 = re.match(r"value=(\d+)\s+count=(\d+)", first)
if m0:
    global_value0 = int(m0.group(1))
    global_count0 = int(m0.group(2))
    print("global value=0 count:", global_value0, global_count0)
    data_lines = lines[1:]
else:
    global_value0 = None
    global_count0 = None
    data_lines = lines

pattern = re.compile(
    r"Silhouette Score \(subsample\):\s+"
    r"(?P<prefix>[0-9_-]+)\s+"
    r"(?P<score>[0-9.]+), sample size "
    r"(?P<n>\d+), K_e (?P<ke>\d+)"
)

records = []
for ln in data_lines:
    m = pattern.search(ln)
    if not m:
        continue
    prefix = m.group("prefix")
    score = float(m.group("score"))
    n = int(m.group("n"))
    ke = int(m.group("ke"))
    n_per_ke = float(n / ke)
    records.append(dict(prefix=prefix, score=score, n=n, ke=ke, n_per_ke=n_per_ke))

df = pd.DataFrame(records)
print(df.head())
print("rows:", len(df))
degenerate = df[(df["score"] == 0.0) & (df["ke"] <= 1)]
print("degenerate rows:", len(degenerate))
print(degenerate.head())

MIN_N = 200   # 好きな値に
small_n = df[df["n"] < MIN_N]
print("small n:", len(small_n))

MIN_N = 500   # subsample がこれ以上
MIN_KE = 3    # 3クラス以上に分かれている
good_mask = (df["n"] >= MIN_N) & (df["ke"] >= MIN_KE)

df_good = df[good_mask].copy()
print("good candidates:", len(df_good))

# スコア順に眺める
print(df_good.sort_values("score", ascending=False).head(30))
# スコア順に眺める
print(df_good.sort_values("score", ascending=True).head(30))

# 「これ以上なら prefix として採用しようかな」という目安
CUTOFF_STRICT = 0.35
CUTOFF_LOOSE  = 0.25

df_strict = df_good[df_good["score"] >= CUTOFF_STRICT]
df_loose = df_good[df_good["score"] >= CUTOFF_LOOSE]

print("strict:", len(df_strict), "loose:", len(df_loose))
