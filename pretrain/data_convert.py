path0 = "/Users/taka/Documents/npy.tar.gz"
path1 = "/Users/taka/Documents/infer_token_ids.pt"

import torch
d = torch.load(path1, map_location="cpu")
kid = d["key_id"].to(torch.int64)
cid = d["cluster_id"].to(torch.int64)

print("same length:", len(kid) == len(cid))
print("kid range:", int(kid.min()), int(kid.max()))
print("cid range:", int(cid.min()), int(cid.max()))
import torch

for k in [2, 3, 11, 5]:  # 頻出キー
    mask = (kid == k)
    sub = cid[mask]
    print("key", k, "count", int(mask.sum()),
          "cid min/max", int(sub.min()), int(sub.max()),
          "unique", int(sub.unique().numel()))
id2safe = d["id2safe"]
for i in [0, 1, 2, 3, 4]:
    k = int(kid[i])
    c = int(cid[i])
    print(i, id2safe[k], c)
