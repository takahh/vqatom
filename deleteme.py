import torch
path1 = "/Users/taka/Documents/infer_token_ids.pt"
import torch
d = torch.load(path1, map_location="cpu")
x = d["cluster_id"].to(torch.int64)
print("cluster_id min/max:", int(x.min()), int(x.max()), "unique~", int(x.unique().numel()))

