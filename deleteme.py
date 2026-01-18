import torch
PATH = "/Users/taka/Downloads/model_epoch_1.pth"
ckpt = torch.load(PATH, map_location="cpu")
sd = ckpt["model"] if "model" in ckpt else ckpt

# codebook / centersっぽい名前を探す
keys = [k for k in sd.keys() if any(s in k.lower() for s in ["codebook", "embed", "center", "centroid", "means", "cluster"])]
print("\n".join(keys[:200]))
print("num matched:", len(keys))
