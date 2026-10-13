import torch

p = "/Users/taka/Downloads/model_epoch_3.pt"
ckpt = torch.load(p, map_location="cpu", weights_only=False)

print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())

print("epoch:", ckpt["epoch"])
print("vocab_size:", ckpt["vocab_size"])
print("pad_id:", ckpt["pad_id"])
print("mask_id:", ckpt["mask_id"])
print("config:", ckpt["config"])
print("global_id_meta:", ckpt["global_id_meta"])