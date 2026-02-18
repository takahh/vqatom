import torch

ckpt_path = "/Users/taka/Downloads/model_epoch_1.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("top-level keys:", len(ckpt))
for k in list(ckpt.keys())[:50]:
    print(k, type(ckpt[k]))
    if 'meta' in k:
        for ky in ['safe_keys', 'global_offsets', 'global_vocab_size', 'pad_id', 'mask_id', 'vocab_size', 'id2safe']:
            print(f"{ky}{ckpt[k][ky]}")
