import torch

ckpt_path = "/Users/taka/Downloads/mlm_ep05.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print(type(ckpt))
if isinstance(ckpt, dict):
    print("keys:", ckpt.keys())
    # よくある候補
    for k in ["state_dict", "model", "model_state_dict", "optimizer", "cfg", "config", "args"]:
        if k in ckpt:
            v = ckpt[k]
            print(k, "type:", type(v))

import torch

ckpt_path = "/Users/taka/Downloads/mlm_ep05.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

state = ckpt["model"]          # これが state_dict
conf  = ckpt["config"]         # モデル設定
base_vocab = int(ckpt["base_vocab"])
vocab_size = int(ckpt["vocab_size"])

PAD_ID  = base_vocab + 0
MASK_ID = base_vocab + 1

print("base_vocab", base_vocab, "vocab_size", vocab_size, "PAD", PAD_ID, "MASK", MASK_ID)
print("config keys:", conf.keys())
print(list(state.keys())[:20])

ckpt_path = "/Users/taka/Downloads/mlm_ep05.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

state = ckpt["model"]
conf  = ckpt["config"]
base_vocab = int(ckpt["base_vocab"])
vocab_size = int(ckpt["vocab_size"])
PAD_ID = base_vocab + 0

enc = PretrainedTokenEncoder(
    vocab_size=vocab_size,
    d_model=conf["d_model"],
    nhead=conf["nhead"],
    layers=conf["layers"],
    dim_ff=conf["dim_ff"],
    dropout=conf["dropout"],
    pad_id=PAD_ID
)

# 下流モデル
model = AffinityRegressor(enc, d_model=conf["d_model"])

# まずは strict=False（回帰headが無いので missing になる）
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", missing[:20], "total", len(missing))
print("unexpected:", unexpected[:20], "total", len(unexpected))

