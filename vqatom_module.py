# vqatom_module.py
# expose: encode_smiles_to_atom_tokens(smiles) -> List[int]

from typing import List
import importlib.util
from pathlib import Path

# あなたの tokenizer 本体
INFER_PY = Path("/Users/taka/PycharmProjects/vqatom/infer_one_smiles.py")

# ===== あなたの実環境に合わせて必要なら修正 =====
CKPT_PATH = "/Users/taka/Documents/model_epoch_3.pt"
DEVICE = -1   # CPUなら -1, GPUなら 0
HIDDEN_DIM = 16
CODEBOOK_SIZE = 10000
STRICT = False
HIDDEN_FEATS = None
EDGE_EMB_DIM = 32
EMA_DECAY = 0.8
MAXN = 100
D_ATTR = 79
# ===============================================

def _import_infer():
    spec = importlib.util.spec_from_file_location("infer_one_smiles_user", str(INFER_PY))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {INFER_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

_mod = None
_initialized = False

def _ensure_initialized():
    global _mod, _initialized

    if _mod is None:
        _mod = _import_infer()

    if _initialized:
        return

    if not hasattr(_mod, "init_tokenizer"):
        raise RuntimeError("infer_one_smiles.py has no init_tokenizer(...)")

    _mod.init_tokenizer(
        ckpt_path=CKPT_PATH,
        device=DEVICE,
        hidden_dim=HIDDEN_DIM,
        codebook_size=CODEBOOK_SIZE,
        strict=STRICT,
        hidden_feats=HIDDEN_FEATS,
        edge_emb_dim=EDGE_EMB_DIM,
        ema_decay=EMA_DECAY,
        maxn=MAXN,
        d_attr=D_ATTR,
    )
    _initialized = True

def encode_smiles_to_atom_tokens(smiles: str) -> List[int]:
    global _mod

    if _mod is None:
        _mod = _import_infer()

    # infer_smiles があっても、その中で未初期化の可能性があるので先に初期化
    _ensure_initialized()

    if hasattr(_mod, "infer_smiles"):
        return list(map(int, _mod.infer_smiles(smiles)))

    if hasattr(_mod, "encode_smiles_to_atom_tokens"):
        return list(map(int, _mod.encode_smiles_to_atom_tokens(smiles)))

    raise RuntimeError(
        "No usable entrypoint found in infer_one_smiles.py. "
        "Need infer_smiles(smiles) or encode_smiles_to_atom_tokens(smiles)."
    )