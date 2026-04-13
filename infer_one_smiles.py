# ---------------------------
# Public API for CSV builders
# ---------------------------
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

for p in [HERE, ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

print("[debug infer_one_smiles] HERE =", HERE)
print("[debug infer_one_smiles] ROOT =", ROOT)
print("[debug infer_one_smiles] sys.path[:5] =", sys.path[:5])
# ---------------------------
# Public API for CSV builders
# ---------------------------
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

_GLOBAL = {
    "model": None,
    "dev": None,
    "maxn": 100,
    "d_attr": 79,
    "offsets": None,   # cache global offsets
}
# infer_one_smiles.py

_MODEL_READY = False

_MODEL_READY = False

def infer_smiles(smiles: str):
    global _MODEL_READY
    if not _MODEL_READY:
        init_tokenizer(
            ckpt_path="/Users/mac/Downloads/model_epoch_3.pt",
            device=-1,
            hidden_dim=16,
            codebook_size=10000,
            strict=False,
            hidden_feats=None,
            edge_emb_dim=32,
            ema_decay=0.8,
            maxn=100,
            d_attr=79,
        )
        _MODEL_READY = True
    return encode_smiles_to_atom_tokens(smiles)
# ---------------------------
# Minimal args for model build
# ---------------------------

def _build_args_for_ckpt(
    hidden_dim: int,
    codebook_size: int,
    edge_emb_dim: int = 32,
    ema_decay: float = 0.8,
):
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        edge_emb_dim=edge_emb_dim,
        ema_decay=ema_decay,
    )

def _load_global_id_meta_any(ckpt_path: str, map_location="cpu") -> Optional[dict]:
    """
    Returns ckpt['global_id_meta'] if present, else None.
    Supports:
      - dict ckpt with 'global_id_meta'
      - dict ckpt with nested dict (same)
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and "global_id_meta" in ckpt and isinstance(ckpt["global_id_meta"], dict):
        return ckpt["global_id_meta"]

    return None

# ---------------------------
# Safe helpers (ported from previous version)
# ---------------------------

def _load_state_dict_any(ckpt_path: str, map_location):
    """
    Supports:
      - raw state_dict
      - dict with 'model'
      - dict with 'state_dict'
      - dict with 'model_state_dict'
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]

    return ckpt

def _as_torch(x, *, dtype=None, device=None):
    import numpy as np
    import torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if dtype is not None:
        x = x.to(dtype)
    if device is not None:
        x = x.to(device)
    return x

def _pad_adj_attr(adj, attr, *, maxn: int = 100, d_attr: int = 79):
    """
    Ensure:
      adj:  (maxn, maxn) float32
      attr: (maxn, d_attr) float32

    Accepts:
      adj:  (N,N)
      attr: (N,d_attr) or flattened (N*d_attr,)
      either numpy or torch
    """
    import torch

    adj = _as_torch(adj, dtype=torch.float32)
    attr = _as_torch(attr, dtype=torch.float32)

    # attr could be flattened
    if attr.dim() == 1:
        if attr.numel() % d_attr != 0:
            raise RuntimeError(
                f"attr is 1D but numel={attr.numel()} not divisible by d_attr={d_attr}"
            )
        attr = attr.view(-1, d_attr)

    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise RuntimeError(f"adj must be square [N,N], got {tuple(adj.shape)}")

    n = int(adj.shape[0])

    if attr.dim() != 2 or int(attr.shape[1]) != d_attr:
        raise RuntimeError(f"attr must be [N,{d_attr}], got {tuple(attr.shape)}")

    if int(attr.shape[0]) != n:
        raise RuntimeError(f"attr rows must match N. N={n}, attr.shape={tuple(attr.shape)}")

    if n > maxn:
        raise RuntimeError(f"N={n} exceeds maxn={maxn}. Increase maxn or change pipeline.")

    adj_pad = torch.zeros((maxn, maxn), dtype=torch.float32)
    attr_pad = torch.zeros((maxn, d_attr), dtype=torch.float32)

    adj_pad[:n, :n] = adj
    attr_pad[:n, :] = attr

    return adj_pad, attr_pad, n

def _compute_global_offsets(model) -> Dict[str, int]:
    """
    Compute global token offsets per safe_key by iterating codebook sizes.
    Uses model.vq._codebook.embed[<safe_key>].shape[0] as K.
    """
    offsets: Dict[str, int] = {}
    cur = 0
    for safe_key in model.vq._codebook.embed.keys():
        K = int(model.vq._codebook.embed[safe_key].shape[0])
        offsets[safe_key] = cur
        cur += K
    return offsets

# ---------------------------
# Model / inference (ported)
# ---------------------------

def load_model(
    ckpt_path: str,
    dev,
    *,
    hidden_dim: int,
    codebook_size: int,
    strict: bool = False,
    hidden_feats: Optional[int] = None,
    edge_emb_dim: int = 32,
    ema_decay: float = 0.8,
):
    """
    Build EquivariantThreeHopGINE and load checkpoint weights.
    """
    # Prevent imported training modules from parsing our argv.
    sys.argv = [sys.argv[0]]

    from models import EquivariantThreeHopGINE

    args = _build_args_for_ckpt(
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        edge_emb_dim=edge_emb_dim,
        ema_decay=ema_decay,
    )

    hf = hidden_feats if hidden_feats is not None else hidden_dim

    model = EquivariantThreeHopGINE(
        in_feats=64,
        hidden_feats=hf,
        out_feats=hidden_dim,
        args=args,
    ).to(dev)

    state = _load_state_dict_any(ckpt_path, map_location=dev)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def infer_one(
    model,
    dev,
    smiles: str,
    *,
    maxn: int = 100,
    d_attr: int = 79,
):
    sys.argv = [sys.argv[0]]

    import torch
    import dgl

    from smiles_to_npy_discretize import smiles_to_graph_with_labels
    from new_train_and_eval import evaluate, convert_to_dgl

    with torch.no_grad():
        adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)

        adj_pad, attr_pad, n = _pad_adj_attr(adj, attr, maxn=maxn, d_attr=d_attr)

        adj_batch = [adj_pad.unsqueeze(0)]
        attr_batch = [attr_pad.unsqueeze(0)]

        _, extended_graphs, masks_dict, attr_matrices_all, _, _ = convert_to_dgl(
            adj_batch,
            attr_batch,
            logger=None,
            start_atom_id=0,
            start_mol_id=0,
            device=str(dev),
        )

        g = extended_graphs[0]
        X = attr_matrices_all[0]

        batched_graph = dgl.batch([g]).to(dev)
        feats = batched_graph.ndata["feat"]
        attr_list = [X.to(dev)]

        _, ids, _ = evaluate(
            model=model,
            g=batched_graph,
            feats=feats,
            epoch=0,
            mask_dict=masks_dict,
            logger=None,
            g_base=None,
            chunk_i=0,
            mode="infer",
            attr_list=attr_list,
        )

        # ids may be (kid,cid,gid,id2safe) or legacy (kid,cid,id2safe)
        if isinstance(ids, (tuple, list)) and len(ids) == 4:
            kid, cid, gid, id2safe = ids
        elif isinstance(ids, (tuple, list)) and len(ids) == 3:
            kid, cid, id2safe = ids
            gid = None
        else:
            raise TypeError(f"infer ids format unexpected: {type(ids)} len={len(ids) if isinstance(ids,(tuple,list)) else 'n/a'}")

        kid_list = torch.as_tensor(kid).reshape(-1).detach().cpu().tolist()
        cid_list = torch.as_tensor(cid).reshape(-1).detach().cpu().tolist()

        # ---- (NEW) prefer offsets from ckpt meta; fallback only if missing ----
        offsets = _GLOBAL.get("offsets")
        if offsets is None:
            # fallback (still safer to sort to match your old convention)
            offsets = {}
            cur = 0
            for safe_key in sorted(model.vq._codebook.embed.keys()):
                K = int(model.vq._codebook.embed[safe_key].shape[0])
                offsets[safe_key] = cur
                cur += K
            _GLOBAL["offsets"] = offsets

        tokens: List[int] = []
        for k, c in zip(kid_list, cid_list):
            safe = id2safe[int(k)]
            if safe not in offsets:
                raise RuntimeError(f"safe_key '{safe}' not in offsets. ckpt/meta mismatch?")
            tokens.append(int(offsets[safe]) + int(c))

        return tokens, kid_list, cid_list, id2safe


def infer_one_from_mol(
    model,
    dev,
    mol,
    *,
    maxn: int = 100,
    d_attr: int = 79,
):
    import torch
    import dgl

    from smiles_to_npy_discretize import mol_to_graph_with_labels
    from new_train_and_eval import evaluate, convert_to_dgl

    with torch.no_grad():

        adj, attr, _ = mol_to_graph_with_labels(mol)

        adj_pad, attr_pad, n = _pad_adj_attr(adj, attr, maxn=maxn, d_attr=d_attr)

        adj_batch = [adj_pad.unsqueeze(0)]
        attr_batch = [attr_pad.unsqueeze(0)]

        _, extended_graphs, masks_dict, attr_matrices_all, _, _ = convert_to_dgl(
            adj_batch,
            attr_batch,
            logger=None,
            start_atom_id=0,
            start_mol_id=0,
            device=str(dev),
        )

        g = extended_graphs[0]
        X = attr_matrices_all[0]

        batched_graph = dgl.batch([g]).to(dev)
        feats = batched_graph.ndata["feat"]
        attr_list = [X.to(dev)]

        _, ids, _ = evaluate(
            model=model,
            g=batched_graph,
            feats=feats,
            epoch=0,
            mask_dict=masks_dict,
            logger=None,
            g_base=None,
            chunk_i=0,
            mode="infer",
            attr_list=attr_list,
        )

        if len(ids) == 4:
            kid, cid, gid, id2safe = ids
        else:
            kid, cid, id2safe = ids

        kid_list = torch.as_tensor(kid).reshape(-1).cpu().tolist()
        cid_list = torch.as_tensor(cid).reshape(-1).cpu().tolist()

        offsets = _GLOBAL["offsets"]

        tokens = []
        for k, c in zip(kid_list, cid_list):
            safe = id2safe[int(k)]
            tokens.append(offsets[safe] + int(c))

        return tokens

# ---------------------------
# Public API (kept)
# ---------------------------

def init_tokenizer(
    ckpt_path: str,
    device: int = 0,
    hidden_dim: int = 16,
    codebook_size: int = 10000,
    strict: bool = False,
    hidden_feats: Optional[int] = None,
    edge_emb_dim: int = 32,
    ema_decay: float = 0.8,
    maxn: int = 100,
    d_attr: int = 79,
):
    import torch

    dev = torch.device(
        f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu"
    )

    # ---- (NEW) read global_id_meta from ckpt (CPU is fine) ----
    meta = _load_global_id_meta_any(ckpt_path, map_location="cpu")
    if meta is not None:
        offsets = meta.get("global_offsets")
        if not isinstance(offsets, dict) or not all(isinstance(v, int) for v in offsets.values()):
            raise RuntimeError(f"ckpt has global_id_meta but global_offsets is invalid: {type(offsets)}")
        _GLOBAL["offsets"] = dict(offsets)  # FIXED OFFSETS from training truth
        _GLOBAL["global_id_meta"] = meta    # optional, for debugging/inspection
    else:
        _GLOBAL["offsets"] = None
        _GLOBAL["global_id_meta"] = None

    model = load_model(
        ckpt_path,
        dev,
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        strict=strict,
        hidden_feats=hidden_feats,
        edge_emb_dim=edge_emb_dim,
        ema_decay=ema_decay,
    )

    _GLOBAL["model"] = model
    _GLOBAL["dev"] = dev
    _GLOBAL["maxn"] = maxn
    _GLOBAL["d_attr"] = d_attr

    # NOTE: ここで offsets を None に “リセットしない”
    #       ckpt内metaがあるならそれを使い続けたいので。
    return model, dev

def encode_smiles_to_atom_tokens(smiles: str) -> List[int]:
    """
    Returns global token ids (List[int]) for this SMILES.
    Requires init_tokenizer() to be called once beforehand.

    IMPORTANT: token order is whatever convert_to_dgl/evaluate produces.
    """
    if _GLOBAL["model"] is None or _GLOBAL["dev"] is None:
        raise RuntimeError("Tokenizer not initialized. Call init_tokenizer(...) first.")

    tokens, _, _, _ = infer_one(
        _GLOBAL["model"],
        _GLOBAL["dev"],
        smiles,
        maxn=_GLOBAL["maxn"],
        d_attr=_GLOBAL["d_attr"],
    )
    return tokens

def infer_smiles(smiles: str):
    return encode_smiles_to_atom_tokens(smiles)