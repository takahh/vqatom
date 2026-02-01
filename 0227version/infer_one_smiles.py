#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-SMILES inference for VQ-Atom (0227version)

This script is designed to be robust against:
- Training modules that call get_args()/parse_args() at import-time
- smiles_to_graph_with_labels returning numpy arrays and/or unpadded shapes
- Checkpoints saved as raw state_dict or as {"model": state_dict, ...}

Pipeline:
  smiles -> (adj, attr) -> pad to (100,100) and (100,79) -> convert_to_dgl
        -> evaluate(mode="infer") -> (kid, cid, id2safe) -> global token ids
"""

import argparse
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple


# ---------------------------
# Minimal args for model build
# ---------------------------

def build_args_for_ckpt(
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


# ---------------------------
# Safe helpers
# ---------------------------

def load_state_dict_any(ckpt_path: str, map_location):
    """Supports raw state_dict OR dict with a 'model' key (or a couple common alternatives)."""
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


def as_torch(x, *, dtype=None, device=None):
    """Convert numpy -> torch; optionally cast dtype and move device."""
    import numpy as np
    import torch

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if dtype is not None:
        x = x.to(dtype)
    if device is not None:
        x = x.to(device)
    return x


def pad_adj_attr(adj, attr, *, maxn: int = 100, d_attr: int = 79):
    """
    Ensure:
      adj:  (maxn, maxn) float32
      attr: (maxn, d_attr) float32
    from possibly:
      adj:  (N,N)
      attr: (N,d_attr) or flattened (N*d_attr,)
    """
    import torch

    adj = as_torch(adj, dtype=torch.float32)
    attr = as_torch(attr, dtype=torch.float32)

    # attr could be flattened
    if attr.dim() == 1:
        if attr.numel() % d_attr != 0:
            raise RuntimeError(f"attr is 1D but numel={attr.numel()} not divisible by d_attr={d_attr}")
        attr = attr.view(-1, d_attr)

    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise RuntimeError(f"adj must be square [N,N], got {tuple(adj.shape)}")

    n = int(adj.shape[0])

    if attr.dim() != 2 or attr.shape[1] != d_attr:
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


def compute_global_offsets(model) -> Dict[str, int]:
    """Compute global token offsets per safe_key by iterating codebook sizes."""
    offsets: Dict[str, int] = {}
    cur = 0
    for safe_key in model.vq._codebook.embed.keys():
        K = int(model.vq._codebook.embed[safe_key].shape[0])
        offsets[safe_key] = cur
        cur += K
    return offsets


# ---------------------------
# Model / inference
# ---------------------------

def load_model(
    ckpt_path: str,
    dev,
    *,
    hidden_dim: int,
    codebook_size: int,
    strict: bool = False,
    hidden_feats: int | None = None,
    edge_emb_dim: int = 32,
    ema_decay: float = 0.8,
):
    """
    Build EquivariantThreeHopGINE and load checkpoint weights.
    """
    # Prevent training modules (imported later) from parsing our argv.
    sys.argv = [sys.argv[0]]

    from models import EquivariantThreeHopGINE

    args = build_args_for_ckpt(
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        edge_emb_dim=edge_emb_dim,
        ema_decay=ema_decay,
    )

    # Internal GINE width
    hf = hidden_feats if hidden_feats is not None else hidden_dim

    model = EquivariantThreeHopGINE(
        in_feats=64,
        hidden_feats=hf,
        out_feats=hidden_dim,
        args=args,
    ).to(dev)

    state = load_state_dict_any(ckpt_path, map_location=dev)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def infer_one(model, dev, smiles: str, *, maxn: int = 100, d_attr: int = 79):
    """
    smiles -> padded matrices -> convert_to_dgl -> evaluate(mode="infer") -> token ids
    """
    # Prevent imported training modules from parsing our argv.
    sys.argv = [sys.argv[0]]

    import torch
    import dgl

    from smiles_to_npy import smiles_to_graph_with_labels
    from new_train_and_eval import evaluate, convert_to_dgl

    with torch.no_grad():
        adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)

        # Ensure shapes expected by convert_to_dgl: [1,100,100] and [1,100,79]
        adj_pad, attr_pad, n = pad_adj_attr(adj, attr, maxn=maxn, d_attr=d_attr)

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
        X = attr_matrices_all[0]  # [N,79] tensor (de-padded)

        batched_graph = dgl.batch([g]).to(dev)
        feats = batched_graph.ndata["feat"]
        attr_list = [X.to(dev)]

        _, (kid, cid, id2safe), _ = evaluate(
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

        offsets = compute_global_offsets(model)

        kid_list = kid.reshape(-1).detach().cpu().tolist()
        cid_list = cid.reshape(-1).detach().cpu().tolist()

        tokens: List[int] = []
        for k, c in zip(kid_list, cid_list):
            safe = id2safe[int(k)]
            tokens.append(offsets[safe] + int(c))

        return tokens, kid_list, cid_list, id2safe


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model_epoch_xx.pth")
    ap.add_argument("--smiles", required=True, help="SMILES string")
    ap.add_argument("--device", type=int, default=0, help="GPU index; -1 for CPU")
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")

    # must match training
    ap.add_argument("--hidden_dim", type=int, default=16)
    ap.add_argument("--codebook_size", type=int, default=10000)

    # optional knobs
    ap.add_argument("--hidden_feats", type=int, default=None, help="GINE internal width (try 128 if mismatch)")
    ap.add_argument("--edge_emb_dim", type=int, default=32)
    ap.add_argument("--ema_decay", type=float, default=0.8)

    # padding expectations from your pipeline
    ap.add_argument("--maxn", type=int, default=100)
    ap.add_argument("--d_attr", type=int, default=79)

    args = ap.parse_args()

    import torch
    dev = torch.device(
        f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        args.ckpt,
        dev,
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        strict=args.strict,
        hidden_feats=args.hidden_feats,
        edge_emb_dim=args.edge_emb_dim,
        ema_decay=args.ema_decay,
    )

    tokens, kid, cid, id2safe = infer_one(
        model,
        dev,
        args.smiles,
        maxn=args.maxn,
        d_attr=args.d_attr,
    )

    print("SMILES:", args.smiles)
    print("TOKENS:", tokens)
    print("kid[:20]:", kid[:20])
    print("cid[:20]:", cid[:20])
    print("id2safe size:", len(id2safe))


if __name__ == "__main__":
    main()
