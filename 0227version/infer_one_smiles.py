#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from types import SimpleNamespace


def build_args_for_ckpt(hidden_dim: int, codebook_size: int,
                        edge_emb_dim: int = 32, ema_decay: float = 0.8):
    """Minimal args object needed by EquivariantThreeHopGINE / VectorQuantize."""
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        edge_emb_dim=edge_emb_dim,
        ema_decay=ema_decay,
    )


def load_state_dict_any(ckpt_path: str, map_location):
    """Supports either raw state_dict or dict with key 'model'."""
    import torch
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def load_model(ckpt_path: str, dev, *, hidden_dim: int, codebook_size: int,
               strict: bool = False, hidden_feats: int | None = None,
               edge_emb_dim: int = 32, ema_decay: float = 0.8):
    import torch

    # IMPORTANT:
    # Some imported training modules call get_args()/parse_args() at import-time.
    # Wipe argv so they don't see our --ckpt/--smiles flags.
    sys.argv = [sys.argv[0]]

    # Delayed imports (after argv sanitization)
    from models import EquivariantThreeHopGINE

    args = build_args_for_ckpt(hidden_dim, codebook_size, edge_emb_dim=edge_emb_dim, ema_decay=ema_decay)

    # GINE internal width. If you get size mismatch, try hidden_feats=128.
    hf = hidden_feats if hidden_feats is not None else args.hidden_dim

    in_feats = 64
    out_feats = args.hidden_dim

    model = EquivariantThreeHopGINE(
        in_feats=in_feats,
        hidden_feats=hf,
        out_feats=out_feats,
        args=args,
    ).to(dev)

    state = load_state_dict_any(ckpt_path, map_location=dev)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def compute_global_offsets(model):
    """Compute global token offsets per safe_key by iterating codebook sizes."""
    offsets = {}
    cur = 0
    for safe_key in model.vq._codebook.embed.keys():
        K = int(model.vq._codebook.embed[safe_key].shape[0])
        offsets[safe_key] = cur
        cur += K
    return offsets


def infer_one(model, dev, smiles: str):
    import torch
    import dgl

    # Delayed imports (after argv sanitization)
    from smiles_to_npy import smiles_to_graph_with_labels
    from new_train_and_eval import evaluate, convert_to_dgl

    with torch.no_grad():
        # 1) padded matrices [100,100] and [100,79]
        adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)

        # 2) convert_to_dgl expects list elements that can .view(-1,100,100)/(..,100,79)
        adj_batch = [adj.unsqueeze(0)]    # [1,100,100]
        attr_batch = [attr.unsqueeze(0)]  # [1,100,79]

        _, extended_graphs, masks_dict, attr_matrices_all, _, _ = convert_to_dgl(
            adj_batch,
            attr_batch,
            logger=None,
            start_atom_id=0,
            start_mol_id=0,
            device=str(dev),
        )

        g = extended_graphs[0]
        X = attr_matrices_all[0]  # de-padded [N,79]

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

        kid_list = kid.reshape(-1).cpu().tolist()
        cid_list = cid.reshape(-1).cpu().tolist()

        tokens = []
        for k, c in zip(kid_list, cid_list):
            safe = id2safe[int(k)]
            tokens.append(offsets[safe] + int(c))

        return tokens, kid_list, cid_list, id2safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model_epoch_xx.pth")
    ap.add_argument("--smiles", required=True, help="SMILES string")
    ap.add_argument("--device", type=int, default=0, help="GPU index; -1 for CPU")
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")
    ap.add_argument("--hidden_dim", type=int, default=16, help="must match training")
    ap.add_argument("--codebook_size", type=int, default=10000, help="must match training")
    ap.add_argument("--hidden_feats", type=int, default=None, help="GINE internal width (try 128 if mismatch)")
    ap.add_argument("--edge_emb_dim", type=int, default=32)
    ap.add_argument("--ema_decay", type=float, default=0.8)
    args = ap.parse_args()

    import torch
    dev = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

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

    tokens, kid, cid, id2safe = infer_one(model, dev, args.smiles)

    print("SMILES:", args.smiles)
    print("TOKENS:", tokens)
    print("kid[:20]:", kid[:20])
    print("cid[:20]:", cid[:20])
    print("id2safe size:", len(id2safe))


if __name__ == "__main__":
    main()
