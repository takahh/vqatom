#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import dgl

from smiles_to_npy import smiles_to_graph_with_labels
from models import EquivariantThreeHopGINE

# ✅ these are in /Users/taka/PycharmProjects/vqatom/0227version/new_train_and_eval.py
from new_train_and_eval import evaluate, convert_to_dgl

from types import SimpleNamespace

def build_args_for_ckpt():
    # From your training command:
    # --codebook_size $CB --hidden_dim $DIM
    # plus any optional defaults used in the model definition
    return SimpleNamespace(
        hidden_dim=16,        # <-- DIM (set to your $DIM used in training)
        codebook_size=10000,  # <-- CB  (set to your $CB used in training)
        edge_emb_dim=32,      # default in model via getattr(args,"edge_emb_dim",32)
        ema_decay=0.8,        # default in VectorQuantize via getattr(args,"ema_decay",0.8)
    )

def load_model(ckpt_path, device="cuda", strict=True):
    import torch

    dev = torch.device(device)

    args = build_args_for_ckpt()

    # hidden_feats is the GINE internal width (NOT args.hidden_dim).
    # In your code, nn1 takes args.hidden_dim -> hidden_feats.
    # Typical setting is hidden_feats == args.hidden_dim, but many runs use larger (e.g., 128).
    # If you trained with DIM=16, you almost certainly kept hidden_feats=16 or 64.
    hidden_feats = args.hidden_dim

    # in_feats/out_feats are not actually used in __init__ beyond signature in this snippet,
    # but keep them consistent.
    in_feats = 64
    out_feats = args.hidden_dim

    model = EquivariantThreeHopGINE(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        args=args,
    ).to(dev)

    ckpt = torch.load(ckpt_path, map_location=dev)

    # support both formats: raw state_dict or {"model": state_dict, ...}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, dev


@torch.no_grad()
def infer_one(model, dev, smiles: str):
    # 1) padded matrices [100,100] and [100,79]
    adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)

    # 2) convert_to_dgl expects list elements that can .view(-1,100,100)/(..,100,79)
    adj_batch = [adj.unsqueeze(0)]      # [1,100,100]
    attr_batch = [attr.unsqueeze(0)]    # [1,100,79]

    base_graphs, extended_graphs, masks_dict, attr_matrices_all, _, _ = convert_to_dgl(
        adj_batch,
        attr_batch,
        logger=None,
        start_atom_id=0,
        start_mol_id=0,
        device=str(dev),
    )

    # single molecule
    g = extended_graphs[0]
    X = attr_matrices_all[0]     # de-padded [N,79]

    # pipeline always batches
    batched_graph = dgl.batch([g]).to(dev)
    feats = batched_graph.ndata["feat"]

    # pipeline passes list aligned with graphs
    attr_list = [X.to(dev)]

    # call evaluate exactly like your infer loop
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

    # offsets for global token ids
    offsets = {}
    cur = 0
    for safe_key in model.vq._codebook.embed.keys():
        K = int(model.vq._codebook.embed[safe_key].shape[0])
        offsets[safe_key] = cur
        cur += K

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
    # ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--device", default=0, type=int, help="GPU index, -1=cpu")
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")
    args = ap.parse_args()

    model, dev = load_model(args.ckpt, device=args.device, strict=args.strict)
    tokens, kid, cid, id2safe = infer_one(model, dev, args.smiles)

    print("SMILES:", args.smiles)
    print("TOKENS:", tokens)
    print("kid[:20]:", kid[:20])
    print("cid[:20]:", cid[:20])
    print("id2safe size:", len(id2safe))


if __name__ == "__main__":
    main()
