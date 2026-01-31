#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import dgl

from smiles_to_npy import smiles_to_graph_with_labels
from models import EquivariantThreeHopGINE
from your_convert_module import convert_to_dgl_single   # <-- change to your actual module
from your_eval_module import evaluate                   # <-- or paste evaluate() into this file


def ensure_codebook_keys_if_possible(model, state_dict):
    """
    If you use dynamic ParameterDict keys inside EuclideanCodebook, strict restore
    can fail unless you pre-create keys. If you already have this helper in your
    codebase, import and use it instead of this stub.
    """
    # If you already have the full implementation, call that one.
    return


def load_model_from_ckpt(ckpt_path: str, device: str = "cuda", strict: bool = True):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev)

    if not (isinstance(ckpt, dict) and "args" in ckpt):
        raise KeyError("Checkpoint must contain ckpt['args'] to rebuild the model exactly.")

    train_args = ckpt["args"]
    state = ckpt["state_dict"] if ("state_dict" in ckpt) else ckpt

    model = EquivariantThreeHopGINE(
        in_feats=train_args.hidden_dim,         # unused in your class; keep same as training
        hidden_feats=train_args.hidden_dim,
        out_feats=train_args.hidden_dim,
        args=train_args,
    ).to(dev)

    # optional: pre-create dynamic VQ codebook keys for strict restore
    ensure_codebook_keys_if_possible(model, state)

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, train_args, dev


@torch.no_grad()
def infer_tokens_for_smiles(model, dev, smiles: str):
    # 1) build adj/attr
    adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)

    # 2) build DGL graphs + node features
    base_g, g, X, meta = convert_to_dgl_single(adj, attr, device=str(dev))

    # IMPORTANT:
    # Your training infer pipeline batches `glist` and uses `batched_graph.ndata["feat"]`.
    # So do the same: dgl.batch([g]).
    batched_graph = dgl.batch([g]).to(dev)
    feats = batched_graph.ndata["feat"]

    # 3) attr_list must be a LIST aligned with the graphs in this batch/chunk
    # In your pipeline: attr_matrices_all[start:start+chunk_size] (a list).
    # For single graph:
    attr_list = [attr.to(dev)]

    # 4) masks: in your infer loop you pass masks_3. If your vq/infer ignores it,
    # you can pass None. To be maximally consistent, pass the real mask_dict if you have it.
    mask_dict = None

    # 5) call evaluate() exactly like your pipeline
    _, (kid, cid, id2safe), _ = evaluate(
        model=model,
        g=batched_graph,
        feats=feats,
        epoch=0,
        mask_dict=mask_dict,
        logger=None,
        g_base=None,
        chunk_i=0,
        mode="infer",
        attr_list=attr_list,
    )

    kid = kid.cpu().tolist()
    cid = cid.cpu().tolist()

    # 6) global token mapping: offsets from codebook key order
    offsets = {}
    cur = 0
    for safe_key in model.vq._codebook.embed.keys():
        K = int(model.vq._codebook.embed[safe_key].shape[0])
        offsets[safe_key] = cur
        cur += K

    tokens = []
    for k, c in zip(kid, cid):
        safe = id2safe[int(k)]
        tokens.append(offsets[safe] + int(c))

    return tokens, kid, cid, id2safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model_epoch_xx.pth")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")
    args = ap.parse_args()

    model, train_args, dev = load_model_from_ckpt(args.ckpt, device=args.device, strict=args.strict)

    tokens, kid, cid, id2safe = infer_tokens_for_smiles(model, dev, args.smiles)

    print("SMILES:", args.smiles)
    print("TOKENS:", tokens)
    # optional debug:
    print("kid (first 20):", kid[:20])
    print("cid (first 20):", cid[:20])
    print("id2safe size:", len(id2safe))


if __name__ == "__main__":
    main()
