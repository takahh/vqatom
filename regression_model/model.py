#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end DTI regression (protein sequence only + ligand VQ-Atom tokens)
using:
  - Protein: small Transformer encoder (train from scratch)
  - Ligand: pretrained MLM Transformer loaded from /Users/taka/Downloads/mlm_ep05.pt
  - Cross-attention: protein queries attend to ligand keys/values
  - Regression head: pooled protein -> affinity (scalar)

Input data (CSV) must contain at least:
  - seq      : protein amino-acid sequence (string)
  - lig_tok  : ligand token ids as space-separated integers (string)
  - y        : regression target (float) e.g., pKd/pKi/log-affinity

Example row:
seq,lig_tok,y
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG, 12 532 9 87 23841 23841, 7.23

Notes:
- lig_tok MUST already be VQ-Atom token ids in the same vocabulary as your pretrain.
- PAD/MASK ids are taken from the checkpoint: PAD=base_vocab, MASK=base_vocab+1.
"""

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Repro
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Metrics (no scipy needed)
# -----------------------------
def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    return float(np.sqrt(np.mean((pred - y) ** 2)))

def _rankdata(x: np.ndarray) -> np.ndarray:
    # average-rank ties, like scipy.stats.rankdata(method="average")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    # handle ties
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    # convert to 1-based ranks (doesn't matter for correlation)
    return ranks + 1.0

def spearmanr(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    rp = _rankdata(pred)
    ry = _rankdata(y)
    rp -= rp.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rp ** 2).sum()) * np.sqrt((ry ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((rp * ry).sum() / denom)


# -----------------------------
# Simple CSV reader
# -----------------------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    import csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


# -----------------------------
# Tokenizers
# -----------------------------
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical
AA_TO_ID = {a: i + 1 for i, a in enumerate(AA_VOCAB)}  # reserve 0 as PAD
AA_PAD = 0
AA_UNK = len(AA_TO_ID) + 1  # 21

def encode_protein(seq: str) -> List[int]:
    # unknowns -> AA_UNK (X, B, Z, etc.)
    ids = []
    for ch in seq.strip().upper():
        ids.append(AA_TO_ID.get(ch, AA_UNK))
    return ids


def parse_lig_tokens(s: str) -> List[int]:
    # allow commas or spaces
    s = s.strip().replace(",", " ")
    if not s:
        return []
    return [int(x) for x in s.split()]


# -----------------------------
# Dataset
# -----------------------------
class DTIDataset(Dataset):
    def __init__(self, csv_path: str):
        rows = read_csv_rows(csv_path)
        self.samples = []
        for r in rows:
            seq = r["seq"].strip()
            lig = r["lig_tok"].strip()
            y = float(r["y"])
            self.samples.append((seq, lig, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, lig_str, y = self.samples[idx]
        p_ids = encode_protein(seq)
        l_ids = parse_lig_tokens(lig_str)
        return {
            "p_ids": torch.tensor(p_ids, dtype=torch.long),
            "l_ids": torch.tensor(l_ids, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
        }


@dataclass
class Batch:
    p_ids: torch.Tensor   # (B, Lp)
    l_ids: torch.Tensor   # (B, Ll)
    y: torch.Tensor       # (B,)


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max([int(x.numel()) for x in seqs]) if seqs else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        out[i, : x.numel()] = x
    return out


def collate_fn(samples: List[Dict[str, torch.Tensor]], aa_pad: int, lig_pad: int) -> Batch:
    p = pad_1d([s["p_ids"] for s in samples], aa_pad)
    l = pad_1d([s["l_ids"] for s in samples], lig_pad)
    y = torch.stack([s["y"] for s in samples], dim=0)
    return Batch(p_ids=p, l_ids=l, y=y)


# -----------------------------
# Models
# -----------------------------
class ProteinEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, dim_ff: int, dropout: float):
        super().__init__()
        vocab_size = AA_UNK + 1  # includes PAD=0
        self.pad_id = AA_PAD
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, p_ids: torch.Tensor) -> torch.Tensor:
        # p_ids: (B, Lp)
        x = self.emb(p_ids)  # (B, Lp, D)
        pad_mask = (p_ids == self.pad_id)  # True at PAD
        h = self.enc(x, src_key_padding_mask=pad_mask)  # (B, Lp, D)
        return h


class PretrainedLigandEncoder(nn.Module):
    """
    Matches your pretrain key names:
      tok.weight
      enc.layers.*.self_attn.in_proj_weight ...
    """
    def __init__(self, ckpt_path: str, device: torch.device, finetune: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.state = ckpt["model"]
        self.conf = ckpt["config"]
        self.base_vocab = int(ckpt["base_vocab"])
        self.vocab_size = int(ckpt["vocab_size"])
        self.pad_id = self.base_vocab + 0
        self.mask_id = self.base_vocab + 1

        d_model = int(self.conf["d_model"])
        nhead = int(self.conf["nhead"])
        layers = int(self.conf["layers"])
        dim_ff = int(self.conf["dim_ff"])
        dropout = float(self.conf["dropout"])

        self.tok = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

        # load weights
        missing, unexpected = self.load_state_dict(self.state, strict=True)
        if len(missing) or len(unexpected):
            # strict=True should match exactly; if not, raise with details
            raise RuntimeError(f"PretrainedLigandEncoder load mismatch. missing={missing[:20]} unexpected={unexpected[:20]}")

        self.to(device)

        if not finetune:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
        else:
            self.train()

    @property
    def d_model(self) -> int:
        return int(self.conf["d_model"])

    def forward(self, l_ids: torch.Tensor) -> torch.Tensor:
        # l_ids: (B, Ll)
        x = self.tok(l_ids)  # (B, Ll, D)
        pad_mask = (l_ids == self.pad_id)
        h = self.enc(x, src_key_padding_mask=pad_mask)  # (B, Ll, D)
        return h


def masked_mean(h: torch.Tensor, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # h: (B,L,D), ids: (B,L)
    mask = (ids != pad_id).float()  # 1 for real, 0 for pad
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B,)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)


class CrossAttnDTIRegressor(nn.Module):
    def __init__(
        self,
        protein_encoder: ProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        cross_nhead: int,
        dropout: float,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder
        d_model = self.lig.d_model  # keep same dim for simplest wiring

        # If protein encoder uses different d_model, project it.
        self.p_proj = None
        if getattr(self.prot, "emb").embedding_dim != d_model:
            self.p_proj = nn.Linear(self.prot.emb.embedding_dim, d_model)

        self.cross = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cross_nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.prot_pad_id = AA_PAD
        self.lig_pad_id = self.lig.pad_id

    def forward(self, p_ids: torch.Tensor, l_ids: torch.Tensor) -> torch.Tensor:
        # encode
        p_h = self.prot(p_ids)  # (B, Lp, Dp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)  # (B, Lp, D)

        # ligand encoder may be frozen; make sure no grad if frozen
        l_h = self.lig(l_ids)  # (B, Ll, D)

        # cross-attn: protein queries attend to ligand keys/values
        # key_padding_mask: True for positions that should be ignored (PAD)
        lig_pad_mask = (l_ids == self.lig_pad_id)  # (B, Ll)
        p2l, _ = self.cross(
            query=p_h,
            key=l_h,
            value=l_h,
            key_padding_mask=lig_pad_mask,
            need_weights=False,
        )  # (B, Lp, D)

        # residual (optional but helps)
        p_h = p_h + p2l

        pooled = masked_mean(p_h, p_ids, self.prot_pad_id)  # (B, D)
        y_hat = self.head(pooled).squeeze(-1)  # (B,)
        return y_hat


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    preds = []
    ys = []
    for batch in loader:
        p = batch.p_ids.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)
        y_hat = model(p, l)
        preds.append(y_hat.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    yy = np.concatenate(ys, axis=0)
    return rmse(pred, yy), spearmanr(pred, yy)


def train_one_epoch(model, loader, optimizer, device, loss_type: str = "huber", huber_delta: float = 1.0, grad_clip: float = 1.0) -> float:
    model.train()
    losses = []
    for batch in loader:
        p = batch.p_ids.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(p, l)

        if loss_type == "mse":
            loss = F.mse_loss(y_hat, y)
        elif loss_type == "mae":
            loss = F.l1_loss(y_hat, y)
        else:
            loss = F.huber_loss(y_hat, y, delta=huber_delta)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True, help="CSV with columns: seq, lig_tok, y")
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="/Users/taka/Downloads/mlm_ep05.pt")
    ap.add_argument("--out_dir", type=str, default="./dti_out")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)

    # protein encoder (small)
    ap.add_argument("--p_d_model", type=int, default=256)
    ap.add_argument("--p_nhead", type=int, default=8)
    ap.add_argument("--p_layers", type=int, default=4)
    ap.add_argument("--p_dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    # cross attn
    ap.add_argument("--cross_nhead", type=int, default=8)

    # finetune ligand encoder or freeze
    ap.add_argument("--finetune_lig", action="store_true")

    # loss
    ap.add_argument("--loss", type=str, default="huber", choices=["huber", "mse", "mae"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # datasets
    train_ds = DTIDataset(args.train_csv)
    valid_ds = DTIDataset(args.valid_csv)

    # ligand pad id must match checkpoint, so we create ligand encoder first to get PAD
    lig_enc = PretrainedLigandEncoder(args.ckpt, device=device, finetune=args.finetune_lig)
    lig_pad = lig_enc.pad_id

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(xs, aa_pad=AA_PAD, lig_pad=lig_pad),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(xs, aa_pad=AA_PAD, lig_pad=lig_pad),
    )

    # protein encoder
    prot_enc = ProteinEncoder(
        d_model=args.p_d_model,
        nhead=args.p_nhead,
        layers=args.p_layers,
        dim_ff=args.p_dim_ff,
        dropout=args.dropout,
    ).to(device)

    model = CrossAttnDTIRegressor(
        protein_encoder=prot_enc,
        ligand_encoder=lig_enc,
        cross_nhead=args.cross_nhead,
        dropout=args.dropout,
    ).to(device)

    # optimizer: only trainable params (lig encoder frozen unless --finetune_lig)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best = {"rmse": 1e9, "spearman": -1e9, "epoch": -1}
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_type=args.loss, huber_delta=args.huber_delta, grad_clip=args.grad_clip
        )
        va_rmse, va_spear = evaluate(model, valid_loader, device)
        print(f"[ep {ep:03d}] train_loss={tr_loss:.4f}  val_RMSE={va_rmse:.4f}  val_Spearman={va_spear:.4f}")

        # save best by RMSE (you can swap to spearman if you prefer)
        if va_rmse < best["rmse"]:
            best.update({"rmse": va_rmse, "spearman": va_spear, "epoch": ep})
            save_path = os.path.join(args.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "lig_config": lig_enc.conf,
                    "lig_base_vocab": lig_enc.base_vocab,
                    "lig_vocab_size": lig_enc.vocab_size,
                },
                save_path,
            )
            print("  saved:", save_path)

    print("BEST:", best)
    with open(os.path.join(args.out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
