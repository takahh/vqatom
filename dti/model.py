#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end DTI regression (protein sequence only + ligand VQ-Atom tokens)

- Protein: ESM-2 (Hugging Face transformers)
- Ligand : pretrained MLM Transformer (loaded from --lig_ckpt)
- Cross-attention: protein queries attend to ligand keys/values
- Regression head: pooled protein -> affinity (scalar)

CSV must contain:
  - seq      : protein amino-acid sequence (string)
  - lig_tok  : ligand token ids as space-separated integers (string)
  - y        : regression target (float) e.g., pKd/pKi/log-affinity

Notes:
- lig_tok MUST already be VQ-Atom token ids in the same vocabulary as your ligand MLM pretrain.
- PAD/MASK ids are taken from the ligand MLM checkpoint:
    PAD = base_vocab + 0
    MASK = base_vocab + 1
"""

import os
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

# NEW: Hugging Face ESM-2
from transformers import AutoTokenizer, EsmModel


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
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

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

    return ranks + 1.0


def spearmanr(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    rp = _rankdata(pred)
    ry = _rankdata(y)
    rp -= rp.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rp**2).sum()) * np.sqrt((ry**2).sum()))
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
# Ligand tokenizer/parser
# -----------------------------
def parse_lig_tokens(s: str) -> List[int]:
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
            if "seq" not in r or "lig_tok" not in r or "y" not in r:
                raise KeyError("CSV must contain columns: seq, lig_tok, y")

            seq = r["seq"].strip()
            lig = r["lig_tok"].strip()
            y = float(r["y"])
            self.samples.append((seq, lig, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, lig_str, y = self.samples[idx]
        l_ids = parse_lig_tokens(lig_str)
        return {
            "seq": seq,  # keep as string; tokenize in collate
            "l_ids": torch.tensor(l_ids, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
        }


@dataclass
class Batch:
    p_input_ids: torch.Tensor  # (B, Lp)
    p_attn_mask: torch.Tensor  # (B, Lp) 1=real 0=pad
    l_ids: torch.Tensor        # (B, Ll)
    y: torch.Tensor            # (B,)


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not seqs:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(x.numel()) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        out[i, : x.numel()] = x
    return out


def collate_fn(samples: List[Dict[str, object]], esm_tokenizer, lig_pad: int) -> Batch:
    # protein (tokenize here)
    seqs = [s["seq"] for s in samples]
    enc = esm_tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,  # keep True; optionally set max_length if you want
    )
    p_input_ids = enc["input_ids"].long()         # (B, Lp)
    p_attn_mask = enc["attention_mask"].long()    # (B, Lp), 1=real, 0=pad

    # ligand (pad int ids)
    l_ids = pad_1d([s["l_ids"] for s in samples], lig_pad)

    # targets
    y = torch.stack([s["y"] for s in samples], dim=0)

    return Batch(p_input_ids=p_input_ids, p_attn_mask=p_attn_mask, l_ids=l_ids, y=y)


# -----------------------------
# Models
# -----------------------------
class PretrainedLigandEncoder(nn.Module):
    """
    Loads the ligand MLM encoder from a checkpoint produced by your MLM pretrain script.

    Expected checkpoint keys:
      ckpt["model"] : state_dict of the MLM model (tok + enc + lm_head)
      ckpt["config"]: args dict including d_model, nhead, layers, dim_ff, dropout
      ckpt["base_vocab"], ckpt["vocab_size"]
    """
    def __init__(self, ckpt_path: str, device: torch.device, finetune: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if not (isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt):
            raise RuntimeError("Ligand checkpoint must be a dict containing at least keys: 'model', 'config'")

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

        missing, unexpected = self.load_state_dict(self.state, strict=False)
        if unexpected:
            print("[ligand] load_state_dict unexpected keys (showing up to 10):", unexpected[:10])
        if missing:
            print("[ligand] load_state_dict missing keys (showing up to 10):", missing[:10])

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
        x = self.tok(l_ids)  # (B, Ll, D)
        pad_mask = (l_ids == self.pad_id)  # True at PAD
        h = self.enc(x, src_key_padding_mask=pad_mask)  # (B, Ll, D)
        return h


def masked_mean_by_attn(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    h: (B,L,D)
    attn_mask: (B,L) 1=real, 0=pad
    """
    m = attn_mask.float()
    denom = m.sum(dim=1).clamp(min=1.0)  # (B,)
    return (h * m.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)


class ESMProteinEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, finetune: bool = False):
        super().__init__()
        self.model_name = model_name
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = int(self.esm.config.hidden_size)

        self.to(device)

        if not finetune:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
        else:
            self.train()

    def forward(self, p_input_ids: torch.Tensor, p_attn_mask: torch.Tensor) -> torch.Tensor:
        # returns last_hidden_state: (B, Lp, H)
        out = self.esm(input_ids=p_input_ids, attention_mask=p_attn_mask)
        return out.last_hidden_state


class CrossAttnDTIRegressor(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        cross_nhead: int,
        dropout: float,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model  # unify to ligand dim

        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

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

        self.lig_pad_id = self.lig.pad_id

    def forward(self, p_input_ids: torch.Tensor, p_attn_mask: torch.Tensor, l_ids: torch.Tensor) -> torch.Tensor:
        p_h = self.prot(p_input_ids, p_attn_mask)  # (B, Lp, Hp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)                 # (B, Lp, D)

        l_h = self.lig(l_ids)                      # (B, Ll, D)

        lig_pad_mask = (l_ids == self.lig_pad_id)  # True at PAD
        p2l, _ = self.cross(
            query=p_h,
            key=l_h,
            value=l_h,
            key_padding_mask=lig_pad_mask,
            need_weights=False,
        )
        p_h = p_h + p2l  # residual

        pooled = masked_mean_by_attn(p_h, p_attn_mask)  # (B, D)
        y_hat = self.head(pooled).squeeze(-1)           # (B,)
        return y_hat


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds = []
    ys = []
    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)
        y_hat = model(p_ids, p_msk, l)
        preds.append(y_hat.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float64)
    yy = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.float64)
    if pred.size == 0:
        return 0.0, 0.0
    return rmse(pred, yy), spearmanr(pred, yy)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(p_ids, p_msk, l)

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


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dti_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, lig_enc: PretrainedLigandEncoder, epoch: int, best: dict):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "lig_config": lig_enc.conf,
            "lig_base_vocab": lig_enc.base_vocab,
            "lig_vocab_size": lig_enc.vocab_size,
        },
        path,
    )


def load_dti_checkpoint(path: str, model: nn.Module, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    return ckpt, missing, unexpected


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_csv", type=str, default=None, help="Train CSV (required unless --eval_only). Columns: seq, lig_tok, y")
    ap.add_argument("--valid_csv", type=str, required=True, help="Validation CSV. Columns: seq, lig_tok, y")
    ap.add_argument("--test_csv", type=str, default=None, help="Optional test CSV to also evaluate (useful with --eval_only).")

    # ligand MLM ckpt
    ap.add_argument("--lig_ckpt", type=str, default="/vqatom/dti/data/mlm_ep05.pt", help="Ligand MLM checkpoint (mlm_epXX.pt)")

    # protein ESM-2
    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="Hugging Face model name for ESM-2")
    ap.add_argument("--finetune_esm", action="store_true", help="If set, allow gradients into ESM-2; otherwise frozen.")

    # DTI ckpt (best.pt) for loading before eval
    ap.add_argument("--dti_ckpt", type=str, default=None, help="Path to a trained DTI checkpoint (e.g., ./dti_out/best.pt). If set, load it before eval.")

    # io
    ap.add_argument("--out_dir", type=str, default="./dti_out")
    ap.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation.")

    # train
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)

    # cross attn
    ap.add_argument("--cross_nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    # finetune ligand encoder or freeze
    ap.add_argument("--finetune_lig", action="store_true")

    # loss
    ap.add_argument("--loss", type=str, default="huber", choices=["huber", "mse", "mae"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    if not args.eval_only and not args.train_csv:
        raise ValueError("--train_csv is required unless --eval_only is set.")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("lig_ckpt:", args.lig_ckpt)
    print("esm_model:", args.esm_model)

    # ESM tokenizer (CPUでOK。collateで呼ぶ)
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model, do_lower_case=False)

    # ligand encoder first (to get lig_pad id)
    lig_enc = PretrainedLigandEncoder(args.lig_ckpt, device=device, finetune=args.finetune_lig)
    lig_pad = lig_enc.pad_id
    print(f"lig_vocab_size={lig_enc.vocab_size} base_vocab={lig_enc.base_vocab} PAD={lig_enc.pad_id} MASK={lig_enc.mask_id}")

    # datasets/loaders
    valid_ds = DTIDataset(args.valid_csv)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad),
    )

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad),
        )

    train_loader = None
    if not args.eval_only:
        train_ds = DTIDataset(args.train_csv)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad),
        )

    # protein encoder (ESM-2)
    prot_enc = ESMProteinEncoder(
        model_name=args.esm_model,
        device=device,
        finetune=args.finetune_esm,
    )

    # full model
    model = CrossAttnDTIRegressor(
        protein_encoder=prot_enc,
        ligand_encoder=lig_enc,
        cross_nhead=args.cross_nhead,
        dropout=args.dropout,
    ).to(device)

    # optionally load DTI checkpoint before eval/train
    if args.dti_ckpt is not None:
        ckpt_obj, missing, unexpected = load_dti_checkpoint(args.dti_ckpt, model, device)
        print(f"Loaded DTI checkpoint: {args.dti_ckpt}")
        if missing:
            print("  [warn] missing keys (up to 20):", missing[:20])
        if unexpected:
            print("  [warn] unexpected keys (up to 20):", unexpected[:20])

    # eval-only mode
    if args.eval_only:
        va_rmse, va_spear = evaluate(model, valid_loader, device)
        print(f"[VALID] RMSE={va_rmse:.4f}  Spearman={va_spear:.4f}")

        if test_loader is not None:
            te_rmse, te_spear = evaluate(model, test_loader, device)
            print(f"[TEST ] RMSE={te_rmse:.4f}  Spearman={te_spear:.4f}")

        summary = {
            "mode": "eval_only",
            "valid": {"rmse": va_rmse, "spearman": va_spear, "csv": args.valid_csv},
            "test": ({"rmse": te_rmse, "spearman": te_spear, "csv": args.test_csv} if test_loader is not None else None),
            "dti_ckpt": args.dti_ckpt,
            "lig_ckpt": args.lig_ckpt,
            "esm_model": args.esm_model,
            "finetune_esm": bool(args.finetune_esm),
            "seed": args.seed,
        }
        save_json(os.path.join(args.out_dir, "eval_only.json"), summary)
        return

    # optimizer: only trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best = {"rmse": 1e9, "spearman": -1e9, "epoch": -1}

    # training loop
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_type=args.loss, huber_delta=args.huber_delta, grad_clip=args.grad_clip
        )
        va_rmse, va_spear = evaluate(model, valid_loader, device)
        print(f"[ep {ep:03d}] train_loss={tr_loss:.4f}  val_RMSE={va_rmse:.4f}  val_Spearman={va_spear:.4f}")

        if va_rmse < best["rmse"]:
            best.update({"rmse": va_rmse, "spearman": va_spear, "epoch": ep})
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best)

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)


if __name__ == "__main__":
    main()
