#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end DTI regression (protein sequence only + ligand VQ-Atom tokens)
+ linear calibration (a,b) fit on a held-out calibration split from TRAIN,
  then applied to VALID/TEST metrics (no leakage from VALID).

- Protein: ESM-2 (Hugging Face transformers)
- Ligand : pretrained MLM Transformer weights (loaded from --lig_ckpt)
          BUT vocab meta (base_vocab/vocab_size/PAD/MASK) can be sourced from discretization ckpt (--vq_ckpt)
- Cross-attention: protein queries attend to ligand keys/values
- Regression head: pooled protein -> affinity (scalar)

CSV must contain:
  - seq      : protein amino-acid sequence (string)
  - lig_tok  : ligand token ids as space-separated integers (string)
  - y        : regression target (float) e.g., pKd/pKi/log-affinity

Calibration:
- Each epoch, fit linear calibration y ≈ a*y_pred + b on TRAIN-CALIB split only.
- Report (and optionally checkpoint-select) calibrated RMSE on VALID/TEST.
- Spearman is invariant to linear transform, so calibration won't change it (numerically tiny diffs only).

IMPORTANT:
- lig_tok MUST be token ids in the SAME global-id space as the vocab meta you use.
  If you pass --vq_ckpt, PAD/MASK/vocab_size come from discretization global_id_meta.
  If you do NOT pass --vq_ckpt, they come from ligand MLM ckpt (legacy behavior).

- When vocab_size differs between MLM weights and discretization vocab meta:
    * We do a SHAPE-SAFE partial load for transformer weights.
    * Embedding tok.weight is copied for the overlapping rows [0:overlap).
      Remaining rows stay randomly initialized.
  This lets you run without index-out-of-range crashes, but note:
    If MLM was trained under a different token-id space, semantic alignment is not guaranteed.
"""

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

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


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    return float(np.mean(np.abs(pred - y)))


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
            ranks[order[i: j + 1]] = avg
        i = j + 1

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


def pearsonr(pred: np.ndarray, y: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    pred -= pred.mean()
    y -= y.mean()
    denom = float(np.sqrt((pred ** 2).sum()) * np.sqrt((y ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((pred * y).sum() / denom)


# -----------------------------
# Calibration (no leakage)
# -----------------------------
def fit_linear_calibration(pred: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ a*pred + b by least squares on calibration split.
    Returns (a, b). Safe if pred variance ~ 0.
    """
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)

    vx = float(np.var(pred))
    if vx < 1e-12:
        return 0.0, float(np.mean(y))

    # a = Cov(x,y) / Var(x)
    cov = float(np.mean((pred - pred.mean()) * (y - y.mean())))
    a = cov / vx
    b = float(np.mean(y) - a * np.mean(pred))
    return float(a), float(b)


def apply_linear_calibration(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a * pred + b).astype(np.float64)


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
            "seq": seq,
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
    max_len = max(int(x.numel()) for x in seqs) if seqs else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        if x.numel() > 0:
            out[i, : x.numel()] = x
    return out


def collate_fn(samples: List[Dict[str, object]], esm_tokenizer, lig_pad: int) -> Batch:
    seqs = [s["seq"] for s in samples]
    enc = esm_tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    p_input_ids = enc["input_ids"].long()
    p_attn_mask = enc["attention_mask"].long()

    l_ids = pad_1d([s["l_ids"] for s in samples], lig_pad)
    y = torch.stack([s["y"] for s in samples], dim=0)
    return Batch(p_input_ids=p_input_ids, p_attn_mask=p_attn_mask, l_ids=l_ids, y=y)


# -----------------------------
# Utilities: vocab meta + safe loading
# -----------------------------
def load_vocab_meta_from_vq_ckpt(vq_ckpt_path: str) -> dict:
    vq = torch.load(vq_ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(vq, dict) and "global_id_meta" in vq):
        raise RuntimeError("vq_ckpt must be a dict containing 'global_id_meta'")

    gim = vq["global_id_meta"]
    for k in ["global_vocab_size", "vocab_size", "pad_id", "mask_id"]:
        if k not in gim:
            raise RuntimeError(f"global_id_meta missing key: {k}")

    return {
        "base_vocab": int(gim["global_vocab_size"]),
        "vocab_size": int(gim["vocab_size"]),
        "pad_id": int(gim["pad_id"]),
        "mask_id": int(gim["mask_id"]),
    }


def load_state_dict_shape_safe(module: nn.Module, state: dict, verbose: bool = True):
    cur = module.state_dict()
    loadable = {}
    skipped = []
    for k, v in state.items():
        if k not in cur:
            skipped.append((k, "missing_in_model"))
            continue
        if cur[k].shape != v.shape:
            skipped.append((k, f"shape {tuple(v.shape)} != {tuple(cur[k].shape)}"))
            continue
        loadable[k] = v

    missing, unexpected = module.load_state_dict(loadable, strict=False)

    if verbose:
        if skipped:
            print("[load_shape_safe] skipped (up to 20):", skipped[:20])
        if unexpected:
            print("[load_shape_safe] unexpected (up to 20):", unexpected[:20])
        if missing:
            print("[load_shape_safe] missing (up to 20):", missing[:20])

    return missing, unexpected, skipped


# -----------------------------
# Models
# -----------------------------
class PretrainedLigandEncoder(nn.Module):
    """
    Loads ligand Transformer encoder weights from MLM ckpt (ckpt_path),
    but can source vocab meta from discretization ckpt (--vq_ckpt).

    Weight loading:
    - encoder weights: shape-safe partial load
    - embedding tok.weight: overlap rows copied
    """
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        finetune: bool = False,
        vq_ckpt_path: Optional[str] = None,
        verbose_load: bool = True,
        debug_index_check: bool = False,
    ):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if not (isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt):
            raise RuntimeError("Ligand checkpoint must be a dict containing at least keys: 'model', 'config'")

        self.state = ckpt["model"]
        self.conf = ckpt["config"]

        # ---- vocab meta
        if vq_ckpt_path is not None:
            vm = load_vocab_meta_from_vq_ckpt(vq_ckpt_path)
            self.base_vocab = int(vm["base_vocab"])
            self.vocab_size = int(vm["vocab_size"])
            self.pad_id = int(vm["pad_id"])
            self.mask_id = int(vm["mask_id"])
            self.vocab_source = f"vq_ckpt:{vq_ckpt_path}"
        else:
            if "base_vocab" not in ckpt or "vocab_size" not in ckpt:
                raise RuntimeError("Ligand MLM ckpt must contain base_vocab and vocab_size if --vq_ckpt is not provided")
            self.base_vocab = int(ckpt["base_vocab"])
            self.vocab_size = int(ckpt["vocab_size"])
            self.pad_id = self.base_vocab + 0
            self.mask_id = self.base_vocab + 1
            self.vocab_source = f"mlm_ckpt:{ckpt_path}"

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

        self.debug_index_check = bool(debug_index_check)

        # ---- Load weights (shape-safe), then handle tok.weight overlap copy
        load_state_dict_shape_safe(self, self.state, verbose=verbose_load)

        mlm_tok_key = "tok.weight"
        if mlm_tok_key in self.state:
            w = self.state[mlm_tok_key]
            if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[1] == d_model:
                overlap = min(int(w.shape[0]), int(self.tok.weight.shape[0]))
                if overlap > 0:
                    with torch.no_grad():
                        self.tok.weight[:overlap].copy_(w[:overlap])
                if verbose_load:
                    print(f"[ligand] tok.weight overlap-copied rows: {overlap} / {self.tok.weight.shape[0]}")
            elif verbose_load:
                print("[ligand] tok.weight exists but shape is unexpected; skipped overlap copy.")

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
        if self.debug_index_check:
            with torch.no_grad():
                li = l_ids.detach()
                li_cpu = li.cpu() if li.is_cuda else li
                bad = ((li_cpu < 0) | (li_cpu >= self.tok.num_embeddings)).nonzero(as_tuple=False)
                if bad.numel() > 0:
                    b0 = bad[0].tolist()
                    v0 = int(li_cpu[b0[0], b0[1]].item())
                    raise RuntimeError(f"Ligand token id out of range: id={v0}, num_embeddings={self.tok.num_embeddings}")

        x = self.tok(l_ids)                      # (B, Ll, D)
        pad_mask = (l_ids == self.pad_id)        # True at PAD
        h = self.enc(x, src_key_padding_mask=pad_mask)  # (B, Ll, D)
        return h


def masked_mean_by_attn(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float()
    denom = m.sum(dim=1).clamp(min=1.0)
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

        d_model = self.lig.d_model

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
# Predict / Eval (raw + calibrated)
# -----------------------------
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    return pred, yy


def eval_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if pred.size == 0:
        return {"rmse": 0.0, "mae": 0.0, "spearman": 0.0, "pearson": 0.0, "pred_mean": 0.0, "pred_std": 0.0}
    return {
        "rmse": rmse(pred, y),
        "mae": mae(pred, y),
        "spearman": spearmanr(pred, y),
        "pearson": pearsonr(pred, y),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


# -----------------------------
# Train
# -----------------------------
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


def save_dti_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    lig_enc: PretrainedLigandEncoder,
    epoch: int,
    best: dict,
    calib: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "calibration": calib,  # {"a":..., "b":..., "fit_on":"train_calib", "epoch":...}
            "lig_config": lig_enc.conf,
            "lig_vocab_source": lig_enc.vocab_source,
            "lig_base_vocab": lig_enc.base_vocab,
            "lig_vocab_size": lig_enc.vocab_size,
            "lig_pad_id": lig_enc.pad_id,
            "lig_mask_id": lig_enc.mask_id,
        },
        path,
    )


def load_dti_checkpoint(path: str, model: nn.Module, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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

    # ligand MLM weights ckpt
    ap.add_argument("--lig_ckpt", type=str, default="/vqatom/data/mlm_ep05.pt", help="Ligand MLM checkpoint (mlm_epXX.pt)")

    # discretization ckpt (vocab meta source)
    ap.add_argument("--vq_ckpt", type=str, default=None, help="Discretization ckpt to source vocab meta (has global_id_meta).")

    # protein ESM-2
    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="Hugging Face model name for ESM-2")
    ap.add_argument("--finetune_esm", action="store_true", help="If set, allow gradients into ESM-2; otherwise frozen.")

    # DTI ckpt for loading before eval/train
    ap.add_argument("--dti_ckpt", type=str, default=None, help="Path to a trained DTI checkpoint (e.g., ./dti_out/best.pt). If set, load it before eval/train.")

    # io
    ap.add_argument("--out_dir", type=str, default="./dti_out")
    ap.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation.")

    # train
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lig_lr_mult", type=float, default=0.1, help="Ligand LR multiplier (lig_lr = lr * lig_lr_mult)")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)

    # calibration split
    ap.add_argument("--calib_frac", type=float, default=0.1, help="Fraction of TRAIN held out for calibration (fit a,b).")
    ap.add_argument("--calib_every", type=int, default=1, help="Fit calibration every N epochs (>=1).")

    # cross attn
    ap.add_argument("--cross_nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    # finetune ligand encoder or freeze
    ap.add_argument("--finetune_lig", action="store_true")

    # debug
    ap.add_argument("--lig_debug_index", action="store_true", help="Crash early if ligand id out-of-range.")

    # loss
    ap.add_argument("--loss", type=str, default="huber", choices=["huber", "mse", "mae"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # scheduler
    ap.add_argument("--plateau", action="store_true", help="Use ReduceLROnPlateau on VALID calibrated RMSE.")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # selection criterion
    ap.add_argument("--select_on", type=str, default="rmse_cal", choices=["rmse_raw", "rmse_cal"],
                    help="Which VALID RMSE to use for best.pt selection.")

    args = ap.parse_args()

    if not args.eval_only and not args.train_csv:
        raise ValueError("--train_csv is required unless --eval_only is set.")
    if not (0.0 <= args.calib_frac < 0.5):
        raise ValueError("--calib_frac must be in [0, 0.5).")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("lig_ckpt:", args.lig_ckpt)
    print("vq_ckpt :", args.vq_ckpt)
    print("esm_model:", args.esm_model)

    # ESM tokenizer
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model, do_lower_case=False)

    # ligand encoder first (to get lig_pad id)
    lig_enc = PretrainedLigandEncoder(
        ckpt_path=args.lig_ckpt,
        device=device,
        finetune=args.finetune_lig,
        vq_ckpt_path=args.vq_ckpt,
        verbose_load=True,
        debug_index_check=bool(args.lig_debug_index),
    )
    lig_pad = lig_enc.pad_id
    print(f"[ligand] vocab_source={lig_enc.vocab_source}")
    print(f"[ligand] vocab_size={lig_enc.vocab_size} base_vocab={lig_enc.base_vocab} PAD={lig_enc.pad_id} MASK={lig_enc.mask_id}")

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
    calib_loader = None
    train_ds = None

    if not args.eval_only:
        train_ds = DTIDataset(args.train_csv)
        n = len(train_ds)
        idx = np.arange(n)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)

        n_cal = int(round(n * args.calib_frac))
        n_cal = max(0, min(n_cal, n - 1))  # keep at least 1 sample for train
        cal_idx = idx[:n_cal]
        tr_idx = idx[n_cal:]

        train_loader = DataLoader(
            Subset(train_ds, tr_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad),
        )
        calib_loader = DataLoader(
            Subset(train_ds, cal_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad),
        )
        print(f"[split] train={len(tr_idx)}  calib={len(cal_idx)}  valid={len(valid_ds)}")

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
    loaded_calib = None
    if args.dti_ckpt is not None:
        ckpt_obj, missing, unexpected = load_dti_checkpoint(args.dti_ckpt, model, device)
        print(f"Loaded DTI checkpoint: {args.dti_ckpt}")
        if missing:
            print("  [warn] missing keys (up to 20):", missing[:20])
        if unexpected:
            print("  [warn] unexpected keys (up to 20):", unexpected[:20])
        loaded_calib = ckpt_obj.get("calibration", None) if isinstance(ckpt_obj, dict) else None
        if loaded_calib is not None:
            print(f"Loaded calibration: a={loaded_calib.get('a')} b={loaded_calib.get('b')} (epoch={loaded_calib.get('epoch')})")

    # eval-only mode (apply loaded calibration if exists)
    if args.eval_only:
        pred_v, y_v = predict(model, valid_loader, device)
        m_raw = eval_metrics(pred_v, y_v)

        if loaded_calib is not None and "a" in loaded_calib and "b" in loaded_calib:
            a = float(loaded_calib["a"])
            b = float(loaded_calib["b"])
            pred_v_cal = apply_linear_calibration(pred_v, a, b)
            m_cal = eval_metrics(pred_v_cal, y_v)
            print(f"[VALID raw] RMSE={m_raw['rmse']:.4f}  Spearman={m_raw['spearman']:.4f}  Pearson={m_raw['pearson']:.4f}  MAE={m_raw['mae']:.4f}")
            print(f"[VALID cal] RMSE={m_cal['rmse']:.4f}  (a={a:.4f}, b={b:.4f})")
        else:
            print(f"[VALID] RMSE={m_raw['rmse']:.4f}  Spearman={m_raw['spearman']:.4f}  Pearson={m_raw['pearson']:.4f}  MAE={m_raw['mae']:.4f}")

        if test_loader is not None:
            pred_t, y_t = predict(model, test_loader, device)
            mt_raw = eval_metrics(pred_t, y_t)
            if loaded_calib is not None and "a" in loaded_calib and "b" in loaded_calib:
                a = float(loaded_calib["a"])
                b = float(loaded_calib["b"])
                pred_t_cal = apply_linear_calibration(pred_t, a, b)
                mt_cal = eval_metrics(pred_t_cal, y_t)
                print(f"[TEST  raw] RMSE={mt_raw['rmse']:.4f}  Spearman={mt_raw['spearman']:.4f}  Pearson={mt_raw['pearson']:.4f}  MAE={mt_raw['mae']:.4f}")
                print(f"[TEST  cal] RMSE={mt_cal['rmse']:.4f}  (a={a:.4f}, b={b:.4f})")
            else:
                print(f"[TEST ] RMSE={mt_raw['rmse']:.4f}  Spearman={mt_raw['spearman']:.4f}  Pearson={mt_raw['pearson']:.4f}  MAE={mt_raw['mae']:.4f}")

        summary = {
            "mode": "eval_only",
            "valid_raw": m_raw,
            "valid_csv": args.valid_csv,
            "test_csv": args.test_csv,
            "dti_ckpt": args.dti_ckpt,
            "calibration_loaded": loaded_calib,
        }
        save_json(os.path.join(args.out_dir, "eval_only.json"), summary)
        return

    # optimizer with LR split (now we can reference model.lig/model.prot directly)
    base_lr = float(args.lr)
    lig_lr = float(args.lr) * float(args.lig_lr_mult)

    lig_params = [p for p in model.lig.parameters() if p.requires_grad]
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("lig."):
            continue
        other_params.append(p)

    print(f"[opt] lig_lr={lig_lr:g} base_lr={base_lr:g}  lig_params={len(lig_params)} other_params={len(other_params)}")
    assert len(other_params) > 0, "No trainable params outside ligand. Did you freeze everything?"

    optimizer = torch.optim.AdamW(
        [
            {"params": lig_params, "lr": lig_lr},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.min_lr),
        )

    # best selection
    best = {"rmse_raw": 1e9, "rmse_cal": 1e9, "spearman": -1e9, "epoch": -1}
    calib_state = {"a": 1.0, "b": 0.0, "fit_on": "train_calib", "epoch": 0}

    # training loop
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_type=args.loss, huber_delta=args.huber_delta, grad_clip=args.grad_clip
        )

        # ---- fit calibration on TRAIN-CALIB every N epochs (or if calib split empty, keep identity)
        if calib_loader is not None and len(calib_loader.dataset) > 0 and (ep % max(1, args.calib_every) == 0):
            pred_c, y_c = predict(model, calib_loader, device)
            # a, b = fit_linear_calibration(pred_c, y_c)
            # calib_state = {"a": float(a), "b": float(b), "fit_on": "train_calib", "epoch": ep}

            a_new, b_new = fit_linear_calibration(pred_c, y_c)

            ema = 0.9
            a_old = float(calib_state["a"])
            b_old = float(calib_state["b"])
            a = ema * a_old + (1 - ema) * float(a_new)
            b = ema * b_old + (1 - ema) * float(b_new)

            calib_state = {"a": float(a), "b": float(b), "fit_on": "train_calib", "epoch": ep}

        a = float(calib_state["a"])
        b = float(calib_state["b"])

        # ---- VALID metrics (raw + calibrated)
        pred_v, y_v = predict(model, valid_loader, device)
        m_raw = eval_metrics(pred_v, y_v)

        pred_v_cal = apply_linear_calibration(pred_v, a, b)
        m_cal = eval_metrics(pred_v_cal, y_v)

        print(
            f"[ep {ep:03d}] train_loss={tr_loss:.4f}  "
            f"val_RMSE={m_raw['rmse']:.4f}  val_RMSE_cal={m_cal['rmse']:.4f}  "
            f"val_Spearman={m_raw['spearman']:.4f}  "
            f"(a={a:.4f}, b={b:.4f})"
        )

        # scheduler step on calibrated RMSE (recommended)
        if scheduler is not None:
            scheduler.step(m_cal["rmse"])

        # selection
        cur_key = "rmse_cal" if args.select_on == "rmse_cal" else "rmse_raw"
        cur_val = float(m_cal["rmse"]) if cur_key == "rmse_cal" else float(m_raw["rmse"])
        best_val = float(best[cur_key])

        if cur_val < best_val:
            best.update(
                {
                    "rmse_raw": float(m_raw["rmse"]),
                    "rmse_cal": float(m_cal["rmse"]),
                    "spearman": float(m_raw["spearman"]),
                    "epoch": ep,
                }
            )
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best, calib=calib_state)
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best, calib=calib_state)

        # optional TEST metrics (if provided) each epoch (can be slow; keep it light)
        if test_loader is not None:
            pred_t, y_t = predict(model, test_loader, device)
            mt_raw = eval_metrics(pred_t, y_t)
            mt_cal = eval_metrics(apply_linear_calibration(pred_t, a, b), y_t)
            print(
                f"         test_RMSE={mt_raw['rmse']:.4f}  test_RMSE_cal={mt_cal['rmse']:.4f}  "
                f"test_Spearman={mt_raw['spearman']:.4f}"
            )

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)
    save_json(os.path.join(args.out_dir, "calibration.json"), calib_state)


if __name__ == "__main__":
    main()
