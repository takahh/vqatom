#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_continuous_mae.py

Continuous-value masked atom modeling for VQ-Atom style pretraining files.

This is the continuous analogue of token MLM:

    token MLM:
        token_ids -> Embedding -> Transformer -> vocab logits -> CE

    continuous MAE:
        atom_features -> Linear -> Transformer -> reconstructed_features -> SmoothL1/MSE

Expected ragged .pt file format:
    required:
        offsets: LongTensor, shape (num_mols + 1,)
    feature tensor, one of:
        atom_feats_flat / feats_flat / x_flat / features_flat / attr_flat / attrs_flat
        shape (total_atoms, feat_dim)

Optional:
    feat_dim: int
    feature_mean: Tensor shape (feat_dim,)
    feature_std : Tensor shape (feat_dim,)

Each item is one molecule with variable number of atoms.
"""

from __future__ import annotations

import os
import glob
import math
import random
import argparse
import json
import time
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Feature-key helpers
# ============================================================
FEATURE_KEYS = (
    "atom_feats_flat",
    "feats_flat",
    "features_flat",
    "x_flat",
    "attr_flat",
    "attrs_flat",
)


def find_feature_key(d: Dict[str, Any], explicit_key: Optional[str] = None) -> str:
    if explicit_key:
        if explicit_key not in d:
            raise KeyError(f"--feature_key={explicit_key!r} not found. Available keys: {sorted(d.keys())}")
        return explicit_key

    for k in FEATURE_KEYS:
        if k in d:
            return k

    raise KeyError(
        "No feature tensor found. Expected one of "
        f"{FEATURE_KEYS}. Available keys: {sorted(d.keys())}"
    )


# ============================================================
# Continuous masking
# ============================================================
def mask_continuous_features(
    x: torch.Tensor,
    attn_keep: torch.Tensor,
    mask_prob: float = 0.30,
    mask_mode: str = "learned",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x:         (B, L, F) float
    attn_keep: (B, L) bool, True=real atom, False=PAD

    returns:
      x_masked: (B, L, F)
      labels:   (B, L, F), original features
      mask_sel: (B, L) bool, True=prediction target

    mask_mode:
      learned: do not change x here; model will replace hidden states with learned mask token.
      zero   : set masked raw input features to 0.0.
      noise  : set masked raw input features to Gaussian noise.
    """
    assert x.dim() == 3
    assert attn_keep.dim() == 2
    assert x.shape[:2] == attn_keep.shape
    assert x.dtype.is_floating_point

    labels = x.clone()

    prob = torch.rand(x.shape[:2], device=x.device, generator=generator)
    mask_sel = (prob < mask_prob) & attn_keep

    x_masked = x.clone()

    if mask_mode == "learned":
        pass
    elif mask_mode == "zero":
        x_masked[mask_sel] = 0.0
    elif mask_mode == "noise":
        noise = torch.randn(x_masked[mask_sel].shape, device=x.device, dtype=x.dtype, generator=generator)
        x_masked[mask_sel] = noise
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return x_masked, labels, mask_sel


# ============================================================
# Dataset: molecule-level sampling from many ragged batch files
# ============================================================
class ContinuousRaggedMolDataset(Dataset):
    """
    Loads pretrain_ragged_batchXXX.pt files and exposes each molecule as one item.
    Each item returns a 2D continuous feature tensor: (num_atoms_in_mol, feat_dim).
    """

    def __init__(
        self,
        root_dir: str,
        pattern: str = "pretrain_ragged_batch*.pt",
        limit_files: Optional[int] = None,
        file_list: Optional[List[str]] = None,
        feature_key: Optional[str] = None,
        normalize: bool = False,
        eps: float = 1e-6,
    ):
        self.root_dir = root_dir
        self.feature_key_arg = feature_key
        self.normalize = bool(normalize)
        self.eps = float(eps)

        if file_list is not None:
            self.files = []
            for p in file_list:
                if os.path.isabs(p):
                    self.files.append(p)
                else:
                    self.files.append(os.path.join(root_dir, p))
            self.files = sorted(self.files)
        else:
            self.files = sorted(glob.glob(os.path.join(root_dir, pattern)))

        if not self.files:
            raise FileNotFoundError(f"No files found in {root_dir} with pattern={pattern}")

        if limit_files is not None:
            self.files = self.files[: int(limit_files)]

        self.index: List[Tuple[int, int]] = []
        self.file_meta: List[Tuple[str, torch.Tensor, str]] = []
        feat_dim_set = set()

        mean_candidates = []
        std_candidates = []

        for fi, fp in enumerate(self.files):
            d = torch.load(fp, map_location="cpu")

            if "offsets" not in d:
                raise KeyError(f"offsets not found in {fp}. Available keys: {sorted(d.keys())}")

            offsets = d["offsets"].to(torch.int64)
            n_mols = offsets.numel() - 1

            fkey = find_feature_key(d, self.feature_key_arg)
            feats = d[fkey]
            if feats.dim() != 2:
                raise RuntimeError(f"Feature tensor {fkey} in {fp} must be 2D, got shape={tuple(feats.shape)}")
            if int(offsets[-1].item()) != int(feats.shape[0]):
                raise RuntimeError(
                    f"offsets[-1] != num feature rows in {fp}: "
                    f"offsets[-1]={int(offsets[-1].item())}, feats.shape[0]={feats.shape[0]}"
                )

            feat_dim_set.add(int(feats.shape[1]))
            self.file_meta.append((fp, offsets, fkey))

            for mi in range(n_mols):
                self.index.append((fi, mi))

            if "feature_mean" in d and "feature_std" in d:
                mean_candidates.append(d["feature_mean"].float())
                std_candidates.append(d["feature_std"].float())

        if len(feat_dim_set) != 1:
            raise RuntimeError(f"feat_dim differs across files: {sorted(feat_dim_set)}")
        self.feat_dim = feat_dim_set.pop()

        self.feature_mean = None
        self.feature_std = None
        if self.normalize:
            if mean_candidates and std_candidates:
                self.feature_mean = torch.stack(mean_candidates, dim=0).mean(dim=0)
                self.feature_std = torch.stack(std_candidates, dim=0).mean(dim=0).clamp_min(self.eps)
                if self.feature_mean.numel() != self.feat_dim or self.feature_std.numel() != self.feat_dim:
                    raise RuntimeError("feature_mean/std dim mismatch")
            else:
                print("[warn] --normalize was set, but feature_mean/feature_std were not found in files.")
                print("[warn] Dataset will run without normalization. Prefer precomputing train-only stats if needed.")

        self._cache_fi = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, fi: int) -> Dict[str, Any]:
        if self._cache_fi == fi and self._cache_data is not None:
            return self._cache_data
        fp, _, _ = self.file_meta[fi]
        d = torch.load(fp, map_location="cpu")
        self._cache_fi = fi
        self._cache_data = d
        return d

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi, mi = self.index[idx]
        d = self._load_file(fi)
        _, offsets, fkey = self.file_meta[fi]

        feats_flat = d[fkey].float()
        s = int(offsets[mi].item())
        e = int(offsets[mi + 1].item())
        x = feats_flat[s:e].clone()

        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std

        return x


def collate_pad_continuous(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    batch: list of (Li, F) FloatTensor sequences

    returns:
      x:         (B, L, F), padded with 0
      attn_keep: (B, L) bool, True=real atom, False=PAD
      lens:      (B,)
    """
    if not batch:
        raise RuntimeError("empty batch")

    feat_dim = int(batch[0].shape[1])
    lens = torch.tensor([x.shape[0] for x in batch], dtype=torch.int64)
    B = len(batch)
    L = int(lens.max().item())

    x_pad = torch.zeros((B, L, feat_dim), dtype=torch.float32)
    for i, x in enumerate(batch):
        if x.dim() != 2 or int(x.shape[1]) != feat_dim:
            raise RuntimeError(f"Bad sample shape: {tuple(x.shape)}")
        x_pad[i, : x.shape[0], :] = x

    attn_keep = torch.arange(L)[None, :] < lens[:, None]
    return x_pad, attn_keep, lens


# ============================================================
# Continuous Transformer MAE
# ============================================================
class ContinuousTransformerMAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        use_type_embedding: bool = False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_model = int(d_model)
        self.use_type_embedding = bool(use_type_embedding)

        self.in_proj = nn.Linear(feat_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        if self.use_type_embedding:
            # 0=normal/pad hidden, 1=masked hidden
            self.type_emb = nn.Embedding(2, d_model)
        else:
            self.type_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feat_dim),
        )

        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mask_sel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:                (B, L, F)
        key_padding_mask: (B, L) bool, True=PAD position
        mask_sel:         (B, L) bool, True=masked target position
        """
        h = self.in_proj(x)

        if mask_sel is not None:
            h = h.clone()
            h[mask_sel] = self.mask_token.to(dtype=h.dtype)

            if self.type_emb is not None:
                type_ids = mask_sel.to(torch.long)
                h = h + self.type_emb(type_ids)

        h = self.enc(h, src_key_padding_mask=key_padding_mask)
        pred = self.out_head(h)
        return pred


# ============================================================
# Utils: logging + seeding
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_log_writer(log_file: Optional[str]):
    if not log_file:
        def _noop(_rec):
            return
        return _noop, None

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    f = open(log_file, "a", buffering=1)

    def _write(rec: dict):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return _write, f


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_sel: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    if not mask_sel.any():
        # Rare but possible with tiny molecules and low mask_prob.
        return pred.sum() * 0.0

    p = pred[mask_sel]
    y = target[mask_sel]

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(p, y)
    if loss_type == "mse":
        return F.mse_loss(p, y)
    if loss_type == "l1":
        return F.l1_loss(p, y)

    raise ValueError(f"Unknown loss_type: {loss_type}")


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, mask_sel: torch.Tensor) -> Dict[str, float]:
    if not mask_sel.any():
        return {"mae": 0.0, "rmse": 0.0, "masked_atoms": 0.0, "masked_values": 0.0}

    p = pred[mask_sel]
    y = target[mask_sel]
    diff = p - y
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "masked_atoms": float(mask_sel.sum().item()),
        "masked_values": float(p.numel()),
    }


# ============================================================
# Train loop
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="pretrain_ragged_batch*.pt")
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--split_json", type=str, default=None, help="Path to split.json with train/valid file lists")
    ap.add_argument("--feature_key", type=str, default=None, help="Explicit feature key in .pt files")
    ap.add_argument("--normalize", action="store_true", help="Use feature_mean/std in files if available")

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.30)
    ap.add_argument("--mask_mode", type=str, default="learned", choices=["learned", "zero", "noise"])
    ap.add_argument("--loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "mse", "l1"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    ap.add_argument("--reset_optim", action="store_true", help="When resuming, re-init optimizer state")
    ap.add_argument("--reset_lr", action="store_true", help="When resuming, set lr from current schedule")

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_type_embedding", action="store_true")

    # io/log
    ap.add_argument("--save_dir", type=str, default="./continuous_mae_ckpt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--deterministic_masking", action="store_true")

    args = ap.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    write_log, log_fh = make_log_writer(args.log_file)

    # dataset / split
    train_files = None
    valid_files = None
    if args.split_json is not None:
        print(f"[info] loading split from {args.split_json}")
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_files = split.get("train", [])
        valid_files = split.get("valid", [])
        print(f"[split] train_files={len(train_files)} valid_files={len(valid_files)}")

    train_ds = ContinuousRaggedMolDataset(
        args.data_dir,
        pattern=args.pattern,
        limit_files=args.limit_files,
        file_list=train_files,
        feature_key=args.feature_key,
        normalize=args.normalize,
    )

    valid_ds = None
    if valid_files:
        valid_ds = ContinuousRaggedMolDataset(
            args.data_dir,
            pattern=args.pattern,
            file_list=valid_files,
            feature_key=args.feature_key,
            normalize=args.normalize,
        )
        if valid_ds.feat_dim != train_ds.feat_dim:
            raise RuntimeError(f"valid feat_dim mismatch: train={train_ds.feat_dim}, valid={valid_ds.feat_dim}")

    feat_dim = int(train_ds.feat_dim)

    print(f"Loaded {len(train_ds)} train molecules from {args.data_dir}")
    if valid_ds is not None:
        print(f"Loaded {len(valid_ds)} valid molecules")
    print(f"feat_dim={feat_dim}")
    print(f"mask_prob={args.mask_prob} mask_mode={args.mask_mode} loss_type={args.loss_type}")
    if args.log_file:
        print(f"logging to: {args.log_file}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad_continuous,
        drop_last=True,
    )

    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_pad_continuous,
            drop_last=False,
        )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise RuntimeError("train_loader is empty. Check batch_size/drop_last/dataset size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousTransformerMAE(
        feat_dim=feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        use_type_embedding=args.use_type_embedding,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    resume_step0 = 0
    resume_lr0 = args.lr
    resume_last_epoch = 0

    if args.resume is not None:
        print(f"[resume] loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")

        if int(ckpt.get("feat_dim", -1)) != int(feat_dim):
            raise RuntimeError(f"[resume] feat_dim mismatch: ckpt={ckpt.get('feat_dim')} current={feat_dim}")

        model.load_state_dict(ckpt["model"], strict=True)

        if (not args.reset_optim) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
                print("[resume] optimizer state loaded")
            except Exception as e:
                print(f"[resume] failed to load optimizer state: {e} (continuing with fresh optim)")

        if "rng" in ckpt:
            try:
                random.setstate(ckpt["rng"]["python"])
                torch.set_rng_state(ckpt["rng"]["torch"])
                if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
                print("[resume] RNG state restored")
            except Exception as e:
                print(f"[resume] failed to restore RNG: {e}")

        global_step = int(ckpt.get("global_step", 0))
        resume_step0 = global_step
        resume_last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resume_last_epoch + 1
        resume_lr0 = float(optim.param_groups[0].get("lr", args.lr))

        print(f"[resume] resumed at epoch={resume_last_epoch}, global_step={global_step}, lr0={resume_lr0:.3e}")

    # ------------------------------------------------------------
    # LR schedule
    # fresh : warmup + cosine to zero over all epochs
    # resume: cosine decay from current lr0 to zero over remaining epochs
    # ------------------------------------------------------------
    if args.resume is None:
        total_steps = args.epochs * steps_per_epoch
        warmup = max(10, int(0.05 * total_steps))

        def lr_now(step: int) -> float:
            if step < warmup:
                fac = (step + 1) / warmup
            else:
                t = (step - warmup) / max(1, (total_steps - warmup))
                fac = 0.5 * (1.0 + math.cos(math.pi * t))
            return args.lr * fac

        total_steps_disp = total_steps
    else:
        remaining_epochs = args.epochs - resume_last_epoch
        if remaining_epochs <= 0:
            raise RuntimeError(f"--epochs must be > ckpt epoch. ckpt={resume_last_epoch} args.epochs={args.epochs}")

        remaining_steps = remaining_epochs * steps_per_epoch
        total_steps_disp = int(resume_step0 + remaining_steps)

        def lr_now(step: int) -> float:
            prog = (step - resume_step0) / max(1, remaining_steps)
            prog = min(max(prog, 0.0), 1.0)
            fac = 0.5 * (1.0 + math.cos(math.pi * prog))
            return resume_lr0 * fac

        if args.reset_lr:
            lr0 = lr_now(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr0
            print(f"[resume] reset_lr: lr set to {lr0:.3e} at step={global_step}")

    model.train()

    mask_gen = None
    if args.deterministic_masking:
        mask_gen = torch.Generator(device=device)

    for ep in range(start_epoch, args.epochs + 1):
        running_loss_sum = 0.0
        running_atoms = 0
        running_mae_sum = 0.0
        running_rmse_sum = 0.0
        running_batches = 0

        for it, (x, attn_keep, lens) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            attn_keep = attn_keep.to(device, non_blocking=True)
            key_padding_mask = ~attn_keep

            if mask_gen is not None:
                step_seed = int(args.seed + ep * 1_000_000 + global_step)
                mask_gen.manual_seed(step_seed)

            x_masked, labels, mask_sel = mask_continuous_features(
                x,
                attn_keep=attn_keep,
                mask_prob=args.mask_prob,
                mask_mode=args.mask_mode,
                generator=mask_gen,
            )

            pred = model(x_masked, key_padding_mask=key_padding_mask, mask_sel=mask_sel)
            loss = reconstruction_loss(pred, labels, mask_sel, loss_type=args.loss_type)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            global_step += 1
            lr_now_val = lr_now(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now_val

            with torch.no_grad():
                metrics = reconstruction_metrics(pred, labels, mask_sel)
                masked_atoms = int(metrics["masked_atoms"])

            running_loss_sum += float(loss.item()) * max(1, masked_atoms)
            running_atoms += masked_atoms
            running_mae_sum += float(metrics["mae"])
            running_rmse_sum += float(metrics["rmse"])
            running_batches += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss_sum / max(1, running_atoms)
                avg_mae = running_mae_sum / max(1, running_batches)
                avg_rmse = running_rmse_sum / max(1, running_batches)

                msg = (
                    f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps_disp} "
                    f"mae_loss={avg_loss:.5f} mae={avg_mae:.5f} rmse={avg_rmse:.5f} "
                    f"lr={lr_now_val:.2e} masked_atoms={running_atoms}"
                )
                print(msg)

                write_log({
                    "time": time.time(),
                    "epoch": ep,
                    "step": global_step,
                    "total_steps": total_steps_disp,
                    "loss": avg_loss,
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                    "lr": lr_now_val,
                    "masked_atoms": running_atoms,
                    "batch_size": args.batch_size,
                    "mask_prob": args.mask_prob,
                    "mask_mode": args.mask_mode,
                    "loss_type": args.loss_type,
                    "deterministic_masking": bool(args.deterministic_masking),
                })

                running_loss_sum = 0.0
                running_atoms = 0
                running_mae_sum = 0.0
                running_rmse_sum = 0.0
                running_batches = 0

        # -------------------------
        # VALID
        # -------------------------
        if valid_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_atoms = 0
            v_mae_sum = 0.0
            v_rmse_sum = 0.0
            v_batches = 0

            with torch.no_grad():
                for v_it, (v_x, v_attn_keep, v_lens) in enumerate(valid_loader, start=1):
                    v_x = v_x.to(device, non_blocking=True)
                    v_attn_keep = v_attn_keep.to(device, non_blocking=True)
                    v_key_padding_mask = ~v_attn_keep

                    if mask_gen is not None:
                        step_seed = int(args.seed + 9_000_000 + v_it)
                        mask_gen.manual_seed(step_seed)

                    v_x_masked, v_labels, v_mask_sel = mask_continuous_features(
                        v_x,
                        attn_keep=v_attn_keep,
                        mask_prob=args.mask_prob,
                        mask_mode=args.mask_mode,
                        generator=mask_gen,
                    )

                    v_pred = model(v_x_masked, key_padding_mask=v_key_padding_mask, mask_sel=v_mask_sel)
                    v_loss = reconstruction_loss(v_pred, v_labels, v_mask_sel, loss_type=args.loss_type)
                    v_metrics = reconstruction_metrics(v_pred, v_labels, v_mask_sel)
                    v_masked_atoms = int(v_metrics["masked_atoms"])

                    v_loss_sum += float(v_loss.item()) * max(1, v_masked_atoms)
                    v_atoms += v_masked_atoms
                    v_mae_sum += float(v_metrics["mae"])
                    v_rmse_sum += float(v_metrics["rmse"])
                    v_batches += 1

            v_avg = v_loss_sum / max(1, v_atoms)
            v_mae = v_mae_sum / max(1, v_batches)
            v_rmse = v_rmse_sum / max(1, v_batches)

            print(
                f"[ep {ep}/{args.epochs}] VALID "
                f"mae_loss={v_avg:.5f} mae={v_mae:.5f} rmse={v_rmse:.5f} masked_atoms={v_atoms}"
            )

            write_log({
                "time": time.time(),
                "event": "valid",
                "epoch": ep,
                "step": global_step,
                "loss": v_avg,
                "mae": v_mae,
                "rmse": v_rmse,
                "masked_atoms": v_atoms,
            })

            model.train()

        # save checkpoint each epoch
        ckpt_path = os.path.join(args.save_dir, f"continuous_mae_ep{ep:02d}.pt")
        torch.save({
            "epoch": ep,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "feat_dim": feat_dim,
            "feature_key": args.feature_key,
            "config": vars(args),
        }, ckpt_path)
        print("saved", ckpt_path)

        write_log({
            "time": time.time(),
            "event": "epoch_end",
            "epoch": ep,
            "step": global_step,
            "ckpt_path": ckpt_path,
        })

    if log_fh is not None:
        log_fh.close()

    print("DONE")


if __name__ == "__main__":
    main()
