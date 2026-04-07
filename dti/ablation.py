#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model:
    z_delta = g(protein, ligand)
    logit   = z_delta
    prob    = sigmoid(logit)

Labels:
    y_bin = 1 if y >= y_thr else 0

Training objective:
    loss = BCEWithLogits(logit, y_bin) + BCEWithLogits(y_bin)

Notes:
- This is a cleaned classification rewrite of the previous regression-style script.
- Regression-only targets such as y_avg_per_protein / delta are no longer used for training.
- AUROC / AP / F1 / EF are computed from probabilities.
"""

from __future__ import annotations

from glob import glob
import os
import json
import math
import random
import csv
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

Y_THR = 7.0
CLS_THR = 0.5


# =========================================================
# Repro
# =========================================================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# CSV utils
# =========================================================
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_csv_random_rows(path: str, n_rows: int, seed: int) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    if n_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), n_rows)
    return [rows[i] for i in idx]


def list_train_shards(shard_dir: str, pattern: str = "train_part_*.csv") -> List[str]:
    paths = sorted(glob(os.path.join(shard_dir, pattern)))
    if not paths:
        raise ValueError(f"No shard files found: dir={shard_dir} pattern={pattern}")
    return paths


def pick_epoch_shards_random(
    shard_paths: List[str],
    train_size: int,
    shard_size: int,
    epoch: int,
    seed: int,
    num_shards_per_epoch: Optional[int] = None,
) -> List[str]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    total = len(shard_paths)
    if total == 0:
        raise ValueError("No shard files found")

    if num_shards_per_epoch is not None:
        num_shards = int(num_shards_per_epoch)
        if num_shards <= 0:
            raise ValueError("num_shards_per_epoch must be positive")
    else:
        if train_size is None or train_size <= 0:
            raise ValueError("train_size must be positive when num_shards_per_epoch is not set")
        num_shards = int(math.ceil(float(train_size) / float(shard_size)))

    rng = random.Random(int(seed) + int(epoch))
    if num_shards <= total:
        return rng.sample(shard_paths, num_shards)
    return [rng.choice(shard_paths) for _ in range(num_shards)]


# =========================================================
# Token utils
# =========================================================
def parse_lig_tokens(s: str) -> List[int]:
    s = str(s).strip().replace(",", " ")
    if not s:
        return []
    return [int(x) for x in s.split()]


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not seqs:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(x.numel()) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        if x.numel() > 0:
            out[i, :x.numel()] = x
    return out


# =========================================================
# Batch / dataset
# =========================================================
from typing import Optional

@dataclass
class Batch:
    p_input_ids: torch.Tensor
    p_attn_mask: torch.Tensor
    l_ids: torch.Tensor
    y_bin: torch.Tensor
    y_reg: Optional[torch.Tensor]

class DTIDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        rows: Optional[List[Dict[str, str]]] = None,
        y_thr: float = Y_THR,
        drop_missing_y: bool = True,
        lig_cls_id: int = 0,   # 必要なら外から渡す
    ):
        if rows is not None:
            raw_rows = rows
        elif csv_path is not None:
            raw_rows = read_csv_rows(csv_path)
        else:
            raise ValueError("Either csv_path or rows must be provided")

        self.lig_cls_id = lig_cls_id
        self.rows: List[Dict[str, object]] = []

        for r in raw_rows:
            seq = (r.get("seq") or "").strip()
            lig_tok = (r.get("lig_tok") or "").strip()
            y_raw = r.get("y", "")

            if not seq or not lig_tok:
                continue

            if y_raw in ("", None):
                if drop_missing_y:
                    continue
                y = float("nan")
            else:
                y = float(y_raw)

            rr = dict(r)
            rr["seq"] = seq
            rr["lig_tok"] = lig_tok
            rr["y"] = y
            rr["y_bin"] = 1.0 if y >= float(y_thr) else 0.0
            self.rows.append(rr)

        if not self.rows:
            raise ValueError("No usable rows in dataset")

        ys = [float(r["y"]) for r in self.rows if not np.isnan(float(r["y"]))]
        if len(ys) == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(np.mean(ys))
            std = float(np.std(ys)) + 1e-6

        for r in self.rows:
            y = float(r["y"])
            if np.isnan(y):
                r["y_reg"] = None
            else:
                r["y_reg"] = (y - mean) / std

    def __len__(self) -> int:
        return len(self.rows)

    def _parse_lig_tok(self, lig_tok: str) -> List[int]:
        return [int(x) for x in lig_tok.split() if x.strip()]

    def __getitem__(self, idx):
        row = self.rows[idx]

        lig_ids = [self.lig_cls_id] + self._parse_lig_tok(row["lig_tok"])

        item = {
            "protein_seq": row["seq"],
            "lig_ids": lig_ids,
            "y_bin": float(row["y_bin"]),
            "y_reg": None if row["y_reg"] is None else float(row["y_reg"]),
        }
        return item


def build_train_dataset_from_shards(shard_paths: List[str], y_thr: float) -> ConcatDataset:
    ds_list = [DTIDataset(csv_path=p, y_thr=float(y_thr), drop_missing_y=True, lig_cls_id=lig_enc.cls_id ) for p in shard_paths]
    if not ds_list:
        raise ValueError("No train shards were loaded")
    return ConcatDataset(ds_list)


def collate_fn(samples, esm_tokenizer, lig_pad, lig_cls):
    p_seqs = [s["protein_seq"] for s in samples]
    l_ids_list = [s["lig_ids"] for s in samples]

    tok = esm_tokenizer(
        p_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    p_input_ids = tok["input_ids"]
    p_attn_mask = tok["attention_mask"]

    max_l = max(len(x) for x in l_ids_list)
    l_ids = torch.full((len(samples), max_l), lig_pad, dtype=torch.long)
    for i, ids in enumerate(l_ids_list):
        l_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    y_bin = torch.tensor([float(s["y_bin"]) for s in samples], dtype=torch.float32)

    has_y_reg = all(("y_reg" in s) and (s["y_reg"] is not None) for s in samples)
    if has_y_reg:
        y_reg = torch.tensor([float(s["y_reg"]) for s in samples], dtype=torch.float32)
    else:
        y_reg = None

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y_bin=y_bin,
        y_reg=y_reg,
    )

# =========================================================
# Checkpoint / vocab utils
# =========================================================
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


# =========================================================
# Encoders
# =========================================================
class PretrainedLigandEncoder(nn.Module):
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
            raise RuntimeError("Ligand checkpoint must contain keys: model, config")

        self.state = ckpt["model"]
        self.conf = ckpt["config"]

        if vq_ckpt_path is not None:
            vm = load_vocab_meta_from_vq_ckpt(vq_ckpt_path)
            self.base_vocab = int(vm["base_vocab"])
            self.vocab_size = int(vm["vocab_size"])
            self.pad_id = int(vm["pad_id"])
            self.mask_id = int(vm["mask_id"])
            self.vocab_source = f"vq_ckpt:{vq_ckpt_path}"
        else:
            if "base_vocab" not in ckpt or "vocab_size" not in ckpt:
                raise RuntimeError("Ligand MLM ckpt must contain base_vocab and vocab_size")
            self.base_vocab = int(ckpt["base_vocab"])
            self.vocab_size = int(ckpt["vocab_size"])
            self.pad_id = self.base_vocab + 0
            self.mask_id = self.base_vocab + 1
            self.vocab_source = f"mlm_ckpt:{ckpt_path}"

        self.cls_id = int(self.vocab_size)
        self.vocab_size = int(self.vocab_size) + 1

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

        load_state_dict_shape_safe(self, self.state, verbose=verbose_load)

        mlm_tok_key = "tok.weight"
        if mlm_tok_key in self.state:
            w = self.state[mlm_tok_key]
            if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[1] == d_model:
                overlap = min(int(w.shape[0]), int(self.tok.weight.shape[0]))
                if overlap > 0:
                    with torch.no_grad():
                        self.tok.weight[:overlap].copy_(w[:overlap])

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
                    raise RuntimeError(
                        f"Ligand token id out of range: id={v0}, num_embeddings={self.tok.num_embeddings}"
                    )

        x = self.tok(l_ids)
        pad_mask = (l_ids == self.pad_id)
        return self.enc(x, src_key_padding_mask=pad_mask)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_one_qk_map(
    model,
    loader,
    device,
    esm_tokenizer,
    sample_idx_in_batch: int = 0,
    show_token_labels: bool = False,
    save_dir: str | None = None,
    prefix: str = "sample",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    batch = next(iter(loader))

    p_ids = batch.p_input_ids.to(device, non_blocking=True)
    p_msk = batch.p_attn_mask.to(device, non_blocking=True)
    l_ids = batch.l_ids.to(device, non_blocking=True)

    with torch.inference_mode():
        logit, yhat_reg, aux = model(p_ids, p_msk, l_ids, return_maps=True)

    p_pad = aux["p_pad"][sample_idx_in_batch].detach().cpu().numpy().astype(bool)  # (Lp,)
    l_pad = aux["l_pad"][sample_idx_in_batch].detach().cpu().numpy().astype(bool)  # (Ll,)

    prob = float(torch.sigmoid(logit[sample_idx_in_batch]).detach().cpu())
    y_bin = float(batch.y_bin[sample_idx_in_batch].detach().cpu())

    y_reg_pred = None
    if yhat_reg is not None:
        y_reg_pred = float(yhat_reg[sample_idx_in_batch].detach().cpu())

    y_reg_true = None
    if getattr(batch, "y_reg", None) is not None:
        y_reg_true = float(batch.y_reg[sample_idx_in_batch].detach().cpu())

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    reg_txt = ""
    if (y_reg_true is not None) and (y_reg_pred is not None):
        reg_txt = f" y_reg={y_reg_true:.3f} pred={y_reg_pred:.3f}"

    # ---------------------------
    # token labels
    # ---------------------------
    p_tok_labels = None
    l_tok_labels = None
    if show_token_labels:
        p_ids_1 = batch.p_input_ids[sample_idx_in_batch].detach().cpu().tolist()
        p_tok_labels_all = esm_tokenizer.convert_ids_to_tokens(p_ids_1)[1:]  # drop CLS

        eos_tok = getattr(esm_tokenizer, "eos_token", None)
        p_tok_labels = []
        for t, is_pad in zip(p_tok_labels_all, p_pad):
            if is_pad:
                continue
            if eos_tok is not None and t == eos_tok:
                continue
            p_tok_labels.append(t)

        l_ids_1 = batch.l_ids[sample_idx_in_batch].detach().cpu().tolist()
        l_tok_labels_all = l_ids_1[1:]  # drop CLS
        l_tok_labels = [str(t) for t, is_pad in zip(l_tok_labels_all, l_pad) if not is_pad]

    # ---------------------------
    # LP: ligand <- protein
    # shape: (Ll, Lp)
    # rows = ligand, cols = protein
    # ---------------------------
    S_lp = aux["qk_scores_lp"][sample_idx_in_batch].detach().float().cpu().numpy()
    A_lp = aux["attn_map_lp"][sample_idx_in_batch].detach().float().cpu().numpy()

    expected_lp = (len(l_pad), len(p_pad))
    if S_lp.shape != expected_lp or A_lp.shape != expected_lp:
        raise ValueError(
            f"Unexpected LP shape: S_lp={S_lp.shape}, A_lp={A_lp.shape}, expected={expected_lp}"
        )

    S_lp_vis = S_lp[~l_pad][:, ~p_pad]
    A_lp_vis = A_lp[~l_pad][:, ~p_pad]

    # ---------------------------
    # PL: protein <- ligand
    # shape: (Lp, Ll)
    # rows = protein, cols = ligand
    # ---------------------------
    S_pl = aux["qk_scores_pl"][sample_idx_in_batch].detach().float().cpu().numpy()
    A_pl = aux["attn_map_pl"][sample_idx_in_batch].detach().float().cpu().numpy()

    expected_pl = (len(p_pad), len(l_pad))
    if S_pl.shape != expected_pl or A_pl.shape != expected_pl:
        raise ValueError(
            f"Unexpected PL shape: S_pl={S_pl.shape}, A_pl={A_pl.shape}, expected={expected_pl}"
        )

    S_pl_vis = S_pl[~p_pad][:, ~l_pad]
    A_pl_vis = A_pl[~p_pad][:, ~l_pad]

    # =========================================================
    # 1) LP QK
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.imshow(S_lp_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"Ligand <- Protein QK | prob={prob:.4f} y_bin={y_bin:.0f}{reg_txt}")
    plt.xlabel("Protein tokens")
    plt.ylabel("Ligand tokens")
    if show_token_labels:
        if p_tok_labels is not None and len(p_tok_labels) <= 80:
            plt.xticks(range(len(p_tok_labels)), p_tok_labels, rotation=90, fontsize=6)
        if l_tok_labels is not None and len(l_tok_labels) <= 80:
            plt.yticks(range(len(l_tok_labels)), l_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_qk_scores_lp.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================
    # 2) LP attention
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.imshow(A_lp_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"Ligand <- Protein attention | prob={prob:.4f} y_bin={y_bin:.0f}")
    plt.xlabel("Protein tokens")
    plt.ylabel("Ligand tokens")
    if show_token_labels:
        if p_tok_labels is not None and len(p_tok_labels) <= 80:
            plt.xticks(range(len(p_tok_labels)), p_tok_labels, rotation=90, fontsize=6)
        if l_tok_labels is not None and len(l_tok_labels) <= 80:
            plt.yticks(range(len(l_tok_labels)), l_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_attn_map_lp.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================
    # 3) PL QK
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.imshow(S_pl_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"Protein <- Ligand QK | prob={prob:.4f} y_bin={y_bin:.0f}{reg_txt}")
    plt.xlabel("Ligand tokens")
    plt.ylabel("Protein tokens")
    if show_token_labels:
        if l_tok_labels is not None and len(l_tok_labels) <= 80:
            plt.xticks(range(len(l_tok_labels)), l_tok_labels, rotation=90, fontsize=6)
        if p_tok_labels is not None and len(p_tok_labels) <= 80:
            plt.yticks(range(len(p_tok_labels)), p_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_qk_scores_pl.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================
    # 4) PL attention
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.imshow(A_pl_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"Protein <- Ligand attention | prob={prob:.4f} y_bin={y_bin:.0f}")
    plt.xlabel("Ligand tokens")
    plt.ylabel("Protein tokens")
    if show_token_labels:
        if l_tok_labels is not None and len(l_tok_labels) <= 80:
            plt.xticks(range(len(l_tok_labels)), l_tok_labels, rotation=90, fontsize=6)
        if p_tok_labels is not None and len(p_tok_labels) <= 80:
            plt.yticks(range(len(p_tok_labels)), p_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_attn_map_pl.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # =========================================================
    # 5) summary lines
    # =========================================================
    # LP: ligandごとの protein attention 平均
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(A_lp_vis.shape[0]), A_lp_vis.mean(axis=1))
    plt.title(f"LP: ligand-token mean attention to protein | prob={prob:.4f}")
    plt.xlabel("Ligand token index")
    plt.ylabel("mean attention")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_lp_lig_token_attention_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # LP: proteinごとの ligand からの平均 attention
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(A_lp_vis.shape[1]), A_lp_vis.mean(axis=0))
    plt.title(f"LP: protein-token mean attention from ligand | prob={prob:.4f}")
    plt.xlabel("Protein token index")
    plt.ylabel("mean attention")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_lp_prot_token_attention_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # PL: proteinごとの ligand attention 平均
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(A_pl_vis.shape[0]), A_pl_vis.mean(axis=1))
    plt.title(f"PL: protein-token mean attention to ligand | prob={prob:.4f}")
    plt.xlabel("Protein token index")
    plt.ylabel("mean attention")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_pl_prot_token_attention_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # PL: ligandごとの protein からの平均 attention
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(A_pl_vis.shape[1]), A_pl_vis.mean(axis=0))
    plt.title(f"PL: ligand-token mean attention from protein | prob={prob:.4f}")
    plt.xlabel("Ligand token index")
    plt.ylabel("mean attention")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_pl_lig_token_attention_mean.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "qk_scores_lp": S_lp_vis,
        "attn_map_lp": A_lp_vis,
        "qk_scores_pl": S_pl_vis,
        "attn_map_pl": A_pl_vis,
        "prob": prob,
        "y_bin": y_bin,
        "y_reg_pred": y_reg_pred,
        "y_reg_true": y_reg_true,
        "protein_tokens": p_tok_labels,
        "ligand_tokens": l_tok_labels,
    }

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


# =========================================================
# Main model
# =========================================================
class QKOnlyDTIClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        dropout: float,
        attn_temp: float = 2.0,
        n_heads: int = 4,
        protein_token_dropout: float = 0.0,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads

        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

        # true multi-head projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # dual V
        self.v_prot_proj = nn.Linear(d_model, d_model, bias=False)  # protein flow
        self.v_lig_proj = nn.Linear(d_model, d_model, bias=False)   # ligand self flow

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)
        self.attn_temp = float(attn_temp)
        self.protein_token_dropout = float(protein_token_dropout)
        self.lig_pad_id = int(self.lig.pad_id)

        # fuse [ctx_from_prot, lig_self]
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        # pooled token summary + ligand CLS
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, D)
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return x

    def _safe_masked_softmax_4d(self, scores: torch.Tensor, key_pad_mask: torch.Tensor) -> torch.Tensor:
        """
        scores:       (B, H, Ll, Lp)
        key_pad_mask: (B, Lp)  True=PAD
        """
        scores = scores.masked_fill(key_pad_mask[:, None, None, :], float("-inf"))

        all_masked = key_pad_mask.all(dim=1)  # (B,)
        if all_masked.any():
            scores = scores.clone()
            scores[all_masked] = 0.0

        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_pad_mask[:, None, None, :], 0.0)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / denom

        return attn

    def forward(
            self,
            p_input_ids: torch.Tensor,
            p_attn_mask: torch.Tensor,
            l_ids: torch.Tensor,
            return_maps: bool = False,
    ):
        aux = {}

        p_h = self.prot(p_input_ids, p_attn_mask)  # (B, Lp_all, Dp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)  # (B, Lp_all, D)

        l_h = self.lig(l_ids)  # (B, Ll_all, D)

        prot_pad_mask = (p_attn_mask == 0)
        lig_pad_mask = (l_ids == self.lig_pad_id)

        # ---- protein special token handling ----
        # ESM: keep CLS separately, remove CLS/EOS from token interaction branch
        p_cls = p_h[:, 0, :]  # (B, D)

        # token ids
        p_tok_ids_all = p_input_ids[:, 1:]  # after CLS
        p_tok_all = p_h[:, 1:, :]
        p_pad_all = prot_pad_mask[:, 1:]

        eos_id = getattr(self.prot.esm.config, "eos_token_id", None)
        if eos_id is None:
            eos_id = 2  # ESM2 usually uses 2 for eos

        p_special = p_pad_all | (p_tok_ids_all == eos_id)

        # keep tensors same length; special positions are just masked out
        p_tok = p_tok_all  # (B, Lp, D)
        p_pad = p_special  # (B, Lp)

        # ligand side: drop CLS only
        l_tok = l_h[:, 1:, :]  # (B, Ll, D)
        l_cls = l_h[:, 0, :]
        l_pad = lig_pad_mask[:, 1:]

        if p_tok.size(1) == 0 or l_tok.size(1) == 0:
            z = torch.cat([l_cls, p_cls], dim=-1)
            logit = self.cls_head(z).squeeze(-1)
            yhat_reg = self.reg_head(z).squeeze(-1)
            if return_maps:
                B = p_input_ids.size(0)
                aux["qk_scores"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["attn_map"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["p_pad"] = p_pad
                aux["l_pad"] = l_pad
                aux["l_imp"] = torch.zeros(B, 0, device=p_input_ids.device)
                aux["attn_entropy"] = torch.tensor(0.0, device=p_input_ids.device)
            return logit, yhat_reg, aux

        if self.training and self.protein_token_dropout > 0.0:
            drop_mask = (torch.rand_like(p_pad.float()) < self.protein_token_dropout) & (~p_pad)
            p_pad = p_pad | drop_mask

            all_dropped = p_pad.all(dim=1)
            if all_dropped.any():
                p_pad = p_pad.clone()
                keep_idx = (~p_special).float().argmax(dim=1)
                for b in torch.where(all_dropped)[0]:
                    p_pad[b, keep_idx[b]] = False

        # ---- projections ----
        # normalize before Q/K so protein-direction contrast becomes stronger
        q_in = torch.nn.functional.normalize(l_tok, dim=-1)
        k_in = torch.nn.functional.normalize(p_tok, dim=-1)

        q = self._split_heads(self.q_proj(q_in))  # (B,H,Ll,Dh)
        k = self._split_heads(self.k_proj(k_in))  # (B,H,Lp,Dh)
        v_prot = self._split_heads(self.v_prot_proj(p_tok))  # (B,H,Lp,Dh)
        v_lig = self._split_heads(self.v_lig_proj(l_tok))  # (B,H,Ll,Dh)

        temp = self.attn_temp + torch.arange(self.n_heads, device=q.device).view(1, -1, 1, 1) * 0.2
        qk_scores = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(self.head_dim) * temp)

        attn_map = self._safe_masked_softmax_4d(qk_scores, p_pad)  # (B,H,Ll,Lp)

        ctx_prot = torch.matmul(attn_map, v_prot)  # (B,H,Ll,Dh)
        lig_self = v_lig  # (B,H,Ll,Dh)

        ctx_prot = self._merge_heads(ctx_prot)  # (B,Ll,D)
        lig_self = self._merge_heads(lig_self)  # (B,Ll,D)

        joint_in = torch.cat([ctx_prot, lig_self], dim=-1)  # (B,Ll,2D)
        joint = self.fuse(joint_in)  # (B,Ll,D)
        gate = torch.sigmoid(self.out_proj(joint))
        joint = l_tok + gate * joint  # (B,Ll,D)

        # ---- ligand token competition: use raw QK, not attn_map ----
        l_importance = qk_scores.max(dim=-1).values  # (B,H,Ll)
        l_importance = l_importance.masked_fill(l_pad.unsqueeze(1), float("-inf"))
        l_weights = torch.softmax(l_importance, dim=-1)  # (B,H,Ll)

        joint_h = joint.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B,H,Ll,D)
        joint_vec_h = (joint_h * l_weights.unsqueeze(-1)).sum(dim=2)  # (B,H,D)
        joint_vec = joint_vec_h.mean(dim=1)  # (B,D)

        bad = ~torch.isfinite(joint_vec).all(dim=1)
        if bad.any():
            joint_vec = joint_vec.clone()
            joint_vec[bad] = 0.0

        z = torch.cat([l_cls, joint_vec], dim=-1)  # (B,2D)

        logit = self.cls_head(z).squeeze(-1)
        yhat_reg = self.reg_head(z).squeeze(-1)

        attn_safe = attn_map.clamp(min=1e-8)
        ent = -(attn_safe * torch.log(attn_safe)).sum(dim=-1)  # (B,H,Ll)
        ent = ent.masked_fill(l_pad[:, None, :], 0.0)
        denom = (~l_pad).sum(dim=1).clamp(min=1)
        aux["attn_entropy"] = (ent.sum(dim=(1, 2)) / (denom * self.n_heads)).mean()

        if return_maps:
            aux["qk_scores"] = qk_scores.mean(dim=1)  # (B,Ll,Lp)
            aux["attn_map"] = attn_map.mean(dim=1)  # (B,Ll,Lp)
            aux["qk_scores_heads"] = qk_scores  # (B,H,Ll,Lp)
            aux["attn_map_heads"] = attn_map  # (B,H,Ll,Lp)
            aux["p_pad"] = p_pad
            aux["l_pad"] = l_pad

            l_imp = qk_scores.max(dim=-1).values.mean(dim=1)  # (B,Ll)
            l_imp = l_imp.masked_fill(l_pad, 0.0)
            aux["l_imp"] = l_imp

        return logit, yhat_reg, aux

# =========================================================
# Metrics / predict
# =========================================================
def compute_pos_weight_from_dataset(ds: Dataset) -> float:
    labels = []
    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            labels.extend(float(sub[i]["y_bin"]) for i in range(len(sub)))
    else:
        labels.extend(float(ds[i]["y_bin"]) for i in range(len(ds)))
    arr = np.asarray(labels, dtype=np.float32)
    pos = float(arr.sum())
    neg = float(len(arr) - pos)
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    prob_list, ybin_list = [], []
    yreg_pred_list, yreg_true_list = [], []
    use_amp = (device.type == "cuda")

    with torch.inference_mode():
        for batch in loader:
            p_ids = batch.p_input_ids.to(device, non_blocking=True)
            p_msk = batch.p_attn_mask.to(device, non_blocking=True)
            l_ids = batch.l_ids.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit, yhat_reg, _ = model(p_ids, p_msk, l_ids)
            else:
                logit, yhat_reg, _ = model(p_ids, p_msk, l_ids)

            prob = torch.sigmoid(logit)

            prob_list.append(prob.detach().float().cpu().numpy())
            ybin_list.append(batch.y_bin.detach().cpu().numpy())
            if batch.y_reg is not None:
                yreg_true_list.append(batch.y_reg.detach().cpu().numpy())
                yreg_pred_list.append(yhat_reg.detach().float().cpu().numpy())

    y_prob = np.concatenate(prob_list, axis=0) if prob_list else np.array([], dtype=np.float64)
    y_bin = np.concatenate(ybin_list, axis=0) if ybin_list else np.array([], dtype=np.float64)
    y_reg_pred = np.concatenate(yreg_pred_list, axis=0) if yreg_pred_list else np.array([], dtype=np.float64)
    y_reg_true = np.concatenate(yreg_true_list, axis=0) if yreg_true_list else np.array([], dtype=np.float64)
    return y_prob, y_bin, y_reg_pred, y_reg_true


def eval_reg_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    if y_pred.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    if len(y_pred) < 2:
        return {"mae": mae, "rmse": rmse, "pearson": 0.0, "spearman": 0.0}

    pearson = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0 and np.std(y_true) > 0 else 0.0

    # simple spearman via rank
    yp_rank = np.argsort(np.argsort(y_pred))
    yt_rank = np.argsort(np.argsort(y_true))
    spearman = float(np.corrcoef(yp_rank, yt_rank)[0, 1]) if np.std(yp_rank) > 0 and np.std(yt_rank) > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }



def eval_metrics(y_pred: np.ndarray, y_bin: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    def enrichment_factor(score: np.ndarray, y01: np.ndarray, frac: float) -> float:
        n = int(len(y01))
        if n == 0:
            return 0.0
        n_pos = int(y01.sum())
        if n_pos == 0:
            return 0.0
        k = max(1, int(math.ceil(n * float(frac))))
        order = np.argsort(-score)
        top_idx = order[:k]
        hits_topk = int(y01[top_idx].sum())
        hit_rate_topk = hits_topk / float(k)
        return float(hit_rate_topk)

    if y_pred.size == 0:
        return {
            "auroc": 0.0,
            "ap": 0.0,
            "f1": 0.0,
            "thr": CLS_THR,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "ef1": 0.0,
            "ef5": 0.0,
            "ef10": 0.0,
        }

    score = y_pred
    y01 = (y_bin > 0.5).astype(np.int32)
    ef1 = enrichment_factor(score, y01, 0.01)
    ef5 = enrichment_factor(score, y01, 0.05)
    ef10 = enrichment_factor(score, y01, 0.10)

    if len(np.unique(y01)) <= 1:
        return {
            "auroc": 0.0,
            "ap": 0.0,
            "f1": 0.0,
            "thr": float(CLS_THR),
            "pred_mean": float(score.mean()),
            "pred_std": float(score.std()),
            "pred_min": float(score.min()),
            "pred_max": float(score.max()),
            "pos_rate": float(y01.mean()) if y01.size else 0.0,
            "pred_pos_rate@thr": float((score >= CLS_THR).mean()) if score.size else 0.0,
            "ef1": float(ef1),
            "ef5": float(ef5),
            "ef10": float(ef10),
        }

    auroc = roc_auc_score(y01, score)
    ap = average_precision_score(y01, score)
    pred01 = (score >= CLS_THR).astype(np.int32)
    f1 = f1_score(y01, pred01)

    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "f1": float(f1),
        "thr": float(CLS_THR),
        "pred_mean": float(score.mean()),
        "pred_std": float(score.std()),
        "pred_min": float(score.min()),
        "pred_max": float(score.max()),
        "pos_rate": float(y01.mean()),
        "pred_pos_rate@thr": float(pred01.mean()),
        "ef1": float(ef1),
        "ef5": float(ef5),
        "ef10": float(ef10),
    }


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
    grad_clip: float = 1.0,
    attn_entropy_lambda: float = 0.0,
    reg_lambda: float = 0.1,
    base_loss_alpha=0.3,
    epoch=1
) -> Dict[str, float]:
    model.train()

    if hasattr(model, "detach_base_for_main"):
        model.detach_base_for_main = (epoch >= 2)
    losses, losses_cls, losses_reg, losses_entropy = [], [], [], []
    use_amp = (device.type == "cuda")

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32)
    )
    reg_loss_fn = nn.SmoothL1Loss(beta=1.0)

    pbar = tqdm(total=len(loader), desc="train", leave=False, dynamic_ncols=True)

    for batch in loader:
        p_ids = batch.p_input_ids.to(device, non_blocking=True)
        p_msk = batch.p_attn_mask.to(device, non_blocking=True)
        l_ids = batch.l_ids.to(device, non_blocking=True)
        y_bin = batch.y_bin.to(device, non_blocking=True).float()

        y_reg = None
        if getattr(batch, "y_reg", None) is not None:
            y_reg = batch.y_reg.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(p_ids, p_msk, l_ids)

                if isinstance(out, tuple) and len(out) == 3:
                    logit, yhat_reg, aux = out
                elif isinstance(out, tuple) and len(out) == 2:
                    logit, aux = out
                    yhat_reg = None
                else:
                    raise ValueError("model must return (logit, aux) or (logit, yhat_reg, aux)")

                loss_cls = bce(logit, y_bin)

                loss_reg = torch.tensor(0.0, device=device)
                if (y_reg is not None) and (yhat_reg is not None):
                    loss_reg = reg_loss_fn(yhat_reg.float(), y_reg)

                loss_entropy = torch.tensor(0.0, device=device)
                if attn_entropy_lambda != 0.0 and "attn_entropy" in aux:
                    loss_entropy = +attn_entropy_lambda * aux["attn_entropy"]

                loss_base = torch.tensor(0.0, device=device)
                if "logit_base" in aux:
                    loss_base = bce(aux["logit_base"], y_bin)

                loss = 0.5 * loss_cls + base_loss_alpha * loss_base + loss_entropy
                if (y_reg is not None) and (yhat_reg is not None):
                    loss = loss + reg_lambda * loss_reg
        else:
            out = model(p_ids, p_msk, l_ids)

            if isinstance(out, tuple) and len(out) == 3:
                logit, yhat_reg, aux = out
            elif isinstance(out, tuple) and len(out) == 2:
                logit, aux = out
                yhat_reg = None
            else:
                raise ValueError("model must return (logit, aux) or (logit, yhat_reg, aux)")

            loss_cls = bce(logit, y_bin)

            loss_reg = torch.tensor(0.0, device=device)
            if (y_reg is not None) and (yhat_reg is not None):
                loss_reg = reg_loss_fn(yhat_reg.float(), y_reg)

            loss_entropy = torch.tensor(0.0, device=device)
            if attn_entropy_lambda != 0.0 and "attn_entropy" in aux:
                loss_entropy = +attn_entropy_lambda * aux["attn_entropy"]
            loss_base = torch.tensor(0.0, device=device)
            if "logit_base" in aux:
                loss_base = bce(aux["logit_base"], y_bin)

            loss = 0.5 * loss_cls + base_loss_alpha * loss_base + loss_entropy
            if (y_reg is not None) and (yhat_reg is not None):
                loss = loss + reg_lambda * loss_reg

        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        losses_cls.append(float(loss_cls.detach().cpu().item()))
        losses_reg.append(float(loss_reg.detach().cpu().item()))
        losses_entropy.append(float(loss_entropy.detach().cpu().item()))

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{losses[-1]:.4f}",
            cls=f"{losses_cls[-1]:.4f}",
            reg=f"{losses_reg[-1]:.4f}",
        )

    pbar.close()

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "loss_cls": float(np.mean(losses_cls)) if losses_cls else 0.0,
        "loss_reg": float(np.mean(losses_reg)) if losses_reg else 0.0,
        "loss_entropy": float(np.mean(losses_entropy)) if losses_entropy else 0.0,
    }

# =========================================================
# Save/load
# =========================================================
def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dti_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, lig_enc: PretrainedLigandEncoder, epoch: int, best: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "model_type": getattr(args, "model_type", "qkonly_cls"),
            "lig_config": lig_enc.conf,
            "lig_vocab_source": lig_enc.vocab_source,
            "lig_base_vocab": lig_enc.base_vocab,
            "lig_vocab_size": lig_enc.vocab_size,
            "lig_pad_id": lig_enc.pad_id,
            "lig_mask_id": lig_enc.mask_id,
            "lig_cls_id": lig_enc.cls_id,
        },
        path,
    )


def load_dti_checkpoint(path: str, model: nn.Module, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    return ckpt, missing, unexpected

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, attn_temp=1.0, qk_norm=True, attn_smooth_eps=0.02):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_temp = attn_temp
        self.qk_norm = qk_norm
        self.attn_smooth_eps = attn_smooth_eps

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, q_in, kv_in, kv_pad_mask=None, return_maps=False):
        B, Lq, _ = q_in.shape
        _, Lk, _ = kv_in.shape

        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)   # ★ VはK側に統一（安全）

        # multi-head
        q = q.view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits * self.scale / self.attn_temp

        if kv_pad_mask is not None:
            mask = kv_pad_mask[:, None, None, :]
            attn_logits = attn_logits.masked_fill(mask, -1e9)

        attn = torch.softmax(attn_logits, dim=-1)

        # ---- mask再適用 + normalize ----
        if kv_pad_mask is not None:
            attn = attn.masked_fill(mask, 0.0)
            denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn = attn / denom

        # ---- smoothing ----
        eps = self.attn_smooth_eps
        if eps > 0:
            if kv_pad_mask is None:
                attn = (1 - eps) * attn + eps / attn.size(-1)
            else:
                valid = (~kv_pad_mask)[:, None, None, :].float()
                valid = valid / valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
                attn = (1 - eps) * attn + eps * valid

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)

        if return_maps:
            return out, {
                "qk_scores": attn_logits.detach(),
                "attn_map": attn.detach(),
            }
        return out

class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.1, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
        )

    def forward(self, x):
        return self.net(x)

class DualStreamBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, attn_temp=1.0, qk_norm=True, attn_smooth_eps=0.02):
        super().__init__()

        self.ln_l_q = nn.LayerNorm(d_model)
        self.ln_l_kv = nn.LayerNorm(d_model)
        self.ln_p_q = nn.LayerNorm(d_model)
        self.ln_p_kv = nn.LayerNorm(d_model)

        self.lig_from_prot = CrossAttention(
            d_model, n_heads, dropout,
            attn_temp=attn_temp,
            qk_norm=qk_norm,
            attn_smooth_eps=attn_smooth_eps,
        )
        self.prot_from_lig = CrossAttention(
            d_model, n_heads, dropout,
            attn_temp=attn_temp,
            qk_norm=qk_norm,
            attn_smooth_eps=attn_smooth_eps,
        )

        self.ln_l_ffn = nn.LayerNorm(d_model)
        self.ln_p_ffn = nn.LayerNorm(d_model)

        self.ff_l = FFN(d_model, dropout)
        self.ff_p = FFN(d_model, dropout)

    def forward(self, l_h, p_h, l_pad=None, p_pad=None, return_maps=False):

        # ---- ligand update ----
        l_q = self.ln_l_q(l_h)
        p_k = self.ln_p_kv(p_h)

        if return_maps:
            l_ctx, aux_lp = self.lig_from_prot(l_q, p_k, p_pad, True)
        else:
            l_ctx = self.lig_from_prot(l_q, p_k, p_pad, False)

        l_h = l_h + l_ctx
        l_h = l_h + self.ff_l(self.ln_l_ffn(l_h))

        # ---- protein update ----
        p_q = self.ln_p_q(p_h)
        l_k = self.ln_l_kv(l_h)

        if return_maps:
            p_ctx, aux_pl = self.prot_from_lig(p_q, l_k, l_pad, True)
        else:
            p_ctx = self.prot_from_lig(p_q, l_k, l_pad, False)

        p_h = p_h + p_ctx
        p_h = p_h + self.ff_p(self.ln_p_ffn(p_h))

        if return_maps:
            return l_h, p_h, {
                "attn_lp": aux_lp["attn_map"],
                "qk_lp": aux_lp["qk_scores"],
                "attn_pl": aux_pl["attn_map"],
                "qk_pl": aux_pl["qk_scores"],
            }

        return l_h, p_h

class DualStreamDTIClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        dropout: float,
        n_heads: int = 4,
        n_layers: int = 2,
        protein_token_dropout: float = 0.0,
        ligand_token_dropout: float = 0.0,
        attn_temp: float = 2.0,
        qk_norm: bool = True,
        attn_smooth_eps: float = 0.02,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model
        self.d_model = d_model
        self.n_heads = int(n_heads)
        self.protein_token_dropout = float(protein_token_dropout)
        self.ligand_token_dropout = float(ligand_token_dropout)
        self.lig_pad_id = int(self.lig.pad_id)

        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)
        self.qk_norm = qk_norm
        self.dropout = dropout
        self.blocks = nn.ModuleList([
            DualStreamBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
                attn_temp=attn_temp,
                qk_norm=self.qk_norm,
            )
            for _ in range(n_layers)
        ])

        # CLSを使わないので 3D = [lig_vec, prot_vec, lig_vec*prot_vec]
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # ★ 追加
        self.base_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.delta_scale = 1.0
        self.detach_base_for_main = False

    def _masked_mean(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        denom = (~pad).sum(dim=1, keepdim=True).clamp(min=1)
        return x.sum(dim=1) / denom

    def _apply_token_dropout(self, pad_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if (not self.training) or drop_prob <= 0.0:
            return pad_mask
        drop_mask = (torch.rand_like(pad_mask.float()) < drop_prob) & (~pad_mask)
        out = pad_mask | drop_mask

        # 全drop防止
        all_dropped = out.all(dim=1)
        if all_dropped.any():
            out = out.clone()
            keep_idx = (~out).float().argmax(dim=1)
            for b in torch.where(all_dropped)[0]:
                out[b, keep_idx[b]] = False
        return out

    def forward(self, p_input_ids, p_attn_mask, l_ids, return_maps: bool = False):
        aux = {}

        p_h = self.prot(p_input_ids, p_attn_mask)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)

        l_h = self.lig(l_ids)

        prot_pad_mask = (p_attn_mask == 0)
        lig_pad_mask = (l_ids == self.lig_pad_id)

        # CLSは切り離すが最終判定には使わない
        p_tok = p_h[:, 1:, :]
        l_tok = l_h[:, 1:, :]

        p_pad = prot_pad_mask[:, 1:]
        l_pad = lig_pad_mask[:, 1:]
        prot_vec_base = self._masked_mean(p_tok, p_pad)

        eos_id = getattr(self.prot.esm.config, "eos_token_id", None)
        if eos_id is None:
            eos_id = 2
        p_tok_ids = p_input_ids[:, 1:]
        p_pad = p_pad | (p_tok_ids == eos_id)

        # protein token dropout
        p_pad = self._apply_token_dropout(p_pad, self.protein_token_dropout)

        # ligand token dropout
        l_pad = self._apply_token_dropout(l_pad, self.ligand_token_dropout)

        last_aux = None
        for blk in self.blocks:
            if return_maps:
                l_tok, p_tok, blk_aux = blk(l_tok, p_tok, l_pad=l_pad, p_pad=p_pad, return_maps=True)
                last_aux = blk_aux  # ★これ追加
            else:
                l_tok, p_tok = blk(l_tok, p_tok, l_pad=l_pad, p_pad=p_pad, return_maps=False)

        lig_vec = self._masked_mean(l_tok, l_pad)
        prot_vec = self._masked_mean(p_tok, p_pad)
        inter_vec = lig_vec * prot_vec

        # --- delta ---
        z_delta = torch.cat([lig_vec, prot_vec, inter_vec], dim=-1)
        logit_delta = self.cls_head(z_delta).squeeze(-1)

        # --- baseline ---
        logit_base = self.base_head(prot_vec_base).squeeze(-1)

        # --- final ---
        # epoch 1: baseline + delta
        # epoch 2+: delta only
        if self.detach_base_for_main:
            logit = logit_base.detach() + 1.5 * logit_delta
        else:
            logit = logit_base + 1.5 * logit_delta

        # regression はそのまま
        yhat_reg = self.reg_head(z_delta).squeeze(-1)

        # --- logging ---
        aux["logit_base"] = logit_base
        aux["logit_delta"] = logit_delta
        aux["logit_full"] = logit

        if last_aux is not None:
            attn_lp = last_aux["attn_lp"]
            qk_lp   = last_aux["qk_lp"]
            attn_pl = last_aux["attn_pl"]
            qk_pl   = last_aux["qk_pl"]

            # entropy は LP/PL 両方にかける
            attn_lp_safe = attn_lp.clamp(min=1e-8)
            ent_lp = -(attn_lp_safe * torch.log(attn_lp_safe)).sum(dim=-1)  # (B,H,Ll)
            ent_lp = ent_lp.masked_fill(l_pad[:, None, :], 0.0)
            denom_l = (~l_pad).sum(dim=1).clamp(min=1)
            ent_lp_mean = (ent_lp.sum(dim=(1, 2)) / (denom_l * self.n_heads))

            attn_pl_safe = attn_pl.clamp(min=1e-8)
            ent_pl = -(attn_pl_safe * torch.log(attn_pl_safe)).sum(dim=-1)  # (B,H,Lp)
            ent_pl = ent_pl.masked_fill(p_pad[:, None, :], 0.0)
            denom_p = (~p_pad).sum(dim=1).clamp(min=1)
            ent_pl_mean = (ent_pl.sum(dim=(1, 2)) / (denom_p * self.n_heads))

            aux["attn_entropy"] = 0.5 * (ent_lp_mean.mean() + ent_pl_mean.mean())

            if return_maps:
                aux["qk_scores_lp"] = qk_lp.mean(dim=1)
                aux["attn_map_lp"] = attn_lp.mean(dim=1)
                aux["qk_scores_lp_heads"] = qk_lp
                aux["attn_map_lp_heads"] = attn_lp

                aux["qk_scores_pl"] = qk_pl.mean(dim=1)
                aux["attn_map_pl"] = attn_pl.mean(dim=1)
                aux["qk_scores_pl_heads"] = qk_pl
                aux["attn_map_pl_heads"] = attn_pl

                # 互換用
                aux["qk_scores"] = qk_lp.mean(dim=1)
                aux["attn_map"] = attn_lp.mean(dim=1)
                aux["qk_scores_heads"] = qk_lp
                aux["attn_map_heads"] = attn_lp

                aux["p_pad"] = p_pad
                aux["l_pad"] = l_pad
        else:
            aux["attn_entropy"] = torch.tensor(0.0, device=p_input_ids.device)
            if return_maps:
                B = p_input_ids.size(0)
                aux["qk_scores_lp"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["attn_map_lp"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["qk_scores_pl"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["attn_map_pl"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["qk_scores_lp_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["attn_map_lp_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["qk_scores_pl_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["attn_map_pl_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["qk_scores"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["attn_map"] = torch.zeros(B, 0, 0, device=p_input_ids.device)
                aux["qk_scores_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["attn_map_heads"] = torch.zeros(B, self.n_heads, 0, 0, device=p_input_ids.device)
                aux["p_pad"] = p_pad
                aux["l_pad"] = l_pad

        return logit, yhat_reg, aux

# =========================================================
# Optimizer
# =========================================================
def build_optimizer_with_llrd(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    base_lr = float(args.lr)
    lig_lr = base_lr * float(args.lig_lr_mult)

    lig_params = [p for p in model.lig.parameters() if p.requires_grad]
    non_esm_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("lig."):
            continue
        if n.startswith("prot.esm."):
            continue
        non_esm_params.append(p)

    param_groups = [
        {"params": lig_params, "lr": lig_lr, "name": "lig"},
        {"params": non_esm_params, "lr": base_lr, "name": "non_esm"},
    ]

    esm = model.prot.esm
    top_lr = base_lr * float(args.esm_lr_mult)
    decay = float(args.llrd_decay)
    min_mult = float(args.esm_min_lr_mult)

    if int(args.freeze_esm_bottom) > 0:
        n_freeze = int(args.freeze_esm_bottom)
        if hasattr(esm, "encoder") and hasattr(esm.encoder, "layer"):
            layers = esm.encoder.layer
            for i in range(min(n_freeze, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False
        else:
            print("[warn] ESM layer freezing requested, but encoder.layer not found; skipping.")

    if args.llrd and any(p.requires_grad for p in esm.parameters()):
        if hasattr(esm, "encoder") and hasattr(esm.encoder, "layer"):
            layers = list(esm.encoder.layer)
            n_layers = len(layers)

            if hasattr(esm, "embeddings"):
                emb_params = [p for p in esm.embeddings.parameters() if p.requires_grad]
                if emb_params:
                    lr_emb = max(top_lr * (decay ** n_layers), top_lr * min_mult)
                    param_groups.append({"params": emb_params, "lr": lr_emb, "name": f"esm.emb lr={lr_emb:g}"})

            for i, layer in enumerate(layers):
                layer_params = [p for p in layer.parameters() if p.requires_grad]
                if not layer_params:
                    continue
                depth_from_top = (n_layers - 1) - i
                lr_i = max(top_lr * (decay ** depth_from_top), top_lr * min_mult)
                param_groups.append({"params": layer_params, "lr": lr_i, "name": f"esm.layer{i} lr={lr_i:g}"})

            other = []
            for n, p in esm.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("embeddings.") or n.startswith("encoder.layer."):
                    continue
                other.append(p)
            if other:
                param_groups.append({"params": other, "lr": top_lr, "name": f"esm.other lr={top_lr:g}"})
        else:
            print("[warn] args.llrd set, but esm.encoder.layer not found; using single ESM group.")
            esm_params = [p for p in esm.parameters() if p.requires_grad]
            if esm_params:
                param_groups.append({"params": esm_params, "lr": top_lr, "name": f"esm.all lr={top_lr:g}"})
    else:
        esm_params = [p for p in esm.parameters() if p.requires_grad]
        if esm_params:
            param_groups.append({"params": esm_params, "lr": top_lr, "name": f"esm.all lr={top_lr:g}"})

    print("[opt groups]")
    for g in param_groups:
        nparam = sum(p.numel() for p in g["params"])
        print(f"  - {g.get('name', 'group')}: lr={g['lr']:.3g} params={len(g['params'])} numel={nparam}")

    return torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--use_train_valid_csv", action="store_true")
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--final_eval_csv", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)
    ap.add_argument("--train_size", type=int, default=None)
    ap.add_argument("--ligand_token_dropout", type=float, default=0.10)
    ap.add_argument("--attn_smooth_eps", type=float, default=0.02)
    ap.add_argument("--train_shard_dir", type=str, default=None)
    ap.add_argument("--train_shard_glob", type=str, default="train_part_*.csv")
    ap.add_argument("--train_shard_size", type=int, default=1000)
    ap.add_argument("--train_num_shards_per_epoch", type=int, default=None)

    ap.add_argument("--lig_ckpt", type=str, required=True)
    ap.add_argument("--vq_ckpt", type=str, default=None)

    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--finetune_esm", action="store_true")
    ap.add_argument("--finetune_lig", action="store_true")
    ap.add_argument("--lig_debug_index", action="store_true")

    ap.add_argument("--dti_ckpt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="./dti_out")
    ap.add_argument("--eval_only", action="store_true")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lig_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--plateau", action="store_true")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--model_type", type=str, default="dual_stream", choices=["qkonly", "dual_stream"])
    ap.add_argument("--dual_stream_layers", type=int, default=2)
    ap.add_argument("--select_on", type=str, default="ap", choices=["ap", "auroc", "f1"])
    ap.add_argument("--protein_token_dropout", type=float, default=0.10)
    ap.add_argument("--llrd", action="store_true")
    ap.add_argument("--llrd_decay", type=float, default=0.95)
    ap.add_argument("--esm_lr_mult", type=float, default=1.0)
    ap.add_argument("--esm_min_lr_mult", type=float, default=0.05)
    ap.add_argument("--freeze_esm_bottom", type=int, default=0)
    ap.add_argument("--attn_temp", type=float, default=2.0)
    ap.add_argument("--attn_entropy_lambda", type=float, default=1e-3)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--y_thr", type=float, default=Y_THR)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--reg_lambda", type=float, default=0.1)
    ap.add_argument("--attn_temp", type=float, default=1.0)
    ap.add_argument("--qk_norm", action="store_true")
    args = ap.parse_args()
    print("DEBUG train_csv:", args.train_csv)
    print("DEBUG train_size:", args.train_size)
    print("DEBUG use_train_valid_csv:", args.use_train_valid_csv)
    if not args.eval_only and not args.train_csv and not args.train_shard_dir:
        raise ValueError("Provide --train_csv or --train_shard_dir unless --eval_only is set.")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model, do_lower_case=False)

    lig_enc = PretrainedLigandEncoder(
        ckpt_path=args.lig_ckpt,
        device=device,
        finetune=args.finetune_lig,
        vq_ckpt_path=args.vq_ckpt,
        verbose_load=True,
        debug_index_check=bool(args.lig_debug_index),
    )
    prot_enc = ESMProteinEncoder(args.esm_model, device=device, finetune=args.finetune_esm)

    if args.model_type == "dual_stream":
        model = DualStreamDTIClassifier(
            protein_encoder=prot_enc,
            ligand_encoder=lig_enc,
            dropout=args.dropout,
            n_heads=args.n_heads,
            n_layers=args.dual_stream_layers,
            protein_token_dropout=args.protein_token_dropout,
            ligand_token_dropout=args.ligand_token_dropout,
            attn_temp=args.attn_temp,
            attn_smooth_eps=args.attn_smooth_eps,
        ).to(device)
    else:
        model = QKOnlyDTIClassifier(
            protein_encoder=prot_enc,
            ligand_encoder=lig_enc,
            dropout=args.dropout,
            attn_temp=args.attn_temp,
            n_heads=args.n_heads,
            protein_token_dropout=args.protein_token_dropout,
        ).to(device)

    if args.dti_ckpt is not None:
        _, missing, unexpected = load_dti_checkpoint(args.dti_ckpt, model, device)
        print(f"Loaded DTI checkpoint: {args.dti_ckpt}")
        if missing:
            print("  [warn] missing keys (up to 20):", missing[:20])
        if unexpected:
            print("  [warn] unexpected keys (up to 20):", unexpected[:20])

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_enc.pad_id, lig_cls=lig_enc.cls_id),
        )

    valid_ds = DTIDataset(args.valid_csv, y_thr=float(args.y_thr), drop_missing_y=True,
    lig_cls_id=lig_enc.cls_id,)
    valid_loader = make_loader(valid_ds, shuffle=False)

    final_eval_loader = None
    if args.final_eval_csv:
        final_eval_ds = DTIDataset(args.final_eval_csv, y_thr=float(args.y_thr), drop_missing_y=True,
    lig_cls_id=lig_enc.cls_id,)
        final_eval_loader = make_loader(final_eval_ds, shuffle=False)

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv, y_thr=float(args.y_thr), drop_missing_y=True,
    lig_cls_id=lig_enc.cls_id)
        test_loader = make_loader(test_ds, shuffle=False)

    optimizer = build_optimizer_with_llrd(model, args) if not args.eval_only else None
    scheduler = None
    if optimizer is not None and args.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.min_lr),
        )

    if args.eval_only:
        yhat_v, yb_v, yhatr_v, yr_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)
        v_r = eval_reg_metrics(yhatr_v, yr_v)
        print("[VALID]", v_m)

        test_m, test_r = None, None
        if test_loader is not None:
            yhat_t, yb_t, yhatr_t, yr_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            test_r = eval_reg_metrics(yhatr_t, yr_t)
            print("[TEST ]", test_m)

        save_json(os.path.join(args.out_dir, "eval_only.json"), {
            "mode": "eval_only",
            "valid": v_m,
            "valid_reg": v_r,
            "test": test_m,
            "test_reg": test_r,
            "valid_csv": args.valid_csv,
            "test_csv": args.test_csv,
            "dti_ckpt": args.dti_ckpt,
        })
        return

    best = {"ap": -1e9, "auroc": -1e9, "f1": -1e9, "epoch": -1}
    fixed_train_loader = None
    fixed_train_ds = None
    train_shard_paths = None

    if args.use_train_valid_csv:
        if args.train_size is None:
            # use full train.csv
            fixed_train_ds = DTIDataset(args.train_csv, y_thr=float(args.y_thr), drop_missing_y=True,
    lig_cls_id=lig_enc.cls_id,)
            fixed_train_loader = make_loader(fixed_train_ds, shuffle=True)
        else:
            # use first N rows
            all_rows = read_csv_rows(args.train_csv)
            train_rows = all_rows[: int(args.train_size)]
            fixed_train_ds = DTIDataset(rows=train_rows, y_thr=float(args.y_thr), drop_missing_y=True,
    lig_cls_id=lig_enc.cls_id,)
            fixed_train_loader = make_loader(fixed_train_ds, shuffle=True)
    else:
        train_shard_paths = list_train_shards(args.train_shard_dir, args.train_shard_glob)

    for ep in range(1, args.epochs + 1):
        qk_save_dir = os.path.join(args.out_dir, "qk_maps")
        if args.use_train_valid_csv and args.train_size is not None:
            epoch_train_ds = fixed_train_ds
            train_loader = fixed_train_loader
        else:
            epoch_train_ds = fixed_train_ds
            train_loader = fixed_train_loader

        print("epoch train n:", len(epoch_train_ds))
        pos_weight = compute_pos_weight_from_dataset(epoch_train_ds)
        print(f"pos_weight={pos_weight:.4f}")

        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            grad_clip=float(args.grad_clip),
            attn_entropy_lambda=float(args.attn_entropy_lambda),
            reg_lambda=float(args.reg_lambda),
            base_loss_alpha=0.3,
            epoch=ep,
        )

        yhat_tr, yb_tr, yhatr_tr, yr_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(yhat_tr, yb_tr)
        tr_r = eval_reg_metrics(yhatr_tr, yr_tr)

        yhat_v, yb_v, yhatr_v, yr_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)
        v_r = eval_reg_metrics(yhatr_v, yr_v)

        if scheduler is not None:
            scheduler.step(float(v_m[args.select_on]))

        if float(v_m[args.select_on]) > float(best[args.select_on]):
            best.update({
                "ap": float(v_m["ap"]),
                "auroc": float(v_m["auroc"]),
                "f1": float(v_m["f1"]),
                "epoch": ep,
            })
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        save_dti_checkpoint(os.path.join(args.out_dir, "last.pt"), model, args, lig_enc, epoch=ep, best=best)

        test_m, test_r = None, None
        if test_loader is not None:
            yhat_t, yb_t, yhatr_t, yr_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            test_r = eval_reg_metrics(yhatr_t, yr_t)

        final_m, final_r = None, None
        if final_eval_loader is not None:
            yhat_f, yb_f, yhatr_f, yr_f = predict(model, final_eval_loader, device)
            final_m = eval_metrics(yhat_f, yb_f)
            final_r = eval_reg_metrics(yhatr_f, yr_f)

        save_json(os.path.join(args.out_dir, f"epoch_{ep:03d}.json"), {
            "epoch": ep,
            "train_stat": tr_stat,
            "train_metrics": tr_m,
            "train_reg_metrics": tr_r,
            "valid_metrics": v_m,
            "valid_reg_metrics": v_r,
            "final_eval_metrics": final_m,
            "final_eval_reg_metrics": final_r if final_eval_loader is not None else None,
            "test_metrics": test_m,
            "test_reg_metrics": test_r if test_loader is not None else None,
            "pos_weight": pos_weight,
        })

        print(f"\n===== Epoch {ep} =====")
        print("[train]",
              f"AUC={tr_m['auroc']:.4f}", f"AP={tr_m['ap']:.4f}", f"F1={tr_m['f1']:.4f}",
              f"RMSE={tr_r['rmse']:.4f}", f"SP={tr_r['spearman']:.4f}")
        print("[valid]",
              f"AUC={v_m['auroc']:.4f}", f"AP={v_m['ap']:.4f}", f"F1={v_m['f1']:.4f}",
              f"RMSE={v_r['rmse']:.4f}", f"SP={v_r['spearman']:.4f}")
        if final_m is not None:
            print("[final]", f"AUC={final_m['auroc']:.4f}", f"AP={final_m['ap']:.4f}", f"F1={final_m['f1']:.4f}", f"EF1={final_m['ef1']:.3f}", f"EF5={final_m['ef5']:.3f}", f"EF10={final_m['ef10']:.3f}")
        if ep in [1, 2]:
            visualize_one_qk_map(
                model=model,
                loader=valid_loader,
                device=device,
                esm_tokenizer=esm_tokenizer,
                sample_idx_in_batch=0,
                show_token_labels=False,
                save_dir=qk_save_dir,
                prefix=f"epoch{ep:03d}_valid_sample0",
            )
    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)


if __name__ == "__main__":
    main()
