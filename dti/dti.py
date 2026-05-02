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
from transformers import AutoTokenizer, EsmModel, get_linear_schedule_with_warmup
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

def read_csv_rows_filter_split(path: str, split_name: str) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    out = []
    for r in rows:
        if (r.get("split") or "").strip() == split_name:
            out.append(r)
    if not out:
        raise ValueError(f"No rows with split={split_name} in {path}")
    return out

def read_csv_random_rows(path: str, n_rows: int, seed: int) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    if n_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), n_rows)
    return [rows[i] for i in idx]

def compute_atom_res_contact_loss_from_aux(
    aux: dict,
    atom_contact_pairs: List[List[Tuple[int, int]]],
    contact_topk: int = 8,
) -> torch.Tensor:
    """
    pair_map: (B, Ll, Lp)
      Ll = ligand atoms/tokens, CLS除去後
      Lp = protein residues, CLS/EOS除去後

    atom_contact_pairs:
      list of [(lig_atom_idx, residue_idx), ...]
      indices are 0-based, matching SDF heavy atom order and protein seq order.
    """
    if "pair_map" not in aux:
        raise ValueError("aux['pair_map'] is missing")

    pair_map = aux["pair_map"].float()  # (B, Ll, Lp)
    l_pad = aux["l_pad"]                # (B, Ll)
    p_pad = aux["p_pad"]                # (B, Lp)

    valid = (~l_pad).unsqueeze(-1) & (~p_pad).unsqueeze(-2)

    vals_valid = pair_map[valid]
    if vals_valid.numel() == 0:
        return pair_map.sum() * 0.0

    mean = vals_valid.mean().detach()
    std = vals_valid.std(unbiased=False).detach().clamp_min(1e-6)
    logits = (pair_map - mean) / std

    losses = []

    B, Ll, Lp = logits.shape

    for b in range(B):
        pos_vals = []

        for ai, ri in atom_contact_pairs[b]:
            # ai: ligand atom index
            # ri: protein residue index
            # pair_map already excludes ligand CLS and protein CLS
            if ai < 0 or ri < 0:
                continue
            if ai >= Ll or ri >= Lp:
                continue
            if bool(l_pad[b, ai]) or bool(p_pad[b, ri]):
                continue

            pos_vals.append(logits[b, ai, ri])

        if not pos_vals:
            continue

        pos_vals = torch.stack(pos_vals)
        k = min(int(contact_topk), pos_vals.numel())
        topk_vals = pos_vals.topk(k).values

        # reward-style guide:
        # contact pair score が大きいほど loss が下がる
        losses.append(-topk_vals.mean())

    if not losses:
        return pair_map.sum() * 0.0

    loss_contact = torch.stack(losses).mean()

    # スコア爆発を少し抑える
    loss_contact = loss_contact + 0.01 * (pair_map[valid] ** 2).mean()

    return loss_contact

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
    contact_mask: Optional[torch.Tensor] = None
    atom_contact_pairs: Optional[List[List[Tuple[int, int]]]] = None

class DTIDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        rows: Optional[List[Dict[str, str]]] = None,
        y_thr: float = Y_THR,
        drop_missing_y: bool = True,
        lig_cls_id: int = 0,
        y_reg_mean: Optional[float] = None,
        y_reg_std: Optional[float] = None,
        ligand_input_type: str = "vqatom",
        smiles_tokenizer=None,
        smiles_col: str = "smiles",
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

            if ligand_input_type == "vqatom":
                lig_text = (r.get("lig_tok") or "").strip()
            elif ligand_input_type == "smiles":
                lig_text = (
                        r.get(smiles_col)
                        or r.get("canonical_smiles")
                        or r.get("smiles")
                        or ""
                ).strip()
            else:
                raise ValueError(f"Unknown ligand_input_type: {ligand_input_type}")

            y_raw = r.get("y", "")

            if not seq or not lig_text:
                continue

            if y_raw in ("", None):
                if drop_missing_y:
                    continue
                y = float("nan")
            else:
                y = float(y_raw)

            rr = dict(r)
            rr["seq"] = seq
            rr["lig_text"] = lig_text
            rr["y"] = y
            rr["y_bin"] = 1.0 if y >= float(y_thr) else 0.0
            self.rows.append(rr)

        if not self.rows:
            raise ValueError("No usable rows in dataset")

        ys = [float(r["y"]) for r in self.rows if not np.isnan(float(r["y"]))]

        if y_reg_mean is None or y_reg_std is None:
            if len(ys) == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = float(np.mean(ys))
                std = float(np.std(ys)) + 1e-6
        else:
            mean = float(y_reg_mean)
            std = float(y_reg_std)

        self.y_reg_mean = mean
        self.y_reg_std = std

        for r in self.rows:
            y = float(r["y"])
            if np.isnan(y):
                r["y_reg"] = None
            else:
                r["y_reg"] = (y - mean) / std

        self.ligand_input_type = ligand_input_type
        self.smiles_tokenizer = smiles_tokenizer
        self.smiles_col = smiles_col

    def __len__(self) -> int:
        return len(self.rows)

    def _parse_lig_tok(self, lig_tok: str) -> List[int]:
        return [int(x) for x in lig_tok.split() if x.strip()]

    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.ligand_input_type == "vqatom":
            lig_ids = [self.lig_cls_id] + self._parse_lig_tok(row["lig_text"])
        else:
            lig_ids = self.smiles_tokenizer.encode(row["lig_text"], add_cls=True)

        contact_mask = str(row.get("contact_mask", "") or "").strip()

        atom_contact_pairs = row.get("atom_contact_pairs", "")

        item = {
            "protein_seq": row["seq"],
            "lig_ids": lig_ids,
            "y_bin": float(row["y_bin"]),
            "y_reg": None if row["y_reg"] is None else float(row["y_reg"]),
            "contact_mask": contact_mask,
            "atom_contact_pairs": atom_contact_pairs,
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
    contact_tensors = []
    has_contact = all(len(str(s.get("contact_mask", "") or "")) > 0 for s in samples)

    if has_contact:
        for s in samples:
            cm = str(s["contact_mask"]).strip()
            contact_tensors.append(torch.tensor([1.0 if c == "1" else 0.0 for c in cm], dtype=torch.float32))

        max_p = p_input_ids.shape[1] - 1  # CLS除去後の長さ
        contact_mask = torch.zeros((len(samples), max_p), dtype=torch.float32)

        for i, cm in enumerate(contact_tensors):
            n = min(cm.numel(), max_p)
            contact_mask[i, :n] = cm[:n]
    else:
        contact_mask = None
    atom_contact_pairs = []

    for s in samples:
        raw = s.get("atom_contact_pairs", "")
        if raw:
            try:
                pairs = json.loads(raw)
                pairs = [(int(a), int(r)) for a, r in pairs]
            except Exception:
                pairs = []
        else:
            pairs = []
        atom_contact_pairs.append(pairs)

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y_bin=y_bin,
        y_reg=y_reg,
        contact_mask=contact_mask,
        atom_contact_pairs=atom_contact_pairs,
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
        load_pretrained: bool = True,  # 追加
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

            # CLS token を追加
            self.cls_id = int(self.vocab_size)
            self.vocab_size = int(self.vocab_size) + 1

            self.vocab_source = f"vq_ckpt:{vq_ckpt_path}"
        else:
            if "base_vocab" in ckpt and "vocab_size" in ckpt:
                self.base_vocab = int(ckpt["base_vocab"])
                self.vocab_size = int(ckpt["vocab_size"])
                self.pad_id = self.base_vocab + 0
                self.mask_id = self.base_vocab + 1
                self.cls_id = int(self.vocab_size)
                self.vocab_size = int(self.vocab_size) + 1
                self.vocab_source = f"mlm_ckpt:{ckpt_path}"

            elif all(k in ckpt for k in ["vocab_size", "pad_id", "mask_id", "cls_id"]):
                # SMILES MLM checkpoint
                self.base_vocab = int(ckpt["vocab_size"])
                self.vocab_size = int(ckpt["vocab_size"])
                self.pad_id = int(ckpt["pad_id"])
                self.mask_id = int(ckpt["mask_id"])
                self.cls_id = int(ckpt["cls_id"])
                self.vocab_source = f"smiles_mlm_ckpt:{ckpt_path}"

            else:
                raise RuntimeError(
                    "Ligand MLM ckpt must contain either "
                    "(base_vocab, vocab_size) or (vocab_size, pad_id, mask_id, cls_id)"
                )

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

        if load_pretrained:
            load_state_dict_shape_safe(self, self.state, verbose=verbose_load)

            mlm_tok_key = "tok.weight"
            if mlm_tok_key in self.state:
                w = self.state[mlm_tok_key]
                if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[1] == d_model:
                    overlap = min(int(w.shape[0]), int(self.tok.weight.shape[0]))
                    if overlap > 0:
                        with torch.no_grad():
                            self.tok.weight[:overlap].copy_(w[:overlap])
        else:
            print("[lig] NO PRETRAIN: random initialized ligand Transformer")
            self.vocab_source = self.vocab_source + ":no_pretrain"

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

def plot_one(mat, title, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)

    if save_path is not None and "lp" in save_path:
        plt.ylabel("Ligand tokens")
        plt.xlabel("Protein tokens")
    else:
        plt.xlabel("Ligand tokens")
        plt.ylabel("Protein tokens")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def visualize_one_qk_map(
    model,
    loader,
    device,
    esm_tokenizer,
    sample_idx_in_batch: int = 0,
    show_token_labels: bool = False,
    save_dir: str | None = None,
    prefix: str = "sample",
    save_headwise: bool = True,
):
    import os
    import torch
    import numpy as np

    model.eval()
    batch = next(iter(loader))

    p_ids = batch.p_input_ids.to(device, non_blocking=True)
    p_msk = batch.p_attn_mask.to(device, non_blocking=True)
    l_ids = batch.l_ids.to(device, non_blocking=True)

    with torch.inference_mode():
        logit, yhat_reg, aux = model(p_ids, p_msk, l_ids, return_maps=True)

    prob = float(torch.sigmoid(logit[sample_idx_in_batch]).detach().cpu())
    y_bin = float(batch.y_bin[sample_idx_in_batch].detach().cpu())

    p_pad = aux["p_pad"][sample_idx_in_batch].detach().cpu().numpy().astype(bool)
    l_pad = aux["l_pad"][sample_idx_in_batch].detach().cpu().numpy().astype(bool)

    def headmean(x):
        if x is None:
            return None
        return x[sample_idx_in_batch].detach().float().cpu().numpy().mean(axis=0)

    def get_heads(x):
        if x is None:
            return None
        return x[sample_idx_in_batch].detach().float().cpu().numpy()

    def trim_lp(mat):
        if mat is None:
            return None
        if mat.ndim == 2:
            if mat.shape[0] == len(l_pad) and mat.shape[1] == len(p_pad):
                return mat[~l_pad][:, ~p_pad]
            elif mat.shape[0] == len(l_pad):
                return mat[~l_pad]
        return mat

    def trim_pl(mat):
        if mat is None:
            return None
        if mat.ndim == 2:
            if mat.shape[0] == len(p_pad) and mat.shape[1] == len(l_pad):
                return mat[~p_pad][:, ~l_pad]
            elif mat.shape[0] == len(p_pad):
                return mat[~p_pad]
        return mat

    def pairctx_to_map(pair_ctx_3d):
        # pair_ctx_3d: (Ll, Lp, Dh) or (Lp, Ll, Dh)
        if pair_ctx_3d is None:
            return None
        return np.linalg.norm(pair_ctx_3d, axis=-1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    base = f"{prefix}_prob{prob:.3f}_y{int(y_bin)}"

    # ===== mean maps =====
    S_lp = trim_lp(headmean(aux.get("lp_qk_logits")))
    A_lp = trim_lp(headmean(aux.get("lp_attn")))
    X_lp = trim_lp(headmean(aux.get("lp_ctx")))


    S_pl = trim_pl(headmean(aux.get("pl_qk_logits")))
    A_pl = trim_pl(headmean(aux.get("pl_attn")))
    X_pl = trim_pl(headmean(aux.get("pl_ctx")))
    P_lp = trim_lp(headmean(aux.get("lp_pair_score")))

    P_pl_raw = headmean(aux.get("pl_pair_score"))
    P_pl = trim_pl(P_pl_raw) if P_pl_raw is not None else None
    PairMap = aux.get("pair_map")
    if PairMap is not None:
        PairMap = PairMap[sample_idx_in_batch].detach().float().cpu().numpy()
        PairMap = trim_lp(PairMap)

    if S_lp is not None:
        plot_one(
            S_lp,
            f"lig <- prot logits(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_lp_logits_mean.png") if save_dir else None,
        )
    if A_lp is not None:
        plot_one(
            A_lp,
            f"lig <- prot attn(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_lp_attn_mean.png") if save_dir else None,
        )
    if X_lp is not None:
        plot_one(
            X_lp,
            f"lig <- prot ctx(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_lp_ctx_mean.png") if save_dir else None,
        )
    if P_lp is not None:
        plot_one(
            P_lp,
            f"lig <- prot pair_ctx_norm(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_lp_pairctx_norm_mean.png") if save_dir else None,
        )
    if PairMap is not None:
        plot_one(
            PairMap,
            f"pair_map(final overlap) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_pair_map.png") if save_dir else None,
        )
    if S_pl is not None:
        plot_one(
            S_pl,
            f"prot <- lig logits(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_pl_logits_mean.png") if save_dir else None,
        )
    if A_pl is not None:
        plot_one(
            A_pl,
            f"prot <- lig attn(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_pl_attn_mean.png") if save_dir else None,
        )
    if X_pl is not None:
        plot_one(
            X_pl,
            f"prot <- lig ctx(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_pl_ctx_mean.png") if save_dir else None,
        )
    if P_pl is not None:
        plot_one(
            P_pl,
            f"prot <- lig pair_ctx_norm(mean) | prob={prob:.4f} y_bin={y_bin:.0f}",
            os.path.join(save_dir, f"{base}_pl_pairctx_norm_mean.png") if save_dir else None,
        )

    # ===== headwise maps =====
    if save_headwise:
        lp_logits_heads = get_heads(aux.get("lp_qk_logits"))     # (H, Ll, Lp)
        lp_attn_heads   = get_heads(aux.get("lp_attn"))          # (H, Ll, Lp)
        lp_ctx_heads    = get_heads(aux.get("lp_ctx"))           # (H, Ll, Dh)

        pl_logits_heads = get_heads(aux.get("pl_qk_logits"))
        pl_attn_heads   = get_heads(aux.get("pl_attn"))
        pl_ctx_heads    = get_heads(aux.get("pl_ctx"))
        lp_pair_heads = get_heads(aux.get("lp_pair_score"))
        pl_pair_heads = get_heads(aux.get("pl_pair_score"))

        if lp_attn_heads is not None:
            n_heads = lp_attn_heads.shape[0]
            for h in range(n_heads):
                S = trim_lp(lp_logits_heads[h]) if lp_logits_heads is not None else None
                A = trim_lp(lp_attn_heads[h]) if lp_attn_heads is not None else None
                X = trim_lp(lp_ctx_heads[h]) if lp_ctx_heads is not None else None
                P = trim_lp(lp_pair_heads[h]) if lp_pair_heads is not None else None
                head_dir = os.path.join(save_dir, f"{base}_head{h:02d}") if save_dir else None
                if head_dir is not None:
                    os.makedirs(head_dir, exist_ok=True)

                if S is not None:
                    plot_one(
                        S,
                        f"lig <- prot logits | head={h} prob={prob:.4f} y_bin={y_bin:.0f}",
                        os.path.join(head_dir, f"lp_logits_h{h:02d}.png")
                    )
                if A is not None:
                    plot_one(
                        A,
                        f"lig <- prot attn | head={h} prob={prob:.4f} y_bin={y_bin:.0f}",
                        os.path.join(head_dir, f"lp_attn_h{h:02d}.png")
                    )
                if X is not None:
                    plot_one(
                        X,
                        f"lig <- prot ctx | head={h} prob={prob:.4f} y_bin={y_bin:.0f}",
                        os.path.join(head_dir, f"lp_ctx_h{h:02d}.png")
                    )
                if P is not None:
                    plot_one(
                        P,
                        f"lig <- prot pair_ctx_norm | head={h} prob={prob:.4f} y_bin={y_bin:.0f}",
                        os.path.join(head_dir, f"lp_pairctx_norm_h{h:02d}.png")
                    )

    return {
        "attn_lp_logits": S_lp,
        "attn_lp": A_lp,
        "attn_lp_x_v": X_lp,
        "attn_lp_pairctx": P_lp,
        "attn_pl_logits": S_pl,
        "attn_pl": A_pl,
        "attn_pl_x_v": X_pl,
        "attn_pl_pairctx": P_pl,
        "pair_map": PairMap,  # ←追加
        "prob": prob,
        "y_bin": y_bin,
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
            if (batch.y_reg is not None) and (yhat_reg is not None):
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
        base_rate = n_pos / float(n)

        return float(hit_rate_topk / base_rate)

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

def compute_contact_loss_from_aux(
    aux: dict,
    contact_mask: torch.Tensor,
    contact_topk: int = 3,
) -> torch.Tensor:
    """
    aux["pair_map"]: (B, Ll, Lp)
    contact_mask:   (B, Lp)
    """
    if "pair_map" not in aux:
        raise ValueError("aux['pair_map'] is missing")

    pair_map = aux["pair_map"].float()  # (B, Ll, Lp)
    p_pad = aux["p_pad"]                # (B, Lp), True=PAD

    # ligand方向にmax poolingして residue-level score にする
    contact_score = pair_map.amax(dim=1)  # (B, Lp)

    L = min(contact_score.shape[1], contact_mask.shape[1], p_pad.shape[1])
    contact_score = contact_score[:, :L]
    contact_target = contact_mask[:, :L].float()
    p_valid = (~p_pad[:, :L])

    valid_vals = contact_score[p_valid]
    if valid_vals.numel() == 0:
        return contact_score.sum() * 0.0

    # batch内z-score。弱いranking guideとして使う
    mean = valid_vals.mean().detach()
    std = valid_vals.std(unbiased=False).detach().clamp_min(1e-6)
    contact_logit = (contact_score - mean) / std

    pos_mask = (contact_target > 0.5) & p_valid

    losses = []
    for b in range(contact_logit.size(0)):
        vals = contact_logit[b][pos_mask[b]]
        if vals.numel() == 0:
            continue
        k = min(int(contact_topk), vals.numel())
        topk_vals = vals.topk(k).values
        losses.append(F.softplus(-topk_vals).mean())

    if not losses:
        return contact_score.sum() * 0.0

    return torch.stack(losses).mean()


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
    scheduler=None,
    grad_clip: float = 1.0,
    attn_entropy_lambda: float = 0.0,
    reg_lambda: float = 0.1,
    sym_lambda: float = 0.0,
    epoch=1,
    contact_lambda: float = 0.0,
    guide_loader: Optional[DataLoader] = None,
    guide_every: int = 1,
    contact_topk: int = 3,
) -> Dict[str, float]:

    model.train()

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    reg_loss_fn = nn.SmoothL1Loss(beta=1.0)

    losses = []
    losses_contact = []

    use_amp = (device.type == "cuda")

    guide_iter = iter(guide_loader) if guide_loader is not None else None

    pbar = tqdm(loader, desc="train", leave=False)

    def attention_symmetry_loss(attn_lp, attn_pl, p_pad, l_pad):
        attn_pl_t = attn_pl.transpose(-1, -2)
        diff = (attn_lp - attn_pl_t) ** 2

        valid = (~l_pad).unsqueeze(1).unsqueeze(-1) & (~p_pad).unsqueeze(1).unsqueeze(2)
        diff = diff.masked_fill(~valid, 0.0)

        return diff.sum() / valid.float().sum().clamp_min(1.0)

    for step, batch in enumerate(pbar):

        # =========================
        # MAIN DTI BATCH
        # =========================
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l_ids = batch.l_ids.to(device)
        y_bin = batch.y_bin.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, yhat_reg, aux = model(p_ids, p_msk, l_ids, return_maps=False)
                loss_cls = bce(logit, y_bin)
                loss = loss_cls
        else:
            logit, yhat_reg, aux = model(p_ids, p_msk, l_ids, return_maps=False)
            loss_cls = bce(logit, y_bin)
            loss = loss_cls

        # =========================
        # GUIDE CONTACT BATCH
        # =========================
        loss_contact = torch.tensor(0.0, device=device)

        if (
                contact_lambda > 0.0
                and guide_loader is not None
                and step % guide_every == 0
        ):
            try:
                g = next(guide_iter)
            except StopIteration:
                guide_iter = iter(guide_loader)
                g = next(guide_iter)

            gp_ids = g.p_input_ids.to(device)
            gp_msk = g.p_attn_mask.to(device)
            gl_ids = g.l_ids.to(device)

            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, g_aux = model(gp_ids, gp_msk, gl_ids, return_maps=True)
                    loss_contact = compute_atom_res_contact_loss_from_aux(
                        g_aux,
                        g.atom_contact_pairs,
                        contact_topk=int(contact_topk),
                    )
            else:
                _, _, g_aux = model(gp_ids, gp_msk, gl_ids, return_maps=True)
                loss_contact = compute_atom_res_contact_loss_from_aux(
                    g_aux,
                    g.atom_contact_pairs,
                    contact_topk=int(contact_topk),
                )

            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, g_aux = model(gp_ids, gp_msk, gl_ids, return_maps=True)
            else:
                _, _, g_aux = model(gp_ids, gp_msk, gl_ids, return_maps=True)

            pair_map = g_aux["pair_map"]           # (B, Ll, Lp)
            p_pad = g_aux["p_pad"]                # (B, Lp)

            # ligand方向max → residue score
            contact_score = pair_map.amax(dim=1)  # (B, Lp)

            L = min(contact_score.shape[1], g_mask.shape[1])
            contact_score = contact_score[:, :L]
            contact_target = g_mask[:, :L]
            p_valid = (~p_pad[:, :L])

            valid_vals = contact_score[p_valid]
            if valid_vals.numel() > 0:
                mean = valid_vals.mean().detach()
                std = valid_vals.std().clamp_min(1e-6).detach()
                contact_logit = (contact_score - mean) / std

                pos_mask = (contact_target > 0.5) & p_valid

                losses_tmp = []
                for b in range(contact_logit.size(0)):
                    vals = contact_logit[b][pos_mask[b]]
                    if vals.numel() == 0:
                        continue
                    k = min(contact_topk, vals.numel())
                    topk = vals.topk(k).values
                    losses_tmp.append(F.softplus(-topk).mean())

                if losses_tmp:
                    loss_contact = torch.stack(losses_tmp).mean()

        # =========================
        # TOTAL LOSS
        # =========================
        loss = loss + contact_lambda * loss_contact

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        losses_contact.append(loss_contact.item())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            contact=f"{loss_contact.item():.4f}"
        )

    return {
        "loss": float(np.mean(losses)),
        "loss_contact": float(np.mean(losses_contact)),
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

import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        attn_temp=1.0,
        qk_norm=True,
        attn_smooth_eps=0.0,
        attn_activation="softmax",
        detach_attn_for_value=False,
        pair_gate_threshold=0.0,
        topk_frac=0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(n_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(n_heads)])
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_temp = attn_temp
        self.qk_norm = qk_norm
        self.attn_smooth_eps = attn_smooth_eps
        self.attn_activation = attn_activation
        self.detach_attn_for_value = detach_attn_for_value
        self.pair_gate_threshold = float(pair_gate_threshold)
        self.topk_frac = float(topk_frac)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _apply_topk_mask(self, attn):
        if self.topk_frac <= 0.0 or self.topk_frac >= 1.0:
            return attn
        B, H, Lq, Lk = attn.shape
        k = max(1, int(math.ceil(Lk * self.topk_frac)))
        _, topk_idx = torch.topk(attn, k=k, dim=-1)
        keep = torch.zeros_like(attn, dtype=torch.bool)
        keep.scatter_(-1, topk_idx, True)
        return attn * keep.to(attn.dtype)

    def forward(self, q_in, k_in, v_in=None, kv_pad_mask=None, return_maps=False):
        import torch.nn.functional as F
        if v_in is None:
            v_in = k_in

        # each: list of (B, L, Dh)
        q_list = [proj(q_in) for proj in self.q_proj]
        k_list = [proj(k_in) for proj in self.k_proj]
        v_list = [proj(v_in) for proj in self.v_proj]

        # -> (B, H, L, Dh)
        q = torch.stack(q_list, dim=1)
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)

        v = torch.tanh(v)

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.scale / max(self.attn_temp, 1e-6))

        if kv_pad_mask is not None:
            mask = kv_pad_mask[:, None, None, :]
            logits = logits.masked_fill(mask, -1e4)

        if self.attn_activation == "softmax":
            attn = torch.softmax(logits, dim=-1)
        elif self.attn_activation == "sigmoid":
            attn = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported attn_activation: {self.attn_activation}")

        if self.attn_smooth_eps > 0 and self.attn_activation == "softmax":
            Lk = attn.size(-1)
            attn = (1.0 - self.attn_smooth_eps) * attn + self.attn_smooth_eps / Lk

        if self.pair_gate_threshold > 0.0:
            attn = attn * (attn >= self.pair_gate_threshold).to(attn.dtype)

        attn = self._apply_topk_mask(attn)
        attn = self.dropout(attn)

        attn_for_v = attn.detach() if self.detach_attn_for_value else attn

        ctx = torch.matmul(attn_for_v, v)  # (B,H,Lq,Dh)

        # lightweight pair score: (B,H,Lq,Lk), no Dh dimension
        v_norm = v.norm(dim=-1)  # (B,H,Lk)
        pair_score = attn_for_v * v_norm.unsqueeze(-2)
        # ctx = ctx * q
        out = self._merge_heads(ctx)  # (B,Lq,D)
        out = self.out_proj(out)

        if return_maps:
            return out, {
                "qk_logits": logits if return_maps else None,
                "attn_map": attn,
                "v_proj": v if return_maps else None,
                "pair_score": pair_score,
                "ctx": ctx,
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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        attn_temp=1.0,
        qk_norm=True,
        detach_attn_for_value=False,
        attn_smooth_eps=0.0,
        attn_activation="softmax",
        pair_gate_threshold=0.0,
        topk_frac=0.0,
    ):
        super().__init__()

        self.ln_l_q = nn.LayerNorm(d_model)
        self.ln_l_kv = nn.LayerNorm(d_model)
        self.ln_p_q = nn.LayerNorm(d_model)
        self.ln_p_kv = nn.LayerNorm(d_model)

        self.lig_from_prot = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_temp=attn_temp,
            qk_norm=qk_norm,
            detach_attn_for_value=detach_attn_for_value,
            attn_smooth_eps=attn_smooth_eps,
            attn_activation=attn_activation,
            pair_gate_threshold=pair_gate_threshold,
            topk_frac=topk_frac,
        )

        self.prot_from_lig = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_temp=attn_temp,
            qk_norm=qk_norm,
            detach_attn_for_value=detach_attn_for_value,
            attn_smooth_eps=attn_smooth_eps,
            attn_activation=attn_activation,
            pair_gate_threshold=pair_gate_threshold,
            topk_frac=topk_frac,
        )

        self.drop = nn.Dropout(dropout)

        self.ffn_l = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_p = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, p_h, l_h, p_pad=None, l_pad=None, return_maps=False):
        # ligand <- protein
        l_q = self.ln_l_q(l_h)
        p_kv = self.ln_p_kv(p_h)
        l_ctx, aux_lp = self.lig_from_prot(
            q_in=l_q,
            k_in=p_kv,
            v_in=p_kv,
            kv_pad_mask=p_pad,
            return_maps=True,
        )

        # protein <- ligand
        p_q = self.ln_p_q(p_h)
        l_kv = self.ln_l_kv(l_h)
        p_ctx, aux_pl = self.prot_from_lig(
            q_in=p_q,
            k_in=l_kv,
            v_in=l_kv,
            kv_pad_mask=l_pad,
            return_maps=True,
        )

        # residual + FFN
        l_h = l_h + self.drop(l_ctx)
        l_h = l_h + self.drop(self.ffn_l(l_h))

        p_h = p_h + self.drop(p_ctx)
        p_h = p_h + self.drop(self.ffn_p(p_h))

        if return_maps:
            return p_h, l_h, {
                "lp_qk_logits": aux_lp["qk_logits"],   # (B,H,Ll,Lp)
                "lp_attn": aux_lp["attn_map"],
                "lp_ctx": aux_lp["ctx"],               # (B,H,Ll,Dh)
                "lp_v": aux_lp["v_proj"],

                "pl_qk_logits": aux_pl["qk_logits"],   # (B,H,Lp,Ll)
                "pl_attn": aux_pl["attn_map"],
                "pl_ctx": aux_pl["ctx"],               # (B,H,Lp,Dh)
                "lp_pair_score": aux_lp["pair_score"],  # (B,H,Ll,Lp)
                "pl_pair_score": aux_pl["pair_score"],  # (B,H,Lp,Ll)
                "pl_v": aux_pl["v_proj"],
            }

        return p_h, l_h

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
        attn_activation: str = "softmax",
        detach_attn_for_value: bool = False,   # <- 追加
        use_cls_in_head: bool = False,         # <- 追加
        use_reg_head: bool = False,            # <- 追加
        pair_gate_threshold: float = 0.5,
        topk_frac: float = 0.1,
        protein_only: bool = False,   # 追加
        pl_lp_overlap: str = "both"
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder
        self.pl_lp_overlap = pl_lp_overlap
        d_model = self.lig.d_model
        self.d_model = d_model
        self.n_heads = int(n_heads)
        self.protein_token_dropout = float(protein_token_dropout)
        self.ligand_token_dropout = float(ligand_token_dropout)
        self.lig_pad_id = int(self.lig.pad_id)
        self.use_cls_in_head = bool(use_cls_in_head)
        self.use_reg_head = bool(use_reg_head)
        self.protein_only = bool(protein_only)
        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

        self.p_ln = nn.LayerNorm(d_model)
        self.l_ln = nn.LayerNorm(d_model)

        self.interaction = DualStreamBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=dropout,
            attn_temp=attn_temp,
            qk_norm=qk_norm,
            detach_attn_for_value=detach_attn_for_value,
            attn_smooth_eps=attn_smooth_eps,
            attn_activation=attn_activation,
            pair_gate_threshold=pair_gate_threshold,
            topk_frac=topk_frac,
        )
        # pair_map classifier
        self.pair_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.AdaptiveMaxPool2d((8, 8)),
        )

        self.pair_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.reg_head = nn.Linear(128, 1) if self.use_reg_head else None

    def _masked_mean(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        denom = (~pad).sum(dim=1, keepdim=True).clamp(min=1)
        return x.sum(dim=1) / denom

    def _apply_token_dropout(self, pad_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if (not self.training) or drop_prob <= 0.0:
            return pad_mask
        drop_mask = (torch.rand_like(pad_mask.float()) < drop_prob) & (~pad_mask)
        out = pad_mask | drop_mask

        all_dropped = out.all(dim=1)
        if all_dropped.any():
            out = out.clone()
            for b in torch.where(all_dropped)[0]:
                keep_idx = (~pad_mask[b]).nonzero(as_tuple=False)
                if keep_idx.numel() > 0:
                    out[b, keep_idx[0, 0]] = False
        return out

    def _masked_max(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
        x_masked = x.masked_fill(pad.unsqueeze(-1), neg_inf)
        out = x_masked.max(dim=1).values
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
        return out

    def _interaction_only_feat(self, lp_pair_ctx, p_pad, l_pad, topk_frac: float = 0.05):
        """
        lp_pair_ctx: (B, H, Ll, Lp, Dh)
        p_pad:       (B, Lp)  True=PAD
        l_pad:       (B, Ll)  True=PAD
        return:      (B, 2H+2)
        """
        # pairwise strength
        pair_strength = torch.norm(lp_pair_ctx, dim=-1)  # (B, H, Ll, Lp)

        # valid mask
        valid = (~l_pad).unsqueeze(1).unsqueeze(-1) & (~p_pad).unsqueeze(1).unsqueeze(2)  # (B,1,Ll,Lp)

        # invalid を 0 に
        pair_strength_masked = pair_strength.masked_fill(~valid, 0.0)

        # per-head mean
        denom = valid.float().sum(dim=(-2, -1)).clamp_min(1.0)  # (B,1)
        head_mean = pair_strength_masked.sum(dim=(-2, -1)) / denom  # (B,H)

        # per-head max
        neg_inf = torch.tensor(float("-inf"), device=pair_strength.device, dtype=pair_strength.dtype)
        pair_for_max = pair_strength.masked_fill(~valid, neg_inf)
        head_max = pair_for_max.amax(dim=(-2, -1))  # (B,H)
        head_max = torch.where(torch.isinf(head_max), torch.zeros_like(head_max), head_max)

        # global mean/max
        global_mean = head_mean.mean(dim=-1, keepdim=True)  # (B,1)
        global_max = head_max.amax(dim=-1, keepdim=True)  # (B,1)

        # optional: global top-k mean
        if topk_frac is not None and topk_frac > 0:
            B, H, Ll, Lp = pair_strength.shape
            flat = pair_for_max.view(B, H * Ll * Lp)
            valid_flat = valid.expand(-1, H, -1, -1).reshape(B, H * Ll * Lp)

            feats_topk = []
            for b in range(B):
                vals = flat[b][valid_flat[b]]
                if vals.numel() == 0:
                    feats_topk.append(torch.zeros(1, device=flat.device, dtype=flat.dtype))
                else:
                    k = max(1, int(vals.numel() * topk_frac))
                    topv, _ = torch.topk(vals, k=k)
                    feats_topk.append(topv.mean().unsqueeze(0))
            topk_mean = torch.cat(feats_topk, dim=0).unsqueeze(-1)  # (B,1)

            feat = torch.cat([head_mean, head_max, global_mean, global_max, topk_mean], dim=-1)
        else:
            feat = torch.cat([head_mean, head_max, global_mean, global_max], dim=-1)

        return feat

    def forward(self, p_input_ids, p_attn_mask, l_ids, return_maps: bool = False):
        aux = {}

        p_h_raw = self.prot(p_input_ids, p_attn_mask)

        if self.p_proj is not None:
            p_h = self.p_proj(p_h_raw)
        else:
            p_h = p_h_raw

        p_h = self.p_ln(p_h)
        l_h = self.lig(l_ids)
        l_h = self.l_ln(l_h)

        if self.protein_only:
            l_h = torch.zeros_like(l_h)

        p_tok = p_h[:, 1:, :]
        l_tok = l_h[:, 1:, :]

        p_pad = (p_attn_mask == 0)[:, 1:]
        l_pad = (l_ids == self.lig_pad_id)[:, 1:]

        p_pad = self._apply_token_dropout(p_pad, self.protein_token_dropout)
        l_pad = self._apply_token_dropout(l_pad, self.ligand_token_dropout)

        eos_id = getattr(self.prot.esm.config, "eos_token_id", 2)
        p_tok_ids = p_input_ids[:, 1:]
        p_pad = p_pad | (p_tok_ids == eos_id)

        # Option Cでは pair_ctx が必要
        p_ctx, l_ctx, inter_aux = self.interaction(
            p_h=p_tok,
            l_h=l_tok,
            p_pad=p_pad,
            l_pad=l_pad,
            return_maps=True,
        )

        def row_zscore(x, eps=1e-6):
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
            return (x - mu) / sd


        lp_score = inter_aux["lp_pair_score"]  # (B,H,Ll,Lp)
        pl_score = inter_aux["pl_pair_score"].transpose(-1, -2)

        if self.pl_lp_overlap == "both":
            pair_map = torch.minimum(lp_score, pl_score)
        elif self.pl_lp_overlap == "lp":
            pair_map = lp_score
        else:
            pair_map = pl_score
        pair_map = pair_map.mean(dim=1)  # (B,Ll,Lp)
        pair_map = torch.nan_to_num(pair_map, nan=0.0, posinf=0.0, neginf=0.0)

        # valid mask
        valid = (~l_pad).unsqueeze(-1) & (~p_pad).unsqueeze(-2)   # (B,Ll,Lp)

        # zero out invalid area
        pair_map = pair_map.masked_fill(~valid, 0.0)

        # optional area normalization to reduce length bias
        denom = valid.float().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        pair_map = pair_map / denom.sqrt()

        # p_ctx: (B, Lp, D)
        # l_ctx: (B, Ll, D)

        Lp = p_ctx.unsqueeze(1)  # (B,1,Lp,D)
        Ll = l_ctx.unsqueeze(2)  # (B,Ll,1,D)

        pair_feat = torch.cat([
            Ll.expand(-1, -1, p_ctx.size(1), -1),
            Lp.expand(-1, l_ctx.size(1), -1, -1),
            Ll * Lp,
            torch.abs(Ll - Lp),
        ], dim=-1)  # (B,Ll,Lp,4D)

        score = pair_map.unsqueeze(-1)  # (B,Ll,Lp,1)

        weighted = pair_feat * score

        denom = score.sum(dim=(1, 2)).clamp_min(1e-6)
        h_mid = weighted.sum(dim=(1, 2)) / denom  # (B,4D)

        logit = self.cls_head(h_mid).squeeze(-1)

        yhat_reg = None
        if self.reg_head is not None:
            yhat_reg = self.reg_head(h_mid).squeeze(-1)

        aux.update(inter_aux)
        aux["p_pad"] = p_pad
        aux["l_pad"] = l_pad
        aux["p_ctx_tok"] = p_ctx
        aux["l_ctx_tok"] = l_ctx
        aux["pair_map"] = pair_map
        aux["delta_logit"] = logit

        if not return_maps:
            aux = {
                "p_pad": p_pad,
                "l_pad": l_pad,
                "delta_logit": logit,
            }

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
    ap.add_argument("--lig_no_pretrain", action="store_true")
    ap.add_argument("--contact_lambda", type=float, default=0.0)
    ap.add_argument("--use_train_valid_csv", action="store_true")
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--final_eval_csv", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)
    ap.add_argument("--train_size", type=int, default=None)
    ap.add_argument("--ligand_token_dropout", type=float, default=0.10)
    ap.add_argument("--attn_smooth_eps", type=float, default=0.0)
    ap.add_argument("--train_shard_dir", type=str, default=None)
    ap.add_argument("--train_shard_glob", type=str, default="train_part_*.csv")
    ap.add_argument("--train_shard_size", type=int, default=1000)
    ap.add_argument("--train_num_shards_per_epoch", type=int, default=None)
    ap.add_argument("--sym_lambda", type=float, default=0.0)
    ap.add_argument("--lig_ckpt", type=str, required=True)
    ap.add_argument("--vq_ckpt", type=str, default=None)
    ap.add_argument("--protein_only", action="store_true")
    ap.add_argument(
        "--attn_activation",
        type=str,
        default="softmax",
        choices=["softmax", "entmax15", "sigmoid"],
    )
    ap.add_argument(
        "--ligand_input_type",
        type=str,
        default="vqatom",
        choices=["vqatom", "smiles"],
    )
    ap.add_argument("--smiles_vocab_path", type=str, default=None)
    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--finetune_esm", action="store_true")
    ap.add_argument("--finetune_lig", action="store_true")
    ap.add_argument("--lig_debug_index", action="store_true")
    ap.add_argument("--pair_gate_threshold", type=float, default=0.0)
    ap.add_argument("--topk_frac", type=float, default=0.0)
    ap.add_argument("--dti_ckpt", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="./dti_out")
    ap.add_argument("--eval_only", action="store_true")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lig_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detach_attn_for_value", action="store_true")
    ap.add_argument("--use_cls_in_head", action="store_true")
    ap.add_argument("--use_reg_head", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--plateau", action="store_true")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--dual_stream_layers", type=int, default=2)
    ap.add_argument("--pl_lp_overlap", type=str, default="ap", choices=["lp", "pl", "both"])
    ap.add_argument("--select_on", type=str, default="ap", choices=["ap", "auroc", "f1"])
    ap.add_argument("--protein_token_dropout", type=float, default=0.10)
    ap.add_argument("--llrd", action="store_true")
    ap.add_argument("--llrd_decay", type=float, default=0.95)
    ap.add_argument("--esm_lr_mult", type=float, default=1.0)
    ap.add_argument("--esm_min_lr_mult", type=float, default=0.05)
    ap.add_argument("--freeze_esm_bottom", type=int, default=0)
    ap.add_argument("--attn_temp", type=float, default=2.0)
    ap.add_argument("--attn_entropy_lambda", type=float, default=0.0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--y_thr", type=float, default=Y_THR)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--reg_lambda", type=float, default=0.1)
    ap.add_argument("--qk_norm", action="store_true")
    ap.add_argument("--guide_csv", type=str, default=None)
    ap.add_argument("--guide_split", type=str, default="guide_train")
    ap.add_argument("--guide_batch_size", type=int, default=8)
    ap.add_argument("--guide_every", type=int, default=1)
    ap.add_argument("--contact_topk", type=int, default=3)
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
    smiles_tokenizer = None
    if args.ligand_input_type == "smiles":
        if args.smiles_vocab_path is None:
            raise ValueError(
                "--smiles_vocab_path is required when --ligand_input_type smiles"
            )

        from smiles_mlm import SmilesCharTokenizer
        smiles_tokenizer = SmilesCharTokenizer.from_json(args.smiles_vocab_path)

    lig_enc = PretrainedLigandEncoder(
        ckpt_path=args.lig_ckpt,
        device=device,
        finetune=True if args.lig_no_pretrain else args.finetune_lig,
        vq_ckpt_path=args.vq_ckpt,
        verbose_load=True,
        debug_index_check=bool(args.lig_debug_index),
        load_pretrained=not args.lig_no_pretrain,
    )
    prot_enc = ESMProteinEncoder(args.esm_model, device=device, finetune=args.finetune_esm)

    model = DualStreamDTIClassifier(
        protein_encoder=prot_enc,
        ligand_encoder=lig_enc,
        dropout=args.dropout,
        n_heads=args.n_heads,
        n_layers=args.dual_stream_layers,
        protein_token_dropout=args.protein_token_dropout,
        ligand_token_dropout=args.ligand_token_dropout,
        attn_temp=args.attn_temp,
        qk_norm=args.qk_norm,
        attn_smooth_eps=args.attn_smooth_eps,
        attn_activation=args.attn_activation,
        detach_attn_for_value=args.detach_attn_for_value,
        use_cls_in_head=args.use_cls_in_head,
        use_reg_head=args.use_reg_head,
        pair_gate_threshold=args.pair_gate_threshold,  # ← 追加
        topk_frac=args.topk_frac,  # ← 追加
        protein_only=args.protein_only,   # 追加
        pl_lp_overlap=args.pl_lp_overlap,
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
            collate_fn=lambda xs: collate_fn(
                xs,
                esm_tokenizer=esm_tokenizer,
                lig_pad=lig_enc.pad_id,
                lig_cls=lig_enc.cls_id,
            ),
        )

    # -------------------------------------------------
    # Build train dataset FIRST so its y_reg mean/std
    # can be reused for valid/final/test.
    # -------------------------------------------------
    fixed_train_loader = None
    fixed_train_ds = None
    train_shard_paths = None
    train_y_mean = 0.0
    train_y_std = 1.0

    if args.use_train_valid_csv:
        if args.train_csv is None:
            raise ValueError("--use_train_valid_csv requires --train_csv")

        if args.train_size is None:
            fixed_train_ds = DTIDataset(
                csv_path=args.train_csv,
                y_thr=float(args.y_thr),
                drop_missing_y=True,
                lig_cls_id=lig_enc.cls_id,
                ligand_input_type=args.ligand_input_type,
                smiles_tokenizer=smiles_tokenizer,
                smiles_col=args.smiles_col,
            )
        else:
            all_rows = read_csv_rows(args.train_csv)
            train_rows = all_rows[: int(args.train_size)]
            fixed_train_ds = DTIDataset(
                rows=train_rows,
                y_thr=float(args.y_thr),
                drop_missing_y=True,
                lig_cls_id=lig_enc.cls_id,
                ligand_input_type=args.ligand_input_type,
                smiles_tokenizer=smiles_tokenizer,
                smiles_col=args.smiles_col,
            )

        fixed_train_loader = make_loader(fixed_train_ds, shuffle=True)
        train_y_mean = float(fixed_train_ds.y_reg_mean)
        train_y_std = float(fixed_train_ds.y_reg_std)

    else:
        train_shard_paths = list_train_shards(args.train_shard_dir, args.train_shard_glob)

        # For shard mode, estimate regression normalization from train_csv if available.
        if args.train_csv is not None:
            if args.train_size is None:
                tmp_train_ds = DTIDataset(
                    csv_path=args.train_csv,
                    y_thr=float(args.y_thr),
                    drop_missing_y=True,
                    lig_cls_id=lig_enc.cls_id,
                    ligand_input_type=args.ligand_input_type,
                    smiles_tokenizer=smiles_tokenizer,
                    smiles_col=args.smiles_col,
                )
            else:
                all_rows = read_csv_rows(args.train_csv)
                train_rows = all_rows[: int(args.train_size)]
                tmp_train_ds = DTIDataset(
                    rows=train_rows,
                    y_thr=float(args.y_thr),
                    drop_missing_y=True,
                    lig_cls_id=lig_enc.cls_id,
                    ligand_input_type=args.ligand_input_type,
                    smiles_tokenizer=smiles_tokenizer,
                    smiles_col=args.smiles_col,
                )
            train_y_mean = float(tmp_train_ds.y_reg_mean)
            train_y_std = float(tmp_train_ds.y_reg_std)
        else:
            print("[warn] shard mode without train_csv: using default y_reg_mean=0.0, y_reg_std=1.0")

    # -------------------------------------------------
    # Build eval datasets using TRAIN normalization
    # -------------------------------------------------
    valid_ds = DTIDataset(
        csv_path=args.valid_csv,
        y_thr=float(args.y_thr),
        drop_missing_y=True,
        lig_cls_id=lig_enc.cls_id,
        y_reg_mean=train_y_mean,
        y_reg_std=train_y_std,
        ligand_input_type=args.ligand_input_type,
        smiles_tokenizer=smiles_tokenizer,
        smiles_col=args.smiles_col,
    )
    valid_loader = make_loader(valid_ds, shuffle=False)

    final_eval_loader = None
    if args.final_eval_csv:
        final_eval_ds = DTIDataset(
            csv_path=args.final_eval_csv,
            y_thr=float(args.y_thr),
            drop_missing_y=True,
            lig_cls_id=lig_enc.cls_id,
            y_reg_mean=train_y_mean,
            y_reg_std=train_y_std,
            ligand_input_type=args.ligand_input_type,
            smiles_tokenizer=smiles_tokenizer,
            smiles_col=args.smiles_col,
        )
        final_eval_loader = make_loader(final_eval_ds, shuffle=False)

    guide_loader = None
    if args.guide_csv is not None and float(args.contact_lambda) > 0.0:
        guide_rows = read_csv_rows_filter_split(args.guide_csv, args.guide_split)
        if len(guide_rows) > 1000:
            rng = random.Random(args.seed)
            guide_rows = rng.sample(guide_rows, 1000)
        guide_ds = DTIDataset(
            rows=guide_rows,
            y_thr=float(args.y_thr),
            drop_missing_y=True,
            lig_cls_id=lig_enc.cls_id,
            y_reg_mean=train_y_mean,
            y_reg_std=train_y_std,
            ligand_input_type=args.ligand_input_type,
            smiles_tokenizer=smiles_tokenizer,
            smiles_col=args.smiles_col,
        )

        guide_loader = DataLoader(
            guide_ds,
            batch_size=int(args.guide_batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda xs: collate_fn(
                xs,
                esm_tokenizer=esm_tokenizer,
                lig_pad=lig_enc.pad_id,
                lig_cls=lig_enc.cls_id,
            ),
        )

        print(f"[guide] loaded {len(guide_ds)} rows from {args.guide_csv} split={args.guide_split}")

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(
            csv_path=args.test_csv,
            y_thr=float(args.y_thr),
            drop_missing_y=True,
            lig_cls_id=lig_enc.cls_id,
            y_reg_mean=train_y_mean,
            y_reg_std=train_y_std,
            ligand_input_type=args.ligand_input_type,
            smiles_tokenizer=smiles_tokenizer,
            smiles_col=args.smiles_col,
        )
        test_loader = make_loader(test_ds, shuffle=False)

    optimizer = build_optimizer_with_llrd(model, args) if not args.eval_only else None
    scheduler = None
    if optimizer is not None:
        if args.use_train_valid_csv:
            steps_per_epoch = len(fixed_train_loader)
        else:
            steps_per_epoch = math.ceil(
                int(args.train_num_shards_per_epoch or math.ceil(args.train_size / args.train_shard_size))
                * int(args.train_shard_size)
                / int(args.batch_size)
            )

        total_steps = int(steps_per_epoch * args.epochs)
        warmup_steps = (
            int(args.warmup_steps)
            if args.warmup_steps is not None
            else int(total_steps * float(args.warmup_ratio))
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"[scheduler] linear warmup: warmup_steps={warmup_steps} total_steps={total_steps}")

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

        save_json(
            os.path.join(args.out_dir, "eval_only.json"),
            {
                "mode": "eval_only",
                "valid": v_m,
                "valid_reg": v_r,
                "test": test_m,
                "test_reg": test_r,
                "valid_csv": args.valid_csv,
                "test_csv": args.test_csv,
                "dti_ckpt": args.dti_ckpt,
                "train_y_reg_mean": train_y_mean,
                "train_y_reg_std": train_y_std,
            },
        )
        return

    best = {"ap": -1e9, "auroc": -1e9, "f1": -1e9, "epoch": -1}

    for ep in range(1, args.epochs + 1):
        qk_save_dir = os.path.join(args.out_dir, "qk_maps")

        if args.use_train_valid_csv:
            epoch_train_ds = fixed_train_ds
            train_loader = fixed_train_loader
        else:
            # Minimal shard-mode support
            epoch_shards = pick_epoch_shards_random(
                shard_paths=train_shard_paths,
                train_size=args.train_size,
                shard_size=args.train_shard_size,
                epoch=ep,
                seed=args.seed,
                num_shards_per_epoch=args.train_num_shards_per_epoch,
            )
            ds_list = [
                DTIDataset(
                    csv_path=p,
                    y_thr=float(args.y_thr),
                    drop_missing_y=True,
                    lig_cls_id=lig_enc.cls_id,
                    y_reg_mean=train_y_mean,
                    y_reg_std=train_y_std,
                    ligand_input_type=args.ligand_input_type,
                    smiles_tokenizer=smiles_tokenizer,
                    smiles_col=args.smiles_col,
                )
                for p in epoch_shards
            ]
            epoch_train_ds = ConcatDataset(ds_list)
            train_loader = make_loader(epoch_train_ds, shuffle=True)

        print("epoch train n:", len(epoch_train_ds))
        pos_weight = compute_pos_weight_from_dataset(epoch_train_ds)
        print(f"pos_weight={pos_weight:.4f}")

        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            pos_weight=pos_weight,
            grad_clip=float(args.grad_clip),
            attn_entropy_lambda=float(args.attn_entropy_lambda),
            reg_lambda=float(args.reg_lambda),
            sym_lambda=float(args.sym_lambda),
            contact_lambda=float(args.contact_lambda),
            guide_loader=guide_loader,
            guide_every=int(args.guide_every),
            contact_topk=int(args.contact_topk),
            epoch=ep,
        )

        yhat_tr, yb_tr, yhatr_tr, yr_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(yhat_tr, yb_tr)
        tr_r = eval_reg_metrics(yhatr_tr, yr_tr)

        yhat_v, yb_v, yhatr_v, yr_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)
        v_r = eval_reg_metrics(yhatr_v, yr_v)

        if float(v_m[args.select_on]) > float(best[args.select_on]):
            best.update(
                {
                    "ap": float(v_m["ap"]),
                    "auroc": float(v_m["auroc"]),
                    "f1": float(v_m["f1"]),
                    "epoch": ep,
                }
            )
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        save_dti_checkpoint(
            os.path.join(args.out_dir, "last.pt"),
            model,
            args,
            lig_enc,
            epoch=ep,
            best=best,
        )

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

        save_json(
            os.path.join(args.out_dir, f"epoch_{ep:03d}.json"),
            {
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
                "train_y_reg_mean": train_y_mean,
                "train_y_reg_std": train_y_std,
            },
        )

        print(f"\n===== Epoch {ep} =====")
        print(
            "[train]",
            f"AUC={tr_m['auroc']:.4f}",
            f"AP={tr_m['ap']:.4f}",
            f"F1={tr_m['f1']:.4f}",
            f"RMSE={tr_r['rmse']:.4f}",
            f"SP={tr_r['spearman']:.4f}",
        )
        print(
            "[valid]",
            f"AUC={v_m['auroc']:.4f}",
            f"AP={v_m['ap']:.4f}",
            f"F1={v_m['f1']:.4f}",
            f"RMSE={v_r['rmse']:.4f}",
            f"SP={v_r['spearman']:.4f}",
        )
        if final_m is not None:
            print(
                "[final]",
                f"AUC={final_m['auroc']:.4f}",
                f"AP={final_m['ap']:.4f}",
                f"F1={final_m['f1']:.4f}",
                f"EF1={final_m['ef1']:.3f}",
                f"EF5={final_m['ef5']:.3f}",
                f"EF10={final_m['ef10']:.3f}",
            )

        def find_pos_neg_indices(batch):
            y = batch.y_bin.detach().cpu().numpy()

            pos_idx = None
            neg_idx = None

            for i, v in enumerate(y):
                if v >= 0.5 and pos_idx is None:
                    pos_idx = i
                if v < 0.5 and neg_idx is None:
                    neg_idx = i
                if (pos_idx is not None) and (neg_idx is not None):
                    break

            return pos_idx, neg_idx

        # -----------------------------
        # epoch loop 内
        # -----------------------------
        qk_save_dir = os.path.join(args.out_dir, "qk_maps")

        # valid loader の最初の batch を取得
        vis_batch = next(iter(valid_loader))

        pos_idx, neg_idx = find_pos_neg_indices(vis_batch)

        targets = []

        if pos_idx is not None:
            targets.append(("pos", pos_idx))

        if neg_idx is not None:
            targets.append(("neg", neg_idx))

        for tag, sidx in targets:
            visualize_one_qk_map(
                model=model,
                loader=valid_loader,
                device=device,
                esm_tokenizer=esm_tokenizer,
                sample_idx_in_batch=sidx,
                show_token_labels=False,
                save_dir=qk_save_dir,
                prefix=f"epoch{ep:03d}_{tag}_sample{sidx}",
            )

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)

if __name__ == "__main__":
    main()
