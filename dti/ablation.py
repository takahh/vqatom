#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end DTI classification (protein sequence only + ligand VQ-Atom tokens)

- Input: protein sequence (ESM2) + ligand token ids (your VQ-Atom ids)
- Task: classify strong vs weak among *binders only*:
    y_bin = 1 if y >= Y_THR else 0
  (rows with label==0 are dropped)

CSV columns (required):
  - seq      : protein AA sequence (string)
  - lig_tok  : ligand token ids as space-separated ints (string)
  - label    : 0/1 binder flag (float/int)
  - y        : affinity/score used to define strong/weak (float)

No distance, no KL, no D.

Outputs:
  model.forward(...) -> (logit (B,), aux={})
"""
from glob import glob
from torch.utils.data import ConcatDataset
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
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, EsmModel

Y_THR = 7.0  # strong if y >= Y_THR


# -----------------------------
# Repro
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_csv_random_rows(path: str, n_rows: int, seed: int) -> List[Dict[str, str]]:
    import csv

    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

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

def compute_valid_size_from_train_size(train_size: int, val_ratio: float) -> int:
    if train_size is None or train_size <= 0:
        raise ValueError("train_size must be positive")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    return int(round(float(train_size) * float(val_ratio)))

def read_csv_head_rows(path: str, n_rows: int) -> List[Dict[str, str]]:
    import csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= n_rows:
                break
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows

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
        chosen = rng.sample(shard_paths, num_shards)
    else:
        chosen = [rng.choice(shard_paths) for _ in range(num_shards)]

    return chosen


def build_train_dataset_from_shards(
    shard_paths: List[str],
    y_thr: float,
):
    ds_list = []
    total_rows = 0

    for p in shard_paths:
        ds = DTIDataset(
            p,
            y_thr=float(y_thr),
            drop_missing_y=True,
        )
        ds_list.append(ds)
        total_rows += len(ds)

    if not ds_list:
        raise ValueError("No train shards were loaded.")

    return ConcatDataset(ds_list)

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
# Ligand token parser
# -----------------------------
def parse_lig_tokens(s: str) -> List[int]:
    s = str(s).strip().replace(",", " ")
    if not s:
        return []
    return [int(x) for x in s.split()]


@dataclass
class Batch:
    p_input_ids: torch.Tensor   # (B, Lp)
    p_attn_mask: torch.Tensor   # (B, Lp)
    l_ids: torch.Tensor         # (B, Ll)
    y_bin: torch.Tensor         # (B,)
    y: torch.Tensor


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not seqs:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(x.numel()) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        if x.numel() > 0:
            out[i, : x.numel()] = x
    return out



def collate_fn(samples, esm_tokenizer, lig_pad: int, lig_cls: int) -> Batch:
    import time, random

    t0 = time.perf_counter()

    seqs = [s["seq"] for s in samples]

    t1 = time.perf_counter()
    enc = esm_tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    t_tok = time.perf_counter() - t1

    t1 = time.perf_counter()
    p_input_ids = enc["input_ids"].long()
    p_attn_mask = enc["attention_mask"].long()

    l_ids_list = []
    for s in samples:
        x = torch.tensor(parse_lig_tokens(s["lig_tok"]), dtype=torch.long)
        if x.numel() == 0:
            x = torch.tensor([lig_cls], dtype=torch.long)
        else:
            x = torch.cat([torch.tensor([lig_cls], dtype=torch.long), x], dim=0)
        l_ids_list.append(x)

    l_ids = pad_1d(l_ids_list, lig_pad)
    y_bin = torch.tensor([s["y_bin"] for s in samples], dtype=torch.float32)
    y = torch.tensor([s["y"] for s in samples], dtype=torch.float32)
    t_rest = time.perf_counter() - t1

    total = time.perf_counter() - t0

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y_bin=y_bin,
        y=y,
    )

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
        # PretrainedLigandEncoder.__init__ の vocab meta 部分の後ろに追加

        # 既存: vocab_size includes PAD+MASK の想定
        # ここに CLS を追加する
        self.cls_id = int(self.vocab_size)  # new id at end
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

        # overlap-copy embedding rows if present
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


class TinyFFNBlock(nn.Module):
    """Transformer-style FFN residual block (PreNorm)."""
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return h   # ★ residual を戻す

class BiasedCrossAttention(nn.Module):
    """Cross-attention using SDPA (scaled_dot_product_attention)."""
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,L,D) -> (B,H,L,Dh)
        B, L, D = x.shape
        return x.view(B, L, self.nhead, self.d_head).transpose(1, 2).contiguous()

    def forward(
        self,
        q: torch.Tensor,                   # (B, Lq, D)
        k: torch.Tensor,                   # (B, Lk, D)
        v: torch.Tensor,                   # (B, Lk, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Lk) True=ignore keys
        logits_bias: Optional[torch.Tensor] = None,        # (B, Lq, Lk) additive bias
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        device = q.device

        qh = self._split_heads(self.q_proj(q))  # (B,H,Lq,Dh)
        kh = self._split_heads(self.k_proj(k))  # (B,H,Lk,Dh)
        vh = self._split_heads(self.v_proj(v))  # (B,H,Lk,Dh)

        attn_mask = None
        if logits_bias is not None:
            attn_mask = logits_bias.to(device=device, dtype=qh.dtype).unsqueeze(1)  # (B,1,Lq,Lk)

        if key_padding_mask is not None:
            kpm2 = key_padding_mask.to(device=device).unsqueeze(1).unsqueeze(1)  # bool
            if attn_mask is None:
                # make float mask
                attn_mask = torch.zeros((B, 1, Lq, kh.size(2)), device=device, dtype=qh.dtype)
            attn_mask = attn_mask.masked_fill(kpm2, torch.finfo(qh.dtype).min)

        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,Dh)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(out)

class DTIDataset(Dataset):
    def __init__(self, csv_path: str = None, rows: list = None, y_thr: float = 7.0, drop_missing_y: bool = True):
        if rows is not None:
            raw_rows = rows
        elif csv_path is not None:
            raw_rows = read_csv_rows(csv_path)
        else:
            raise ValueError("Either csv_path or rows must be provided")

        self.rows = []
        n_all = 0
        n_drop_seq = 0
        n_drop_lig = 0
        n_drop_y = 0

        for r in raw_rows:
            n_all += 1

            seq = (r.get("seq") or "").strip()
            lig_tok = (r.get("lig_tok") or "").strip()
            y_raw = r.get("y", "")

            if not seq:
                n_drop_seq += 1
                continue
            if not lig_tok:
                n_drop_lig += 1
                continue
            if y_raw in ("", None):
                if drop_missing_y:
                    n_drop_y += 1
                    continue
                y = float("nan")
            else:
                y = float(y_raw)

            rr = dict(r)
            rr["y"] = y
            rr["y_bin"] = 1 if y >= float(y_thr) else 0
            self.rows.append(rr)

        if not self.rows:
            raise ValueError("No usable rows in dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

class QKOnlyDTIClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        dropout: float,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model

        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # self.reg_head = nn.Linear(2 * d_model, 1)
        self.lig_pad_id = int(self.lig.pad_id)

    def forward(self, p_input_ids, p_attn_mask, l_ids):
        aux = {}

        p_h = self.prot(p_input_ids, p_attn_mask)   # (B,Lp,Hp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)                  # (B,Lp,D)

        l_h = self.lig(l_ids)                       # (B,Ll,D)

        prot_pad_mask = (p_attn_mask == 0)
        lig_pad_mask = (l_ids == self.lig_pad_id)

        p_tok = p_h[:, 1:, :]
        l_tok = l_h[:, 1:, :]
        p_cls = p_h[:, 0, :]
        l_cls = l_h[:, 0, :]
        p_pad = prot_pad_mask[:, 1:]
        l_pad = lig_pad_mask[:, 1:]

        # fallback
        if p_tok.size(1) == 0 or l_tok.size(1) == 0:
            p_cls = p_h[:, 0, :]
            l_cls = l_h[:, 0, :]
            z = torch.cat([p_cls, l_cls], dim=-1)

            logit = self.head(z).squeeze(-1)
            aux["y_hat"] = self.reg_head(z).squeeze(-1)
            return logit, aux

        q = self.q_proj(p_tok)                      # (B,Lp-1,D)
        k = self.k_proj(l_tok)                      # (B,Ll-1,D)

        S = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))  # (B,Lp-1,Ll-1)

        # mask ligand PAD columns only with large negative value
        S = S.masked_fill(l_pad.unsqueeze(1), -1e9)

        # protein PAD rows must NOT become all -inf before softmax
        # set them to 0 first, then zero them out after softmax
        S = S.masked_fill(p_pad.unsqueeze(-1), 0.0)

        # if ligand side is all-pad for a sample, keep whole matrix zero
        bad = (p_pad.all(dim=1) | l_pad.all(dim=1)).view(-1, 1, 1)
        S = torch.where(bad, torch.zeros_like(S), S)

        A = torch.softmax(S, dim=-1)  # (B,Lp-1,Ll-1)

        # zero out invalid positions AFTER softmax
        A = A.masked_fill(p_pad.unsqueeze(-1), 0.0)
        A = A.masked_fill(l_pad.unsqueeze(1), 0.0)

        # token importance from map
        p_imp = A.max(dim=-1).values   # (B,Lp-1)
        l_imp = A.max(dim=1).values    # (B,Ll-1)
        # p_imp = 0.5 * A.max(dim=-1).values + 0.5 * A.mean(dim=-1)
        # l_imp = 0.5 * A.max(dim=1).values + 0.5 * A.mean(dim=1)

        p_imp = p_imp.masked_fill(p_pad, 0.0)
        l_imp = l_imp.masked_fill(l_pad, 0.0)

        p_imp = p_imp / p_imp.sum(dim=1, keepdim=True).clamp(min=1e-6)
        l_imp = l_imp / l_imp.sum(dim=1, keepdim=True).clamp(min=1e-6)

        p_sum = torch.bmm(p_imp.unsqueeze(1), p_tok).squeeze(1)  # (B,D)
        l_sum = torch.bmm(l_imp.unsqueeze(1), l_tok).squeeze(1)  # (B,D)
        z = torch.cat([p_cls, l_cls, p_sum, l_sum], dim=-1)

        # z = torch.cat([p_mix, l_mix], dim=-1)  # (B, 2D)

        logit = self.head(z).squeeze(-1)
        aux["y_hat"] = self.reg_head(z).squeeze(-1)

        aux["p_cls_norm"] = p_cls.norm(dim=-1).detach()
        aux["l_cls_norm"] = l_cls.norm(dim=-1).detach()
        aux["p_sum_norm"] = p_sum.norm(dim=-1).detach()
        aux["l_sum_norm"] = l_sum.norm(dim=-1).detach()

        return logit, aux

# -----------------------------
# Eval / Train
# -----------------------------
def predict(model, loader, device):
    model.eval()
    logits, ybins = [], []
    use_amp = (device.type == "cuda")

    with torch.inference_mode():
        for batch in loader:
            p_ids = batch.p_input_ids.to(device, non_blocking=True)
            p_msk = batch.p_attn_mask.to(device, non_blocking=True)
            l_ids = batch.l_ids.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit, _ = model(p_ids, p_msk, l_ids)
            else:
                logit, _ = model(p_ids, p_msk, l_ids)

            logits.append(logit.detach().float().cpu().numpy())
            ybins.append(batch.y_bin.detach().cpu().numpy())

    logit = np.concatenate(logits, axis=0) if logits else np.array([], dtype=np.float64)
    yb = np.concatenate(ybins, axis=0) if ybins else np.array([], dtype=np.float64)
    return logit, yb

def eval_metrics(logit: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    - Computes AUROC / AP
    - Finds best-F1 threshold on a grid
    - Computes EF at 1%, 5%, 10%
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    def enrichment_factor(prob: np.ndarray, y01: np.ndarray, frac: float) -> float:
        n = int(len(y01))
        if n == 0:
            return 0.0

        n_pos = int(y01.sum())
        if n_pos == 0:
            return 0.0

        k = max(1, int(math.ceil(n * float(frac))))
        order = np.argsort(-prob)   # descending
        top_idx = order[:k]
        hits_topk = int(y01[top_idx].sum())

        hit_rate_topk = hits_topk / float(k)
        base_rate = n_pos / float(n)
        return float(hit_rate_topk / base_rate) if base_rate > 0 else 0.0

    if logit.size == 0:
        return {
            "auroc": 0.0, "ap": 0.0,
            "f1": 0.0, "thr": 0.5,
            "prob_mean": 0.0, "prob_std": 0.0,
            "ef1": 0.0, "ef5": 0.0, "ef10": 0.0,
        }

    prob = 1.0 / (1.0 + np.exp(-logit))
    if not np.isfinite(prob).all():
        nbad = np.sum(~np.isfinite(prob))
        print(f"[warn] prob has non-finite: {nbad}/{prob.size}; applying nan_to_num")
        prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)

    y01 = (y > 0.5).astype(np.int32)

    ef1 = enrichment_factor(prob, y01, 0.01)
    ef5 = enrichment_factor(prob, y01, 0.05)
    ef10 = enrichment_factor(prob, y01, 0.10)

    if len(np.unique(y01)) <= 1:
        return {
            "auroc": 0.0, "ap": 0.0,
            "f1": 0.0, "thr": 0.5,
            "prob_mean": float(prob.mean()),
            "prob_std": float(prob.std()),
            "prob_min": float(prob.min()),
            "prob_max": float(prob.max()),
            "pos_rate": float(y01.mean()) if y01.size else 0.0,
            "pred_pos_rate@thr": 0.0,
            "ef1": float(ef1),
            "ef5": float(ef5),
            "ef10": float(ef10),
        }

    auroc = roc_auc_score(y01, prob)
    ap = average_precision_score(y01, prob)

    thrs = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        np.array([0.5], dtype=np.float64),
    ]))

    best_f1 = -1.0
    best_thr = 0.5
    for t in thrs:
        pred01 = (prob >= float(t)).astype(np.int32)
        f1 = f1_score(y01, pred01)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(t)

    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "f1": float(best_f1),
        "thr": float(best_thr),
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
        "pos_rate": float(y01.mean()),
        "pred_pos_rate@thr": float((prob >= best_thr).mean()),
        "ef1": float(ef1),
        "ef5": float(ef5),
        "ef10": float(ef10),
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    reg_lambda: float = 0.0,
    log_interval: int = 50,
) -> Dict[str, float]:
    import time

    model.train()
    losses, losses_cls, losses_reg = [], [], []
    use_amp = (device.type == "cuda")

    t_data = 0.0
    t_h2d = 0.0
    t_fwd = 0.0
    t_bwd = 0.0
    t_opt = 0.0
    n_steps = 0

    it = iter(loader)

    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t_data += time.perf_counter() - t0

        t0 = time.perf_counter()
        p_ids = batch.p_input_ids.to(device, non_blocking=True)
        p_msk = batch.p_attn_mask.to(device, non_blocking=True)
        l_ids = batch.l_ids.to(device, non_blocking=True)
        y_bin = batch.y_bin.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_h2d += time.perf_counter() - t0

        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, aux = model(p_ids, p_msk, l_ids)
                loss_cls = F.binary_cross_entropy_with_logits(logit, y_bin)
                y_hat = aux["y_hat"]
                loss_reg = F.smooth_l1_loss(y_hat, y)
                loss = loss_cls + reg_lambda * loss_reg
        else:
            logit, aux = model(p_ids, p_msk, l_ids)
            loss_cls = F.binary_cross_entropy_with_logits(logit, y_bin)
            y_hat = aux["y_hat"]
            loss_reg = F.smooth_l1_loss(y_hat, y)
            loss = loss_cls + reg_lambda * loss_reg
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        if grad_clip and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt += time.perf_counter() - t0

        losses.append(float(loss.detach().cpu().item()))
        losses_cls.append(float(loss_cls.detach().cpu().item()))
        losses_reg.append(float(loss_reg.detach().cpu().item()))
        n_steps += 1

        # if n_steps % log_interval == 0:
        #     print(
        #         f"[train step {n_steps}] "
        #         f"data={t_data/n_steps:.3f}s "
        #         f"h2d={t_h2d/n_steps:.3f}s "
        #         f"fwd={t_fwd/n_steps:.3f}s "
        #         f"bwd={t_bwd/n_steps:.3f}s "
        #         f"opt={t_opt/n_steps:.3f}s"
        #     )

    # print(
    #     f"[train epoch timing] "
    #     f"data={t_data:.2f}s "
    #     f"h2d={t_h2d:.2f}s "
    #     f"fwd={t_fwd:.2f}s "
    #     f"bwd={t_bwd:.2f}s "
    #     f"opt={t_opt:.2f}s "
    #     f"steps={n_steps}"
    # )

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "loss_cls": float(sum(losses_cls) / max(1, len(losses_cls))),
        "loss_reg": float(sum(losses_reg) / max(1, len(losses_reg))),
    }

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dti_checkpoint(path: str, model: nn.Module, args: argparse.Namespace,
                        lig_enc: PretrainedLigandEncoder, epoch: int, best: dict):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "model_type": getattr(args, "model_type", "cross"),
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


# -----------------------------
# Optimizer with LLRD (ESM)
# -----------------------------
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

    # freeze bottom N layers
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

            # embeddings group
            if hasattr(esm, "embeddings"):
                emb_params = [p for p in esm.embeddings.parameters() if p.requires_grad]
                if emb_params:
                    lr_emb = max(top_lr * (decay ** n_layers), top_lr * min_mult)
                    param_groups.append({"params": emb_params, "lr": lr_emb, "name": f"esm.emb lr={lr_emb:g}"})

            # encoder layers bottom->top
            for i, layer in enumerate(layers):
                layer_params = [p for p in layer.parameters() if p.requires_grad]
                if not layer_params:
                    continue
                depth_from_top = (n_layers - 1) - i
                lr_i = top_lr * (decay ** depth_from_top)
                lr_i = max(lr_i, top_lr * min_mult)
                param_groups.append({"params": layer_params, "lr": lr_i, "name": f"esm.layer{i} lr={lr_i:g}"})

            # other params
            other = []
            for n, p in esm.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("embeddings."):
                    continue
                if n.startswith("encoder.layer."):
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
        print(f"  - {g.get('name','group')}: lr={g['lr']:.3g} params={len(g['params'])} numel={nparam}")

    return torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

def rows_to_dataset(rows, esm_tokenizer, y_thr: float = 7.0):
    samples = []
    dropped_no_y = 0
    dropped_bad = 0

    for r in rows:
        try:
            seq = str(r["seq"]).strip()
            lig = str(r["lig_tok"]).strip()
        except Exception:
            dropped_bad += 1
            continue

        y_str = str(r.get("y", "")).strip()
        if y_str == "":
            dropped_no_y += 1
            continue

        y_val = float(y_str)
        y_bin = 1.0 if y_val >= float(y_thr) else 0.0
        pdbid = str(r.get("pdbid", "")).strip()

        enc = esm_tokenizer(seq, return_tensors="pt", truncation=True)
        p_input_ids = enc["input_ids"][0].long()
        p_attn_mask = enc["attention_mask"][0].long()
        l_ids = torch.tensor(parse_lig_tokens(lig), dtype=torch.long)

        samples.append({
            "p_input_ids": p_input_ids,
            "p_attn_mask": p_attn_mask,
            "l_ids": l_ids,
            "y_bin": torch.tensor(float(y_bin), dtype=torch.float32),
            "y": torch.tensor(float(y_val), dtype=torch.float32),
            "pdbid": pdbid,
        })

    ds = DTIDataset.__new__(DTIDataset)
    ds.samples = samples
    print(f"[rows_to_dataset] kept={len(samples)} dropped_no_y={dropped_no_y} dropped_bad={dropped_bad}")
    return ds

def split_rows_train_valid(
    rows,
    val_ratio: float = 0.15,
    seed: int = 0,
    stratified: bool = False,
):
    rng = random.Random(seed)
    val_ratio = max(0.10, min(0.20, float(val_ratio)))

    if not stratified:
        rows = rows[:]
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * val_ratio)))
        n_val = min(n_val, max(1, len(rows) - 1))
        valid_rows = rows[:n_val]
        train_rows = rows[n_val:]
        return train_rows, valid_rows

    pos_rows = []
    neg_rows = []
    bad_rows = []

    for r in rows:
        try:
            y = float(str(r.get("y", "")).strip())
            y_bin = 1 if y >= Y_THR else 0
            if y_bin == 1:
                pos_rows.append(r)
            else:
                neg_rows.append(r)
        except Exception:
            bad_rows.append(r)

    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)

    n_val_pos = max(1, int(round(len(pos_rows) * val_ratio))) if len(pos_rows) > 1 else 0
    n_val_neg = max(1, int(round(len(neg_rows) * val_ratio))) if len(neg_rows) > 1 else 0

    valid_rows = pos_rows[:n_val_pos] + neg_rows[:n_val_neg]
    train_rows = pos_rows[n_val_pos:] + neg_rows[n_val_neg:]

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def downsample_rows(rows, max_samples: Optional[int], seed: int = 0, stratified: bool = False):
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)

    if not stratified:
        idx = rng.sample(range(len(rows)), max_samples)
        return [rows[i] for i in idx]

    pos_rows = []
    neg_rows = []
    for r in rows:
        try:
            y = float(str(r.get("y", "")).strip())
            y_bin = 1 if y >= Y_THR else 0
            if y_bin == 1:
                pos_rows.append(r)
            else:
                neg_rows.append(r)
        except Exception:
            continue

    n_total = len(pos_rows) + len(neg_rows)
    if n_total == 0:
        return []

    target_pos = int(round(max_samples * (len(pos_rows) / n_total)))
    target_neg = max_samples - target_pos
    target_pos = min(target_pos, len(pos_rows))
    target_neg = min(target_neg, len(neg_rows))

    cur = target_pos + target_neg
    if cur < max_samples:
        remain = max_samples - cur
        extra_pos = min(remain, len(pos_rows) - target_pos)
        target_pos += extra_pos
        remain -= extra_pos
        extra_neg = min(remain, len(neg_rows) - target_neg)
        target_neg += extra_neg

    keep = rng.sample(pos_rows, target_pos) + rng.sample(neg_rows, target_neg)
    rng.shuffle(keep)
    return keep

import time

def tsec():
    return time.perf_counter()

def tprint(name: str, t0: float):
    dt = time.perf_counter() - t0
    return dt

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_shard_size", type=int, default=1000,
                    help="Rows per train shard CSV")
    # data
    ap.add_argument(
        "--use_train_valid_csv",
        action="store_true",
        help="Use train_csv/valid_csv directly instead of shard-based training. With --train_size, resample train rows each epoch; otherwise use all train_csv rows."
    )
    ap.add_argument("--train_csv", type=str, default=None, help="Base train CSV (required unless --eval_only)")
    ap.add_argument("--valid_csv", type=str, required=True, help="Validation CSV (all rows will be used)")
    ap.add_argument("--final_eval_csv", type=str, default=None, help="Final evaluation CSV run every epoch")
    ap.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="If set with --use_train_valid_csv, sample this many training rows per epoch. If unset, use all rows in train_csv."
    )
    ap.add_argument("--split_seed", type=int, default=0,
                    help="Seed for train/val splitting")
    ap.add_argument("--stratified_split", action="store_true",
                    help="Preserve class balance when splitting train/val")
    ap.add_argument("--test_csv", type=str, default=None, help="Optional extra test CSV")
    ap.add_argument("--y_thr", type=float, default=Y_THR, help="Strong threshold: y>=y_thr => y_bin=1 (binders only)")
    ap.add_argument("--cross_layers", type=int, default=1)
    ap.add_argument("--train_max_samples", type=int, default=None,
                    help="If set, randomly subsample train dataset to this many samples")
    ap.add_argument("--train_stratified", action="store_true",
                    help="Use stratified subsampling for train set")
    # ligand MLM weights ckpt
    ap.add_argument("--lig_ckpt", type=str, required=True, help="Ligand MLM checkpoint")

    # discretization ckpt (vocab meta source)
    ap.add_argument("--vq_ckpt", type=str, default=None, help="Discretization ckpt to source vocab meta (has global_id_meta).")

    # protein ESM-2
    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--finetune_esm", action="store_true")

    # DTI ckpt
    ap.add_argument("--dti_ckpt", type=str, default=None)

    # io
    ap.add_argument("--out_dir", type=str, default="./dti_out")
    ap.add_argument("--eval_only", action="store_true")

    # train
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lig_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)

    # cross attn
    ap.add_argument("--cross_nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    # finetune ligand
    ap.add_argument("--finetune_lig", action="store_true")
    ap.add_argument("--lig_debug_index", action="store_true")

    # loss
    ap.add_argument("--grad_clip", type=float, default=1.0)
    from glob import glob
    from torch.utils.data import ConcatDataset
    ap.add_argument("--train_shard_dir", type=str, default=None,
                    help="Directory containing sharded train CSVs, e.g. train_part_*.csv")
    ap.add_argument("--train_shard_glob", type=str, default="train_part_*.csv",
                    help="Glob pattern inside --train_shard_dir")
    ap.add_argument("--train_num_shards_per_epoch", type=int, default=None)
    ap.add_argument("--train_shuffle_shards", action="store_true",
                    help="Shuffle shard order every epoch")
    # scheduler
    ap.add_argument("--plateau", action="store_true")
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # selection criterion
    ap.add_argument("--select_on", type=str, default="ap", choices=["ap", "auroc", "f1"])

    # ESM LLRD
    ap.add_argument("--llrd", action="store_true")
    ap.add_argument("--llrd_decay", type=float, default=0.95)
    ap.add_argument("--esm_lr_mult", type=float, default=1.0)
    ap.add_argument("--esm_min_lr_mult", type=float, default=0.05)
    ap.add_argument("--freeze_esm_bottom", type=int, default=0)
    ap.add_argument("--model_type", type=str, default="qkonly",
                    choices=["qkonly"])
    args = ap.parse_args()
    import time

    if not args.eval_only and not args.train_csv:
        raise ValueError("--train_csv is required unless --eval_only is set.")

    if not args.eval_only and (not args.valid_csv):
        raise ValueError("Provide --valid_csv, or use --auto_split_val.")

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
    lig_pad = lig_enc.pad_id
    assert lig_enc.cls_id != lig_enc.pad_id
    assert lig_enc.cls_id != lig_enc.mask_id

    # -----------------------------
    # datasets
    # -----------------------------
    train_ds = None
    valid_ds = None
    train_shard_paths = None

    if args.eval_only:
        if not args.valid_csv:
            raise ValueError("--eval_only requires --valid_csv")
        valid_ds = DTIDataset(
            args.valid_csv,
            y_thr=float(args.y_thr),
            drop_missing_y=True,
        )
    else:
        if not args.valid_csv:
            raise ValueError("Provide --valid_csv")

        valid_ds = DTIDataset(
            args.valid_csv,
            y_thr=float(args.y_thr),
            drop_missing_y=True,
        )

        if args.use_train_valid_csv:
            # full-train fixed mode: build once outside epoch loop
            if args.train_size is None:
                train_ds = DTIDataset(
                    args.train_csv,
                    y_thr=float(args.y_thr),
                    drop_missing_y=True,
                )
        else:
            if args.train_shard_dir:
                train_shard_paths = list_train_shards(args.train_shard_dir, args.train_shard_glob)
            else:
                raise ValueError("This mode requires --train_shard_dir")

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
    )

    train_loader = None
    # loader_num_workers = min(3, os.cpu_count() or 1)
    # pin_memory = (device.type == "cuda")
    loader_num_workers = 0
    pin_memory = False
    if train_ds is not None:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=loader_num_workers,
            pin_memory=False,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
        )

    final_eval_loader = None
    if args.final_eval_csv:
        final_eval_ds = DTIDataset(args.final_eval_csv, y_thr=float(args.y_thr), drop_missing_y=True)
        final_eval_loader = DataLoader(
            final_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
        )

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv, y_thr=float(args.y_thr), drop_missing_y=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
        )

    def pos_rate(ds):
        ys = []
        for i in range(len(ds)):
            v = ds[i]["y_bin"]
            if hasattr(v, "item"):
                v = float(v.item())
            else:
                v = float(v)
            ys.append(v)
        return sum(1 for y in ys if y > 0.5) / max(1, len(ys))

    prot_enc = ESMProteinEncoder(
        model_name=args.esm_model,
        device=device,
        finetune=args.finetune_esm,
    )
    esm = prot_enc.esm
    if args.model_type == "cross":
        model = CrossAttnDTIClassifier(
            protein_encoder=prot_enc,
            ligand_encoder=lig_enc,
            cross_nhead=args.cross_nhead,
            dropout=args.dropout,
            cross_layers=int(args.cross_layers),
        ).to(device)
    elif args.model_type == "qkonly":
        model = QKOnlyDTIClassifier(
            protein_encoder=prot_enc,
            ligand_encoder=lig_enc,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.dti_ckpt is not None:
        _, missing, unexpected = load_dti_checkpoint(args.dti_ckpt, model, device)
        print(f"Loaded DTI checkpoint: {args.dti_ckpt}")
        if missing:
            print("  [warn] missing keys (up to 20):", missing[:20])
        if unexpected:
            print("  [warn] unexpected keys (up to 20):", unexpected[:20])

    if args.eval_only:
        logit_v, yb_v = predict(model, valid_loader, device)
        m = eval_metrics(logit_v, yb_v)

        if test_loader is not None:
            logit_t, y_t = predict(model, test_loader, device)
            mt = eval_metrics(logit_t, y_t)
            print("[TEST ]", mt)

        summary = {"mode": "eval_only", "valid": m, "valid_csv": args.valid_csv,
                   "test_csv": args.test_csv, "dti_ckpt": args.dti_ckpt}
        save_json(os.path.join(args.out_dir, "eval_only.json"), summary)
        return

    optimizer = build_optimizer_with_llrd(model, args)

    scheduler = None
    if args.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.min_lr),
        )

    best = {"ap": -1e9, "auroc": -1e9, "f1": -1e9, "epoch": -1}
    for ep in range(1, args.epochs + 1):

        if args.use_train_valid_csv and args.train_size is not None:
            epoch_seed = int(args.split_seed) + int(ep)

            train_rows = read_csv_random_rows(
                args.train_csv,
                int(args.train_size),
                seed=epoch_seed,
            )
            epoch_train_ds = DTIDataset(
                rows=train_rows,
                y_thr=float(args.y_thr),
                drop_missing_y=True,
            )

            train_loader = DataLoader(
                epoch_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=loader_num_workers,
                pin_memory=pin_memory,
                collate_fn=lambda xs: collate_fn(
                    xs,
                    esm_tokenizer=esm_tokenizer,
                    lig_pad=lig_pad,
                    lig_cls=lig_enc.cls_id,
                ),
            )

            print("epoch train n:", len(epoch_train_ds))

        elif args.use_train_valid_csv:
            # fixed full-train mode; loader already built outside loop
            print("epoch train n:", len(train_ds))

        elif train_shard_paths is not None:
            chosen_shards = pick_epoch_shards_random(
                train_shard_paths,
                train_size=int(args.train_size),
                shard_size=int(args.train_shard_size),
                epoch=ep,
                seed=int(args.seed),
                num_shards_per_epoch=args.train_num_shards_per_epoch,
            )

            epoch_train_ds = build_train_dataset_from_shards(
                chosen_shards,
                y_thr=float(args.y_thr),
            )

            train_loader = DataLoader(
                epoch_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=loader_num_workers,
                pin_memory=pin_memory,
                collate_fn=lambda xs: collate_fn(
                    xs,
                    esm_tokenizer=esm_tokenizer,
                    lig_pad=lig_pad,
                    lig_cls=lig_enc.cls_id,
                ),
            )

            print("epoch train n:", len(epoch_train_ds))
        else:
            raise ValueError("No training data source configured")

        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(args.grad_clip),
            reg_lambda=args.reg_lambda,
        )

        do_train_eval = True

        if do_train_eval:
            logit_tr, yb_tr = predict(model, train_loader, device)

            tr_m = eval_metrics(logit_tr, yb_tr)
        else:
            tr_m = {
                "ap": float("nan"),
                "auroc": float("nan"),
                "f1": float("nan"),
                "ef1": float("nan"),
                "ef5": float("nan"),
                "ef10": float("nan"),
            }

        logit_v, yb_v = predict(model, valid_loader, device)

        prob = 1.0 / (1.0 + np.exp(-logit_v))
        if prob.size:
            print("prob min/mean/max:", float(prob.min()), float(prob.mean()), float(prob.max()))

        v_m = eval_metrics(logit_v, yb_v)

        if scheduler is not None:
            scheduler.step(float(v_m[args.select_on]))

        cur_key = args.select_on
        cur_val = float(v_m[cur_key])
        best_val = float(best[cur_key])

        if cur_val > best_val:
            best.update({
                "ap": float(v_m["ap"]),
                "auroc": float(v_m["auroc"]),
                "f1": float(v_m["f1"]),
                "epoch": ep
            })
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best)

        if test_loader is not None:
            logit_t, yb_t = predict(model, test_loader, device)

            mt = eval_metrics(logit_t, yb_t)
            print("[TEST ]", mt)

        final_m = None
        if final_eval_loader is not None:
            logit_f, yb_f = predict(model, final_eval_loader, device)

            final_m = eval_metrics(logit_f, yb_f)

        epoch_summary = {
            "epoch": ep,
            "train_stat": tr_stat,
            "train_metrics": tr_m,
            "valid_metrics": v_m,
            "final_eval_metrics": final_m,
        }
        save_json(os.path.join(args.out_dir, f"epoch_{ep:03d}.json"), epoch_summary)

        print(f"\n===== Epoch {ep} =====")

        print(
            "[train]",
            f"AUC={tr_m['auroc']:.4f}",
            f"AP={tr_m['ap']:.4f}",
            f"F1={tr_m['f1']:.4f}",
            f"EF1={tr_m['ef1']:.3f}",
            f"EF5={tr_m['ef5']:.3f}",
            f"EF10={tr_m['ef10']:.3f}",
        )

        print(
            "[valid]",
            f"AUC={v_m['auroc']:.4f}",
            f"AP={v_m['ap']:.4f}",
            f"F1={v_m['f1']:.4f}",
            f"EF1={v_m['ef1']:.3f}",
            f"EF5={v_m['ef5']:.3f}",
            f"EF10={v_m['ef10']:.3f}",
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

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)


if __name__ == "__main__":
    main()