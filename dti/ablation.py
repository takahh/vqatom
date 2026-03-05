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
    # Protein tokenize (unchanged)
    seqs = [str(s["seq"]) for s in samples]
    enc = esm_tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
    p_input_ids = enc["input_ids"].long()
    p_attn_mask = enc["attention_mask"].long()

    # Ligand: prepend CLS then pad
    l_ids_list = []
    for s in samples:
        x = s["l_ids"]
        if x.numel() == 0:
            x = torch.tensor([lig_cls], dtype=torch.long)
        else:
            x = torch.cat([torch.tensor([lig_cls], dtype=torch.long), x], dim=0)
        l_ids_list.append(x)

    l_ids = pad_1d(l_ids_list, lig_pad)

    y_bin = torch.stack([s["y_bin"] for s in samples], dim=0).float()
    y = torch.stack([s["y"] for s in samples], dim=0).float()

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y_bin=y_bin,
        y=y
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
            kpm2 = key_padding_mask.to(device=device).unsqueeze(1).unsqueeze(1)  # (B,1,1,Lk)
            if attn_mask is None:
                attn_mask = kpm2
            else:
                attn_mask = attn_mask.masked_fill(kpm2, float("-inf"))

        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,Dh)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(out)


class DTIDataset(Dataset):
    """
    label-free dataset:
      - requires: seq, lig_tok, y
      - defines binary label as (y >= y_thr)
    """
    def __init__(self, csv_path: str, y_thr: float = 7.0, drop_missing_y: bool = True):
        rows = read_csv_rows(csv_path)
        self.samples = []

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
                if drop_missing_y:
                    dropped_no_y += 1
                    continue
                y_val = float("-inf")
            else:
                y_val = float(y_str)

            y_bin = 1.0 if y_val >= float(y_thr) else 0.0
            pdbid = str(r.get("pdbid", "")).strip()

            self.samples.append((seq, lig, y_bin, y_val, pdbid))

        print(
            f"[DTIDataset] {csv_path}: kept={len(self.samples)} "
            f"dropped_no_y={dropped_no_y} dropped_bad={dropped_bad}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, lig_str, y_bin, y_val, pdbid = self.samples[idx]
        l_ids = parse_lig_tokens(lig_str)
        return {
            "seq": seq,
            "l_ids": torch.tensor(l_ids, dtype=torch.long),
            "y_bin": torch.tensor(float(y_bin), dtype=torch.float32),
            "y": torch.tensor(float(y_val), dtype=torch.float32),
            "pdbid": pdbid,
        }

class CrossAttnDTIClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        cross_nhead: int,
        dropout: float,
        cross_layers: int = 1,
    ):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model
        self.cross_layers = int(cross_layers)
        assert self.cross_layers >= 1

        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

        self.cross_p_from_l = nn.ModuleList([
            BiasedCrossAttention(d_model=d_model, nhead=cross_nhead, dropout=dropout)
            for _ in range(self.cross_layers)
        ])
        self.cross_l_from_p = nn.ModuleList([
            BiasedCrossAttention(d_model=d_model, nhead=cross_nhead, dropout=dropout)
            for _ in range(self.cross_layers)
        ])

        self.post_ln_p = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])
        self.post_ln_l = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])

        self.ffn_p = nn.ModuleList([TinyFFNBlock(d_model=d_model, dropout=dropout, mult=4) for _ in range(self.cross_layers)])
        self.ffn_l = nn.ModuleList([TinyFFNBlock(d_model=d_model, dropout=dropout, mult=4) for _ in range(self.cross_layers)])

        self.ln_p_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])
        self.ln_l_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])
        self.ln_p_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])
        self.ln_l_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.cross_layers)])
        self.drop = nn.Dropout(dropout)

        self.shared = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_cls = nn.Linear(d_model, 1)
        self.head_reg = nn.Linear(d_model, 1)

        self.lig_pad_id = int(self.lig.pad_id)
        self.int_proj_p = nn.Linear(d_model, d_model, bias=False)
        self.int_proj_l = nn.Linear(d_model, d_model, bias=False)
        self.int_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mix_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    def forward(
            self,
            p_input_ids: torch.Tensor,
            p_attn_mask: torch.Tensor,
            l_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}

        # ----------------------------
        # Encode
        # ----------------------------
        p_h = self.prot(p_input_ids, p_attn_mask)  # (B,Lp,Hp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)  # (B,Lp,D)
        l_h = self.lig(l_ids)  # (B,Ll,D)

        prot_pad_mask = (p_attn_mask == 0)  # (B,Lp) True=PAD (ignore as keys)
        lig_pad_mask = (l_ids == self.lig_pad_id)  # (B,Ll) True=PAD (ignore as keys)

        # ----------------------------
        # Cross-attention blocks (PreNorm + Residual)
        # NOTE: this assumes TinyFFNBlock returns FFN(x) (no internal residual)
        # ----------------------------
        for li in range(self.cross_layers):
            # p <- l
            p_attn = self.cross_p_from_l[li](
                q=self.ln_p_attn[li](p_h),
                k=self.ln_l_attn[li](l_h),
                v=self.ln_l_attn[li](l_h),
                key_padding_mask=lig_pad_mask,
            )
            p_h = p_h + self.drop(p_attn)
            p_h = p_h + self.ffn_p[li](self.ln_p_ffn[li](p_h))

            # l <- p
            l_attn = self.cross_l_from_p[li](
                q=self.ln_l_attn[li](l_h),
                k=self.ln_p_attn[li](p_h),
                v=self.ln_p_attn[li](p_h),
                key_padding_mask=prot_pad_mask,
            )
            l_h = l_h + self.drop(l_attn)
            l_h = l_h + self.ffn_l[li](self.ln_l_ffn[li](l_h))

        # ----------------------------
        # Interaction pooling (token-level) + CLS pooling
        # ----------------------------
        p_cls = p_h[:, 0, :]  # (B,D)
        l_cls = l_h[:, 0, :]  # (B,D)

        # Optional: drop special tokens (protein CLS; ligand CLS) from interaction map
        p_tok = p_h[:, 1:, :]  # (B,Lp-1,D)
        l_tok = l_h[:, 1:, :]  # (B,Ll-1,D)
        p_pad = prot_pad_mask[:, 1:]  # (B,Lp-1)
        l_pad = lig_pad_mask[:, 1:]  # (B,Ll-1)

        # If sequence becomes empty (rare), fall back to CLS-only
        if p_tok.size(1) == 0 or l_tok.size(1) == 0:
            z = self.shared(torch.cat([p_cls, l_cls], dim=-1))
            logit = self.head_cls(z).squeeze(-1)
            aux["y_hat"] = self.head_reg(z).squeeze(-1)
            return logit, aux

        # Score matrix S: (B, Lp-1, Ll-1)
        # Uses bilinear-ish form; set up these layers in __init__:
        #   self.int_proj_p = nn.Linear(D, D, bias=False)
        #   self.int_proj_l = nn.Linear(D, D, bias=False)
        #   self.int_head   = nn.Sequential(LN(2D), Linear(2D,D), GELU(), Dropout)
        p_i = self.int_proj_p(p_tok)
        l_i = self.int_proj_l(l_tok)
        S = torch.matmul(p_i, l_i.transpose(1, 2))

        # Mask pads -> -inf so max ignores them
        S = S.masked_fill(p_pad.unsqueeze(-1), float("-inf"))
        S = S.masked_fill(l_pad.unsqueeze(1), float("-inf"))

        # after masking S with -inf
        # If a whole row/col is fully masked, max becomes -inf; clamp them to 0.
        p_all_pad = p_pad.all(dim=1)  # (B,)
        l_all_pad = l_pad.all(dim=1)  # (B,)
        if p_all_pad.any() or l_all_pad.any():
            # For samples with all-pad tokens, just zero out S so pooled features become 0.
            bad = (p_all_pad | l_all_pad).view(-1, 1, 1)
            S = torch.where(bad, torch.zeros_like(S), S)

        # "best match" pooling
        p_best = S.max(dim=2).values  # (B, Lp-1)
        l_best = S.max(dim=1).values  # (B, Ll-1)

        # mask-aware mean of best-match scores -> scalars (B,)
        p_best = p_best.masked_fill(p_pad, 0.0)
        l_best = l_best.masked_fill(l_pad, 0.0)
        p_den = (~p_pad).sum(dim=1).clamp(min=1)
        l_den = (~l_pad).sum(dim=1).clamp(min=1)
        p_feat = p_best.sum(dim=1) / p_den
        l_feat = l_best.sum(dim=1) / l_den

        aux["int_p_feat"] = p_feat.detach()
        aux["int_l_feat"] = l_feat.detach()

        # Turn (B,) scalars into a (B,2D) vector and fuse with CLS head
        D = p_cls.size(-1)
        int_vec = torch.cat(
            [p_feat.unsqueeze(-1).expand(-1, D), l_feat.unsqueeze(-1).expand(-1, D)],
            dim=-1,
        )  # (B, 2D)

        z_cls = self.shared(torch.cat([p_cls, l_cls], dim=-1))  # (B,D)
        z_int = self.int_head(int_vec)  # (B,D)
        alpha = torch.sigmoid(self.mix_logit)  # 0..1
        z = (1 - alpha) * z_cls + alpha * z_int
        aux["mix_alpha"] = alpha.detach()

        logit = self.head_cls(z).squeeze(-1)  # (B,)
        aux["y_hat"] = self.head_reg(z).squeeze(-1)
        return logit, aux

# -----------------------------
# Eval / Train
# -----------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ybins = [], []
    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l_ids = batch.l_ids.to(device)
        logit, _ = model(p_ids, p_msk, l_ids)
        logits.append(logit.detach().cpu().numpy())
        ybins.append(batch.y_bin.detach().cpu().numpy())
    logit = np.concatenate(logits, axis=0) if logits else np.array([], dtype=np.float64)
    yb = np.concatenate(ybins, axis=0) if ybins else np.array([], dtype=np.float64)
    return logit, yb


def eval_metrics(logit: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    - Computes AUROC / AP (threshold-free)
    - Finds best-F1 threshold on a grid, and reports that best F1 + thr.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    if logit.size == 0:
        return {
            "auroc": 0.0, "ap": 0.0,
            "f1": 0.0, "thr": 0.5,
            "prob_mean": 0.0, "prob_std": 0.0,
        }

    prob = 1.0 / (1.0 + np.exp(-logit))
    y01 = (y > 0.5).astype(np.int32)

    if len(np.unique(y01)) <= 1:
        return {
            "auroc": 0.0, "ap": 0.0,
            "f1": 0.0, "thr": 0.5,
            "prob_mean": float(prob.mean()), "prob_std": float(prob.std()),
        }

    auroc = roc_auc_score(y01, prob)
    ap = average_precision_score(y01, prob)

    # --- threshold search for best F1 ---
    # avoid degenerate ends; include 0.5 anyway
    thrs = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),  # ~0.005 step
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
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    reg_lambda: float = 0.0,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []

    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l_ids = batch.l_ids.to(device)
        y_bin = batch.y_bin.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logit, aux = model(p_ids, p_msk, l_ids)
            loss_cls = F.binary_cross_entropy_with_logits(logit, y_bin)
            y = batch.y.to(device)
            y_hat = aux["y_hat"]
            loss_reg = F.smooth_l1_loss(y_hat, y)
            loss = loss_cls + reg_lambda * loss_reg
        loss.backward()

        if grad_clip and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return {"loss": float(sum(losses) / max(1, len(losses)))}


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
def build_optimizer_with_llrd(model: CrossAttnDTIClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_csv", type=str, default=None, help="Train CSV (required unless --eval_only)")
    ap.add_argument("--valid_csv", type=str, required=True, help="Validation CSV")
    ap.add_argument("--test_csv", type=str, default=None, help="Optional test CSV")
    ap.add_argument("--y_thr", type=float, default=Y_THR, help="Strong threshold: y>=y_thr => y_bin=1 (binders only)")
    ap.add_argument("--cross_layers", type=int, default=1)

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

    args = ap.parse_args()

    if not args.eval_only and not args.train_csv:
        raise ValueError("--train_csv is required unless --eval_only is set.")

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("lig_ckpt:", args.lig_ckpt)
    print("vq_ckpt :", args.vq_ckpt)
    print("esm_model:", args.esm_model)
    print("y_thr:", float(args.y_thr))

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
    print(f"[ligand] vocab_source={lig_enc.vocab_source}")
    print(f"[ligand] vocab_size={lig_enc.vocab_size} base_vocab={lig_enc.base_vocab} PAD={lig_enc.pad_id} MASK={lig_enc.mask_id}")
    print(f"[ligand] CLS={lig_enc.cls_id} PAD={lig_enc.pad_id} MASK={lig_enc.mask_id} vocab={lig_enc.vocab_size}")
    assert lig_enc.cls_id != lig_enc.pad_id
    assert lig_enc.cls_id != lig_enc.mask_id

    valid_ds = DTIDataset(args.valid_csv, y_thr=float(args.y_thr), drop_missing_y=True)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
    )

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv, y_thr=float(args.y_thr))
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
        )

    train_loader = None
    train_ds = None
    if not args.eval_only:
        train_ds = DTIDataset(args.train_csv, y_thr=float(args.y_thr), drop_missing_y=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
        )

        def pos_rate(ds):
            ys = [float(ds[i]["y_bin"].item()) for i in range(len(ds))]
            return sum(1 for y in ys if y > 0.5) / max(1, len(ys))

        print("train n:", len(train_ds), "pos_rate:", pos_rate(train_ds))
        print("valid n:", len(valid_ds), "pos_rate:", pos_rate(valid_ds))
    else:
        print("valid n:", len(valid_ds))

    prot_enc = ESMProteinEncoder(
        model_name=args.esm_model,
        device=device,
        finetune=args.finetune_esm,
    )
    print("finetune_esm:", args.finetune_esm)
    esm = prot_enc.esm
    n_train = sum(p.requires_grad for p in esm.parameters())
    n_all = sum(1 for _ in esm.parameters())
    print(f"[debug] ESM requires_grad: {n_train}/{n_all}")

    model = CrossAttnDTIClassifier(
        protein_encoder=prot_enc,
        ligand_encoder=lig_enc,
        cross_nhead=args.cross_nhead,
        dropout=args.dropout,
        cross_layers=int(args.cross_layers),
    ).to(device)

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
        print("[VALID]", m)

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
        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(args.grad_clip),
            reg_lambda=args.reg_lambda,
        )

        logit_tr, yb_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(logit_tr, yb_tr)

        logit_v, yb_v = predict(model, valid_loader, device)
        prob = 1.0 / (1.0 + np.exp(-logit_v))
        print("pos_rate(valid):", float((yb_v > 0.5).mean()) if yb_v.size else 0.0)
        print("pred_pos_rate@0.5:", float((prob >= 0.5).mean()) if prob.size else 0.0)
        if prob.size:
            print("prob min/mean/max:", float(prob.min()), float(prob.mean()), float(prob.max()))

        v_m = eval_metrics(logit_v, yb_v)

        print(
            f"[ep {ep:03d}] "
            f"train_loss={tr_stat['loss']:.4f} "
            f"train_AP={tr_m['ap']:.4f} train_AUROC={tr_m['auroc']:.4f} train_F1={tr_m['f1']:.4f} "
            f"val_AP={v_m['ap']:.4f} val_AUROC={v_m['auroc']:.4f} val_F1={v_m['f1']:.4f} (thr={v_m['thr']:.3f})"
        )

        if scheduler is not None:
            scheduler.step(float(v_m[args.select_on]))

        cur_key = args.select_on
        cur_val = float(v_m[cur_key])
        best_val = float(best[cur_key])

        if cur_val > best_val:
            best.update({"ap": float(v_m["ap"]), "auroc": float(v_m["auroc"]), "f1": float(v_m["f1"]), "epoch": ep})
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best)

        if test_loader is not None:
            logit_t, yb_t = predict(model, test_loader, device)
            mt = eval_metrics(logit_t, yb_t)
            print("[TEST ]", mt)

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)


if __name__ == "__main__":
    main()