#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end DTI regression (protein sequence only + ligand VQ-Atom tokens)
+ optional distance bias in cross-attention logits
+ optional auxiliary KL loss: guide ONLY the FIRST head (head0) of p<-l cross-attn
  using a target distribution derived from distances.

CSV columns (minimum):
  - seq      : protein amino-acid sequence (string)
  - lig_tok  : ligand token ids as space-separated integers (string)
  - y        : regression target (float)

Optional CSV columns (for distance):
  - pdbid    : PDB ID (string)
  - dist_ok  : 1 to enable dist loading for this row, else 0

Distance file:
  {dist_dir}/{pdbid}.npz
  must contain key: "D" (float32) shape (n_res, n_lig)
  Example you showed: D (349, 31)

Alignment:
  - ESM tokenization includes special tokens; distance rows correspond to residues.
  - We place residue row i into protein token position (i + p_off), with p_off=1.

Outputs:
  model.forward(...) -> Tuple[y_hat (B,), aux dict]
  aux may include: "attn_kl_head0_p_from_l" (scalar tensor)
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
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    vx = float(np.var(pred))
    if vx < 1e-12:
        return 0.0, float(np.mean(y))
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
    s = str(s).strip().replace(",", " ")
    if not s:
        return []
    return [int(x) for x in s.split()]


# -----------------------------
# Dataset
# -----------------------------
class DTIDataset(Dataset):
    def __init__(self, csv_path: str, y_thr: float = 7.0):
        self.y_thr = y_thr

        rows = read_csv_rows(csv_path)
        print("[csv]", csv_path, "columns=", list(rows[0].keys()))
        print("[csv] sample pdbid raw=", rows[0].get("pdbid", None))

        self.samples = []

        for r in rows:
            if "seq" not in r or "lig_tok" not in r or "y" not in r:
                raise KeyError("CSV must contain columns: seq, lig_tok, y")

            seq = r["seq"].strip()
            lig = r["lig_tok"].strip()

            y = float(r["y"])
            y_bin = 1.0 if (y >= self.y_thr) else 0.0

            pdbid = r.get("pdbid", "").strip()

            self.samples.append((seq, lig, y_bin, pdbid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, lig_str, y_bin, pdbid = self.samples[idx]
        l_ids = parse_lig_tokens(lig_str)
        return {
            "seq": seq,
            "l_ids": torch.tensor(l_ids, dtype=torch.long),
            "y": torch.tensor(y_bin, dtype=torch.float32),
            "pdbid": pdbid,
        }

def _load_dist_D_npz(pdbid: str, dist_dir: str):
    """
    returns D: (n_res, n_lig) float32 or None
    """
    if not pdbid:
        return None
    path = os.path.join(dist_dir, f"{pdbid}.npz")
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if "D" not in z:
            return None
        D = z["D"].astype(np.float32, copy=False)
        if D.ndim != 2:
            return None
        return D
    except Exception:
        return None

@dataclass
class Batch:
    p_input_ids: torch.Tensor        # (B, Lp)
    p_attn_mask: torch.Tensor        # (B, Lp) 1=real 0=pad
    l_ids: torch.Tensor              # (B, Ll)
    y: torch.Tensor                  # (B,)

    pdbid: List[str]
    dist_ok: torch.Tensor            # (B,) long 0/1

    dist_bias_pl: Optional[torch.Tensor]  # (B, Lp, Ll) float32
    dist_bias_lp: Optional[torch.Tensor]  # (B, Ll, Lp) float32

    dist_res_target_p: Optional[torch.Tensor]  # (B, Lp) float32 (target prob over protein tokens)
    dist_res_mask_p: Optional[torch.Tensor]    # (B, Lp) bool   (where target defined)

def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not seqs:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(x.numel()) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        if x.numel() > 0:
            out[i, : x.numel()] = x
    return out

def _load_dist_profile_npz(pdbid: str, dist_dir: str):
    """
    returns (d_mean, row_mask) or (None, None)
      d_mean : (L_seq,) float32
      row_mask: (L_seq,) uint8
    """
    if not pdbid:
        return None, None
    path = os.path.join(dist_dir, f"{pdbid}.npz")
    if not os.path.isfile(path):
        return None, None
    try:
        z = np.load(path, allow_pickle=False)
        if ("d_mean" not in z) or ("row_mask" not in z):
            return None, None
        d = z["d_mean"].astype(np.float32, copy=False)
        m = z["row_mask"].astype(np.uint8, copy=False)
        if d.ndim != 1 or m.ndim != 1 or d.shape[0] != m.shape[0]:
            return None, None
        return d, m
    except Exception:
        return None, None


def collate_fn(
    samples: List[Dict[str, object]],
    esm_tokenizer,
    lig_pad: int,
    use_dist_profile: bool,
    dist_dir: str,
    dist_sigma: float,
    dist_beta: float,
    dist_cut: float,
    dist_bidir: bool,
    attn_sigma: float,   # ★追加（KLターゲット用）
) -> Batch:
    seqs = [str(s["seq"]) for s in samples]
    enc = esm_tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    p_input_ids = enc["input_ids"].long()         # (B, Lp)
    p_attn_mask = enc["attention_mask"].long()    # (B, Lp)

    l_ids_list = [s["l_ids"] for s in samples]
    l_ids = pad_1d(l_ids_list, lig_pad)           # (B, Ll)

    y = torch.stack([s["y"] for s in samples], dim=0)

    pdbid_list = [str(s.get("pdbid", "") or "") for s in samples]
    dist_ok = torch.tensor(
        [1 if (pid and os.path.isfile(os.path.join(dist_dir, f"{pid}.npz"))) else 0 for pid in pdbid_list],
        dtype=torch.long
    )

    # defaults (must exist in Batch)
    dist_bias_pl = None
    dist_bias_lp = None
    dist_res_target_p = None
    dist_res_mask_p = None

    if use_dist_profile:
        B, Lp = p_input_ids.shape
        _, Ll = l_ids.shape

        # We'll build both bias matrices; later optionally drop one if dist_bidir=False
        dist_bias_pl_full = torch.zeros((B, Lp, Ll), dtype=torch.float32)
        dist_bias_lp_full = torch.zeros((B, Ll, Lp), dtype=torch.float32)

        dist_res_target_p = torch.zeros((B, Lp), dtype=torch.float32)
        dist_res_mask_p = torch.zeros((B, Lp), dtype=torch.bool)

        p_off = 1
        max_res_tokens = max(0, Lp - 2)  # exclude CLS/EOS

        for bi, pdbid in enumerate(pdbid_list):
            if int(dist_ok[bi].item()) != 1:
                continue

            D_np = _load_dist_D_npz(pdbid, dist_dir)
            if D_np is None:
                continue

            n_res, n_lig = D_np.shape

            # align lengths
            R = min(int(n_res), int(max_res_tokens))
            A = min(int(n_lig), int(Ll))

            if bi == 0:
                print(f"[dist] pdbid={pdbid} D={n_res}x{n_lig} Lp={Lp} Ll={Ll} -> R={R} A={A}")
            if R <= 0 or A <= 0:
                continue
            D = torch.from_numpy(D_np[:R, :A]).float()  # (R, A)

            # -------------------------
            # (1) logits bias from D (R,A) -> place into token grids
            # close -> higher bias
            # -------------------------
            if dist_cut is not None and float(dist_cut) > 0:
                far = (D > float(dist_cut))
            else:
                far = None

            sim = torch.exp(-D / float(dist_sigma))  # (R,A)
            if far is not None:
                sim = sim.masked_fill(far, 0.0)

            bias_RA = float(dist_beta) * sim  # (R,A)

            dist_bias_pl_full[bi, p_off:p_off+R, :A] = bias_RA
            dist_bias_lp_full[bi, :A, p_off:p_off+R] = bias_RA.transpose(0, 1)

            # -------------------------
            # (2) KL target over protein residues from D
            # Use residue closest distance: d_res[i] = min_j D[i,j]
            # target ~ exp(-d_res/attn_sigma), mask invalid/pad
            # -------------------------
            d_res = D.min(dim=1).values  # (R,)
            # valid residue token positions (exclude PAD)
            prot_not_pad = (p_attn_mask[bi, p_off:p_off+R] == 1)

            # if you want also mask by cut, uncomment:
            # if dist_cut is not None and float(dist_cut) > 0:
            #     prot_not_pad = prot_not_pad & (d_res <= float(dist_cut))

            ok = prot_not_pad
            if ok.sum().item() <= 0:
                continue

            w = torch.exp(-d_res / float(attn_sigma))  # (R,)
            w = w.masked_fill(~ok, 0.0)
            Z = w.sum().clamp(min=1e-12)
            t = w / Z  # (R,)

            dist_res_target_p[bi, p_off:p_off+R] = t
            dist_res_mask_p[bi, p_off:p_off+R] = ok

        # apply bidir toggle:
        # - dist_bidir=True : bias both directions
        # - dist_bidir=False: bias only l<-p (lp), leave pl=None
        if bool(dist_bidir):
            dist_bias_pl = dist_bias_pl_full
            dist_bias_lp = dist_bias_lp_full
        else:
            dist_bias_pl = None
            dist_bias_lp = dist_bias_lp_full

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y=y,
        pdbid=pdbid_list,
        dist_ok=dist_ok,
        dist_bias_pl=dist_bias_pl,
        dist_bias_lp=dist_bias_lp,
        dist_res_target_p=dist_res_target_p,
        dist_res_mask_p=dist_res_mask_p,
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


class TinyFFNBlock(nn.Module):
    """
    Transformer-style FFN residual block (PreNorm).
    x -> x + Dropout( W2( Dropout(GELU(W1(LN(x)))) ) )
    """
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
        return x + h


class BiasedCrossAttention(nn.Module):
    """
    Cross attention using SDPA (scaled_dot_product_attention) with optional logits bias.
    Also can return head0 attention weights for auxiliary loss (manual compute for head0).
    """
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
        # x: (B, L, D) -> (B, H, L, Dh)
        B, L, D = x.shape
        x = x.view(B, L, self.nhead, self.d_head).transpose(1, 2).contiguous()
        return x

    def forward(
        self,
        q: torch.Tensor,                   # (B, Lq, D)
        k: torch.Tensor,                   # (B, Lk, D)
        v: torch.Tensor,                   # (B, Lk, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Lk) True=ignore
        logits_bias: Optional[torch.Tensor] = None,        # (B, Lq, Lk) additive to logits
        return_attn_head0: bool = False,
    ):
        B, Lq, D = q.shape
        _, Lk, _ = k.shape
        device = q.device

        qh = self._split_heads(self.q_proj(q))  # (B,H,Lq,Dh)
        kh = self._split_heads(self.k_proj(k))  # (B,H,Lk,Dh)
        vh = self._split_heads(self.v_proj(v))  # (B,H,Lk,Dh)

        # ---- optional manual head0 attn (for aux loss)
        attn_head0 = None
        if return_attn_head0:
            q0 = qh[:, 0]  # (B, Lq, Dh)
            k0 = kh[:, 0]  # (B, Lk, Dh)
            logits0 = torch.matmul(q0, k0.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,Lq,Lk)

            if key_padding_mask is not None:
                logits0 = logits0.masked_fill(key_padding_mask.to(device=device).unsqueeze(1), float("-inf"))

            if logits_bias is not None:
                logits0 = logits0 + logits_bias.to(device=device, dtype=logits0.dtype)

            attn_head0 = torch.softmax(logits0, dim=-1)  # (B,Lq,Lk)

        # ---- build SDPA attn_mask with bias + padding
        # SDPA attn_mask shape can be (B,H,Lq,Lk) boolean or float additive.
        attn_mask = None
        if logits_bias is not None:
            # broadcast bias to heads
            bias = logits_bias.to(device=device, dtype=qh.dtype).unsqueeze(1)  # (B,1,Lq,Lk)
            attn_mask = bias

        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=device)  # (B, Lk) True=ignore
            # boolean mask for SDPA needs shape broadcastable to (B,H,Lq,Lk)
            kpm2 = kpm.unsqueeze(1).unsqueeze(1)  # (B,1,1,Lk)
            if attn_mask is None:
                attn_mask = kpm2
            else:
                # float mask already: set ignored keys to -inf
                attn_mask = attn_mask.masked_fill(kpm2, float("-inf"))

        # ---- SDPA
        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,Dh)

        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.o_proj(out)

        if return_attn_head0:
            return out, attn_head0
        return out
class CrossAttnDTIRegressor(nn.Module):
    def __init__(
        self,
        protein_encoder: ESMProteinEncoder,
        ligand_encoder: PretrainedLigandEncoder,
        cross_nhead: int,
        dropout: float,
        attn_sigma: float = 3.0,
        cross_layers: int = 1,
        attn_kl_layers: int = 1,  # ★追加
        attn_kl_reduce: str = "mean",  # ★追加
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

        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.lig_pad_id = self.lig.pad_id
        self.attn_sigma = float(attn_sigma)
        self.attn_kl_layers = int(attn_kl_layers)
        self.attn_kl_reduce = str(attn_kl_reduce)

    def forward(
        self,
        p_input_ids: torch.Tensor,
        p_attn_mask: torch.Tensor,
        l_ids: torch.Tensor,
        dist_bias_pl: Optional[torch.Tensor] = None,
        dist_bias_lp: Optional[torch.Tensor] = None,
        dist_res_target_p: Optional[torch.Tensor] = None,  # (B,Lp)
        dist_res_mask_p: Optional[torch.Tensor] = None,    # (B,Lp)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}

        p_h = self.prot(p_input_ids, p_attn_mask)  # (B,Lp,Hp)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)                 # (B,Lp,D)
        l_h = self.lig(l_ids)                      # (B,Ll,D)

        prot_pad_mask = (p_attn_mask == 0)         # (B,Lp) True=PAD
        lig_pad_mask = (l_ids == self.lig_pad_id)  # (B,Ll) True=PAD

        # stacked blocks
        kls_lp: List[torch.Tensor] = []  # ★追加：層ごとのKLを集める

        for li in range(self.cross_layers):
            # p <- l
            p_from_l = self.cross_p_from_l[li](
                q=p_h, k=l_h, v=l_h,
                key_padding_mask=lig_pad_mask,
                logits_bias=dist_bias_pl,
                return_attn_head0=False,
            )
            p_h = self.post_ln_p[li](p_h + p_from_l)
            p_h = self.ffn_p[li](p_h)

            # l <- p : ★指定層だけ head0 を取る
            want_kl_this_layer = (
                (self.attn_kl_layers > 0) and (li < self.attn_kl_layers)
                and (dist_res_target_p is not None) and (dist_res_mask_p is not None)
            )

            if want_kl_this_layer:
                l_from_p, attn0_lp = self.cross_l_from_p[li](
                    q=l_h, k=p_h, v=p_h,
                    key_padding_mask=prot_pad_mask,
                    logits_bias=dist_bias_lp,
                    return_attn_head0=True,
                )
            else:
                l_from_p = self.cross_l_from_p[li](
                    q=l_h, k=p_h, v=p_h,
                    key_padding_mask=prot_pad_mask,
                    logits_bias=dist_bias_lp,
                    return_attn_head0=False,
                )
                attn0_lp = None

            l_h = self.post_ln_l[li](l_h + l_from_p)
            l_h = self.ffn_l[li](l_h)

            # KL (layer selected): residue importance via l<-p head0
            if want_kl_this_layer and (attn0_lp is not None):
                # attn0_lp: (B, Ll, Lp) softmax over Lp
                A = attn0_lp.to(device=p_h.device, dtype=p_h.dtype).clamp(min=1e-12)

                # average over ligand queries (exclude PAD queries)
                q_ok = (~lig_pad_mask).to(device=p_h.device)   # (B,Ll)
                q_ok_f = q_ok.float()
                denom_q = q_ok_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
                Abar = (A * q_ok_f.unsqueeze(-1)).sum(dim=1) / denom_q    # (B,Lp)

                T = dist_res_target_p.to(device=p_h.device, dtype=p_h.dtype).clamp(min=1e-12)  # (B,Lp)
                M = dist_res_mask_p.to(device=p_h.device)                                      # (B,Lp)

                Tm = T.masked_fill(~M, 0.0)
                Am = Abar.masked_fill(~M, 1e-12)

                Zt = Tm.sum(dim=1, keepdim=True).clamp(min=1e-12)
                Za = Am.sum(dim=1, keepdim=True).clamp(min=1e-12)
                Tm = Tm / Zt
                Am = Am / Za

                kl_b = (Tm * (torch.log(Tm.clamp(min=1e-12)) - torch.log(Am.clamp(min=1e-12)))).sum(dim=1)  # (B,)
                kl = kl_b.mean()
                kls_lp.append(kl)

                # 層別ログも欲しければ残す
                aux[f"res_kl_head0_l_from_p_layer{li}"] = kl

        # ★集約（aux["res_kl"] を統一キーに）
        if kls_lp:
            if self.attn_kl_reduce == "sum":
                aux["res_kl"] = torch.stack(kls_lp, dim=0).sum()
            else:
                aux["res_kl"] = torch.stack(kls_lp, dim=0).mean()
        else:
            # ないときは入れない（train側で get する）
            pass

        # pool & head
        p_pool = masked_mean_by_attn(p_h, p_attn_mask)         # (B,D)
        l_attn_mask = (~lig_pad_mask).long()                   # (B,Ll)
        l_pool = masked_mean_by_attn(l_h, l_attn_mask)         # (B,D)

        y_hat = self.head(torch.cat([p_pool, l_pool], dim=-1)).squeeze(-1)
        return y_hat, aux


# -----------------------------
# Predict / Eval (raw + calibrated
# ---- predict は「logitを返す」(中身はほぼ同じ)
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    ys = []
    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)

        logit, _aux = model(
            p_ids, p_msk, l,
            dist_bias_pl=(batch.dist_bias_pl.to(device) if batch.dist_bias_pl is not None else None),
            dist_bias_lp=(batch.dist_bias_lp.to(device) if batch.dist_bias_lp is not None else None),
            dist_res_target_p=(batch.dist_res_target_p.to(device) if batch.dist_res_target_p is not None else None),
            dist_res_mask_p=(batch.dist_res_mask_p.to(device) if batch.dist_res_mask_p is not None else None),
        )
        logits.append(logit.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    logit = np.concatenate(logits, axis=0) if logits else np.array([], dtype=np.float64)
    yy = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.float64)
    return logit, yy


def eval_metrics(logit: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

    prob = 1.0 / (1.0 + np.exp(-logit))
    y01 = (y > 0.5).astype(np.int32)

    if len(np.unique(y01)) <= 1:
        return {"auroc": 0.0, "ap": 0.0, "f1": 0.0, "thr": 0.5,
                "prob_mean": float(prob.mean()), "prob_std": float(prob.std())}

    auroc = roc_auc_score(y01, prob)
    ap = average_precision_score(y01, prob)

    prec, rec, thr = precision_recall_curve(y01, prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    # precision_recall_curve のthrは長さが1短いので、この取り方が安全
    best_i = int(np.argmax(f1s[:-1]))  # thr と同じ長さに揃える
    best_thr = float(thr[best_i]) if thr.size > 0 else 0.5

    pred01 = (prob >= best_thr).astype(np.int32)
    f1 = f1_score(y01, pred01)

    return {"auroc": float(auroc), "ap": float(ap), "f1": float(f1), "thr": float(best_thr),
            "prob_mean": float(prob.mean()), "prob_std": float(prob.std())}

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    grad_clip: float = 1.0,
    attn_lambda: float = 0.0,
    pos_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    base_losses: List[float] = []
    kls: List[float] = []

    use_kl = float(attn_lambda) > 0.0
    pw = torch.tensor([float(pos_weight)], device=device)

    for batch in loader:
        p_ids = batch.p_input_ids.to(device)
        p_msk = batch.p_attn_mask.to(device)
        l = batch.l_ids.to(device)
        y = batch.y.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_hat, aux = model(
            p_ids, p_msk, l,
            dist_bias_pl=(batch.dist_bias_pl.to(device) if batch.dist_bias_pl is not None else None),
            dist_bias_lp=(batch.dist_bias_lp.to(device) if batch.dist_bias_lp is not None else None),
            dist_res_target_p=(batch.dist_res_target_p.to(device) if batch.dist_res_target_p is not None else None),
            dist_res_mask_p=(batch.dist_res_mask_p.to(device) if batch.dist_res_mask_p is not None else None),
        )

        # ---- base loss
        logit = y_hat  # (B,)
        base_loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
        loss = base_loss

        # ---- optional KL (already aggregated in aux["res_kl"])
        if use_kl:
            kl = aux.get("res_kl", None)
            if (kl is not None) and torch.isfinite(kl).all():
                loss = loss + float(attn_lambda) * kl
        else:
            kl = None  # for logging

        loss.backward()
        if grad_clip and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        # ---- logging
        losses.append(float(loss.detach().cpu().item()))
        base_losses.append(float(base_loss.detach().cpu().item()))
        if use_kl and (kl is not None) and torch.isfinite(kl).all():
            kls.append(float(kl.detach().cpu().item()))

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "base_loss": float(sum(base_losses) / max(1, len(base_losses))),
        "res_kl": float(sum(kls) / max(1, len(kls))) if kls else 0.0,
    }

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
            "calibration": calib,
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
# Optimizer with LLRD (ESM)
# -----------------------------
def build_optimizer_with_llrd(model: CrossAttnDTIRegressor, args: argparse.Namespace) -> torch.optim.Optimizer:
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

    opt = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    return opt


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_weight", type=float, default=1.0,
                    help="BCE pos_weight. >1 boosts positive class (useful for imbalance).")
    # aux attn KL (head0 only)
    # ★追加：何層をKL対象にするか
    ap.add_argument("--attn_kl_layers", type=int, default=1,
                    help="Apply distance-guided KL on the first N cross layers (l<-p head0). 0 disables.")
    ap.add_argument("--attn_kl_reduce", type=str, default="mean", choices=["mean", "sum"],
                    help="How to aggregate KL over layers when attn_kl_layers>1.")
    # data
    ap.add_argument("--y_thr", type=float, default=7.0, help="Binarize y: y>=thr -> 1 else 0")
    ap.add_argument("--train_csv", type=str, default=None, help="Train CSV (required unless --eval_only)")
    ap.add_argument("--valid_csv", type=str, required=True, help="Validation CSV")
    ap.add_argument("--test_csv", type=str, default=None, help="Optional test CSV")
    ap.add_argument("--cross_layers", type=int, default=1, help="Number of cross-attention blocks")

    # ligand MLM weights ckpt
    ap.add_argument("--lig_ckpt", type=str, default="/vqatom/data/mlm_ep05.pt", help="Ligand MLM checkpoint")

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

    # calibration
    ap.add_argument("--calib_frac", type=float, default=0.1)
    ap.add_argument("--calib_every", type=int, default=1)

    # cross attn
    ap.add_argument("--cross_nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

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

    # distance bias
    ap.add_argument("--use_dist_profile", action="store_true")
    ap.add_argument("--dist_dir", type=str, default="../data/dist_mat")
    ap.add_argument("--dist_sigma", type=float, default=3.0)
    ap.add_argument("--dist_beta", type=float, default=0.2)
    ap.add_argument("--dist_cut", type=float, default=12.0)
    ap.add_argument("--dist_bidir", action="store_true")

    # aux attn KL (head0 only)
    ap.add_argument("--attn_lambda", type=float, default=0.0, help="Aux KL weight for head0 attention guidance")
    ap.add_argument("--attn_sigma", type=float, default=3.0, help="Sigma for target exp(-D/sigma) in KL loss")

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
    print("use_dist_profile:", bool(args.use_dist_profile), "dist_dir:", args.dist_dir)
    print("attn_lambda:", float(args.attn_lambda), "attn_sigma:", float(args.attn_sigma))

    # ESM tokenizer
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model, do_lower_case=False)

    # ligand encoder
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
    valid_ds = DTIDataset(args.valid_csv, y_thr=args.y_thr)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: collate_fn(
            xs,
            esm_tokenizer=esm_tokenizer,
            lig_pad=lig_pad,
            use_dist_profile=bool(args.use_dist_profile),
            dist_dir=str(args.dist_dir),
            dist_sigma=float(args.dist_sigma),
            dist_beta=float(args.dist_beta),
            dist_cut=float(args.dist_cut),
            dist_bidir=bool(args.dist_bidir),
            attn_sigma=float(args.attn_sigma),
        ),
    )

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(
                xs,
                esm_tokenizer=esm_tokenizer,
                lig_pad=lig_pad,
                use_dist_profile=bool(args.use_dist_profile),
                dist_dir=str(args.dist_dir),
                dist_sigma=float(args.dist_sigma),
                dist_beta=float(args.dist_beta),
                dist_cut=float(args.dist_cut),
                dist_bidir=bool(args.dist_bidir),
                attn_sigma=float(args.attn_sigma),
            ),
        )

    train_loader = None
    calib_loader = None
    train_ds = None

    if not args.eval_only:
        train_ds = DTIDataset(args.train_csv, y_thr=args.y_thr)
        valid_ds = DTIDataset(args.valid_csv, y_thr=args.y_thr)  # 既に作ってるならこの再定義は削除でもOK
        n = len(train_ds)
        idx = np.arange(n)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)

        n_cal = int(round(n * args.calib_frac))
        n_cal = max(0, min(n_cal, n - 1))
        cal_idx = idx[:n_cal]
        tr_idx = idx[n_cal:]

        train_loader = DataLoader(
            Subset(train_ds, tr_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(
                xs,
                esm_tokenizer=esm_tokenizer,
                lig_pad=lig_pad,
                use_dist_profile=bool(args.use_dist_profile),
                dist_dir=str(args.dist_dir),
                dist_sigma=float(args.dist_sigma),
                dist_beta=float(args.dist_beta),
                dist_cut=float(args.dist_cut),
                dist_bidir=bool(args.dist_bidir),
                attn_sigma=float(args.attn_sigma),
            ),
        )
        calib_loader = DataLoader(
            Subset(train_ds, cal_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda xs: collate_fn(
                xs,
                esm_tokenizer=esm_tokenizer,
                lig_pad=lig_pad,
                use_dist_profile=bool(args.use_dist_profile),
                dist_dir=str(args.dist_dir),
                dist_sigma=float(args.dist_sigma),
                dist_beta=float(args.dist_beta),
                dist_cut=float(args.dist_cut),
                dist_bidir=bool(args.dist_bidir),
                attn_sigma=float(args.attn_sigma),
            ),
        )
        print(f"[split] train={len(tr_idx)}  calib={len(cal_idx)}  valid={len(valid_ds)}")

    # protein encoder
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
        attn_sigma=float(args.attn_sigma),
        cross_layers=int(args.cross_layers),
        attn_kl_layers=int(args.attn_kl_layers),      # ★追加
        attn_kl_reduce=str(args.attn_kl_reduce),      # ★追加
    ).to(device)

    # optionally load DTI checkpoint
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

    # eval-only
    if args.eval_only:
        logit_v, y_v = predict(model, valid_loader, device)
        m = eval_metrics(logit_v, y_v)
        print(f"[VALID] AP={m['ap']:.4f} AUROC={m['auroc']:.4f} F1={m['f1']:.4f} (thr={m['thr']:.3f}) "
              f"prob_mean={m['prob_mean']:.4f} prob_std={m['prob_std']:.4f}")

        if test_loader is not None:
            logit_t, y_t = predict(model, test_loader, device)
            mt = eval_metrics(logit_t, y_t)
            print(f"[TEST ] AP={mt['ap']:.4f} AUROC={mt['auroc']:.4f} F1={mt['f1']:.4f} (thr={mt['thr']:.3f})")

        summary = {"mode": "eval_only", "valid": m, "valid_csv": args.valid_csv,
                   "test_csv": args.test_csv, "dti_ckpt": args.dti_ckpt}
        save_json(os.path.join(args.out_dir, "eval_only.json"), summary)
        return

    # optimizer
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

    # training loop
    best = {"ap": -1e9, "auroc": -1e9, "f1": -1e9, "epoch": -1}

    for ep in range(1, args.epochs + 1):
        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(args.grad_clip),
            attn_lambda=float(args.attn_lambda),
            pos_weight=float(args.pos_weight),
        )

        print(f"res_kl={tr_stat['res_kl']:.6f} attn_lambda={args.attn_lambda}")

        logit_tr, y_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(logit_tr, y_tr)

        logit_v, y_v = predict(model, valid_loader, device)
        v_m = eval_metrics(logit_v, y_v)

        print(
            f"[ep {ep:03d}] "
            f"train_loss={tr_stat['loss']:.4f} (base={tr_stat['base_loss']:.4f}) "
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
            # calibは空 dict で渡す（関数を変えたくなければ）
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best, calib={})
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best, calib={})

        if test_loader is not None:
            logit_t, y_t = predict(model, test_loader, device)
            t_m = eval_metrics(logit_t, y_t)
            print(f"         test_AP={t_m['ap']:.4f} test_AUROC={t_m['auroc']:.4f} test_F1={t_m['f1']:.4f}")

    print("BEST:", best)
    save_json(os.path.join(args.out_dir, "best.json"), best)


if __name__ == "__main__":
    main()