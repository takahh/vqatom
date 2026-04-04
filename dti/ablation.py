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
@dataclass
class Batch:
    p_input_ids: torch.Tensor
    p_attn_mask: torch.Tensor
    l_ids: torch.Tensor
    y_bin: torch.Tensor


class DTIDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        rows: Optional[List[Dict[str, str]]] = None,
        y_thr: float = Y_THR,
        drop_missing_y: bool = True,
    ):
        if rows is not None:
            raw_rows = rows
        elif csv_path is not None:
            raw_rows = read_csv_rows(csv_path)
        else:
            raise ValueError("Either csv_path or rows must be provided")

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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.rows[idx]


def build_train_dataset_from_shards(shard_paths: List[str], y_thr: float) -> ConcatDataset:
    ds_list = [DTIDataset(csv_path=p, y_thr=float(y_thr), drop_missing_y=True) for p in shard_paths]
    if not ds_list:
        raise ValueError("No train shards were loaded")
    return ConcatDataset(ds_list)


def collate_fn(samples, esm_tokenizer, lig_pad: int, lig_cls: int) -> Batch:
    seqs = [s["seq"] for s in samples]
    enc = esm_tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
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
    y_bin = torch.tensor([float(s["y_bin"]) for s in samples], dtype=torch.float32)
    return Batch(p_input_ids=p_input_ids, p_attn_mask=p_attn_mask, l_ids=l_ids, y_bin=y_bin)


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
    model.eval()
    batch = next(iter(loader))

    p_ids = batch.p_input_ids.to(device, non_blocking=True)
    p_msk = batch.p_attn_mask.to(device, non_blocking=True)
    l_ids = batch.l_ids.to(device, non_blocking=True)

    with torch.inference_mode():
        logit, aux = model(p_ids, p_msk, l_ids, return_maps=True)

    S = aux["qk_scores"][sample_idx_in_batch].float().cpu().numpy()      # (Ll, Lp)
    A = aux["attn_map"][sample_idx_in_batch].float().cpu().numpy()       # (Ll, Lp)
    l_imp = aux["lig_importance"][sample_idx_in_batch].float().cpu().numpy()
    p_pad = aux["p_pad"][sample_idx_in_batch].cpu().numpy().astype(bool)
    l_pad = aux["l_pad"][sample_idx_in_batch].cpu().numpy().astype(bool)

    # pad removal
    S_vis = S[~l_pad][:, ~p_pad]
    A_vis = A[~l_pad][:, ~p_pad]
    l_imp_vis = l_imp[~l_pad]

    # token labels
    p_tok_labels = None
    l_tok_labels = None
    if show_token_labels:
        p_ids_1 = batch.p_input_ids[sample_idx_in_batch].cpu().tolist()
        l_ids_1 = batch.l_ids[sample_idx_in_batch].cpu().tolist()

        p_tok_labels = esm_tokenizer.convert_ids_to_tokens(p_ids_1)
        p_tok_labels = p_tok_labels[1:]  # drop CLS
        p_tok_labels = [t for t, is_pad in zip(p_tok_labels, p_pad) if not is_pad]

        l_tok_labels = [str(x) for x in l_ids_1[1:]]  # drop ligand CLS
        l_tok_labels = [t for t, is_pad in zip(l_tok_labels, l_pad) if not is_pad]

    prob = float(torch.sigmoid(logit[sample_idx_in_batch]).detach().cpu())
    y_bin = float(batch.y_bin[sample_idx_in_batch])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1) QK score map
    plt.figure(figsize=(10, 6))
    plt.imshow(S_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"QK score map | prob={prob:.4f} y={y_bin:.0f}")
    plt.xlabel("Protein tokens")
    plt.ylabel("Ligand tokens")
    if show_token_labels and p_tok_labels is not None and l_tok_labels is not None:
        if len(p_tok_labels) <= 80:
            plt.xticks(range(len(p_tok_labels)), p_tok_labels, rotation=90, fontsize=6)
        if len(l_tok_labels) <= 80:
            plt.yticks(range(len(l_tok_labels)), l_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_qk_scores.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Attention map
    plt.figure(figsize=(10, 6))
    plt.imshow(A_vis, aspect="auto")
    plt.colorbar()
    plt.title(f"Attention map | prob={prob:.4f} y={y_bin:.0f}")
    plt.xlabel("Protein tokens")
    plt.ylabel("Ligand tokens")
    if show_token_labels and p_tok_labels is not None and l_tok_labels is not None:
        if len(p_tok_labels) <= 80:
            plt.xticks(range(len(p_tok_labels)), p_tok_labels, rotation=90, fontsize=6)
        if len(l_tok_labels) <= 80:
            plt.yticks(range(len(l_tok_labels)), l_tok_labels, fontsize=6)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_attn_map.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Ligand importance
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(l_imp_vis)), l_imp_vis)
    plt.title(f"Ligand importance | prob={prob:.4f} y={y_bin:.0f}")
    plt.xlabel("Ligand token index")
    plt.ylabel("importance")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_lig_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "qk_scores": S_vis,
        "attn_map": A_vis,
        "lig_importance": l_imp_vis,
        "prob": prob,
        "y_bin": y_bin,
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
    def __init__(self, protein_encoder: ESMProteinEncoder, ligand_encoder: PretrainedLigandEncoder, dropout: float):
        super().__init__()
        self.prot = protein_encoder
        self.lig = ligand_encoder

        d_model = self.lig.d_model
        self.p_proj = None
        if self.prot.hidden_size != d_model:
            self.p_proj = nn.Linear(self.prot.hidden_size, d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.lig_pad_id = int(self.lig.pad_id)

        self.v_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            p_input_ids: torch.Tensor,
            p_attn_mask: torch.Tensor,
            l_ids: torch.Tensor,
            return_maps: bool = False,
    ):
        aux = {}
        p_h = self.prot(p_input_ids, p_attn_mask)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)
        l_h = self.lig(l_ids)

        prot_pad_mask = (p_attn_mask == 0)
        lig_pad_mask = (l_ids == self.lig_pad_id)

        p_tok = p_h[:, 1:, :]
        l_tok = l_h[:, 1:, :]
        l_cls = l_h[:, 0, :]
        p_pad = prot_pad_mask[:, 1:]
        l_pad = lig_pad_mask[:, 1:]

        if p_tok.size(1) == 0 or l_tok.size(1) == 0:
            l_sum = torch.zeros(l_h.size(0), l_h.size(-1), device=l_h.device, dtype=l_h.dtype)
            z_delta = self.delta_head(l_sum).squeeze(-1)
            logit = z_delta
            aux["z_delta"] = z_delta
            aux["logit"] = logit
            return logit, aux

        # Q = ligand, K = protein, V = ligand
        q = self.q_proj(l_tok)
        k = self.k_proj(p_tok)
        S = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))

        S = S.masked_fill(p_pad.unsqueeze(1), -1e9)
        S = S.masked_fill(l_pad.unsqueeze(-1), 0.0)

        bad = (p_pad.all(dim=1) | l_pad.all(dim=1)).view(-1, 1, 1)
        S = torch.where(bad, torch.zeros_like(S), S)

        A = torch.softmax(S, dim=-1)  # softmax over protein
        A = A.masked_fill(l_pad.unsqueeze(-1), 0.0)
        A = A.masked_fill(p_pad.unsqueeze(1), 0.0)
        context = torch.bmm(A, p_tok)  # (B, Ll, D)
        # ligand token importance
        l_imp = A.mean(dim=-1).masked_fill(l_pad, 0.0)
        l_imp = l_imp / l_imp.sum(dim=1, keepdim=True).clamp(min=1e-6)

        l_sum = torch.bmm(l_imp.unsqueeze(1), context).squeeze(1)  # (B, D)

        # z_delta_in = torch.cat([l_cls, l_sum], dim=-1)
        z_delta = self.delta_head(l_sum).squeeze(-1)

        logit = z_delta
        aux["z_delta"] = z_delta
        aux["logit"] = logit

        if return_maps:
            aux["qk_scores"] = S.detach()  # before softmax
            aux["attn_map"] = A.detach()  # after softmax
            aux["lig_importance"] = l_imp.detach()
            aux["p_pad"] = p_pad.detach()
            aux["l_pad"] = l_pad.detach()

        return logit, aux

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

            prob = torch.sigmoid(logit)
            prob_list.append(prob.detach().float().cpu().numpy())
            ybin_list.append(batch.y_bin.detach().cpu().numpy())

    y_prob = np.concatenate(prob_list, axis=0) if prob_list else np.array([], dtype=np.float64)
    y_bin = np.concatenate(ybin_list, axis=0) if ybin_list else np.array([], dtype=np.float64)
    return y_prob, y_bin


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
) -> Dict[str, float]:
    model.train()
    losses, losses_main = [], []
    use_amp = (device.type == "cuda")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

    pbar = tqdm(total=len(loader), desc="train", leave=False, dynamic_ncols=True)
    for batch in loader:
        p_ids = batch.p_input_ids.to(device, non_blocking=True)
        p_msk = batch.p_attn_mask.to(device, non_blocking=True)
        l_ids = batch.l_ids.to(device, non_blocking=True)
        y_bin = batch.y_bin.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, aux = model(p_ids, p_msk, l_ids)
                loss_main = bce(logit, y_bin)
                loss = loss_main
        else:
            logit, aux = model(p_ids, p_msk, l_ids)
            loss_main = bce(logit, y_bin)
            loss = loss_main

        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        losses_main.append(float(loss_main.detach().cpu().item()))
        pbar.update(1)
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    pbar.close()
    return {
        "loss": float(np.mean(losses)),
        "loss_main": float(np.mean(losses_main)),
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

    ap.add_argument("--select_on", type=str, default="ap", choices=["ap", "auroc", "f1"])

    ap.add_argument("--llrd", action="store_true")
    ap.add_argument("--llrd_decay", type=float, default=0.95)
    ap.add_argument("--esm_lr_mult", type=float, default=1.0)
    ap.add_argument("--esm_min_lr_mult", type=float, default=0.05)
    ap.add_argument("--freeze_esm_bottom", type=int, default=0)

    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--y_thr", type=float, default=Y_THR)
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
    model = QKOnlyDTIClassifier(protein_encoder=prot_enc, ligand_encoder=lig_enc, dropout=args.dropout).to(device)

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

    valid_ds = DTIDataset(args.valid_csv, y_thr=float(args.y_thr), drop_missing_y=True)
    valid_loader = make_loader(valid_ds, shuffle=False)

    final_eval_loader = None
    if args.final_eval_csv:
        final_eval_ds = DTIDataset(args.final_eval_csv, y_thr=float(args.y_thr), drop_missing_y=True)
        final_eval_loader = make_loader(final_eval_ds, shuffle=False)

    test_loader = None
    if args.test_csv:
        test_ds = DTIDataset(args.test_csv, y_thr=float(args.y_thr), drop_missing_y=True)
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
        yhat_v, yb_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)
        print("[VALID]", v_m)

        test_m = None
        if test_loader is not None:
            yhat_t, yb_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            print("[TEST ]", test_m)

        save_json(os.path.join(args.out_dir, "eval_only.json"), {
            "mode": "eval_only",
            "valid": v_m,
            "test": test_m,
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
            fixed_train_ds = DTIDataset(args.train_csv, y_thr=float(args.y_thr), drop_missing_y=True)
            fixed_train_loader = make_loader(fixed_train_ds, shuffle=True)
        else:
            # use first N rows
            all_rows = read_csv_rows(args.train_csv)
            train_rows = all_rows[: int(args.train_size)]
            fixed_train_ds = DTIDataset(rows=train_rows, y_thr=float(args.y_thr), drop_missing_y=True)
            fixed_train_loader = make_loader(fixed_train_ds, shuffle=True)
    else:
        train_shard_paths = list_train_shards(args.train_shard_dir, args.train_shard_glob)

    for ep in range(1, args.epochs + 1):
        qk_save_dir = os.path.join(args.out_dir, "qk_maps")
        if args.use_train_valid_csv and args.train_size is not None:
            epoch_train_ds = fixed_train_ds
            train_loader = fixed_train_loader
        elif args.use_train_valid_csv:
            epoch_train_ds = fixed_train_ds
            train_loader = fixed_train_loader
        else:
            chosen_shards = pick_epoch_shards_random(
                train_shard_paths,
                train_size=int(args.train_size) if args.train_size is not None else len(train_shard_paths) * args.train_shard_size,
                shard_size=int(args.train_shard_size),
                epoch=ep,
                seed=int(args.seed),
                num_shards_per_epoch=args.train_num_shards_per_epoch,
            )
            epoch_train_ds = build_train_dataset_from_shards(chosen_shards, y_thr=float(args.y_thr))
            train_loader = make_loader(epoch_train_ds, shuffle=True)

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
        )

        yhat_tr, yb_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(yhat_tr, yb_tr)

        yhat_v, yb_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)

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

        test_m = None
        if test_loader is not None:
            yhat_t, yb_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            print("[TEST ]", test_m)

        final_m = None
        if final_eval_loader is not None:
            yhat_f, yb_f = predict(model, final_eval_loader, device)
            final_m = eval_metrics(yhat_f, yb_f)

        save_json(os.path.join(args.out_dir, f"epoch_{ep:03d}.json"), {
            "epoch": ep,
            "train_stat": tr_stat,
            "train_metrics": tr_m,
            "valid_metrics": v_m,
            "final_eval_metrics": final_m,
            "test_metrics": test_m,
            "pos_weight": pos_weight,
        })

        print(f"\n===== Epoch {ep} =====")
        print("[train]", f"AUC={tr_m['auroc']:.4f}", f"AP={tr_m['ap']:.4f}", f"F1={tr_m['f1']:.4f}", f"EF1={tr_m['ef1']:.3f}", f"EF5={tr_m['ef5']:.3f}", f"EF10={tr_m['ef10']:.3f}")
        print("[valid]", f"AUC={v_m['auroc']:.4f}", f"AP={v_m['ap']:.4f}", f"F1={v_m['f1']:.4f}", f"EF1={v_m['ef1']:.3f}", f"EF5={v_m['ef5']:.3f}", f"EF10={v_m['ef10']:.3f}")
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
