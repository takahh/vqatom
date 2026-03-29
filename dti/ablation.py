#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DTI regression/classification with explicit protein baseline + ligand delta.

Model:
    y_base  = f(protein)
    y_delta = g(protein, ligand)
    y_hat   = y_base + y_delta

Training targets:
    y_base  -> y_avg_per_protein
    y_delta -> delta
    y_hat   -> y

Evaluation:
    final score = y_hat
    binary label = 1 if y >= Y_THR else 0
    AUC/AP/F1/EF are computed from y_hat against y_bin

Expected CSV columns:
  required:
    - seq
    - lig_tok
    - y
  recommended:
    - y_avg_per_protein
    - delta

If y_avg_per_protein / delta are missing:
  - y_avg_per_protein becomes NaN
  - delta becomes NaN unless computable
"""

from glob import glob
import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm

Y_THR = 7.0


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
    import csv
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


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
) -> ConcatDataset:
    ds_list = []
    for p in shard_paths:
        ds = DTIDataset(
            csv_path=p,
            y_thr=float(y_thr),
            drop_missing_y=True,
        )
        ds_list.append(ds)

    if not ds_list:
        raise ValueError("No train shards were loaded.")

    return ConcatDataset(ds_list)


# =========================================================
# Ligand token parser
# =========================================================
def parse_lig_tokens(s: str) -> List[int]:
    s = str(s).strip().replace(",", " ")
    if not s:
        return []
    return [int(x) for x in s.split()]


# =========================================================
# Batch / collate
# =========================================================
@dataclass
class Batch:
    p_input_ids: torch.Tensor
    p_attn_mask: torch.Tensor
    l_ids: torch.Tensor
    y_bin: torch.Tensor
    y: torch.Tensor
    y_avg: torch.Tensor
    delta: torch.Tensor


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    if not seqs:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(int(x.numel()) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        if x.numel() > 0:
            out[i, :x.numel()] = x
    return out


def collate_fn(samples, esm_tokenizer, lig_pad: int, lig_cls: int) -> Batch:
    seqs = [s["seq"] for s in samples]

    enc = esm_tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
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
    y_avg = torch.tensor([s["y_avg_per_protein"] for s in samples], dtype=torch.float32)
    delta = torch.tensor([s["delta"] for s in samples], dtype=torch.float32)

    return Batch(
        p_input_ids=p_input_ids,
        p_attn_mask=p_attn_mask,
        l_ids=l_ids,
        y_bin=y_bin,
        y=y,
        y_avg=y_avg,
        delta=delta,
    )


# =========================================================
# Vocab/meta utils
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
# Models
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
        h = self.enc(x, src_key_padding_mask=pad_mask)
        return h


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


class DTIDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        rows: Optional[list] = None,
        y_thr: float = 7.0,
        drop_missing_y: bool = True,
    ):
        if rows is not None:
            raw_rows = rows
        elif csv_path is not None:
            raw_rows = read_csv_rows(csv_path)
        else:
            raise ValueError("Either csv_path or rows must be provided")

        self.rows = []
        n_drop_seq = 0
        n_drop_lig = 0
        n_drop_y = 0

        for r in raw_rows:
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

            y_avg_raw = r.get("y_avg_per_protein", "")
            delta_raw = r.get("delta", "")

            if y_avg_raw in ("", None):
                y_avg = float("nan")
            else:
                y_avg = float(y_avg_raw)

            if delta_raw in ("", None):
                delta = y - y_avg if math.isfinite(y_avg) else float("nan")
            else:
                delta = float(delta_raw)

            rr = dict(r)
            rr["y"] = y
            rr["y_bin"] = 1 if y >= float(y_thr) else 0
            rr["y_avg_per_protein"] = y_avg
            rr["delta"] = delta
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

        self.base_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.lig_pad_id = int(self.lig.pad_id)

    def forward(self, p_input_ids, p_attn_mask, l_ids):
        aux = {}

        p_h = self.prot(p_input_ids, p_attn_mask)
        if self.p_proj is not None:
            p_h = self.p_proj(p_h)

        l_h = self.lig(l_ids)

        prot_pad_mask = (p_attn_mask == 0)
        lig_pad_mask = (l_ids == self.lig_pad_id)

        p_tok = p_h[:, 1:, :]
        l_tok = l_h[:, 1:, :]
        p_cls = p_h[:, 0, :]
        l_cls = l_h[:, 0, :]
        p_pad = prot_pad_mask[:, 1:]
        l_pad = lig_pad_mask[:, 1:]

        y_base = self.base_head(p_cls).squeeze(-1)

        if p_tok.size(1) == 0 or l_tok.size(1) == 0:
            z = torch.cat([p_cls, l_cls, p_cls, l_cls], dim=-1)
            y_delta = self.delta_head(z).squeeze(-1)
            y_hat = y_base + y_delta
            aux["y_base"] = y_base
            aux["y_delta"] = y_delta
            aux["y_hat"] = y_hat
            return y_hat, aux

        q = self.q_proj(p_tok)
        k = self.k_proj(l_tok)

        S = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        S = S.masked_fill(l_pad.unsqueeze(1), -1e9)
        S = S.masked_fill(p_pad.unsqueeze(-1), 0.0)

        bad = (p_pad.all(dim=1) | l_pad.all(dim=1)).view(-1, 1, 1)
        S = torch.where(bad, torch.zeros_like(S), S)

        A = torch.softmax(S, dim=-1)
        A = A.masked_fill(p_pad.unsqueeze(-1), 0.0)
        A = A.masked_fill(l_pad.unsqueeze(1), 0.0)

        p_imp = A.max(dim=-1).values
        l_imp = A.max(dim=1).values

        p_imp = p_imp.masked_fill(p_pad, 0.0)
        l_imp = l_imp.masked_fill(l_pad, 0.0)

        p_imp = p_imp / p_imp.sum(dim=1, keepdim=True).clamp(min=1e-6)
        l_imp = l_imp / l_imp.sum(dim=1, keepdim=True).clamp(min=1e-6)

        p_sum = torch.bmm(p_imp.unsqueeze(1), p_tok).squeeze(1)
        l_sum = torch.bmm(l_imp.unsqueeze(1), l_tok).squeeze(1)

        z = torch.cat([p_cls, l_cls, p_sum, l_sum], dim=-1)

        y_delta = self.delta_head(z).squeeze(-1)
        y_hat = y_base + y_delta

        aux["y_base"] = y_base
        aux["y_delta"] = y_delta
        aux["y_hat"] = y_hat
        return y_hat, aux


# =========================================================
# Predict / metrics
# =========================================================
def predict(model, loader, device):
    model.eval()
    yhat_list, ybin_list, y_list = [], [], []
    use_amp = (device.type == "cuda")

    with torch.inference_mode():
        for batch in loader:
            p_ids = batch.p_input_ids.to(device, non_blocking=True)
            p_msk = batch.p_attn_mask.to(device, non_blocking=True)
            l_ids = batch.l_ids.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y_hat, _ = model(p_ids, p_msk, l_ids)
            else:
                y_hat, _ = model(p_ids, p_msk, l_ids)

            yhat_list.append(y_hat.detach().float().cpu().numpy())
            ybin_list.append(batch.y_bin.detach().cpu().numpy())
            y_list.append(batch.y.detach().cpu().numpy())

    y_hat = np.concatenate(yhat_list, axis=0) if yhat_list else np.array([], dtype=np.float64)
    y_bin = np.concatenate(ybin_list, axis=0) if ybin_list else np.array([], dtype=np.float64)
    y = np.concatenate(y_list, axis=0) if y_list else np.array([], dtype=np.float64)
    return y_hat, y_bin, y


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
        return float(hit_rate_topk / base_rate) if base_rate > 0 else 0.0

    if y_pred.size == 0:
        return {
            "auroc": 0.0, "ap": 0.0, "f1": 0.0, "thr": Y_THR,
            "pred_mean": 0.0, "pred_std": 0.0, "ef1": 0.0, "ef5": 0.0, "ef10": 0.0,
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
            "thr": float(Y_THR),
            "pred_mean": float(score.mean()),
            "pred_std": float(score.std()),
            "pred_min": float(score.min()),
            "pred_max": float(score.max()),
            "pos_rate": float(y01.mean()) if y01.size else 0.0,
            "pred_pos_rate@thr": float((score >= Y_THR).mean()) if score.size else 0.0,
            "ef1": float(ef1),
            "ef5": float(ef5),
            "ef10": float(ef10),
        }

    auroc = roc_auc_score(y01, score)
    ap = average_precision_score(y01, score)
    pred01 = (score >= Y_THR).astype(np.int32)
    f1 = f1_score(y01, pred01)

    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "f1": float(f1),
        "thr": float(Y_THR),
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
        epoch: int,
        grad_clip: float = 1.0,
        reg_lambda: float = 1.0,
        base_lambda: float = 1.0,
        delta_lambda: float = 1.0,
) -> Dict[str, float]:
    model.train()
    losses, losses_y, losses_base, losses_delta = [], [], [], []
    use_amp = (device.type == "cuda")

    pbar = tqdm(total=len(loader), desc="train", leave=False, dynamic_ncols=True)
    cur_base_lambda = 0.3
    cur_delta_lambda = 1.0

    for batch in loader:
        p_ids = batch.p_input_ids.to(device, non_blocking=True)
        p_msk = batch.p_attn_mask.to(device, non_blocking=True)
        l_ids = batch.l_ids.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        y_avg = batch.y_avg.to(device, non_blocking=True)
        delta = batch.delta.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_hat, aux = model(p_ids, p_msk, l_ids)
                y_base = aux["y_base"]
                y_delta = aux["y_delta"]

                loss_y = F.smooth_l1_loss(y_hat, y)
                loss_base = F.smooth_l1_loss(y_base, y_avg)
                delta_target = y - y_base.detach()
                loss_delta = F.smooth_l1_loss(y_delta, delta_target)
                loss = reg_lambda * loss_y + cur_base_lambda * loss_base + cur_delta_lambda * loss_delta
        else:
            y_hat, aux = model(p_ids, p_msk, l_ids)
            y_base = aux["y_base"]
            y_delta = aux["y_delta"]

            loss_y = F.smooth_l1_loss(y_hat, y)
            loss_base = F.smooth_l1_loss(y_base, y_avg)
            delta_target = y - y_base.detach()
            loss_delta = F.smooth_l1_loss(y_delta, delta_target)
            loss = reg_lambda * loss_y + cur_base_lambda * loss_base + cur_delta_lambda * loss_delta

        loss.backward()
        if grad_clip and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        losses_y.append(float(loss_y.detach().cpu().item()))
        losses_base.append(float(loss_base.detach().cpu().item()))
        losses_delta.append(float(loss_delta.detach().cpu().item()))

        pbar.update(1)
        pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    pbar.close()

    return {
        "loss": float(np.mean(losses)),
        "loss_y": float(np.mean(losses_y)),
        "loss_base": float(np.mean(losses_base)),
        "loss_delta": float(np.mean(losses_delta)),
    }


# =========================================================
# Save/load
# =========================================================
def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dti_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    lig_enc: PretrainedLigandEncoder,
    epoch: int,
    best: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "model_type": getattr(args, "model_type", "qkonly"),
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
                lr_i = top_lr * (decay ** depth_from_top)
                lr_i = max(lr_i, top_lr * min_mult)
                param_groups.append({"params": layer_params, "lr": lr_i, "name": f"esm.layer{i} lr={lr_i:g}"})

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

    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--base_lambda", type=float, default=1.0)
    ap.add_argument("--delta_lambda", type=float, default=1.0)

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
    lig_pad = lig_enc.pad_id

    prot_enc = ESMProteinEncoder(
        model_name=args.esm_model,
        device=device,
        finetune=args.finetune_esm,
    )
    model = QKOnlyDTIClassifier(
        protein_encoder=prot_enc,
        ligand_encoder=lig_enc,
        dropout=args.dropout,
    ).to(device)

    if args.dti_ckpt is not None:
        _, missing, unexpected = load_dti_checkpoint(args.dti_ckpt, model, device)
        print(f"Loaded DTI checkpoint: {args.dti_ckpt}")
        if missing:
            print("  [warn] missing keys (up to 20):", missing[:20])
        if unexpected:
            print("  [warn] unexpected keys (up to 20):", unexpected[:20])

    # -----------------------------
    # datasets/loaders
    # -----------------------------
    valid_ds = DTIDataset(args.valid_csv, y_thr=float(args.y_thr), drop_missing_y=True)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
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

    train_ds = None
    train_shard_paths = None
    train_loader = None
    loader_num_workers = 0
    pin_memory = False

    if not args.eval_only:
        if args.use_train_valid_csv:
            if args.train_size is None:
                train_ds = DTIDataset(args.train_csv, y_thr=float(args.y_thr), drop_missing_y=True)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=loader_num_workers,
                    pin_memory=pin_memory,
                    collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
                )
        else:
            train_shard_paths = list_train_shards(args.train_shard_dir, args.train_shard_glob)

    if args.eval_only:
        yhat_v, yb_v, y_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)
        print("[VALID]", v_m)

        test_m = None
        if test_loader is not None:
            yhat_t, yb_t, y_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            print("[TEST ]", test_m)

        summary = {
            "mode": "eval_only",
            "valid": v_m,
            "test": test_m,
            "valid_csv": args.valid_csv,
            "test_csv": args.test_csv,
            "dti_ckpt": args.dti_ckpt,
        }
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
            train_rows = read_csv_random_rows(args.train_csv, int(args.train_size), seed=epoch_seed)
            epoch_train_ds = DTIDataset(rows=train_rows, y_thr=float(args.y_thr), drop_missing_y=True)
            train_loader = DataLoader(
                epoch_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=loader_num_workers,
                pin_memory=pin_memory,
                collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
            )
            print("epoch train n:", len(epoch_train_ds))

        elif args.use_train_valid_csv:
            print("epoch train n:", len(train_ds))

        elif train_shard_paths is not None:
            chosen_shards = pick_epoch_shards_random(
                train_shard_paths,
                train_size=int(args.train_size) if args.train_size is not None else len(train_shard_paths) * args.train_shard_size,
                shard_size=int(args.train_shard_size),
                epoch=ep,
                seed=int(args.seed),
                num_shards_per_epoch=args.train_num_shards_per_epoch,
            )
            epoch_train_ds = build_train_dataset_from_shards(chosen_shards, y_thr=float(args.y_thr))
            train_loader = DataLoader(
                epoch_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=loader_num_workers,
                pin_memory=pin_memory,
                collate_fn=lambda xs: collate_fn(xs, esm_tokenizer=esm_tokenizer, lig_pad=lig_pad, lig_cls=lig_enc.cls_id),
            )
            print("epoch train n:", len(epoch_train_ds))

        else:
            raise ValueError("No training data source configured")

        tr_stat = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=ep,
            grad_clip=float(args.grad_clip),
            reg_lambda=float(args.reg_lambda),
            base_lambda=float(args.base_lambda),
            delta_lambda=float(args.delta_lambda),
        )

        yhat_tr, yb_tr, y_tr = predict(model, train_loader, device)
        tr_m = eval_metrics(yhat_tr, yb_tr)

        yhat_v, yb_v, y_v = predict(model, valid_loader, device)
        v_m = eval_metrics(yhat_v, yb_v)

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
                "epoch": ep,
            })
            save_path = os.path.join(args.out_dir, "best.pt")
            save_dti_checkpoint(save_path, model, args, lig_enc, epoch=ep, best=best)
            print("  saved:", save_path)

        last_path = os.path.join(args.out_dir, "last.pt")
        save_dti_checkpoint(last_path, model, args, lig_enc, epoch=ep, best=best)

        test_m = None
        if test_loader is not None:
            yhat_t, yb_t, y_t = predict(model, test_loader, device)
            test_m = eval_metrics(yhat_t, yb_t)
            print("[TEST ]", test_m)

        final_m = None
        if final_eval_loader is not None:
            yhat_f, yb_f, y_f = predict(model, final_eval_loader, device)
            final_m = eval_metrics(yhat_f, yb_f)

        epoch_summary = {
            "epoch": ep,
            "train_stat": tr_stat,
            "train_metrics": tr_m,
            "valid_metrics": v_m,
            "final_eval_metrics": final_m,
            "test_metrics": test_m,
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