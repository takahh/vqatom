#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import csv
import json
import random
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


def split_rows(rows: List[Dict[str, str]], valid_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not (0.0 <= valid_ratio < 1.0):
        raise ValueError("valid_ratio must satisfy 0 <= valid_ratio < 1")

    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_valid = int(len(idx) * valid_ratio)
    valid_idx = set(idx[:n_valid])

    train_rows, valid_rows = [], []
    for i, r in enumerate(rows):
        if i in valid_idx:
            valid_rows.append(r)
        else:
            train_rows.append(r)

    if len(train_rows) == 0:
        raise ValueError("Train split became empty")
    return train_rows, valid_rows


# =========================================================
# Tokenizer
# =========================================================
class SmilesCharTokenizer:
    PAD = "[PAD]"
    MASK = "[MASK]"
    CLS = "[CLS]"
    UNK = "[UNK]"

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {int(v): k for k, v in self.stoi.items()}

        for tok in [self.PAD, self.MASK, self.CLS, self.UNK]:
            if tok not in self.stoi:
                raise ValueError(f"Missing special token: {tok}")

        self.pad_id = int(self.stoi[self.PAD])
        self.mask_id = int(self.stoi[self.MASK])
        self.cls_id = int(self.stoi[self.CLS])
        self.unk_id = int(self.stoi[self.UNK])
        self.vocab_size = int(len(self.stoi))

    @classmethod
    def build_from_rows(cls, rows: List[Dict[str, str]]) -> "SmilesCharTokenizer":
        chars = set()
        for r in rows:
            smi = (r.get("smiles") or "").strip()
            if smi:
                chars.update(list(smi))

        toks = [cls.PAD, cls.MASK, cls.CLS, cls.UNK] + sorted(chars)
        stoi = {tok: i for i, tok in enumerate(toks)}
        return cls(stoi)

    def to_json(self, path: str) -> None:
        obj = {
            "stoi": self.stoi,
            "tokens": [self.itos[i] for i in range(len(self.itos))],
            "pad_id": self.pad_id,
            "mask_id": self.mask_id,
            "cls_id": self.cls_id,
            "unk_id": self.unk_id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SmilesCharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["stoi"])

    def encode(self, smiles: str, add_cls: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in smiles]
        if add_cls:
            ids = [self.cls_id] + ids
        return ids


# =========================================================
# Dataset
# =========================================================
class SmilesMLMDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, str]],
        tokenizer: SmilesCharTokenizer,
        mask_prob: float = 0.15,
        max_len: Optional[int] = None,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.mask_prob = float(mask_prob)
        self.max_len = max_len

        self.smiles_list = []
        for r in rows:
            smi = (r.get("smiles") or "").strip()
            if smi:
                self.smiles_list.append(smi)

        if not self.smiles_list:
            raise ValueError("No usable smiles found")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def _truncate(self, ids: List[int]) -> List[int]:
        if self.max_len is None or len(ids) <= self.max_len:
            return ids
        return ids[: self.max_len]

    def _mask_ids(self, ids: List[int]):
        x = torch.tensor(ids, dtype=torch.long)
        y = x.clone()

        maskable = torch.ones_like(x, dtype=torch.bool)
        maskable[0] = False  # CLSはmaskしない

        rand = torch.rand(x.shape)
        do_mask = (rand < self.mask_prob) & maskable

        rand2 = torch.rand(x.shape)
        replace_mask = do_mask & (rand2 < 0.8)
        replace_rand = do_mask & (rand2 >= 0.8) & (rand2 < 0.9)

        x[replace_mask] = self.tokenizer.mask_id

        if replace_rand.any():
            low = 4
            high = self.tokenizer.vocab_size
            x[replace_rand] = torch.randint(low, high, (int(replace_rand.sum().item()),), dtype=torch.long)

        y[~do_mask] = -100
        return x, y

    def __getitem__(self, idx: int):
        smi = self.smiles_list[idx]
        ids = self.tokenizer.encode(smi, add_cls=True)
        ids = self._truncate(ids)
        input_ids, labels = self._mask_ids(ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "smiles": smi,
        }


def collate_smiles_mlm(samples: List[dict], pad_id: int) -> dict:
    max_len = max(int(s["input_ids"].numel()) for s in samples)
    B = len(samples)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, s in enumerate(samples):
        n = int(s["input_ids"].numel())
        input_ids[i, :n] = s["input_ids"]
        labels[i, :n] = s["labels"]
        attention_mask[i, :n] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "smiles": [s["smiles"] for s in samples],
    }


# =========================================================
# Model
# =========================================================
class SmilesTransformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 512,
        nhead: int = 16,
        layers: int = 10,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        self.tok = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(self.max_len, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_ff),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len={self.max_len}")

        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(input_ids) + self.pos(pos_ids)
        x = self.ln(x)

        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.enc(x, src_key_padding_mask=pad_mask)
        logits = self.head(h)
        return logits


# =========================================================
# Logging / save
# =========================================================
def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_checkpoint(path: str, model: nn.Module, args, tokenizer: SmilesCharTokenizer, epoch: int, best: dict):
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "config": {
                "d_model": int(args.d_model),
                "nhead": int(args.nhead),
                "layers": int(args.layers),
                "dim_ff": int(args.dim_ff),
                "dropout": float(args.dropout),
                "max_len": int(args.max_len),
            },
            "vocab_size": int(tokenizer.vocab_size),
            "pad_id": int(tokenizer.pad_id),
            "mask_id": int(tokenizer.mask_id),
            "cls_id": int(tokenizer.cls_id),
            "unk_id": int(tokenizer.unk_id),
            "args": vars(args),
            "best": dict(best),
        },
        path,
    )


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = (labels != -100)
    n = int(mask.sum().item())
    if n == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == labels[mask]).float().mean().item())


# =========================================================
# Train / eval
# =========================================================
def run_epoch(model, loader, optimizer, device, grad_clip=1.0, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = (device.type == "cuda")

    losses, accs = [], []
    pbar = tqdm(loader, desc="train" if train else "valid", leave=False, dynamic_ncols=True)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attn)
                loss = ce(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
        else:
            logits = model(input_ids, attn)
            loss = ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        acc = masked_accuracy(logits.detach(), labels)

        losses.append(float(loss.detach().cpu().item()))
        accs.append(float(acc))

        pbar.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.4f}")

    pbar.close()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mlm_acc": float(np.mean(accs)) if accs else 0.0,
    }


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_file", type=str, default=None)

    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--nhead", type=int, default=16)
    ap.add_argument("--dim_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    if args.log_file is None:
        args.log_file = os.path.join(args.out_dir, "train_log.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_csv_rows(args.data_csv)
    rows = [r for r in rows if (r.get("smiles") or "").strip()]
    if not rows:
        raise ValueError("No rows with smiles found")

    train_rows, valid_rows = split_rows(rows, valid_ratio=float(args.valid_ratio), seed=int(args.split_seed))
    print(f"[split] train={len(train_rows)} valid={len(valid_rows)}")

    tokenizer = SmilesCharTokenizer.build_from_rows(train_rows)
    vocab_path = os.path.join(args.out_dir, "smiles_vocab.json")
    tokenizer.to_json(vocab_path)
    print(f"[vocab] size={tokenizer.vocab_size} saved={vocab_path}")

    train_ds = SmilesMLMDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        mask_prob=float(args.mask_prob),
        max_len=int(args.max_len),
    )
    valid_ds = SmilesMLMDataset(
        rows=valid_rows,
        tokenizer=tokenizer,
        mask_prob=float(args.mask_prob),
        max_len=int(args.max_len),
    ) if len(valid_rows) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda xs: collate_smiles_mlm(xs, pad_id=tokenizer.pad_id),
    )
    valid_loader = None if valid_ds is None else DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda xs: collate_smiles_mlm(xs, pad_id=tokenizer.pad_id),
    )

    model = SmilesTransformerMLM(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        layers=int(args.layers),
        dim_ff=int(args.dim_ff),
        dropout=float(args.dropout),
        max_len=int(args.max_len),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best = {"loss": 1e18, "mlm_acc": -1.0, "epoch": -1}

    for ep in range(1, int(args.epochs) + 1):
        print(f"\n===== Epoch {ep} =====")

        tr = run_epoch(model, train_loader, optimizer, device, grad_clip=float(args.grad_clip), train=True)
        print(f"[train] loss={tr['loss']:.4f} mlm_acc={tr['mlm_acc']:.4f}")

        va = None
        if valid_loader is not None:
            va = run_epoch(model, valid_loader, None, device, grad_clip=0.0, train=False)
            print(f"[valid] loss={va['loss']:.4f} mlm_acc={va['mlm_acc']:.4f}")

        stat = {
            "epoch": ep,
            "train": tr,
            "valid": va,
        }
        append_jsonl(args.log_file, stat)

        save_checkpoint(
            os.path.join(args.out_dir, "last.pt"),
            model=model,
            args=args,
            tokenizer=tokenizer,
            epoch=ep,
            best=best,
        )

        score_loss = va["loss"] if va is not None else tr["loss"]
        if score_loss < best["loss"]:
            best = {
                "loss": float(score_loss),
                "mlm_acc": float(va["mlm_acc"] if va is not None else tr["mlm_acc"]),
                "epoch": int(ep),
            }
            save_checkpoint(
                os.path.join(args.out_dir, "best.pt"),
                model=model,
                args=args,
                tokenizer=tokenizer,
                epoch=ep,
                best=best,
            )
            print(f"  saved: {os.path.join(args.out_dir, 'best.pt')}")

    print("\nDone")
    print("Best:", best)
    print("Log:", args.log_file)


if __name__ == "__main__":
    main()