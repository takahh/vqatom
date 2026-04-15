#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import random
import argparse
import json
import time
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Tokenizer
# ============================================================
class SmilesCharTokenizer:
    PAD = "[PAD]"
    MASK = "[MASK]"
    CLS = "[CLS]"
    UNK = "[UNK]"

    def __init__(self, stoi: dict):
        self.stoi = dict(stoi)
        self.itos = {int(v): k for k, v in self.stoi.items()}

        for tok in [self.PAD, self.MASK, self.CLS, self.UNK]:
            if tok not in self.stoi:
                raise ValueError(f"Missing special token in vocab: {tok}")

        self.pad_id = int(self.stoi[self.PAD])
        self.mask_id = int(self.stoi[self.MASK])
        self.cls_id = int(self.stoi[self.CLS])
        self.unk_id = int(self.stoi[self.UNK])
        self.vocab_size = int(len(self.stoi))

    @classmethod
    def from_json(cls, path: str) -> "SmilesCharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "stoi" not in obj:
            raise ValueError("vocab json must contain key 'stoi'")
        return cls(obj["stoi"])

    def encode(self, smiles: str, add_cls: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in smiles]
        if add_cls:
            ids = [self.cls_id] + ids
        return ids


# ============================================================
# MLM masking
# ============================================================
def mlm_mask_tokens(
    input_ids: torch.Tensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input_ids: (B, L) int64 with PAD included
    returns:
      masked_input_ids: (B, L)
      labels: (B, L) where non-masked = -100
    """
    assert input_ids.dtype == torch.int64

    labels = torch.full_like(input_ids, -100)

    is_pad = (input_ids == pad_id)
    is_cls = (input_ids == (mask_id - 1)) if False else torch.zeros_like(input_ids, dtype=torch.bool)
    # CLSは pad_id/mask_id から自動推定しない。あとで明示処理する方が安全。

    prob = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_sel = (prob < mask_prob) & (~is_pad)

    labels[mask_sel] = input_ids[mask_sel]

    r = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)

    m_mask = mask_sel & (r < 0.80)
    m_rand = mask_sel & (r >= 0.80) & (r < 0.90)
    # keep = remaining selected

    masked = input_ids.clone()
    masked[m_mask] = mask_id

    if m_rand.any():
        rand_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(int(m_rand.sum().item()),),
            device=input_ids.device,
            dtype=torch.int64,
            generator=generator,
        )
        masked[m_rand] = rand_ids

    return masked, labels


def mlm_mask_tokens_with_special_control(
    input_ids: torch.Tensor,
    pad_id: int,
    mask_id: int,
    cls_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full_like(input_ids, -100)

    is_pad = (input_ids == pad_id)
    is_cls = (input_ids == cls_id)

    prob = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_sel = (prob < mask_prob) & (~is_pad) & (~is_cls)

    labels[mask_sel] = input_ids[mask_sel]

    r = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    m_mask = mask_sel & (r < 0.80)
    m_rand = mask_sel & (r >= 0.80) & (r < 0.90)

    masked = input_ids.clone()
    masked[m_mask] = mask_id

    if m_rand.any():
        rand_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(int(m_rand.sum().item()),),
            device=input_ids.device,
            dtype=torch.int64,
            generator=generator,
        )
        masked[m_rand] = rand_ids

    return masked, labels


# ============================================================
# Dataset
# ============================================================
class SmilesShardDataset(Dataset):
    """
    Loads smiles_pretrain_batchXXX.pt files and exposes each SMILES as one item.

    Each shard file is expected to contain either:
      {"smiles": ["CCO", "c1ccccc1", ...]}
    or
      {"rows": [{"smiles": "CCO"}, {"smiles": "c1ccccc1"}, ...]}
    """

    def __init__(self, root_dir: str, tokenizer: SmilesCharTokenizer, file_list: List[str]):
        self.root_dir = root_dir
        self.tokenizer = tokenizer

        self.files = []
        for p in file_list:
            if os.path.isabs(p):
                self.files.append(p)
            else:
                self.files.append(os.path.join(root_dir, p))
        self.files = sorted(self.files)

        if not self.files:
            raise FileNotFoundError("No shard files provided")

        self.index = []
        self.file_meta = []

        for fi, fp in enumerate(self.files):
            d = torch.load(fp, map_location="cpu", weights_only=False)

            smiles_list = self._extract_smiles_list(d)
            self.file_meta.append((fp, len(smiles_list)))

            for mi in range(len(smiles_list)):
                self.index.append((fi, mi))

        self._cache_fi = None
        self._cache_smiles = None

    def _extract_smiles_list(self, d) -> List[str]:
        if isinstance(d, dict):
            if "smiles" in d:
                smiles_list = [str(x).strip() for x in d["smiles"] if str(x).strip()]
                return smiles_list
            if "rows" in d:
                smiles_list = []
                for r in d["rows"]:
                    smi = ""
                    if isinstance(r, dict):
                        smi = str(r.get("smiles", "")).strip()
                    if smi:
                        smiles_list.append(smi)
                return smiles_list

        raise ValueError("Each shard .pt must contain key 'smiles' or 'rows'")

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi: int) -> List[str]:
        if self._cache_fi == fi and self._cache_smiles is not None:
            return self._cache_smiles

        fp, _ = self.file_meta[fi]
        d = torch.load(fp, map_location="cpu", weights_only=False)
        smiles_list = self._extract_smiles_list(d)

        self._cache_fi = fi
        self._cache_smiles = smiles_list
        return smiles_list

    def __getitem__(self, idx):
        fi, mi = self.index[idx]
        smiles_list = self._load_file(fi)
        smi = smiles_list[mi]
        ids = self.tokenizer.encode(smi, add_cls=True)
        return torch.tensor(ids, dtype=torch.int64)


def collate_pad(batch: List[torch.Tensor], pad_id: int):
    lens = torch.tensor([x.numel() for x in batch], dtype=torch.int64)
    B = len(batch)
    L = int(lens.max().item())

    input_ids = torch.full((B, L), pad_id, dtype=torch.int64)
    for i, seq in enumerate(batch):
        input_ids[i, :seq.numel()] = seq

    attn_keep = (torch.arange(L)[None, :] < lens[:, None])  # True for real token
    return input_ids, attn_keep, lens


# ============================================================
# Simple Transformer MLM
# ============================================================
class TransformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, key_padding_mask):
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len={self.max_len}")

        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(input_ids) + self.pos(pos_ids)
        x = self.enc(x, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Utils
# ============================================================
def set_all_seeds(seed: int):
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


def make_batch_file_list(prefix: str, start: int, end: int) -> List[str]:
    if start < 0 or end < 0:
        raise ValueError("start/end must be non-negative")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")
    return [f"{prefix}_batch{i:03d}.pt" for i in range(start, end + 1)]


# ============================================================
# Train loop
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--vocab_json", type=str, required=True)
    ap.add_argument("--file_prefix", type=str, default="smiles_pretrain")
    ap.add_argument("--train_start", type=int, required=True)
    ap.add_argument("--train_end", type=int, required=True)
    ap.add_argument("--valid_start", type=int, default=None)
    ap.add_argument("--valid_end", type=int, default=None)

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--reset_optim", action="store_true")
    ap.add_argument("--reset_lr", action="store_true")

    # model
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=16)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--dim_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=512)

    # io/log
    ap.add_argument("--save_dir", type=str, default="./smiles_mlm_ckpt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--deterministic_masking", action="store_true")

    args = ap.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    write_log, log_fh = make_log_writer(args.log_file)

    tokenizer = SmilesCharTokenizer.from_json(args.vocab_json)
    PAD_ID = tokenizer.pad_id
    MASK_ID = tokenizer.mask_id
    CLS_ID = tokenizer.cls_id
    vocab_size = tokenizer.vocab_size

    train_files = make_batch_file_list(args.file_prefix, args.train_start, args.train_end)

    valid_files = None
    if (args.valid_start is not None) and (args.valid_end is not None):
        valid_files = make_batch_file_list(args.file_prefix, args.valid_start, args.valid_end)

    train_ds = SmilesShardDataset(
        root_dir=args.data_dir,
        tokenizer=tokenizer,
        file_list=train_files,
    )

    valid_ds = None
    if valid_files is not None:
        valid_ds = SmilesShardDataset(
            root_dir=args.data_dir,
            tokenizer=tokenizer,
            file_list=valid_files,
        )

    print(f"Loaded {len(train_ds)} train sequences from {args.data_dir}")
    if valid_ds is not None:
        print(f"Loaded {len(valid_ds)} valid sequences")
    print(f"vocab_size={vocab_size} PAD_ID={PAD_ID} MASK_ID={MASK_ID} CLS_ID={CLS_ID}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_pad(batch, PAD_ID),
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
            collate_fn=lambda batch: collate_pad(batch, PAD_ID),
            drop_last=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMLM(
        vocab_size=vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    resume_step0 = 0
    resume_lr0 = args.lr
    resume_last_epoch = 0

    if args.resume is not None:
        print(f"[resume] loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        if int(ckpt.get("vocab_size", -1)) != int(vocab_size):
            raise RuntimeError(f"[resume] vocab_size mismatch: ckpt={ckpt.get('vocab_size')} current={vocab_size}")

        model.load_state_dict(ckpt["model"], strict=True)

        if (not args.reset_optim) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
                print("[resume] optimizer state loaded")
            except Exception as e:
                print(f"[resume] failed to load optimizer state: {e}")

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

    steps_per_epoch = len(train_loader)

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
            raise RuntimeError(
                f"--epochs must be > ckpt epoch. ckpt={resume_last_epoch} args.epochs={args.epochs}"
            )

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
        running_loss = 0.0
        running_tok = 0

        for it, (input_ids, attn_keep, lens) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_keep = attn_keep.to(device, non_blocking=True)
            key_padding_mask = ~attn_keep

            if mask_gen is not None:
                step_seed = int(args.seed + ep * 1_000_000 + global_step)
                mask_gen.manual_seed(step_seed)

            masked_input, labels = mlm_mask_tokens_with_special_control(
                input_ids=input_ids,
                pad_id=PAD_ID,
                mask_id=MASK_ID,
                cls_id=CLS_ID,
                vocab_size=vocab_size,
                mask_prob=args.mask_prob,
                generator=mask_gen,
            )

            logits = model(masked_input, key_padding_mask=key_padding_mask)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            global_step += 1

            lr_now_val = lr_now(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now_val

            with torch.no_grad():
                masked_positions = int((labels != -100).sum().item())

            running_loss += float(loss.item()) * masked_positions
            running_tok += masked_positions

            if global_step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_tok)
                ppl = math.exp(min(20.0, avg_loss))

                msg = (
                    f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps_disp} "
                    f"mlm_loss={avg_loss:.4f} ppl~{ppl:.2f} lr={lr_now_val:.2e} "
                    f"masked_tokens={running_tok}"
                )
                print(msg)

                write_log({
                    "time": time.time(),
                    "epoch": ep,
                    "step": global_step,
                    "total_steps": total_steps_disp,
                    "mlm_loss": avg_loss,
                    "ppl": ppl,
                    "lr": lr_now_val,
                    "masked_tokens": running_tok,
                    "batch_size": args.batch_size,
                    "mask_prob": args.mask_prob,
                    "deterministic_masking": bool(args.deterministic_masking),
                })

                running_loss, running_tok = 0.0, 0

        if valid_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_tok = 0

            with torch.no_grad():
                for v_it, (v_input_ids, v_attn_keep, v_lens) in enumerate(valid_loader, start=1):
                    v_input_ids = v_input_ids.to(device, non_blocking=True)
                    v_attn_keep = v_attn_keep.to(device, non_blocking=True)
                    v_key_padding_mask = ~v_attn_keep

                    if mask_gen is not None:
                        step_seed = int(args.seed + 9_000_000 + v_it)
                        mask_gen.manual_seed(step_seed)

                    v_masked, v_labels = mlm_mask_tokens_with_special_control(
                        input_ids=v_input_ids,
                        pad_id=PAD_ID,
                        mask_id=MASK_ID,
                        cls_id=CLS_ID,
                        vocab_size=vocab_size,
                        mask_prob=args.mask_prob,
                        generator=mask_gen,
                    )

                    v_logits = model(v_masked, key_padding_mask=v_key_padding_mask)
                    v_loss = F.cross_entropy(
                        v_logits.reshape(-1, vocab_size),
                        v_labels.reshape(-1),
                        ignore_index=-100,
                    )

                    v_masked_positions = int((v_labels != -100).sum().item())
                    v_loss_sum += float(v_loss.item()) * v_masked_positions
                    v_tok += v_masked_positions

            v_avg = v_loss_sum / max(1, v_tok)
            v_ppl = math.exp(min(20.0, v_avg))
            print(f"[ep {ep}/{args.epochs}] VALID mlm_loss={v_avg:.4f} ppl~{v_ppl:.2f} masked_tokens={v_tok}")

            write_log({
                "time": time.time(),
                "event": "valid",
                "epoch": ep,
                "step": global_step,
                "mlm_loss": v_avg,
                "ppl": v_ppl,
                "masked_tokens": v_tok,
            })

            model.train()

        ckpt_path = os.path.join(args.save_dir, f"smiles_mlm_ep{ep:02d}.pt")
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
            "vocab_size": vocab_size,
            "pad_id": PAD_ID,
            "mask_id": MASK_ID,
            "cls_id": CLS_ID,
            "config": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "layers": args.layers,
                "dim_ff": args.dim_ff,
                "dropout": args.dropout,
                "max_len": args.max_len,
            },
            "args": vars(args),
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