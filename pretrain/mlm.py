#!/usr/bin/env python3
import os, glob, math, random, argparse, json, time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# MLM masking
# ============================================================
def mlm_mask_tokens(
    input_ids: torch.Tensor,
    base_vocab: int,
    mask_prob: float = 0.30,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    input_ids: (B, L) int64 with PAD included (PAD_ID = base_vocab+0)
    returns:
      masked_input_ids: (B, L)
      labels: (B, L) where non-masked = -100
      vocab_size: base_vocab + 2  (PAD + MASK)
    Notes:
      - Mask positions are sampled randomly each call.
      - If `generator` is provided, sampling becomes deterministic w.r.t that generator.
    """
    assert input_ids.dtype == torch.int64

    PAD_ID  = base_vocab + 0
    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2

    labels = torch.full_like(input_ids, -100)

    # don't mask PAD
    is_pad = (input_ids == PAD_ID)

    # choose positions to predict
    prob = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_sel = (prob < mask_prob) & (~is_pad)

    labels[mask_sel] = input_ids[mask_sel]

    # among selected: 80% -> MASK, 10% -> random, 10% -> keep
    r = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)

    m_mask = mask_sel & (r < 0.80)                          # MASK
    m_rand = mask_sel & (r >= 0.80) & (r < 0.90)            # RAND
    # KEEP is remaining selected positions

    masked = input_ids.clone()
    masked[m_mask] = MASK_ID

    if m_rand.any():
        masked[m_rand] = torch.randint(
            low=0,
            high=base_vocab,
            size=(int(m_rand.sum().item()),),
            device=input_ids.device,
            dtype=torch.int64,
            generator=generator,
        )

    return masked, labels, vocab_size
# ============================================================
# Dataset: molecule-level sampling from many batch files
# ============================================================
class RaggedMolDataset(Dataset):
    """
    Loads pretrain_ragged_batchXXX.pt files and exposes each molecule as one item.
    Each item returns a 1D token sequence (len = num_atoms_in_mol).
    """

    def __init__(self, root_dir,
                 pattern="pretrain_ragged_batch*.pt",
                 limit_files=None,
                 file_list=None):

        if file_list is not None:
            # file_list can contain relative paths
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
            raise FileNotFoundError(f"No files found in {root_dir}")

        if limit_files is not None:
            self.files = self.files[:int(limit_files)]

        self.index = []
        self.file_meta = []
        base_vocab_set = set()

        for fi, fp in enumerate(self.files):
            d = torch.load(fp, map_location="cpu")
            offsets = d["offsets"].to(torch.int64)
            n_mols = offsets.numel() - 1
            self.file_meta.append((fp, offsets))

            for mi in range(n_mols):
                self.index.append((fi, mi))

            base_vocab_set.add(int(d["base_vocab"]))

        if len(base_vocab_set) != 1:
            raise RuntimeError(f"base_vocab differs across files: {sorted(base_vocab_set)}")

        self.base_vocab = base_vocab_set.pop()

        self._cache_fi = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi):
        if self._cache_fi == fi and self._cache_data is not None:
            return self._cache_data
        fp, _ = self.file_meta[fi]
        d = torch.load(fp, map_location="cpu")
        self._cache_fi = fi
        self._cache_data = d
        return d

    def __getitem__(self, idx):
        fi, mi = self.index[idx]
        d = self._load_file(fi)

        tokens_flat = d["tokens_flat"].to(torch.int64)
        offsets = d["offsets"].to(torch.int64)

        s = int(offsets[mi].item())
        e = int(offsets[mi + 1].item())
        seq = tokens_flat[s:e].clone()
        return seq

def collate_pad(batch, base_vocab: int):
    """
    batch: list of 1D LongTensor sequences (variable len)
    returns:
      input_ids (B,L) padded with PAD
      attn_keep (B,L) bool where True=real token, False=pad
      lens (B,)
    """
    PAD_ID = base_vocab + 0
    lens = torch.tensor([x.numel() for x in batch], dtype=torch.int64)
    B = len(batch)
    L = int(lens.max().item())
    input_ids = torch.full((B, L), PAD_ID, dtype=torch.int64)
    for i, seq in enumerate(batch):
        input_ids[i, :seq.numel()] = seq
    attn_keep = (torch.arange(L)[None, :] < lens[:, None])  # True for real tokens
    return input_ids, attn_keep, lens


# ============================================================
# Simple Transformer MLM
# ============================================================
class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)

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
        """
        input_ids: (B,L)
        key_padding_mask: (B,L) bool, True for PAD positions (PyTorch convention)
        """
        x = self.tok(input_ids)  # (B,L,D)
        x = self.enc(x, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)  # (B,L,V)
        return logits


# ============================================================
# Utils: logging + seeding
# ============================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_log_writer(log_file: Optional[str]):
    """
    Returns a function write(rec: dict) -> None
    that appends JSONL to log_file, or a no-op if None.
    """
    if not log_file:
        def _noop(_rec):  # type: ignore
            return
        return _noop, None

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    f = open(log_file, "a", buffering=1)  # line-buffered

    def _write(rec: dict):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return _write, f


# ============================================================
# Train loop
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Path to split.json (with train/valid file lists)"
    )
    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    # io/log
    ap.add_argument("--save_dir", type=str, default="./mlm_ckpt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--log_file", type=str, default=None, help="JSONL file path to append training logs")
    ap.add_argument(
        "--deterministic_masking",
        action="store_true",
        help="Make MLM masking deterministic per (epoch, step) given --seed (useful for debugging).",
    )

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

    train_ds = RaggedMolDataset(
        args.data_dir,
        limit_files=args.limit_files,
        file_list=train_files
    )

    valid_ds = None
    if valid_files:
        valid_ds = RaggedMolDataset(
            args.data_dir,
            file_list=valid_files
        )

    # vocab info from train dataset
    base_vocab = train_ds.base_vocab
    PAD_ID = base_vocab + 0
    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2

    print(f"Loaded {len(train_ds)} train molecules from {args.data_dir}")
    if valid_ds is not None:
        print(f"Loaded {len(valid_ds)} valid molecules")
    print(f"base_vocab={base_vocab} PAD_ID={PAD_ID} MASK_ID={MASK_ID} vocab_size={vocab_size}")
    if args.log_file:
        print(f"logging to: {args.log_file}")

    # dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_pad(batch, base_vocab),
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
            collate_fn=lambda batch: collate_pad(batch, base_vocab),
            drop_last=False,
        )

    # device/model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine schedule with warmup
    total_steps = args.epochs * len(train_loader)
    warmup = max(10, int(0.05 * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, (total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    global_step = 0
    model.train()

    # helper for deterministic masking
    mask_gen = None
    if args.deterministic_masking:
        mask_gen = torch.Generator(device=device)

    for ep in range(1, args.epochs + 1):
        running_loss = 0.0
        running_tok = 0

        for it, (input_ids, attn_keep, lens) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_keep = attn_keep.to(device, non_blocking=True)

            # Transformer expects key_padding_mask True at PAD positions
            key_padding_mask = ~attn_keep

            # deterministic masking option: reseed generator each step
            if mask_gen is not None:
                # stable mix: seed + epoch*1e6 + global_step
                step_seed = int(args.seed + ep * 1_000_000 + global_step)
                mask_gen.manual_seed(step_seed)

            masked_input, labels, vocab_size2 = mlm_mask_tokens(
                input_ids,
                base_vocab=base_vocab,
                mask_prob=args.mask_prob,
                generator=mask_gen,
            )
            assert vocab_size2 == vocab_size

            logits = model(masked_input, key_padding_mask=key_padding_mask)  # (B,L,V)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            global_step += 1  # increment first

            # lr schedule
            lr_now = args.lr * lr_factor(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            # stats
            with torch.no_grad():
                masked_positions = int((labels != -100).sum().item())

            running_loss += float(loss.item()) * masked_positions
            running_tok += masked_positions

            if global_step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_tok)
                ppl = math.exp(min(20.0, avg_loss))

                msg = (f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps} "
                       f"mlm_loss={avg_loss:.4f} ppl~{ppl:.2f} lr={lr_now:.2e} "
                       f"masked_tokens={running_tok}")
                print(msg)

                write_log({
                    "time": time.time(),
                    "epoch": ep,
                    "step": global_step,
                    "total_steps": total_steps,
                    "mlm_loss": avg_loss,
                    "ppl": ppl,
                    "lr": lr_now,
                    "masked_tokens": running_tok,
                    "batch_size": args.batch_size,
                    "mask_prob": args.mask_prob,
                    "deterministic_masking": bool(args.deterministic_masking),
                })

                running_loss, running_tok = 0.0, 0
        # -------------------------
        # VALID
        # -------------------------
        if valid_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_tok = 0

            with torch.no_grad():
                for v_it, (v_input_ids, v_attn_keep, v_lens) in enumerate(valid_loader, start=1):
                    v_input_ids = v_input_ids.to(device, non_blocking=True)
                    v_attn_keep = v_attn_keep.to(device, non_blocking=True)
                    v_key_padding_mask = ~v_attn_keep

                    # deterministic masking option (use different stream than train)
                    if mask_gen is not None:
                        step_seed = int(args.seed + 9_000_000 + v_it)
                        mask_gen.manual_seed(step_seed)

                    v_masked, v_labels, _ = mlm_mask_tokens(
                        v_input_ids,
                        base_vocab=base_vocab,
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

        # save checkpoint each epoch
        ckpt_path = os.path.join(args.save_dir, f"mlm_ep{ep:02d}.pt")
        torch.save({
            "epoch": ep,
            "global_step": global_step,
            "model": model.state_dict(),
            "base_vocab": base_vocab,
            "vocab_size": vocab_size,
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
