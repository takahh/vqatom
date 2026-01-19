#!/usr/bin/env python3
import os, glob, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def mlm_mask_tokens(input_ids, base_vocab: int, mask_prob=0.30):
    """
    input_ids: (B, L) int64 with PAD included
    returns:
      masked_input_ids: (B, L)
      labels: (B, L) where non-masked = -100
      vocab_size: base_vocab + 2  (PAD + MASK)
    """
    assert input_ids.dtype == torch.int64

    PAD_ID  = base_vocab + 0
    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2

    labels = torch.full_like(input_ids, -100)

    # don't mask PAD
    is_pad = (input_ids == PAD_ID)

    # choose positions to predict
    prob = torch.rand(input_ids.shape, device=input_ids.device)
    mask_sel = (prob < mask_prob) & (~is_pad)

    labels[mask_sel] = input_ids[mask_sel]

    # among selected:
    # 80% -> MASK, 10% -> random, 10% -> keep
    r = torch.rand(input_ids.shape, device=input_ids.device)

    # MASK
    m_mask = mask_sel & (r < 0.80)
    # RAND
    m_rand = mask_sel & (r >= 0.80) & (r < 0.90)
    # KEEP is remaining

    masked = input_ids.clone()
    masked[m_mask] = MASK_ID
    if m_rand.any():
        masked[m_rand] = torch.randint(
            low=0, high=base_vocab,
            size=(int(m_rand.sum().item()),),
            device=input_ids.device,
            dtype=torch.int64
        )

    return masked, labels, vocab_size


# ----------------------------
# Dataset: molecule-level sampling from many batch files
# ----------------------------
class RaggedMolDataset(Dataset):
    """
    Loads pretrain_ragged_batchXXX.pt files and exposes each molecule as one item.
    Each item returns a 1D token sequence (len = num_atoms_in_mol).
    """
    def __init__(self, root_dir, pattern="pretrain_ragged_batch*.pt", limit_files=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No files matched: {os.path.join(root_dir, pattern)}")
        if limit_files is not None:
            self.files = self.files[:int(limit_files)]

        # Build index: global_idx -> (file_i, mol_i)
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
            # たぶん同一 vocab のはずだが、念のためチェック
            raise RuntimeError(f"base_vocab differs across files: {sorted(base_vocab_set)}")
        self.base_vocab = base_vocab_set.pop()

        # LRU-ish cache for speed (1 file at a time)
        self._cache_fi = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi):
        if self._cache_fi == fi and self._cache_data is not None:
            return self._cache_data
        fp, _offsets = self.file_meta[fi]
        d = torch.load(fp, map_location="cpu")
        # keep in memory
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
        seq = tokens_flat[s:e].clone()  # (len,)
        return seq


def collate_pad(batch, base_vocab: int):
    """
    batch: list of 1D LongTensor sequences (variable len)
    returns padded input_ids (B,L), attn_mask (B,L) where True=keep, False=pad
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


# ----------------------------
# Simple Transformer MLM
# ----------------------------
class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
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


# ----------------------------
# Train loop
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="./mlm_ckpt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit_files", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    ds = RaggedMolDataset(args.data_dir, limit_files=args.limit_files)
    base_vocab = ds.base_vocab
    PAD_ID = base_vocab + 0
    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2
    print(f"Loaded {len(ds)} molecules from {args.data_dir}")
    print(f"base_vocab={base_vocab} PAD_ID={PAD_ID} MASK_ID={MASK_ID} vocab_size={vocab_size}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_pad(batch, base_vocab),
        drop_last=True,
    )

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

    # simple cosine schedule with warmup
    total_steps = args.epochs * len(dl)
    warmup = max(10, int(0.05 * total_steps))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, (total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    global_step = 0
    model.train()

    for ep in range(1, args.epochs + 1):
        running_loss = 0.0
        running_tok = 0

        for it, (input_ids, attn_keep, lens) in enumerate(dl, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_keep = attn_keep.to(device, non_blocking=True)

            # PyTorch transformer expects key_padding_mask True at PAD positions
            key_padding_mask = ~attn_keep

            # apply MLM masking on padded batch
            masked_input, labels, vocab_size2 = mlm_mask_tokens(
                input_ids, base_vocab=base_vocab, mask_prob=args.mask_prob
            )
            assert vocab_size2 == vocab_size

            masked_input = masked_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(masked_input, key_padding_mask=key_padding_mask)  # (B,L,V)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # lr schedule
            for pg in optim.param_groups:
                pg["lr"] = args.lr * lr_factor(global_step)

            global_step += 1

            # stats
            with torch.no_grad():
                masked_positions = (labels != -100).sum().item()
            running_loss += float(loss.item()) * masked_positions
            running_tok += masked_positions

            if global_step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_tok)
                ppl = math.exp(min(20.0, avg_loss))
                print(f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps} "
                      f"mlm_loss={avg_loss:.4f} ppl~{ppl:.2f} lr={optim.param_groups[0]['lr']:.2e}")
                running_loss, running_tok = 0.0, 0

        # save checkpoint each epoch
        ckpt_path = os.path.join(args.save_dir, f"mlm_ep{ep:02d}.pt")
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "base_vocab": base_vocab,
            "vocab_size": vocab_size,
            "config": vars(args),
        }, ckpt_path)
        print("saved", ckpt_path)

    print("DONE")

if __name__ == "__main__":
    main()
