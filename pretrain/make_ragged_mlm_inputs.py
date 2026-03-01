#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import tarfile
import io
import tempfile
from typing import List, Tuple

import numpy as np
import torch


# =============================================================================
# Paths
# =============================================================================
path0 = "/Users/taka/Downloads/pretrain_no_id.tar.gz"      # tar.gz with attr_*.npy + smiles_*.txt
path1 = "/Users/taka/Downloads/infer_token_ids.pt"         # has d["global_id"] (flat or any shape)
vq_ckpt = "/Users/taka/Downloads/model_epoch_3.pt"         # to get base_vocab (PAD/MASK)
out_dir = "/Users/taka/Documents/pretrain_ragged"
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Expected shapes
# =============================================================================
MAX_ATOMS = 100
ATTR_DIM = 79


# =============================================================================
# Helpers
# =============================================================================
def read_bytes_from_tar(tar: tarfile.TarFile, name: str) -> bytes:
    f = tar.extractfile(name)
    if f is None:
        raise FileNotFoundError(f"Not found in tar: {name}")
    return f.read()


def detect_prefix_and_batches(names: List[str]) -> Tuple[str, List[int]]:
    """
    Detect prefix directory inside tar for attr_###.npy and return (prefix, batches).
    Supports:
      - "discret_50k/attr_0.npy"
      - "pretrain_ragged/attr_0.npy"
      - "attr_0.npy" (no prefix)
    """
    first = None
    for n in names:
        m = re.search(r"^(?P<prefix>.*/)?attr_(?P<idx>\d+)\.npy$", n)
        if m:
            first = m
            break
    if first is None:
        sample = "\n".join(names[:80])
        raise RuntimeError("No attr_###.npy found in tar. First entries:\n" + sample)

    prefix = first.group("prefix") or ""
    pat = re.compile(r"^" + re.escape(prefix) + r"attr_(\d+)\.npy$")
    batches = sorted({int(pat.search(n).group(1)) for n in names if pat.search(n)})
    if not batches:
        raise RuntimeError(f"Failed to enumerate batches with prefix={repr(prefix)}")
    return prefix, batches


def load_attr_from_bytes(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def normalize_attr_shape(attr: np.ndarray, batch: int) -> Tuple[np.ndarray, int]:
    """
    Accept both:
      - (n_mols, MAX_ATOMS, ATTR_DIM)
      - (n_mols*MAX_ATOMS, ATTR_DIM)
    Return: (attr3, n_mols) where attr3 is (n_mols, MAX_ATOMS, ATTR_DIM)
    """
    if attr.ndim == 3:
        n_mols, max_atoms, attr_dim = attr.shape
        if max_atoms != MAX_ATOMS or attr_dim != ATTR_DIM:
            raise RuntimeError(
                f"attr shape mismatch at batch {batch}: {attr.shape} expected (*,{MAX_ATOMS},{ATTR_DIM})"
            )
        return attr, n_mols

    if attr.ndim == 2:
        n_rows, attr_dim = attr.shape
        if attr_dim != ATTR_DIM:
            raise RuntimeError(
                f"attr shape mismatch at batch {batch}: {attr.shape} expected (*,{ATTR_DIM})"
            )
        if n_rows % MAX_ATOMS != 0:
            raise RuntimeError(
                f"attr rows not divisible by MAX_ATOMS at batch {batch}: rows={n_rows} MAX_ATOMS={MAX_ATOMS}"
            )
        n_mols = n_rows // MAX_ATOMS
        attr3 = attr.reshape(n_mols, MAX_ATOMS, ATTR_DIM)
        return attr3, n_mols

    raise RuntimeError(f"attr ndim unexpected at batch {batch}: shape={attr.shape}")


def parse_smiles(txt: str) -> List[str]:
    return [x.strip() for x in txt.splitlines() if x.strip()]


def atomic_torch_save(obj, path: str):
    """
    Write to temp file then rename (atomic on same filesystem).
    Prevents partially-written .pt that causes "unexpected pos" errors.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =============================================================================
# Load token stream
# =============================================================================
d = torch.load(path1, map_location="cpu")
if "global_id" not in d:
    raise KeyError(f"{path1} missing key 'global_id'. keys={list(d.keys())}")

token_stream_raw = d["global_id"]
token_stream = token_stream_raw.to(torch.int64).reshape(-1).clone()

total_tokens = int(token_stream.numel())
neg_tokens_total = int((token_stream < 0).sum().item())

# =============================================================================
# Load vocab info
# =============================================================================
vq = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
base_vocab = int(vq["base_vocab"])   # e.g. 180678
PAD_ID = int(base_vocab + 0)
MASK_ID = int(base_vocab + 1)

print("[token_stream] total_tokens:", total_tokens, "neg_tokens_total:", neg_tokens_total)
print("[vocab] base_vocab:", base_vocab, "PAD_ID:", PAD_ID, "MASK_ID:", MASK_ID)


# =============================================================================
# Inspect tar & detect prefix/batches
# =============================================================================
with tarfile.open(path0, "r:gz") as tar:
    names = tar.getnames()

prefix, batches = detect_prefix_and_batches(names)
print("[tar] prefix:", repr(prefix))
print("[tar] num batches:", len(batches), "first/last:", batches[:3], batches[-3:])

# sanity for first batch
for nf in [f"{prefix}attr_{batches[0]}.npy", f"{prefix}smiles_{batches[0]}.txt"]:
    if nf not in names:
        raise RuntimeError(f"Expected file not found in tar: {nf}")


# =============================================================================
# Main conversion
# =============================================================================
pos = 0  # cursor in token_stream

with tarfile.open(path0, "r:gz") as tar:
    for b in batches:
        attr_name = f"{prefix}attr_{b}.npy"
        smiles_name = f"{prefix}smiles_{b}.txt"

        # --- load attr
        raw_attr = read_bytes_from_tar(tar, attr_name)
        try:
            attr = load_attr_from_bytes(raw_attr)
        except ValueError as e:
            raise RuntimeError(
                f"Failed to np.load {attr_name}. If this npy requires pickle, set allow_pickle=True. Original error: {e}"
            )

        attr3, n_mols = normalize_attr_shape(attr, b)

        # --- load smiles
        raw_smiles = read_bytes_from_tar(tar, smiles_name).decode("utf-8", errors="replace")
        smiles = parse_smiles(raw_smiles)
        if len(smiles) != n_mols:
            raise RuntimeError(f"batch {b}: smiles count != n_mols : smiles={len(smiles)} n_mols={n_mols}")

        # --- compute lengths by "non-zero row" detection
        row_is_real = (np.abs(attr3).sum(axis=2) > 0)  # (n_mols, MAX_ATOMS)
        lengths = row_is_real.sum(axis=1).astype(np.int64)  # (n_mols,)

        if (lengths <= 0).any():
            bad = np.where(lengths <= 0)[0][:10].tolist()
            raise RuntimeError(f"batch {b}: found molecules with length<=0 at indices {bad}. attr may be all-zero.")
        if int(lengths.max()) > MAX_ATOMS:
            raise RuntimeError(f"batch {b}: length exceeds MAX_ATOMS? max_length={int(lengths.max())} MAX_ATOMS={MAX_ATOMS}")

        need = int(lengths.sum())
        remain = total_tokens - pos

        # debug: how many -1 are in this slice (before replacement)
        slice_end = pos + need
        if slice_end > total_tokens:
            # print diagnostics then fail
            print("[exhaust] batch", b, "need", need, "pos", pos, "remain", remain, "total_tokens", total_tokens)
            raise RuntimeError(
                f"token stream exhausted at batch {b}: need {need}, remain {remain}, stream_len={total_tokens}"
            )

        neg_in_slice = int((token_stream[pos:slice_end] < 0).sum().item())

        # --- slice tokens for this batch
        tokens_flat = token_stream[pos:slice_end].clone()
        pos = slice_end

        # -1 を MASK に置換（削除しない）
        if neg_in_slice > 0:
            tokens_flat = torch.where(
                tokens_flat >= 0,
                tokens_flat,
                torch.full_like(tokens_flat, MASK_ID),
            )

        tokens_flat = tokens_flat.to(torch.int32)

        # --- offsets (ragged)
        offsets_np = np.zeros(n_mols + 1, dtype=np.int64)
        offsets_np[1:] = np.cumsum(lengths)
        offsets = torch.from_numpy(offsets_np)

        out_path = os.path.join(out_dir, f"pretrain_ragged_batch{b:03d}.pt")
        atomic_torch_save(
            {
                "batch": int(b),
                "tokens_flat": tokens_flat,              # (sum(lengths),)
                "offsets": offsets,                      # (n_mols+1,)
                "lengths": torch.from_numpy(lengths),    # (n_mols,)
                "base_vocab": int(base_vocab),
                "pad_id": int(PAD_ID),
                "mask_id": int(MASK_ID),
                "smiles": smiles,                        # list[str]
                "neg_in_slice": int(neg_in_slice),
            },
            out_path,
        )

        print(
            f"[saved] {os.path.basename(out_path)} | mols={n_mols} need={need} "
            f"| neg_in_slice={neg_in_slice} "
            f"| pos={pos}/{total_tokens} "
            f"| len[min/mean/max]={int(lengths.min())}/{float(lengths.mean()):.2f}/{int(lengths.max())}"
        )

print("[DONE] consumed:", pos, "/", total_tokens)
if pos != total_tokens:
    print("[WARN] consumed != total_tokens (ordering mismatch? extra tokens?)")