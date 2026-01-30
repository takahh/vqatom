#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import tarfile
import io
from typing import List, Tuple, Optional

import numpy as np
import torch


# =============================================================================
# Paths (user says path0/path1/path2 are correct)
# =============================================================================
path0 = "/Users/taka/Downloads/pretrain_no_id.tar.gz"      # tar.gz with attr_*.npy + smiles_*.txt
path1 = "/Users/taka/Documents/infer_token_ids.pt"         # has d["cluster_id"] (global IDs)
out_dir = "/Users/taka/Documents/pretrain_ragged"          # output directory
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
    # find first attr_###.npy
    first = None
    for n in names:
        m = re.search(r"^(?P<prefix>.*/)?attr_(?P<idx>\d+)\.npy$", n)
        if m:
            first = m
            break
    if first is None:
        # give a hint
        sample = "\n".join(names[:50])
        raise RuntimeError(
            "No attr_###.npy found in tar. First 50 entries:\n" + sample
        )

    prefix = first.group("prefix") or ""  # e.g. "discret_50k/" or ""
    pat = re.compile(r"^" + re.escape(prefix) + r"attr_(\d+)\.npy$")
    batches = sorted({int(pat.search(n).group(1)) for n in names if pat.search(n)})
    if not batches:
        raise RuntimeError(f"Failed to enumerate batches with prefix={repr(prefix)}")
    return prefix, batches


def load_attr_from_bytes(raw: bytes) -> np.ndarray:
    """Load .npy from bytes (no pickle unless needed)."""
    # If your npy truly needs pickle, change allow_pickle to True.
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


# =============================================================================
# Load token stream
# =============================================================================
d = torch.load(path1, map_location="cpu")
if "cluster_id" not in d:
    raise KeyError(f"{path1} missing key 'cluster_id'. keys={list(d.keys())}")

cluster_stream = d["cluster_id"].to(torch.int64)  # global IDs (flattened over all real atoms)
if cluster_stream.numel() == 0:
    raise RuntimeError("cluster_stream is empty")

base_vocab = int(cluster_stream.max().item()) + 1
print("cluster_stream_len:", int(cluster_stream.numel()), "base_vocab:", base_vocab)


# =============================================================================
# Inspect tar & detect prefix/batches
# =============================================================================
with tarfile.open(path0, "r:gz") as tar:
    names = tar.getnames()

prefix, batches = detect_prefix_and_batches(names)
print("Detected prefix in tar:", repr(prefix))
print("num batches:", len(batches), "first/last:", batches[:3], batches[-3:])

# Optional: quick existence sanity
need_files = [f"{prefix}attr_{batches[0]}.npy", f"{prefix}smiles_{batches[0]}.txt"]
for nf in need_files:
    if nf not in names:
        raise RuntimeError(f"Expected file not found in tar: {nf}")


# =============================================================================
# Main conversion
# =============================================================================
pos = 0  # position in cluster_stream

with tarfile.open(path0, "r:gz") as tar:
    for b in batches:
        attr_name = f"{prefix}attr_{b}.npy"
        smiles_name = f"{prefix}smiles_{b}.txt"

        # --- load attr
        raw_attr = read_bytes_from_tar(tar, attr_name)
        try:
            attr = load_attr_from_bytes(raw_attr)
        except ValueError as e:
            # If allow_pickle=False fails, you can switch to True above.
            raise RuntimeError(
                f"Failed to np.load {attr_name}. If this npy requires pickle, set allow_pickle=True. Original error: {e}"
            )

        attr3, n_mols = normalize_attr_shape(attr, b)

        # --- load smiles
        raw_smiles = read_bytes_from_tar(tar, smiles_name).decode("utf-8", errors="replace")
        smiles = parse_smiles(raw_smiles)
        if len(smiles) != n_mols:
            raise RuntimeError(
                f"batch {b}: smiles count != n_mols : smiles={len(smiles)} n_mols={n_mols}"
            )

        # --- compute per-mol atom lengths via zero-row detection
        # safer than !=0 for floats: use >0 threshold
        row_is_real = (np.abs(attr3).sum(axis=2) > 0)  # (n_mols, MAX_ATOMS)
        lengths = row_is_real.sum(axis=1).astype(np.int64)  # (n_mols,)

        if (lengths <= 0).any():
            bad = np.where(lengths <= 0)[0][:10].tolist()
            raise RuntimeError(
                f"batch {b}: found molecules with length<=0 at indices {bad}. "
                f"This usually means your 'real atom' detection is wrong or attr is all-zeros."
            )
        if int(lengths.max()) > MAX_ATOMS:
            raise RuntimeError(
                f"batch {b}: length exceeds MAX_ATOMS? max_length={int(lengths.max())} MAX_ATOMS={MAX_ATOMS}"
            )

        need = int(lengths.sum())
        remain = int(cluster_stream.numel()) - pos
        if need > remain:
            raise RuntimeError(
                f"token stream exhausted at batch {b}: need {need}, remain {remain}, stream_len={int(cluster_stream.numel())}"
            )

        # --- slice flat tokens for this batch
        tokens_flat = cluster_stream[pos:pos + need].clone()
        pos += need

        # --- offsets (ragged)
        offsets_np = np.zeros(n_mols + 1, dtype=np.int64)
        offsets_np[1:] = np.cumsum(lengths)
        offsets = torch.from_numpy(offsets_np)

        out_path = os.path.join(out_dir, f"pretrain_ragged_batch{b:03d}.pt")
        torch.save(
            {
                "batch": int(b),
                "tokens_flat": tokens_flat,              # (sum(lengths),)
                "offsets": offsets,                      # (n_mols+1,)
                "lengths": torch.from_numpy(lengths),    # (n_mols,)
                "base_vocab": int(base_vocab),
                "smiles": smiles,                        # list[str]
            },
            out_path,
        )

        print(
            f"saved {out_path} | mols={n_mols} need={need} "
            f"| pos={pos}/{int(cluster_stream.numel())} "
            f"| len[min/mean/max]={int(lengths.min())}/{float(lengths.mean()):.2f}/{int(lengths.max())}"
        )

print("DONE consumed:", pos, "/", int(cluster_stream.numel()))
if pos != int(cluster_stream.numel()):
    print("WARNING: consumed != stream_len (ordering mismatch? extra tokens? different padding rule?)")
