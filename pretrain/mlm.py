#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_vqa_gnn_mlm_clean.py

GNN version of VQ-Atom token MLM pretraining.

Original Transformer MLM:
    token_ids -> Embedding -> Transformer -> vocab logits -> CE

This GNN MLM:
    token_ids + molecular graph -> Embedding -> GIN/GINE -> vocab logits -> CE

Expected ragged .pt format:
    required:
      offsets: LongTensor, shape (num_mols + 1,)
      tokens_flat: LongTensor, shape (total_atoms,)
      base_vocab: int

    required for GNN:
      edge_index-like tensor, one of:
        edge_index / edge_index_flat / edges / edges_flat / bond_index / bond_index_flat
        shape (2, total_edges) or (total_edges, 2)

    optional:
      edge_offsets / bond_offsets / edge_ptr / bond_ptr, shape (num_mols + 1,)
      edge_attr / edge_attr_flat / bond_attr / bond_attr_flat / edge_feats_flat / bond_feats_flat
      shape (total_edges, edge_dim)

edge_index_format:
      If d.get("edge_index_format") == "local_per_molecule", edge_index values are
      already local node indices for each molecule. In that case edge_offsets is
      required and this script does NOT subtract atom offsets.
      Otherwise edge_index is treated as global atom indices and converted to local
      indices by subtracting the molecule atom offset.

If edge_offsets is absent, this script filters global edge_index by atom slice [s, e).
That is slower but robust. For local_per_molecule edge_index, edge_offsets is required.
"""

from __future__ import annotations

import os
import glob
import math
import json
import time
import random
import argparse
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINConv, GINEConv, global_mean_pool
except Exception as e:
    raise ImportError(
        "This GNN version requires torch_geometric. Install it before running.\n"
        "Example: pip install torch_geometric"
    ) from e


EDGE_INDEX_KEYS = (
    "edge_index",
    "edge_index_flat",
    "edges",
    "edges_flat",
    "bond_index",
    "bond_index_flat",
)

EDGE_OFFSET_KEYS = (
    "edge_offsets",
    "bond_offsets",
    "edge_ptr",
    "bond_ptr",
)

EDGE_ATTR_KEYS = (
    "edge_attr",
    "edge_attr_flat",
    "bond_attr",
    "bond_attr_flat",
    "edge_feats_flat",
    "bond_feats_flat",
)


def find_key(d: Dict[str, Any], keys: Tuple[str, ...], explicit: Optional[str], name: str) -> str:
    if explicit:
        if explicit not in d:
            raise KeyError(f"{explicit!r} not found for {name}. Available keys: {sorted(d.keys())}")
        return explicit
    for k in keys:
        if k in d:
            return k
    raise KeyError(f"No {name} key found. Expected one of {keys}. Available keys: {sorted(d.keys())}")


def find_optional_key(d: Dict[str, Any], keys: Tuple[str, ...], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        if explicit not in d:
            raise KeyError(f"{explicit!r} not found. Available keys: {sorted(d.keys())}")
        return explicit
    for k in keys:
        if k in d:
            return k
    return None


def normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = edge_index.long()
    if edge_index.dim() != 2:
        raise RuntimeError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        return edge_index.contiguous()
    if edge_index.shape[1] == 2:
        return edge_index.t().contiguous()
    raise RuntimeError(f"edge_index must have shape (2,E) or (E,2), got {tuple(edge_index.shape)}")


# ============================================================
# MLM masking for flattened PyG node batch
# ============================================================
def mlm_mask_node_tokens(
    input_ids: torch.Tensor,
    base_vocab: int,
    mask_prob: float = 0.30,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    input_ids: (N,) int64 node token IDs. No PAD is needed in PyG batch.

    returns:
      masked_input_ids: (N,)
      labels:           (N,), non-masked = -100
      vocab_size:       base_vocab + 2  (PAD + MASK; PAD kept for checkpoint compatibility)
      mask_sel:         (N,) bool
    """
    assert input_ids.dtype == torch.int64

    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2

    labels = torch.full_like(input_ids, -100)
    prob = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_sel = prob < mask_prob
    labels[mask_sel] = input_ids[mask_sel]

    r = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    m_mask = mask_sel & (r < 0.80)
    m_rand = mask_sel & (r >= 0.80) & (r < 0.90)

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

    return masked, labels, vocab_size, mask_sel


# ============================================================
# Dataset
# ============================================================
class RaggedMolGraphDataset(Dataset):
    """
    Loads pretrain_ragged_batchXXX.pt and exposes each molecule as one PyG Data.
    Each item contains:
      token_ids:  (num_atoms,)
      edge_index: (2, num_edges), local node indices
      edge_attr:  optional (num_edges, edge_dim)
    """

    def __init__(
        self,
        root_dir: str,
        pattern: str = "pretrain_ragged_batch*.pt",
        limit_files: Optional[int] = None,
        file_list: Optional[List[str]] = None,
        edge_index_key: Optional[str] = None,
        edge_attr_key: Optional[str] = None,
        edge_offsets_key: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.edge_index_key_arg = edge_index_key
        self.edge_attr_key_arg = edge_attr_key
        self.edge_offsets_key_arg = edge_offsets_key

        if file_list is not None:
            self.files = sorted([p if os.path.isabs(p) else os.path.join(root_dir, p) for p in file_list])
        else:
            self.files = sorted(glob.glob(os.path.join(root_dir, pattern)))

        if not self.files:
            raise FileNotFoundError(f"No files found in {root_dir} with pattern={pattern}")

        if limit_files is not None:
            self.files = self.files[: int(limit_files)]

        self.index: List[Tuple[int, int]] = []
        self.file_meta: List[Dict[str, Any]] = []
        base_vocab_set = set()
        edge_dim_set = set()

        for fi, fp in enumerate(self.files):
            d = torch.load(fp, map_location="cpu")

            for req in ("offsets", "tokens_flat", "base_vocab"):
                if req not in d:
                    raise KeyError(f"{req!r} not found in {fp}. Available keys: {sorted(d.keys())}")

            offsets = d["offsets"].to(torch.int64)
            tokens_flat = d["tokens_flat"].to(torch.int64)
            n_mols = offsets.numel() - 1
            max_node = int(offsets[-1].item())

            if int(tokens_flat.numel()) != max_node:
                raise RuntimeError(
                    f"tokens_flat length != offsets[-1] in {fp}: {tokens_flat.numel()} vs {max_node}"
                )

            base_vocab_set.add(int(d["base_vocab"]))

            ekey = find_key(d, EDGE_INDEX_KEYS, self.edge_index_key_arg, "edge_index")
            edge_index = normalize_edge_index(d[ekey])

            # Supported formats:
            #   global             : edge_index uses flattened atom indices for the whole file.
            #   local_per_molecule : edge_index uses local atom indices within each molecule,
            #                        and edge_offsets tells which edges belong to each molecule.
            edge_index_format = str(d.get("edge_index_format", "global"))
            if edge_index_format not in {"global", "local_per_molecule"}:
                raise RuntimeError(
                    f"Unsupported edge_index_format={edge_index_format!r} in {fp}. "
                    "Expected 'global' or 'local_per_molecule'."
                )

            eokey = find_optional_key(d, EDGE_OFFSET_KEYS, self.edge_offsets_key_arg)
            if eokey is not None:
                eo = d[eokey].to(torch.int64)
                if eo.numel() != n_mols + 1:
                    raise RuntimeError(f"{eokey} must have shape num_mols+1, got {eo.numel()} vs {n_mols+1}")
            elif edge_index_format == "local_per_molecule":
                raise RuntimeError(
                    f"{fp} has edge_index_format='local_per_molecule' but no edge_offsets. "
                    "Cannot assign local edges to molecules without edge_offsets."
                )

            if edge_index.numel() > 0:
                emin = int(edge_index.min().item())
                emax = int(edge_index.max().item())
                if emin < 0:
                    raise RuntimeError(f"edge_index in {fp} has negative index: min={emin}")
                if edge_index_format == "global" and emax >= max_node:
                    raise RuntimeError(
                        f"edge_index in {fp} appears not to use global atom indices in [0, {max_node}). "
                        "Check AtomFeat/Adj correspondence."
                    )
                # For local_per_molecule, max is checked per molecule in __getitem__,
                # because each edge slice has its own local node count.

            eattr_key = find_optional_key(d, EDGE_ATTR_KEYS, self.edge_attr_key_arg)
            if eattr_key is not None:
                edge_attr = d[eattr_key]
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr[:, None]
                if edge_attr.dim() != 2:
                    raise RuntimeError(f"edge_attr {eattr_key} must be 2D, got {tuple(edge_attr.shape)}")
                if edge_attr.shape[0] != edge_index.shape[1]:
                    raise RuntimeError(
                        f"edge_attr rows != edges in {fp}: {edge_attr.shape[0]} vs {edge_index.shape[1]}"
                    )
                edge_dim_set.add(int(edge_attr.shape[1]))

            self.file_meta.append({
                "fp": fp,
                "offsets": offsets,
                "edge_index_key": ekey,
                "edge_offsets_key": eokey,
                "edge_attr_key": eattr_key,
                "edge_index_format": edge_index_format,
            })

            for mi in range(n_mols):
                self.index.append((fi, mi))

        if len(base_vocab_set) != 1:
            raise RuntimeError(f"base_vocab differs across files: {sorted(base_vocab_set)}")
        self.base_vocab = base_vocab_set.pop()

        if len(edge_dim_set) > 1:
            raise RuntimeError(f"edge_dim differs across files: {sorted(edge_dim_set)}")
        self.edge_dim = edge_dim_set.pop() if edge_dim_set else 0

        self._cache_fi = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, fi: int) -> Dict[str, Any]:
        if self._cache_fi == fi and self._cache_data is not None:
            return self._cache_data
        d = torch.load(self.file_meta[fi]["fp"], map_location="cpu")
        self._cache_fi = fi
        self._cache_data = d
        return d

    def __getitem__(self, idx: int) -> Data:
        fi, mi = self.index[idx]
        meta = self.file_meta[fi]
        d = self._load_file(fi)

        offsets = meta["offsets"]
        s = int(offsets[mi].item())
        e = int(offsets[mi + 1].item())
        n = e - s

        token_ids = d["tokens_flat"][s:e].to(torch.int64).clone()

        edge_index_all = normalize_edge_index(d[meta["edge_index_key"]])
        edge_index_format = str(meta.get("edge_index_format", "global"))

        if meta["edge_offsets_key"] is not None:
            eo = d[meta["edge_offsets_key"]].to(torch.int64)
            es = int(eo[mi].item())
            ee = int(eo[mi + 1].item())
            edge_index = edge_index_all[:, es:ee].clone()
            edge_slice = slice(es, ee)

            # global indices need conversion to local molecule indices.
            # local_per_molecule indices are already local, so do NOT subtract s.
            if edge_index_format == "global":
                edge_index = edge_index - s
            elif edge_index_format == "local_per_molecule":
                pass
            else:
                raise RuntimeError(f"Unsupported edge_index_format={edge_index_format!r}")
        else:
            if edge_index_format == "local_per_molecule":
                raise RuntimeError(
                    "edge_offsets are required when edge_index_format='local_per_molecule'."
                )
            src = edge_index_all[0]
            dst = edge_index_all[1]
            keep = (src >= s) & (src < e) & (dst >= s) & (dst < e)
            edge_index = edge_index_all[:, keep].clone() - s
            edge_slice = keep

        if edge_index.numel() > 0:
            emin = int(edge_index.min().item())
            emax = int(edge_index.max().item())
            if emin < 0 or emax >= n:
                raise RuntimeError(
                    "Local edge_index is out of range after slicing. "
                    f"file={meta['fp']} mol={mi} n={n} "
                    f"edge_index_format={edge_index_format} min={emin} max={emax}"
                )

        edge_attr = None
        if meta["edge_attr_key"] is not None:
            ea = d[meta["edge_attr_key"]].float()
            if ea.dim() == 1:
                ea = ea[:, None]
            edge_attr = ea[edge_slice].clone()

        return Data(token_ids=token_ids, edge_index=edge_index.long(), edge_attr=edge_attr, num_nodes=n)


# ============================================================
# GNN MLM model
# ============================================================
class VQAtomGNNMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        aggr: str = "sum",
        dropout: float = 0.1,
        use_graph_context: bool = False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.gnn_layers = int(gnn_layers)
        self.aggr = str(aggr)
        self.use_graph_context = bool(use_graph_context)

        if self.aggr not in {"sum", "mean", "max"}:
            raise ValueError(f"aggr must be one of sum/mean/max, got: {self.aggr}")

        self.tok = nn.Embedding(vocab_size, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            if self.edge_dim > 0:
                self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim, aggr=self.aggr))
            else:
                self.convs.append(GINConv(mlp, aggr=self.aggr))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if self.use_graph_context:
            self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.graph_proj = None

        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.tok(input_ids)

        for conv, norm in zip(self.convs, self.norms):
            h0 = h
            if self.edge_dim > 0:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + h0

        if self.graph_proj is not None:
            g = global_mean_pool(h, batch)
            h = h + self.graph_proj(g)[batch]

        logits = self.lm_head(h)
        return logits


# ============================================================
# Utils
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_log_writer(log_file: Optional[str]):
    if not log_file:
        return (lambda _rec: None), None
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    f = open(log_file, "a", buffering=1)

    def _write(rec: dict):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return _write, f


@torch.no_grad()
def compute_mlm_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    mask = labels != -100
    n = int(mask.sum().item())
    if n == 0:
        return {"masked_tokens": 0.0, "acc": 0.0}
    pred = logits[mask].argmax(dim=-1)
    acc = (pred == labels[mask]).float().mean().item()
    return {"masked_tokens": float(n), "acc": float(acc)}


@torch.no_grad()
def valid_epoch(model, loader, device, args, base_vocab: int, vocab_size: int, mask_gen=None) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    tok_sum = 0
    acc_sum = 0.0
    batches = 0

    for it, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        if mask_gen is not None:
            mask_gen.manual_seed(int(args.seed + 9_000_000 + it))

        masked_input, labels, vocab_size2, _ = mlm_mask_node_tokens(
            batch.token_ids,
            base_vocab=base_vocab,
            mask_prob=args.mask_prob,
            generator=mask_gen,
        )
        assert vocab_size2 == vocab_size

        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        logits = model(masked_input, batch.edge_index, batch.batch, edge_attr=edge_attr)
        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        metrics = compute_mlm_metrics(logits, labels)
        masked_tokens = int(metrics["masked_tokens"])
        loss_sum += float(loss.item()) * max(1, masked_tokens)
        tok_sum += masked_tokens
        acc_sum += float(metrics["acc"])
        batches += 1

    avg = loss_sum / max(1, tok_sum)
    ppl = math.exp(min(20.0, avg))
    acc = acc_sum / max(1, batches)
    return {"mlm_loss": avg, "ppl": ppl, "acc": acc, "masked_tokens": float(tok_sum)}


# ============================================================
# Train loop
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="pretrain_ragged_batch*.pt")
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--split_json", type=str, default=None, help="Path to split.json with train/valid file lists")
    ap.add_argument("--edge_index_key", type=str, default=None)
    ap.add_argument("--edge_attr_key", type=str, default=None)
    ap.add_argument("--edge_offsets_key", type=str, default=None)

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    ap.add_argument("--reset_optim", action="store_true", help="When resuming, re-init optimizer state")
    ap.add_argument("--reset_lr", action="store_true", help="When resuming, set lr from current schedule")

    # model
    ap.add_argument("--gnn_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--aggr", type=str, default="sum", choices=["sum", "mean", "max"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_graph_context", action="store_true")

    # io/log
    ap.add_argument("--save_dir", type=str, default="./vqa_gnn_mlm_ckpt")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--deterministic_masking", action="store_true")

    args = ap.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    write_log, log_fh = make_log_writer(args.log_file)

    train_files = None
    valid_files = None
    if args.split_json is not None:
        print(f"[info] loading split from {args.split_json}")
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_files = split.get("train", [])
        valid_files = split.get("valid", [])
        print(f"[split] train_files={len(train_files)} valid_files={len(valid_files)}")

    train_ds = RaggedMolGraphDataset(
        args.data_dir,
        pattern=args.pattern,
        limit_files=args.limit_files,
        file_list=train_files,
        edge_index_key=args.edge_index_key,
        edge_attr_key=args.edge_attr_key,
        edge_offsets_key=args.edge_offsets_key,
    )

    valid_ds = None
    if valid_files:
        valid_ds = RaggedMolGraphDataset(
            args.data_dir,
            pattern=args.pattern,
            file_list=valid_files,
            edge_index_key=args.edge_index_key,
            edge_attr_key=args.edge_attr_key,
            edge_offsets_key=args.edge_offsets_key,
        )
        if int(valid_ds.base_vocab) != int(train_ds.base_vocab):
            raise RuntimeError(f"valid base_vocab mismatch: train={train_ds.base_vocab}, valid={valid_ds.base_vocab}")
        if int(valid_ds.edge_dim) != int(train_ds.edge_dim):
            raise RuntimeError(f"valid edge_dim mismatch: train={train_ds.edge_dim}, valid={valid_ds.edge_dim}")

    base_vocab = int(train_ds.base_vocab)
    PAD_ID = base_vocab + 0
    MASK_ID = base_vocab + 1
    vocab_size = base_vocab + 2
    edge_dim = int(train_ds.edge_dim)

    print(f"Loaded {len(train_ds)} train molecules from {args.data_dir}")
    if valid_ds is not None:
        print(f"Loaded {len(valid_ds)} valid molecules")
    print(f"base_vocab={base_vocab} PAD_ID={PAD_ID} MASK_ID={MASK_ID} vocab_size={vocab_size}")
    print(f"edge_dim={edge_dim} mask_prob={args.mask_prob}")
    if args.log_file:
        print(f"logging to: {args.log_file}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
            drop_last=False,
        )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise RuntimeError("train_loader is empty. Check batch_size/drop_last/dataset size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAtomGNNMLM(
        vocab_size=vocab_size,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        aggr=args.aggr,
        dropout=args.dropout,
        use_graph_context=args.use_graph_context,
    ).to(device)
    print(
        f"model: GNN layers={args.gnn_layers} hidden_dim={args.hidden_dim} "
        f"aggr={args.aggr} dropout={args.dropout} use_graph_context={args.use_graph_context}"
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    resume_step0 = 0
    resume_lr0 = args.lr
    resume_last_epoch = 0

    if args.resume is not None:
        print(f"[resume] loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")

        if int(ckpt.get("base_vocab", -1)) != int(base_vocab):
            raise RuntimeError(f"[resume] base_vocab mismatch: ckpt={ckpt.get('base_vocab')} current={base_vocab}")
        if int(ckpt.get("vocab_size", -1)) != int(vocab_size):
            raise RuntimeError(f"[resume] vocab_size mismatch: ckpt={ckpt.get('vocab_size')} current={vocab_size}")
        if int(ckpt.get("edge_dim", edge_dim)) != int(edge_dim):
            raise RuntimeError(f"[resume] edge_dim mismatch: ckpt={ckpt.get('edge_dim')} current={edge_dim}")

        model.load_state_dict(ckpt["model"], strict=True)

        if (not args.reset_optim) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
                print("[resume] optimizer state loaded")
            except Exception as e:
                print(f"[resume] failed to load optimizer state: {e} (continuing with fresh optim)")

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

    # LR schedule
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
            raise RuntimeError(f"--epochs must be > ckpt epoch. ckpt={resume_last_epoch} args.epochs={args.epochs}")
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
        running_acc = 0.0
        running_batches = 0

        for it, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)

            if mask_gen is not None:
                step_seed = int(args.seed + ep * 1_000_000 + global_step)
                mask_gen.manual_seed(step_seed)

            masked_input, labels, vocab_size2, _ = mlm_mask_node_tokens(
                batch.token_ids,
                base_vocab=base_vocab,
                mask_prob=args.mask_prob,
                generator=mask_gen,
            )
            assert vocab_size2 == vocab_size

            edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
            logits = model(masked_input, batch.edge_index, batch.batch, edge_attr=edge_attr)

            loss = F.cross_entropy(logits, labels, ignore_index=-100)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            global_step += 1
            lr_now_val = lr_now(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now_val

            with torch.no_grad():
                metrics = compute_mlm_metrics(logits, labels)
                masked_positions = int(metrics["masked_tokens"])
                acc = float(metrics["acc"])

            running_loss += float(loss.item()) * max(1, masked_positions)
            running_tok += masked_positions
            running_acc += acc
            running_batches += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / max(1, running_tok)
                ppl = math.exp(min(20.0, avg_loss))
                avg_acc = running_acc / max(1, running_batches)
                msg = (
                    f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps_disp} "
                    f"mlm_loss={avg_loss:.4f} ppl~{ppl:.2f} acc={avg_acc:.4f} "
                    f"lr={lr_now_val:.2e} masked_tokens={running_tok}"
                )
                print(msg)

                write_log({
                    "time": time.time(),
                    "epoch": ep,
                    "step": global_step,
                    "total_steps": total_steps_disp,
                    "mlm_loss": avg_loss,
                    "ppl": ppl,
                    "acc": avg_acc,
                    "lr": lr_now_val,
                    "masked_tokens": running_tok,
                    "batch_size": args.batch_size,
                    "mask_prob": args.mask_prob,
                    "deterministic_masking": bool(args.deterministic_masking),
                })
                running_loss = 0.0
                running_tok = 0
                running_acc = 0.0
                running_batches = 0

        if valid_loader is not None:
            v = valid_epoch(model, valid_loader, device, args, base_vocab, vocab_size, mask_gen=mask_gen)
            print(
                f"[ep {ep}/{args.epochs}] VALID "
                f"mlm_loss={v['mlm_loss']:.4f} ppl~{v['ppl']:.2f} "
                f"acc={v['acc']:.4f} masked_tokens={int(v['masked_tokens'])}"
            )
            write_log({
                "time": time.time(),
                "event": "valid",
                "epoch": ep,
                "step": global_step,
                **v,
            })
            model.train()

        ckpt_path = os.path.join(args.save_dir, f"vqa_gnn_mlm_ep{ep:02d}.pt")
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
            "base_vocab": base_vocab,
            "vocab_size": vocab_size,
            "edge_dim": edge_dim,
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
