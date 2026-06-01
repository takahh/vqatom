#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_continuous_gnn_mae.py

GNN version of continuous masked atom modeling.

Input:
  pretrain_ragged_batch*.pt files with:
    required:
      offsets: LongTensor, shape (num_mols + 1,)
      atom feature tensor:
        atom_feats_flat / feats_flat / features_flat / x_flat / attr_flat / attrs_flat
        shape (total_atoms, feat_dim)

    required for GNN:
      edge_index-like tensor, one of:
        edge_index / edge_index_flat / edges / edges_flat / bond_index / bond_index_flat
        shape (2, total_edges) or (total_edges, 2)
      optional:
        edge_offsets / bond_offsets, shape (num_mols + 1,)
        edge_attr / edge_attr_flat / bond_attr / bond_attr_flat / edge_feats_flat
        shape (total_edges, edge_dim)

If edge_offsets is absent, this script filters global edge_index by atom slice [s, e).
That is slower but robust.
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


FEATURE_KEYS = (
    "atom_feats_flat",
    "feats_flat",
    "features_flat",
    "x_flat",
    "attr_flat",
    "attrs_flat",
)

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


def get_edge_index_format(d: Dict[str, Any]) -> str:
    """
    Supported formats:
      - global: edge_index stores global atom indices over the flattened file.
      - local_per_molecule: edge_index stores local atom indices for each molecule,
        concatenated according to edge_offsets. In this case DO NOT subtract atom offset.
    """
    fmt = str(d.get("edge_index_format", "global"))
    if fmt not in {"global", "local_per_molecule"}:
        print(f"[warn] unknown edge_index_format={fmt!r}; treating as global")
        fmt = "global"
    return fmt


def mask_continuous_nodes(
    x: torch.Tensor,
    mask_prob: float = 0.30,
    mask_mode: str = "learned",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = x.clone()
    prob = torch.rand((x.shape[0],), device=x.device, generator=generator)
    mask_sel = prob < mask_prob

    # ensure at least one target per graph is handled outside if needed;
    # at batch level this almost never becomes empty.
    x_masked = x.clone()
    if mask_mode == "learned":
        pass
    elif mask_mode == "zero":
        x_masked[mask_sel] = 0.0
    elif mask_mode == "noise":
        x_masked[mask_sel] = torch.randn(
            x_masked[mask_sel].shape,
            device=x_masked.device,
            dtype=x_masked.dtype,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return x_masked, labels, mask_sel


class ContinuousRaggedMolGraphDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        pattern: str = "pretrain_ragged_batch*.pt",
        limit_files: Optional[int] = None,
        file_list: Optional[List[str]] = None,
        feature_key: Optional[str] = None,
        edge_index_key: Optional[str] = None,
        edge_attr_key: Optional[str] = None,
        edge_offsets_key: Optional[str] = None,
        normalize: bool = False,
        eps: float = 1e-6,
    ):
        self.root_dir = root_dir
        self.feature_key_arg = feature_key
        self.edge_index_key_arg = edge_index_key
        self.edge_attr_key_arg = edge_attr_key
        self.edge_offsets_key_arg = edge_offsets_key
        self.normalize = bool(normalize)
        self.eps = float(eps)

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
        feat_dim_set = set()
        edge_dim_set = set()
        mean_candidates = []
        std_candidates = []

        for fi, fp in enumerate(self.files):
            d = torch.load(fp, map_location="cpu")
            if "offsets" not in d:
                raise KeyError(f"offsets not found in {fp}. Available keys: {sorted(d.keys())}")

            offsets = d["offsets"].to(torch.int64)
            n_mols = offsets.numel() - 1

            fkey = find_key(d, FEATURE_KEYS, self.feature_key_arg, "feature")
            feats = d[fkey]
            if feats.dim() != 2:
                raise RuntimeError(f"Feature tensor {fkey} in {fp} must be 2D, got {tuple(feats.shape)}")
            if int(offsets[-1].item()) != int(feats.shape[0]):
                raise RuntimeError(
                    f"offsets[-1] != num feature rows in {fp}: "
                    f"{int(offsets[-1].item())} vs {feats.shape[0]}"
                )
            feat_dim_set.add(int(feats.shape[1]))

            ekey = find_key(d, EDGE_INDEX_KEYS, self.edge_index_key_arg, "edge_index")
            edge_index = normalize_edge_index(d[ekey])
            edge_index_format = get_edge_index_format(d)

            max_node = int(offsets[-1].item())
            if edge_index.numel() > 0 and int(edge_index.min()) < 0:
                raise RuntimeError(f"edge_index in {fp} contains negative indices.")

            # For global format, values must be flattened atom indices.
            # For local_per_molecule format, values are local atom indices and can be
            # much smaller than offsets[-1], so the global max check is not valid here.
            if edge_index_format == "global":
                if edge_index.numel() > 0 and int(edge_index.max()) >= max_node:
                    raise RuntimeError(
                        f"edge_index in {fp} appears not to use global atom indices in [0, {max_node}). "
                        "Check AtomFeat/Adj correspondence."
                    )

            eokey = find_optional_key(d, EDGE_OFFSET_KEYS, self.edge_offsets_key_arg)
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

            if eokey is not None:
                eo = d[eokey].to(torch.int64)
                if eo.numel() != n_mols + 1:
                    raise RuntimeError(f"{eokey} must have shape num_mols+1, got {eo.numel()} vs {n_mols+1}")

            self.file_meta.append({
                "fp": fp,
                "offsets": offsets,
                "feature_key": fkey,
                "edge_index_key": ekey,
                "edge_index_format": edge_index_format,
                "edge_offsets_key": eokey,
                "edge_attr_key": eattr_key,
            })

            for mi in range(n_mols):
                self.index.append((fi, mi))

            if "feature_mean" in d and "feature_std" in d:
                mean_candidates.append(d["feature_mean"].float())
                std_candidates.append(d["feature_std"].float())

        if len(feat_dim_set) != 1:
            raise RuntimeError(f"feat_dim differs across files: {sorted(feat_dim_set)}")
        self.feat_dim = feat_dim_set.pop()

        if len(edge_dim_set) > 1:
            raise RuntimeError(f"edge_dim differs across files: {sorted(edge_dim_set)}")
        self.edge_dim = edge_dim_set.pop() if edge_dim_set else 0

        self.feature_mean = None
        self.feature_std = None
        if self.normalize:
            if mean_candidates and std_candidates:
                self.feature_mean = torch.stack(mean_candidates, dim=0).mean(dim=0)
                self.feature_std = torch.stack(std_candidates, dim=0).mean(dim=0).clamp_min(self.eps)
            else:
                print("[warn] --normalize set, but feature_mean/feature_std were not found. Running unnormalized.")

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

        x = d[meta["feature_key"]][s:e].float().clone()
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std

        edge_index_all = normalize_edge_index(d[meta["edge_index_key"]])
        edge_index_format = str(meta.get("edge_index_format", "global"))

        if meta["edge_offsets_key"] is not None:
            eo = d[meta["edge_offsets_key"]].to(torch.int64)
            es = int(eo[mi].item())
            ee = int(eo[mi + 1].item())

            if edge_index_format == "local_per_molecule":
                # Already local to this molecule. Do NOT subtract atom offset.
                edge_index = edge_index_all[:, es:ee].clone()
            else:
                # Global flattened atom indices. Convert to local molecule indices.
                edge_index = edge_index_all[:, es:ee].clone() - s
            edge_slice = slice(es, ee)
        else:
            if edge_index_format == "local_per_molecule":
                raise RuntimeError(
                    "edge_index_format='local_per_molecule' requires edge_offsets/edge_ptr "
                    "so per-molecule edges can be sliced."
                )

            src = edge_index_all[0]
            dst = edge_index_all[1]
            keep = (src >= s) & (src < e) & (dst >= s) & (dst < e)
            edge_index = edge_index_all[:, keep].clone() - s
            edge_slice = keep

        if edge_index.numel() > 0 and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
            raise RuntimeError(
                f"Local edge_index is out of range after slicing: "
                f"mol={mi} n={n} min={int(edge_index.min())} max={int(edge_index.max())} "
                f"format={edge_index_format}. Check offsets/edge_index."
            )

        edge_attr = None
        if meta["edge_attr_key"] is not None:
            ea = d[meta["edge_attr_key"]].float()
            if ea.dim() == 1:
                ea = ea[:, None]
            edge_attr = ea[edge_slice].clone()

        return Data(x=x, edge_index=edge_index.long(), edge_attr=edge_attr)


class ContinuousGNNMAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        aggr: str = "sum",
        dropout: float = 0.1,
        use_graph_context: bool = False,
    ):
        super().__init__()
        if aggr not in {"sum", "mean", "max"}:
            raise ValueError(f"Unsupported aggr={aggr!r}. Use one of: sum, mean, max")

        self.feat_dim = int(feat_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.gnn_layers = int(gnn_layers)
        self.aggr = str(aggr)
        self.use_graph_context = bool(use_graph_context)

        self.in_proj = nn.Linear(feat_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            if self.edge_dim > 0:
                self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim, aggr=aggr))
            else:
                self.convs.append(GINConv(mlp, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if self.use_graph_context:
            self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.graph_proj = None

        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask_sel: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_proj(x)

        if mask_sel is not None:
            h = h.clone()
            h[mask_sel] = self.mask_token.to(dtype=h.dtype)

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

        return self.out_head(h)


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


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask_sel: torch.Tensor, loss_type: str) -> torch.Tensor:
    if not mask_sel.any():
        return pred.sum() * 0.0
    p = pred[mask_sel]
    y = target[mask_sel]
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(p, y)
    if loss_type == "mse":
        return F.mse_loss(p, y)
    if loss_type == "l1":
        return F.l1_loss(p, y)
    raise ValueError(loss_type)


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, mask_sel: torch.Tensor) -> Dict[str, float]:
    if not mask_sel.any():
        return {"mae": 0.0, "rmse": 0.0, "masked_atoms": 0.0, "masked_values": 0.0}
    diff = pred[mask_sel] - target[mask_sel]
    return {
        "mae": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff ** 2).mean()).item()),
        "masked_atoms": float(mask_sel.sum().item()),
        "masked_values": float(diff.numel()),
    }


@torch.no_grad()
def valid_epoch(model, loader, device, args, mask_gen=None) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    atoms = 0
    mae_sum = 0.0
    rmse_sum = 0.0
    batches = 0

    for it, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        if mask_gen is not None:
            mask_gen.manual_seed(int(args.seed + 9_000_000 + it))

        x_masked, labels, mask_sel = mask_continuous_nodes(
            batch.x,
            mask_prob=args.mask_prob,
            mask_mode=args.mask_mode,
            generator=mask_gen,
        )
        edge_attr = batch.edge_attr if getattr(batch, "edge_attr", None) is not None else None
        pred = model(
            x_masked,
            edge_index=batch.edge_index,
            edge_attr=edge_attr,
            batch=batch.batch,
            mask_sel=mask_sel,
        )
        loss = reconstruction_loss(pred, labels, mask_sel, args.loss_type)
        m = reconstruction_metrics(pred, labels, mask_sel)

        loss_sum += float(loss.item()) * max(1, int(m["masked_atoms"]))
        atoms += int(m["masked_atoms"])
        mae_sum += m["mae"]
        rmse_sum += m["rmse"]
        batches += 1

    model.train()
    return {
        "loss": loss_sum / max(1, atoms),
        "mae": mae_sum / max(1, batches),
        "rmse": rmse_sum / max(1, batches),
        "masked_atoms": atoms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="pretrain_ragged_batch*.pt")
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--split_json", type=str, default=None)
    ap.add_argument("--feature_key", type=str, default=None)
    ap.add_argument("--edge_index_key", type=str, default=None)
    ap.add_argument("--edge_attr_key", type=str, default=None)
    ap.add_argument("--edge_offsets_key", type=str, default=None)
    ap.add_argument("--normalize", action="store_true")

    # GNN architecture. These are the only depth/width args used by the model.
    ap.add_argument("--gnn_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--aggr", type=str, default="sum", choices=["sum", "mean", "max"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_graph_context", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.30)
    ap.add_argument("--mask_mode", type=str, default="learned", choices=["learned", "zero", "noise"])
    ap.add_argument("--loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "mse", "l1"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--reset_optim", action="store_true")
    ap.add_argument("--reset_lr", action="store_true")


    ap.add_argument("--save_dir", type=str, default="./continuous_gnn_mae_ckpt")
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

    train_ds = ContinuousRaggedMolGraphDataset(
        args.data_dir,
        pattern=args.pattern,
        limit_files=args.limit_files,
        file_list=train_files,
        feature_key=args.feature_key,
        edge_index_key=args.edge_index_key,
        edge_attr_key=args.edge_attr_key,
        edge_offsets_key=args.edge_offsets_key,
        normalize=args.normalize,
    )
    valid_ds = None
    if valid_files:
        valid_ds = ContinuousRaggedMolGraphDataset(
            args.data_dir,
            pattern=args.pattern,
            file_list=valid_files,
            feature_key=args.feature_key,
            edge_index_key=args.edge_index_key,
            edge_attr_key=args.edge_attr_key,
            edge_offsets_key=args.edge_offsets_key,
            normalize=args.normalize,
        )

    print(f"Loaded {len(train_ds)} train molecules from {args.data_dir}")
    if valid_ds is not None:
        print(f"Loaded {len(valid_ds)} valid molecules")
    print(f"feat_dim={train_ds.feat_dim} edge_dim={train_ds.edge_dim}")
    print(f"mask_prob={args.mask_prob} mask_mode={args.mask_mode} loss_type={args.loss_type}")
    print(
        f"GNN config: gnn_layers={args.gnn_layers} hidden_dim={args.hidden_dim} "
        f"aggr={args.aggr} dropout={args.dropout} use_graph_context={args.use_graph_context}"
    )

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
    model = ContinuousGNNMAE(
        feat_dim=train_ds.feat_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        aggr=args.aggr,
        dropout=args.dropout,
        use_graph_context=args.use_graph_context,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    resume_step0 = 0
    resume_lr0 = args.lr
    resume_last_epoch = 0

    if args.resume is not None:
        print(f"[resume] loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        if int(ckpt.get("feat_dim", -1)) != int(train_ds.feat_dim):
            raise RuntimeError(f"[resume] feat_dim mismatch: ckpt={ckpt.get('feat_dim')} current={train_ds.feat_dim}")
        if int(ckpt.get("edge_dim", -1)) != int(train_ds.edge_dim):
            raise RuntimeError(f"[resume] edge_dim mismatch: ckpt={ckpt.get('edge_dim')} current={train_ds.edge_dim}")

        model.load_state_dict(ckpt["model"], strict=True)
        if (not args.reset_optim) and ("optim" in ckpt):
            optim.load_state_dict(ckpt["optim"])
        if "rng" in ckpt:
            try:
                random.setstate(ckpt["rng"]["python"])
                torch.set_rng_state(ckpt["rng"]["torch"])
                if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            except Exception as e:
                print(f"[resume] failed to restore RNG: {e}")

        global_step = int(ckpt.get("global_step", 0))
        resume_step0 = global_step
        resume_last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resume_last_epoch + 1
        resume_lr0 = float(optim.param_groups[0].get("lr", args.lr))
        print(f"[resume] resumed at epoch={resume_last_epoch}, global_step={global_step}, lr0={resume_lr0:.3e}")

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
            print(f"[resume] reset_lr: lr set to {lr0:.3e}")

    model.train()
    mask_gen = torch.Generator(device=device) if args.deterministic_masking else None

    for ep in range(start_epoch, args.epochs + 1):
        running_loss_sum = 0.0
        running_atoms = 0
        running_mae_sum = 0.0
        running_rmse_sum = 0.0
        running_batches = 0

        for it, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)

            if mask_gen is not None:
                mask_gen.manual_seed(int(args.seed + ep * 1_000_000 + global_step))

            x_masked, labels, mask_sel = mask_continuous_nodes(
                batch.x,
                mask_prob=args.mask_prob,
                mask_mode=args.mask_mode,
                generator=mask_gen,
            )

            edge_attr = batch.edge_attr if getattr(batch, "edge_attr", None) is not None else None
            pred = model(
                x_masked,
                edge_index=batch.edge_index,
                edge_attr=edge_attr,
                batch=batch.batch,
                mask_sel=mask_sel,
            )
            loss = reconstruction_loss(pred, labels, mask_sel, args.loss_type)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            global_step += 1
            lr_now_val = lr_now(global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_now_val

            with torch.no_grad():
                m = reconstruction_metrics(pred, labels, mask_sel)
                masked_atoms = int(m["masked_atoms"])

            running_loss_sum += float(loss.item()) * max(1, masked_atoms)
            running_atoms += masked_atoms
            running_mae_sum += float(m["mae"])
            running_rmse_sum += float(m["rmse"])
            running_batches += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss_sum / max(1, running_atoms)
                avg_mae = running_mae_sum / max(1, running_batches)
                avg_rmse = running_rmse_sum / max(1, running_batches)
                msg = (
                    f"[ep {ep}/{args.epochs}] step {global_step}/{total_steps_disp} "
                    f"gnn_mae_loss={avg_loss:.5f} mae={avg_mae:.5f} rmse={avg_rmse:.5f} "
                    f"lr={lr_now_val:.2e} masked_atoms={running_atoms}"
                )
                print(msg)
                write_log({
                    "time": time.time(),
                    "epoch": ep,
                    "step": global_step,
                    "loss": avg_loss,
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                    "lr": lr_now_val,
                    "masked_atoms": running_atoms,
                    "batch_size": args.batch_size,
                    "mask_prob": args.mask_prob,
                    "mask_mode": args.mask_mode,
                    "loss_type": args.loss_type,
                })
                running_loss_sum = 0.0
                running_atoms = 0
                running_mae_sum = 0.0
                running_rmse_sum = 0.0
                running_batches = 0

        if valid_loader is not None:
            vm = valid_epoch(model, valid_loader, device, args, mask_gen=mask_gen)
            print(
                f"[ep {ep}/{args.epochs}] VALID "
                f"gnn_mae_loss={vm['loss']:.5f} mae={vm['mae']:.5f} "
                f"rmse={vm['rmse']:.5f} masked_atoms={vm['masked_atoms']}"
            )
            write_log({
                "time": time.time(),
                "event": "valid",
                "epoch": ep,
                "step": global_step,
                **vm,
            })

        ckpt_path = os.path.join(args.save_dir, f"continuous_gnn_mae_ep{ep:02d}.pt")
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
            "feat_dim": train_ds.feat_dim,
            "edge_dim": train_ds.edge_dim,
            "gnn_layers": args.gnn_layers,
            "hidden_dim": args.hidden_dim,
            "aggr": args.aggr,
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
