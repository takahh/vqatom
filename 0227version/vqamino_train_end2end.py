#!/usr/bin/env python3
"""
End-to-end VQ-Amino training from preprocessed sequence graphs.

Usage:
  python vqamino_preprocess.py --input proteins.fasta --output data/vqamino_train.pt
  python vqamino_train_end2end.py --data data/vqamino_train.pt --epochs 20

This is a sequence-only starter pipeline:
  residue features + sequence k-hop graph -> GNN contextualization -> per-AA VQ codebooks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

AA20X = "ACDEFGHIKLMNPQRSTVWYX"


class VQAminoDataset(Dataset):
    def __init__(self, path: str):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        self.examples = obj["examples"] if isinstance(obj, dict) and "examples" in obj else obj
        if len(self.examples) == 0:
            raise ValueError("empty dataset")
        self.feature_dim = int(self.examples[0]["x"].shape[1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_graphs(batch: List[dict]) -> dict:
    xs, aa_ids, edges, edge_types, batch_vec = [], [], [], [], []
    seqs, ids = [], []
    offset = 0
    for b, ex in enumerate(batch):
        x = ex["x"].float()
        n = x.shape[0]
        xs.append(x)
        aa_ids.append(ex["aa_id"].long())
        if ex["edge_index"].numel() > 0:
            edges.append(ex["edge_index"].long() + offset)
            edge_types.append(ex["edge_type"].long())
        batch_vec.append(torch.full((n,), b, dtype=torch.long))
        seqs.append(ex.get("seq", "")); ids.append(ex.get("id", str(b)))
        offset += n
    out = {
        "x": torch.cat(xs, dim=0),
        "aa_id": torch.cat(aa_ids, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "seqs": seqs,
        "ids": ids,
    }
    if edges:
        out["edge_index"] = torch.cat(edges, dim=1)
        out["edge_type"] = torch.cat(edge_types, dim=0)
    else:
        out["edge_index"] = torch.empty((2, 0), dtype=torch.long)
        out["edge_type"] = torch.empty((0,), dtype=torch.long)
    return out


class SimpleGraphConv(nn.Module):
    def __init__(self, dim: int, max_edge_type: int = 3):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.edge_emb = nn.Embedding(max_edge_type + 1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = h[src] + self.edge_emb(edge_type.clamp(0, self.edge_emb.num_embeddings - 1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((h.shape[0],), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(1)
        return self.norm(F.relu(self.self_lin(h) + self.neigh_lin(agg)))


class ResidueGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, max_edge_type: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SimpleGraphConv(hidden_dim, max_edge_type=max_edge_type) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.in_proj(x))
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_type)
        return F.normalize(self.out(h), dim=-1, eps=1e-8)


class PerAAVectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 64, n_aa: int = 21, beta: float = 0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.n_aa = n_aa
        self.beta = beta
        self.codebooks = nn.Parameter(torch.randn(n_aa, codebook_size, dim) * 0.02)

    def forward(self, z: torch.Tensor, aa_id: torch.Tensor) -> dict:
        zq = torch.empty_like(z)
        token_id = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
        losses = []
        usage: Dict[int, int] = {}
        for aa in aa_id.unique(sorted=True):
            a = int(aa.item())
            if a < 0 or a >= self.n_aa:
                continue
            mask = aa_id == a
            zi = z[mask]
            cb = F.normalize(self.codebooks[a], dim=-1, eps=1e-8)
            # cosine distance because z and cb are normalized
            sim = zi @ cb.t()
            idx = sim.argmax(dim=1)
            qi = cb.index_select(0, idx)
            zq[mask] = qi
            token_id[mask] = a * self.codebook_size + idx
            usage[a] = int(idx.unique().numel())
            commit = F.mse_loss(zi, qi.detach())
            codebook = F.mse_loss(qi, zi.detach())
            losses.append(self.beta * commit + codebook)
        if not losses:
            loss = z.sum() * 0.0
        else:
            loss = torch.stack(losses).mean()
        # straight-through quantized output, useful for future decoder/task heads
        z_st = z + (zq - z).detach()
        return {"loss": loss, "z_q": z_st, "token_id": token_id, "usage": usage}

    @property
    def vocab_size(self) -> int:
        return self.n_aa * self.codebook_size + 2  # plus PAD/MASK convention


class VQAminoModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, codebook_size: int, beta: float, max_edge_type: int):
        super().__init__()
        self.encoder = ResidueGNN(in_dim, hidden_dim, layers, max_edge_type=max_edge_type)
        self.vq = PerAAVectorQuantizer(hidden_dim, codebook_size=codebook_size, beta=beta)

    def forward(self, batch: dict) -> dict:
        z = self.encoder(batch["x"], batch["edge_index"], batch["edge_type"])
        out = self.vq(z, batch["aa_id"])
        # small variance regularizer discourages all latents from becoming identical
        std_loss = F.relu(0.5 - z.std(dim=0)).mean()
        out["loss"] = out["loss"] + 0.01 * std_loss
        out["z"] = z
        return out


@torch.no_grad()
def save_tokens(model: VQAminoModel, loader: DataLoader, device: torch.device, path: str) -> None:
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        token = out["token_id"].detach().cpu()
        start = 0
        for sid, seq in zip(batch["ids"], batch["seqs"]):
            n = len(seq)
            rows.append({"id": sid, "seq": seq, "token_ids": token[start:start+n].tolist()})
            start += n
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "aa_vocab": AA20X, "codebook_size_per_aa": model.vq.codebook_size, "vocab_size": model.vq.vocab_size}, path)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/vqamino_train.pt")
    p.add_argument("--out_dir", default="runs/vqamino")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=64, help="codes per amino-acid type")
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_every", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = VQAminoDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    device = torch.device(args.device)
    # max edge type is inferred from preprocessing k_hop; 3 works for default and clamps larger if needed.
    model = VQAminoModel(ds.feature_dim, args.hidden_dim, args.layers, args.codebook_size, args.beta, max_edge_type=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args) | {"feature_dim": ds.feature_dim, "vocab_size": model.vq.vocab_size}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    for ep in range(1, args.epochs + 1):
        model.train()
        total, steps = 0.0, 0
        usage_sum: Dict[int, List[int]] = {}
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()); steps += 1
            for k, v in out["usage"].items():
                usage_sum.setdefault(k, []).append(v)
        mean_loss = total / max(1, steps)
        mean_usage = {AA20X[k]: round(sum(v)/len(v), 2) for k, v in usage_sum.items() if k < len(AA20X)}
        print(f"epoch={ep:04d} loss={mean_loss:.6f} mean_used_codes={mean_usage}")
        if ep % args.save_every == 0 or ep == args.epochs:
            ckpt_path = out_dir / f"model_epoch_{ep:04d}.pt"
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep}, ckpt_path)
            save_tokens(model, eval_loader, device, str(out_dir / f"vqamino_tokens_epoch_{ep:04d}.pt"))
            print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
