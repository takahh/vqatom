#!/usr/bin/env python3
"""
VQ-Amino preprocessing: raw protein sequences -> residue graphs/features (.pt).

Input formats
-------------
1) FASTA:
   >protein_id
   MKTFFV...

2) CSV/TSV with at least one sequence column:
   id,sequence
   P12345,MKTFFV...

3) TXT: one sequence per line.

Output
------
A torch file containing a list of examples:
  {
    'id': str,
    'seq': str,
    'x': FloatTensor [L, F],
    'aa_id': LongTensor [L],
    'edge_index': LongTensor [2, E],
    'edge_type': LongTensor [E],   # hop distance: 1..k
  }
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID: Dict[str, int] = {aa: i for i, aa in enumerate(AA20)}
UNK_ID = len(AA20)

# Simple residue descriptors. Values are intentionally rough/scaled, not a biochemical oracle.
# columns: hydrophobicity(KD)/4.5, charge, polar, aromatic, sulfur, small, proline, glycine, mass/200
AA_PROPS: Dict[str, Tuple[float, ...]] = {
    "A": ( 1.8/4.5,  0, 0, 0, 0, 1, 0, 0,  89.09/200),
    "C": ( 2.5/4.5,  0, 0, 0, 1, 1, 0, 0, 121.16/200),
    "D": (-3.5/4.5, -1, 1, 0, 0, 0, 0, 0, 133.10/200),
    "E": (-3.5/4.5, -1, 1, 0, 0, 0, 0, 0, 147.13/200),
    "F": ( 2.8/4.5,  0, 0, 1, 0, 0, 0, 0, 165.19/200),
    "G": (-0.4/4.5,  0, 0, 0, 0, 1, 0, 1,  75.07/200),
    "H": (-3.2/4.5,  0.5, 1, 1, 0, 0, 0, 0, 155.16/200),
    "I": ( 4.5/4.5,  0, 0, 0, 0, 0, 0, 0, 131.17/200),
    "K": (-3.9/4.5,  1, 1, 0, 0, 0, 0, 0, 146.19/200),
    "L": ( 3.8/4.5,  0, 0, 0, 0, 0, 0, 0, 131.17/200),
    "M": ( 1.9/4.5,  0, 0, 0, 1, 0, 0, 0, 149.21/200),
    "N": (-3.5/4.5,  0, 1, 0, 0, 0, 0, 0, 132.12/200),
    "P": (-1.6/4.5,  0, 0, 0, 0, 0, 1, 0, 115.13/200),
    "Q": (-3.5/4.5,  0, 1, 0, 0, 0, 0, 0, 146.15/200),
    "R": (-4.5/4.5,  1, 1, 0, 0, 0, 0, 0, 174.20/200),
    "S": (-0.8/4.5,  0, 1, 0, 0, 1, 0, 0, 105.09/200),
    "T": (-0.7/4.5,  0, 1, 0, 0, 1, 0, 0, 119.12/200),
    "V": ( 4.2/4.5,  0, 0, 0, 0, 0, 0, 0, 117.15/200),
    "W": (-0.9/4.5,  0, 0, 1, 0, 0, 0, 0, 204.23/200),
    "Y": (-1.3/4.5,  0, 1, 1, 0, 0, 0, 0, 181.19/200),
}
N_PROP = len(next(iter(AA_PROPS.values())))


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    cur_id, cur_seq = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(cur_seq)))
            cur_id = line[1:].split()[0] or f"seq_{len(records)}"
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq)))
    return records


def read_table(path: Path, seq_col: str, id_col: str | None) -> List[Tuple[str, str]]:
    sample = path.read_text(errors="ignore")[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    with path.open(newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        out = []
        for i, row in enumerate(reader):
            sid = row.get(id_col, "") if id_col else ""
            sid = sid or f"seq_{i}"
            seq = row[seq_col]
            out.append((sid, seq))
    return out


def read_input(path: Path, seq_col: str, id_col: str | None) -> List[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        return read_fasta(path)
    if suffix in {".csv", ".tsv"}:
        return read_table(path, seq_col=seq_col, id_col=id_col)
    # txt fallback: one sequence per line
    records = []
    for i, line in enumerate(path.read_text().splitlines()):
        seq = line.strip()
        if seq and not seq.startswith("#"):
            records.append((f"seq_{i}", seq))
    return records


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in seq.upper().replace(" ", "") if ch.isalpha())


def residue_features(seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([AA_TO_ID.get(a, UNK_ID) for a in seq], dtype=torch.long)
    onehot = torch.zeros((len(seq), len(AA20) + 1), dtype=torch.float32)
    onehot[torch.arange(len(seq)), ids] = 1.0
    props = []
    for a in seq:
        props.append(AA_PROPS.get(a, (0.0,) * N_PROP))
    prop_t = torch.tensor(props, dtype=torch.float32) if props else torch.empty((0, N_PROP))
    # normalized absolute and relative position features
    if len(seq) > 1:
        pos = torch.linspace(0.0, 1.0, len(seq)).unsqueeze(1)
    else:
        pos = torch.zeros((len(seq), 1), dtype=torch.float32)
    nterm = (torch.arange(len(seq)) == 0).float().unsqueeze(1)
    cterm = (torch.arange(len(seq)) == len(seq) - 1).float().unsqueeze(1)
    x = torch.cat([onehot, prop_t, pos, nterm, cterm], dim=1)
    return x, ids


def sequence_edges(length: int, k: int = 3, self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst, etype = [], [], []
    for hop in range(1, k + 1):
        for i in range(length - hop):
            j = i + hop
            src += [i, j]
            dst += [j, i]
            etype += [hop, hop]
    if self_loops:
        for i in range(length):
            src.append(i); dst.append(i); etype.append(0)
    if not src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(etype, dtype=torch.long)


def build_examples(records: Iterable[Tuple[str, str]], max_len: int, min_len: int, k_hop: int) -> List[dict]:
    examples = []
    skipped = 0
    for sid, raw_seq in records:
        seq = clean_sequence(raw_seq)
        if len(seq) < min_len:
            skipped += 1
            continue
        if max_len > 0 and len(seq) > max_len:
            seq = seq[:max_len]
        x, aa_id = residue_features(seq)
        edge_index, edge_type = sequence_edges(len(seq), k=k_hop, self_loops=True)
        examples.append({"id": sid, "seq": seq, "x": x, "aa_id": aa_id, "edge_index": edge_index, "edge_type": edge_type})
    print(f"built {len(examples)} examples; skipped {skipped}")
    return examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="FASTA/CSV/TSV/TXT containing protein sequences")
    p.add_argument("--output", default="data/vqamino_train.pt")
    p.add_argument("--seq_col", default="sequence")
    p.add_argument("--id_col", default="id")
    p.add_argument("--max_len", type=int, default=1024, help="truncate longer sequences; <=0 disables")
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--k_hop", type=int, default=3, help="connect i to i +/- hop for hop=1..k")
    args = p.parse_args()

    in_path = Path(args.input)
    records = read_input(in_path, seq_col=args.seq_col, id_col=args.id_col)
    examples = build_examples(records, max_len=args.max_len, min_len=args.min_len, k_hop=args.k_hop)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": examples, "aa_vocab": AA20 + "X", "feature_dim": examples[0]["x"].shape[1] if examples else None}, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
