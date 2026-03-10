#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
from typing import Optional
from Bio.PDB import MMCIFParser, is_aa
from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1

PDBID = "1a4k"

CSV_PATH = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv"
NPZ_PATH = f"/Users/taka/Downloads/dist_mats_npz_mean/{PDBID}.npz"
CIF_PATH = f"/Users/taka/Downloads/{PDBID.upper()}.cif"

CHAINS_TO_CHECK = ["B", "H"]
TOPK = 30


def residue_to_aa1(res) -> str:
    name3 = (res.get_resname() or "").strip().upper()
    return protein_letters_3to1.get(name3.capitalize(), "X")


def get_csv_seq(pdbid: str, csv_path: str) -> Optional[str]:
    pdbid = pdbid.lower()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pdbid"].lower() == pdbid:
                return row["seq"].strip()
    return None

def print_seq_block(seq: str, width: int = 20):
    print("CSV protein sequence:")
    print("----------------------")
    for i in range(0, len(seq), width):
        block = seq[i:i+width]
        print(f"{i:4d}-{i+len(block)-1:4d}  {block}")
    print()

def build_seqidx_to_residue_mapping(csv_seq: str, residues: list):
    pdb_seq = "".join(residue_to_aa1(r) for r in residues)

    aln = pairwise2.align.globalms(
        csv_seq,
        pdb_seq,
        2.0,   # match
        -1.0,  # mismatch
        -5.0,  # gap open
        -0.5,  # gap extend
        one_alignment_only=True,
    )
    if not aln:
        raise RuntimeError("alignment failed")

    aln_csv, aln_pdb, score, _, _ = aln[0]

    mapping = {}
    i_csv = 0
    i_pdb = 0

    for a, b in zip(aln_csv, aln_pdb):
        if a != "-" and b != "-":
            mapping[i_csv] = residues[i_pdb]
            i_csv += 1
            i_pdb += 1
        elif a != "-" and b == "-":
            i_csv += 1
        elif a == "-" and b != "-":
            i_pdb += 1

        if i_csv >= len(csv_seq):
            break

    return mapping, score, pdb_seq


def main():
    csv_seq = get_csv_seq(PDBID, CSV_PATH)
    if csv_seq is None:
        raise RuntimeError(f"pdbid not found in CSV: {PDBID}")

    arr = np.load(NPZ_PATH, allow_pickle=True)

    if "d_min" in arr:
        d_min = arr["d_min"]
        d_key = "d_min"
    elif "d_mean" in arr:
        d_min = arr["d_mean"]   # 今のファイルでは名前だけ古い可能性
        d_key = "d_mean(as_d_min)"
    else:
        raise RuntimeError("Neither d_min nor d_mean found in npz")

    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure(PDBID, CIF_PATH)

    chain_maps = {}
    chain_scores = {}

    for ch in CHAINS_TO_CHECK:
        if ch not in st[0]:
            print(f"[warn] chain {ch} not found")
            continue

        residues = [r for r in st[0][ch] if is_aa(r, standard=False)]
        mapping, score, pdb_seq = build_seqidx_to_residue_mapping(csv_seq, residues)
        chain_maps[ch] = mapping
        chain_scores[ch] = {
            "score": score,
            "pdb_seq_len": len(pdb_seq),
            "mapped": len(mapping),
        }

    print(f"PDBID: {PDBID}")
    print(f"CSV seq len: {len(csv_seq)}")
    print(f"distance key: {d_key}")
    print(f"distance len: {len(d_min)}")
    print()

    for ch, info in chain_scores.items():
        print(f"chain {ch}: score={info['score']:.1f} pdb_seq_len={info['pdb_seq_len']} mapped={info['mapped']}")
    print()

    top_idx = np.argsort(d_min)[:TOPK]

    rows = []
    for seq_idx in top_idx:
        row = {
            "seq_idx": int(seq_idx),
            "csv_aa": csv_seq[int(seq_idx)] if int(seq_idx) < len(csv_seq) else "?",
            "d_min": float(d_min[int(seq_idx)]),
        }

        for ch in CHAINS_TO_CHECK:
            mp = chain_maps.get(ch, {})
            res = mp.get(int(seq_idx), None)
            if res is None:
                row[f"{ch}_resi"] = None
                row[f"{ch}_resn"] = None
                row[f"{ch}_aa"] = None
            else:
                row[f"{ch}_resi"] = int(res.id[1])
                row[f"{ch}_resn"] = res.resname
                row[f"{ch}_aa"] = residue_to_aa1(res)

        rows.append(row)

    df = pd.DataFrame(rows)

    print("Top seq_idx mapped to chain B/H")
    print("--------------------------------")
    print(df.to_string(index=False))

    out_csv = f"/Users/taka/Downloads/{PDBID}_seqidx_to_chainBH_top{TOPK}.csv"
    df.to_csv(out_csv, index=False)
    print()
    print_seq_block(csv_seq, 20)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()