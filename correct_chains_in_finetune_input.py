#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read:
  /Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv

For each row:
  - read pdbid
  - read current seq
  - read the corresponding mmCIF from /Users/taka/Downloads/{PDBID}.cif
  - detect ligand heavy atoms
  - rank protein chains by ligand contact
  - choose the best chain
  - extract the chosen chain sequence
  - replace seq with the chosen chain sequence

Write:
  /Users/taka/Downloads/dti_stage1_seq_ligtok_y_reselected_seq.csv

Also write a log:
  /Users/taka/Downloads/dti_stage1_seq_ligtok_y_reselected_seq.log.csv

The log records whether seq changed or not.
"""

import csv
import os
from typing import List, Optional, Tuple, Dict, Any
import urllib.request

import numpy as np
from Bio.PDB import MMCIFParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1

IN_CSV = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv"
OUT_CSV = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y_reselected_seq.csv"
LOG_CSV = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y_reselected_seq.log.csv"
CIF_DIR = "/Users/taka/Downloads"

LIGAND_EXCLUDE = {"HOH", "WAT", "DOD"}
CONTACT_CUTOFF = 4.5


def residue_to_aa1(res) -> str:
    name3 = (res.get_resname() or "").strip().upper()
    return protein_letters_3to1.get(name3.capitalize(), "X")


def get_protein_residues(chain) -> List:
    return [r for r in chain if is_aa(r, standard=False)]


def chain_sequence(chain) -> str:
    residues = get_protein_residues(chain)
    return "".join(residue_to_aa1(r) for r in residues)

def ensure_cif_downloaded(pdbid: str) -> str:
    """
    Ensure mmCIF exists locally. Download if missing.
    """
    pdbid = pdbid.lower()
    cif_path = os.path.join(CIF_DIR, f"{pdbid}.cif")
    if os.path.exists(cif_path):
        return cif_path

    url = f"https://files.rcsb.org/download/{pdbid}.cif"

    print(f"[download] {pdbid} -> {cif_path}")
    import time
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, cif_path)
            break
        except:
            time.sleep(1)

    return cif_path

def detect_ligand_atoms(structure) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Very simple ligand detection:
      - exclude amino acids
      - exclude water
      - merge all remaining heavy atoms as ligand atoms
    """
    lig_coords = []
    lig_meta = []

    for model in structure:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=False):
                    continue

                resname = (res.get_resname() or "").strip().upper()
                if resname in LIGAND_EXCLUDE:
                    continue

                atom_count = 0
                for atom in res:
                    elem = (atom.element or "").strip().upper()
                    if elem == "H":
                        continue
                    lig_coords.append(atom.get_coord())
                    atom_count += 1

                if atom_count > 0:
                    lig_meta.append({
                        "model": model.id,
                        "chain": chain.id,
                        "resname": resname,
                        "resseq": int(res.id[1]),
                        "atom_count": atom_count,
                    })

    if not lig_coords:
        raise RuntimeError("No ligand heavy atoms detected")

    return np.asarray(lig_coords, dtype=np.float32), lig_meta


def rank_chains_by_ligand_contact(structure, lig_coords, cutoff=4.5):
    """
    Choose chain by:
      1) number of residues with min distance < cutoff
      2) soft score sum(exp(-d/4))
      3) smallest best_contact
    """
    rows = []

    for model in structure:
        for chain in model:
            residues = get_protein_residues(chain)
            if not residues:
                continue

            best = 1e9
            contact = 0
            soft_score = 0.0

            for res in residues:
                coords = [a.get_coord() for a in res.get_atoms()]
                if not coords:
                    continue

                coords = np.asarray(coords, dtype=np.float32)
                d = np.linalg.norm(
                    coords[:, None, :] - lig_coords[None, :, :],
                    axis=2
                )
                dmin = float(d.min())

                best = min(best, dmin)
                if dmin < cutoff:
                    contact += 1
                soft_score += float(np.exp(-dmin / 4.0))

            rows.append({
                "model": model.id,
                "chain": chain.id,
                "best_contact": float(best),
                "contact_count": int(contact),
                "soft_score": float(soft_score),
            })

    if not rows:
        raise RuntimeError("No protein chains found")

    rows.sort(
        key=lambda x: (x["contact_count"], x["soft_score"], -x["best_contact"]),
        reverse=True,
    )
    return rows


def process_one_pdb(pdbid: str) -> Dict[str, Any]:
    cif_path = ensure_cif_downloaded(pdbid)
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"mmCIF not found: {cif_path}")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdbid, cif_path)

    lig_coords, lig_meta = detect_ligand_atoms(structure)
    chain_rows = rank_chains_by_ligand_contact(structure, lig_coords, cutoff=CONTACT_CUTOFF)

    best = chain_rows[0]
    best_chain = best["chain"]
    best_model = best["model"]

    chosen_chain = structure[best_model][best_chain]
    new_seq = chain_sequence(chosen_chain)

    return {
        "new_seq": new_seq,
        "chosen_chain": best_chain,
        "chosen_model": best_model,
        "best_contact": best["best_contact"],
        "contact_count": best["contact_count"],
        "soft_score": best["soft_score"],
        "ligand_summary": ";".join(
            f"{x['chain']}:{x['resname']}{x['resseq']}({x['atom_count']})" for x in lig_meta
        ),
    }


def main():
    with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError("Input CSV has no header")
    if "pdbid" not in fieldnames or "seq" not in fieldnames:
        raise RuntimeError("Input CSV must contain columns: pdbid, seq")

    out_rows = []
    log_rows = []

    for i, row in enumerate(rows, start=1):
        pdbid = str(row.get("pdbid", "")).strip().lower()
        old_seq = str(row.get("seq", "")).strip()

        log = {
            "row_index": i,
            "pdbid": pdbid,
            "old_seq_len": len(old_seq),
            "new_seq_len": "",
            "changed": "",
            "status": "",
            "chosen_chain": "",
            "chosen_model": "",
            "best_contact": "",
            "contact_count": "",
            "soft_score": "",
            "ligand_summary": "",
            "message": "",
        }

        new_row = dict(row)

        try:
            if not pdbid:
                raise ValueError("empty pdbid")

            info = process_one_pdb(pdbid)
            new_seq = info["new_seq"]

            new_row["seq"] = new_seq

            changed = (new_seq != old_seq)

            log.update({
                "new_seq_len": len(new_seq),
                "changed": int(changed),
                "status": "ok",
                "chosen_chain": info["chosen_chain"],
                "chosen_model": info["chosen_model"],
                "best_contact": f"{info['best_contact']:.4f}",
                "contact_count": info["contact_count"],
                "soft_score": f"{info['soft_score']:.4f}",
                "ligand_summary": info["ligand_summary"],
                "message": "sequence_changed" if changed else "sequence_unchanged",
            })

        except Exception as e:
            log.update({
                "new_seq_len": len(old_seq),
                "changed": "",
                "status": "error",
                "message": repr(e),
            })
            # keep original seq on error

        out_rows.append(new_row)
        log_rows.append(log)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log_fieldnames = [
        "row_index",
        "pdbid",
        "old_seq_len",
        "new_seq_len",
        "changed",
        "status",
        "chosen_chain",
        "chosen_model",
        "best_contact",
        "contact_count",
        "soft_score",
        "ligand_summary",
        "message",
    ]
    with open(LOG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("saved:", OUT_CSV)
    print("saved:", LOG_CSV)


if __name__ == "__main__":
    main()