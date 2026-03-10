#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np

PDBID = "1a4k"

CSV_PATH = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv"
DIST_PATH = f"/Users/taka/Downloads/dist_mats_npz_mean/{PDBID}.npz"


def get_sequence(pdbid):
    pdbid = pdbid.lower()

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["pdbid"].lower() == pdbid:
                return row["seq"].strip()

    return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, is_aa

PDBID = "1a4k"
CSV_PATH = "/Users/taka/Downloads/dti_stage1_seq_ligtok_y.csv"
DIST_PATH = f"/Users/taka/Downloads/dist_mats_npz_mean/{PDBID}.npz"

# どちらか存在する方を使う
STRUCTURE_PATH = f"/Users/taka/Downloads/{PDBID.upper()}.cif"
# STRUCTURE_PATH = f"/Users/taka/Downloads/{PDBID.upper()}.pdb"

LIGAND_RESNAME = "FRA"
CUTOFF = 4.5  # Å


def get_sequence(pdbid):
    pdbid = pdbid.lower()
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pdbid"].lower() == pdbid:
                return row["seq"].strip()
    return None


def load_structure(path, pdbid="X"):
    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(pdbid, path)


def get_ligand_residues(structure, ligand_resname="FRA"):
    ligs = []
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag = res.id[0]
                if hetflag.strip() != "" and res.resname.strip() == ligand_resname:
                    ligs.append(res)
    return ligs


def atom_dist(a, b):
    return np.linalg.norm(a.coord - b.coord)


def get_residues_near_ligand(structure, ligand_resname="FRA", cutoff=4.5):
    ligs = get_ligand_residues(structure, ligand_resname)
    out = []

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    continue

                dmin = float("inf")
                hit = False
                for latom in [a for lig in ligs for a in lig.get_atoms()]:
                    for patom in res.get_atoms():
                        d = atom_dist(patom, latom)
                        if d < dmin:
                            dmin = d
                        if d <= cutoff:
                            hit = True

                if hit:
                    out.append({
                        "chain": chain.id,
                        "resi": res.id[1],
                        "icode": res.id[2].strip(),
                        "resn": res.resname,
                        "dmin": dmin,
                    })

    out.sort(key=lambda x: (x["dmin"], x["chain"], x["resi"], x["icode"]))
    return out


def aa3_to_1(resn):
    table = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
        "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
        "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
        "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    return table.get(resn.upper(), "?")


def main():

    # --------------------------------------------------------------------------------
    # PDBファイルから読み取った近接アミノ酸残基ランキング
    # --------------------------------------------------------------------------------
    seq = get_sequence(PDBID)
    if seq is None:
        print("sequence not found in CSV")
        return

    arr = np.load(DIST_PATH, allow_pickle=True)
    d_mean = arr["d_mean"]

    print(f"PDBID: {PDBID}")
    print(f"CSV seq length : {len(seq)}")
    print(f"d_mean length  : {len(d_mean)}")
    print()

    idx = np.argsort(d_mean)[:5]
    print("Top 5 by d_mean")
    print("----------------")
    top5 = []
    for i in idx:
        aa = seq[i] if i < len(seq) else "?"
        top5.append((int(i), aa, float(d_mean[i])))
        print(f"seq_idx={i:3d}  aa={aa}  d_mean={d_mean[i]:.3f}")
    print()

    structure = load_structure(STRUCTURE_PATH, PDBID)
    near = get_residues_near_ligand(structure, ligand_resname=LIGAND_RESNAME, cutoff=CUTOFF)

    print(f"Residues within {CUTOFF:.1f} Å of {LIGAND_RESNAME}")
    print("-----------------------------------")
    for x in near[:30]:
        aa1 = aa3_to_1(x["resn"])
        icode = x["icode"] if x["icode"] else ""
        print(
            f"chain={x['chain']}  resi={x['resi']}{icode:1s}  "
            f"resn={x['resn']:>3s}({aa1})  dmin={x['dmin']:.3f}"
        )

    print()
    print("Note:")
    print("- seq_idx is your CSV/d_mean index (0-based)")
    print("- chain/resi is the actual PDB residue numbering")
    print("- These are not guaranteed to match directly")
    print("- If needed, next step is building seq_idx -> chain/resi mapping")

    # --------------------------------------------------------------------------------
    # csv から、データファイルから読み取った近接アミノ酸残基ランキング
    # --------------------------------------------------------------------------------
    # 距離読み込み
    arr = np.load(DIST_PATH, allow_pickle=True)
    d_mean = arr["d_mean"]

    # 配列取得
    seq = get_sequence(PDBID)

    print("PDBID:", PDBID)
    print("Sequence length:", len(seq))
    print("d_mean length :", len(d_mean))
    print()

    # 小さい順 index
    idx = np.argsort(d_mean)[:5]

    print("Top 5 closest residues:")
    print("-----------------------")

    for i in idx:
        aa = seq[i] if i < len(seq) else "?"
        print(
            f"res_idx={i:3d}  aa={aa}  d_mean={d_mean[i]:.3f}"
        )

if __name__ == "__main__":
    main()