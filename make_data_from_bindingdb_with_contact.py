#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import gzip
import json
import random
import argparse
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

MIN_RESOLUTION = 2.5

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O",
}

WATER = {"HOH", "WAT", "H2O", "DOD"}
COMMON_IONS = {
    "NA", "K", "CL", "MG", "MN", "CA", "ZN", "FE", "CU", "CO", "NI",
    "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "FMT", "DMS",
}


def open_text_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def get_resolution_from_cif(cif_path):
    try:
        d = MMCIF2Dict(cif_path)
        v = d.get("_refine.ls_d_res_high")
        if isinstance(v, list):
            v = v[0]
        if v in (None, "?", "."):
            return None
        return float(v)
    except Exception:
        return None

def norm_smiles(s):
    if not s:
        return ""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return ""
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return ""


def inchikey_from_mol(m):
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return ""


def mol_has_3d(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    if not conf.Is3D():
        return False
    coords = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([p.x, p.y, p.z])
    coords = np.asarray(coords, dtype=float)
    return np.isfinite(coords).all() and coords.std() > 1e-3


def load_sdf_index(sdf_path, max_mols=None):
    """
    index:
      monomer_id -> mol
      canonical_smiles -> mol
      inchikey -> mol
    """
    by_monomer = {}
    by_smiles = {}
    by_inchikey = {}

    suppl = Chem.ForwardSDMolSupplier(
        sdf_path,
        sanitize=True,
        removeHs=False,
    )

    n = 0
    kept = 0
    for mol in tqdm(suppl, desc="index SDF"):
        n += 1
        if mol is None:
            continue
        if not mol_has_3d(mol):
            continue

        props = {p: str(mol.GetProp(p)) for p in mol.GetPropNames()}

        monomer = (
            props.get("BindingDB MonomerID")
            or props.get("MonomerID")
            or props.get("monomerid")
            or props.get("Ligand ID")
            or ""
        ).strip()

        smi = (
            props.get("SMILES")
            or props.get("Ligand SMILES")
            or props.get("smiles")
            or ""
        ).strip()

        can = norm_smiles(smi)
        if not can:
            try:
                can = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            except Exception:
                can = ""
        try:
            mol_noh = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            continue

        ik = inchikey_from_mol(mol_noh)
        if monomer:
            by_monomer[monomer] = mol
        if can:
            by_smiles[can] = mol
        if ik:
            by_inchikey[ik] = mol

        kept += 1
        if max_mols and kept >= max_mols:
            break

    print(f"[SDF] scanned={n} kept_3d={kept}")
    print(f"[SDF] by_monomer={len(by_monomer)} by_smiles={len(by_smiles)} by_inchikey={len(by_inchikey)}")
    return by_monomer, by_smiles, by_inchikey


def guess_col(fieldnames, candidates):
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for f in fieldnames:
        fl = f.lower()
        for cand in candidates:
            if cand.lower() in fl:
                return f
    return None


def parse_pdb_ids(s):
    if not s:
        return []
    ids = re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", str(s))
    return sorted(set(x.upper() for x in ids))


def parse_affinity_nm(row, cols):
    for c in cols:
        if not c:
            continue
        raw = row.get(c, "")
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        s = s.replace(",", "")
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        if not m:
            continue
        try:
            return c, float(m.group(1))
        except Exception:
            pass
    return "", None


def nm_to_pchem(nm):
    if nm is None or nm <= 0:
        return None
    return -np.log10(nm * 1e-9)


def residue_seq(chain):
    seq = []
    res_indices = []
    for res in chain:
        if not is_aa(res, standard=False):
            continue
        name = res.get_resname().upper()
        aa = AA3_TO_1.get(name)
        if aa is None:
            continue
        seq.append(aa)
        res_indices.append(res.get_id())
    return "".join(seq), res_indices


def is_candidate_ligand_residue(res):
    name = res.get_resname().strip().upper()
    hetflag = res.get_id()[0].strip()

    if not hetflag:
        return False
    if name in WATER:
        return False
    if name in COMMON_IONS:
        return False

    atoms = list(res.get_atoms())
    heavy = [a for a in atoms if (a.element or "").upper() != "H"]
    if len(heavy) < 5:
        return False
    return True


def pdb_line_for_atom(serial, atom, resname, chain_id, resseq):
    coord = atom.coord
    atom_name = atom.get_name()
    elem = (atom.element or atom_name[0]).strip().upper()
    return (
        f"HETATM{serial:5d} {atom_name:<4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"  1.00 20.00          {elem:>2s}\n"
    )


def residue_to_pdb_block(residue, chain_id="A"):
    resname = residue.get_resname().strip()
    resseq = residue.get_id()[1]
    lines = []
    for i, atom in enumerate(residue.get_atoms(), start=1):
        lines.append(pdb_line_for_atom(i, atom, resname, chain_id, resseq))
    lines.append("END\n")
    return "".join(lines)


def match_sdf_to_pdb_ligand(sdf_mol, residue, chain_id):
    """
    returns:
      pdb_mol, sdf_to_pdb_atom_idx dict, pdb_to_sdf_atom_idx dict
    """
    sdf_noh = Chem.RemoveHs(sdf_mol)

    block = residue_to_pdb_block(residue, chain_id=chain_id)
    pdb_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if pdb_mol is None:
        return None

    pdb_noh = Chem.RemoveHs(pdb_mol, sanitize=False)

    try:
        pdb_bo = AllChem.AssignBondOrdersFromTemplate(sdf_noh, pdb_noh)
    except Exception:
        pdb_bo = pdb_noh

    match = pdb_bo.GetSubstructMatch(sdf_noh)
    if not match or len(match) != sdf_noh.GetNumAtoms():
        match = sdf_noh.GetSubstructMatch(pdb_bo)
        if match:
            return None

    sdf_to_pdb = {int(i): int(match[i]) for i in range(len(match))}
    pdb_to_sdf = {v: k for k, v in sdf_to_pdb.items()}

    return pdb_bo, sdf_to_pdb, pdb_to_sdf


def compute_contacts(structure, chain, ligand_residue, pdb_to_sdf, cutoff=4.5):
    seq, res_ids = residue_seq(chain)
    if not seq:
        return None

    res_id_to_j = {rid: j for j, rid in enumerate(res_ids)}

    protein_atoms = []
    atom_to_res_j = {}
    for res in chain:
        if res.get_id() not in res_id_to_j:
            continue
        j = res_id_to_j[res.get_id()]
        for atom in res.get_atoms():
            protein_atoms.append(atom)
            atom_to_res_j[atom] = j

    if not protein_atoms:
        return None

    ns = NeighborSearch(protein_atoms)

    contact_mask = [0] * len(seq)
    atom_contact_pairs = set()

    pdb_atom_idx = 0
    for atom in ligand_residue.get_atoms():
        elem = (atom.element or "").upper()
        if elem == "H":
            continue

        if pdb_atom_idx not in pdb_to_sdf:
            pdb_atom_idx += 1
            continue

        sdf_i = pdb_to_sdf[pdb_atom_idx]
        near_atoms = ns.search(atom.coord, cutoff, level="A")

        for pa in near_atoms:
            j = atom_to_res_j.get(pa)
            if j is None:
                continue
            contact_mask[j] = 1
            atom_contact_pairs.add((int(sdf_i), int(j)))

        pdb_atom_idx += 1

    contact_n = int(sum(contact_mask))
    return {
        "seq": seq,
        "contact_mask": "".join("1" if x else "0" for x in contact_mask),
        "contact_n": contact_n,
        "atom_contact_pairs": sorted(atom_contact_pairs),
    }


def find_best_ligand_match_in_structure(sdf_mol, structure):
    best = None

    for model in structure:
        for chain in model:
            chain_id = chain.id
            seq, _ = residue_seq(chain)
            if not seq:
                continue

            for res in chain:
                if not is_candidate_ligand_residue(res):
                    continue

                matched = match_sdf_to_pdb_ligand(sdf_mol, res, chain_id)
                if matched is None:
                    continue

                pdb_mol, sdf_to_pdb, pdb_to_sdf = matched
                contacts = compute_contacts(
                    structure=structure,
                    chain=chain,
                    ligand_residue=res,
                    pdb_to_sdf=pdb_to_sdf,
                    cutoff=4.5,
                )
                if contacts is None:
                    continue

                score = contacts["contact_n"]
                if best is None or score > best["contact_n"]:
                    best = {
                        **contacts,
                        "chain_id": chain_id,
                        "ligand_resname": res.get_resname().strip(),
                        "sdf_to_pdb_atom_map": sdf_to_pdb,
                    }

    return best


def get_cif_path(mmcif_dir, pdb_id):
    pdb_id = pdb_id.upper()
    candidates = [
        os.path.join(mmcif_dir, f"{pdb_id}.cif"),
        os.path.join(mmcif_dir, f"{pdb_id}.cif.gz"),
        os.path.join(mmcif_dir, f"{pdb_id.lower()}.cif"),
        os.path.join(mmcif_dir, f"{pdb_id.lower()}.cif.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def parse_structure_cached(parser, cache, mmcif_dir, pdb_id):
    if pdb_id in cache:
        return cache[pdb_id]

    path = get_cif_path(mmcif_dir, pdb_id)
    if path is None:
        cache[pdb_id] = None
        return None

    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                structure = parser.get_structure(pdb_id, f)
        else:
            structure = parser.get_structure(pdb_id, path)
        cache[pdb_id] = structure
        return structure
    except Exception as e:
        cache[pdb_id] = None
        return None


def encode_vqatom(smiles):
    """
    あなたの環境の関数名に合わせる。
    失敗したら空にして drop。
    """
    try:
        from vqatom_module import encode_smiles_to_atom_tokens
        toks = encode_smiles_to_atom_tokens(smiles)
        return " ".join(str(int(x)) for x in toks)
    except Exception:
        try:
            from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL
            if not _GLOBAL:
                init_tokenizer()
            out = infer_one(smiles)
            toks = out.get("tokens") or out.get("atom_tokens") or out.get("lig_tok")
            return " ".join(str(int(x)) for x in toks)
        except Exception:
            return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bindingdb_tsv", default="/Users/taka/Desktop/BindingDB_All.tsv")
    ap.add_argument("--sdf", default="/Users/taka/Documents/BindingDB_All_3D.sdf")
    ap.add_argument("--mmcif_dir", default="/Users/taka/Documents/mmcif")
    ap.add_argument("--out_csv", default="/Users/taka/Desktop/bindingdb_sdf_pdb_contacts_2000.csv")
    ap.add_argument("--target_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--min_contact_n", type=int, default=3)
    ap.add_argument("--max_scan_rows", type=int, default=500000)
    ap.add_argument("--shuffle_rows", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    by_monomer, by_smiles, by_inchikey = load_sdf_index(args.sdf)

    with open_text_maybe_gz(args.bindingdb_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("Could not read TSV header")

        col_smiles = guess_col(fieldnames, ["Ligand SMILES", "SMILES", "smiles"])
        col_seq = guess_col(fieldnames, ["BindingDB Target Chain Sequence", "Target Chain Sequence", "Sequence"])
        col_pdb = guess_col(fieldnames, ["PDB ID(s) for Ligand-Target Complex", "PDB ID", "PDB IDs"])
        col_monomer = guess_col(fieldnames, ["BindingDB MonomerID", "MonomerID"])
        aff_cols = [
            guess_col(fieldnames, ["Ki (nM)"]),
            guess_col(fieldnames, ["Kd (nM)"]),
            guess_col(fieldnames, ["IC50 (nM)"]),
            guess_col(fieldnames, ["EC50 (nM)"]),
        ]

        print("[cols]")
        print(" smiles :", col_smiles)
        print(" seq    :", col_seq)
        print(" pdb    :", col_pdb)
        print(" monomer:", col_monomer)
        print(" aff    :", aff_cols)

        rows = []
        for i, r in enumerate(reader):
            if i >= args.max_scan_rows:
                break
            rows.append(r)

    if args.shuffle_rows:
        random.shuffle(rows)

    parser = MMCIFParser(QUIET=True)
    struct_cache = {}

    out_fields = [
        "seq",
        "smiles",
        "lig_tok",
        "y",
        "aff_type",
        "aff_nm",
        "pdb_id",
        "chain_id",
        "ligand_resname",
        "contact_mask",
        "contact_n",
        "atom_contact_pairs",
        "sdf_to_pdb_atom_map",
    ]

    kept = 0
    stats = defaultdict(int)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        writer.writeheader()

        for r in tqdm(rows, desc="build dataset"):
            if kept >= args.target_n:
                break

            smiles = (r.get(col_smiles) or "").strip() if col_smiles else ""
            seq_tsv = (r.get(col_seq) or "").strip() if col_seq else ""
            pdb_ids = parse_pdb_ids(r.get(col_pdb, "")) if col_pdb else []
            monomer = (r.get(col_monomer) or "").strip() if col_monomer else ""

            if not smiles or not seq_tsv or not pdb_ids:
                stats["missing_basic"] += 1
                continue

            can = norm_smiles(smiles)
            if not can:
                stats["bad_smiles"] += 1
                continue

            sdf_mol = None
            if monomer and monomer in by_monomer:
                sdf_mol = by_monomer[monomer]
            elif can in by_smiles:
                sdf_mol = by_smiles[can]
            else:
                try:
                    m = Chem.MolFromSmiles(smiles)
                    ik = inchikey_from_mol(m)
                    sdf_mol = by_inchikey.get(ik)
                except Exception:
                    sdf_mol = None

            if sdf_mol is None:
                stats["no_sdf_match"] += 1
                continue

            aff_type, aff_nm = parse_affinity_nm(r, aff_cols)
            y = nm_to_pchem(aff_nm)
            if y is None:
                stats["no_affinity"] += 1
                continue

            lig_tok = encode_vqatom(can)
            if not lig_tok:
                stats["token_fail"] += 1
                continue

            best = None
            best_pdb = None

            for pdb_id in pdb_ids:
                structure = parse_structure_cached(parser, struct_cache, args.mmcif_dir, pdb_id)
                cif_path = get_cif_path(args.mmcif_dir, pdb_id)
                if cif_path is None:
                    continue

                res = get_resolution_from_cif(cif_path)
                if res is None or res > MIN_RESOLUTION:
                    stats["bad_resolution"] += 1
                    continue
                if structure is None:
                    stats["no_cif"] += 1
                    continue

                hit = find_best_ligand_match_in_structure(sdf_mol, structure)
                if hit is None:
                    stats["no_lig_match"] += 1
                    continue

                if hit["contact_n"] < args.min_contact_n:
                    stats["too_few_contacts"] += 1
                    continue

                if best is None or hit["contact_n"] > best["contact_n"]:
                    best = hit
                    best_pdb = pdb_id

            if best is None:
                continue
            # sequence consistency check（簡易）
            if seq_tsv not in best["seq"] and best["seq"] not in seq_tsv:
                stats["seq_mismatch"] += 1
                continue

            writer.writerow({
                "seq": best["seq"],
                "smiles": can,
                "lig_tok": lig_tok,
                "y": f"{y:.6f}",
                "aff_type": aff_type,
                "aff_nm": aff_nm,
                "pdb_id": best_pdb,
                "chain_id": best["chain_id"],
                "ligand_resname": best["ligand_resname"],
                "contact_mask": best["contact_mask"],
                "contact_n": best["contact_n"],
                "atom_contact_pairs": json.dumps(best["atom_contact_pairs"], ensure_ascii=False),
                "sdf_to_pdb_atom_map": json.dumps(best["sdf_to_pdb_atom_map"], ensure_ascii=False),
            })

            kept += 1
            stats["kept"] += 1

    print("\n[DONE]", args.out_csv)
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()