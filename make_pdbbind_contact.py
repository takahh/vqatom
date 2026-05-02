#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import shutil
import hashlib
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa


AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O",
}


def seq_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


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


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def collect_existing_sequences(train_csv, valid_csv, final_csv):
    seq_to_existing_split = {}
    split_to_seqs = defaultdict(set)

    for split, path in [
        ("train", train_csv),
        ("valid", valid_csv),
        ("final", final_csv),
    ]:
        rows = read_csv_rows(path)
        if not rows:
            continue
        col_seq = guess_col(rows[0].keys(), ["seq", "sequence", "protein_sequence"])
        if col_seq is None:
            raise RuntimeError(f"Cannot find seq column in {path}")

        for r in rows:
            seq = (r.get(col_seq) or "").strip()
            if not seq:
                continue
            split_to_seqs[split].add(seq)
            seq_to_existing_split[seq] = split

    return split_to_seqs, seq_to_existing_split


def load_mmseqs_cluster_tsv(cluster_tsv):
    """
    MMseqs cluster TSV is usually:
      representative_id<TAB>member_id

    This function only records member_id -> representative_id.
    If your previous FASTA IDs were raw sequences or sha1 IDs, this will work.
    If not, mmseqs easy-search below still performs the actual 40% overlap check.
    """
    member_to_cluster = {}
    if not cluster_tsv or not os.path.exists(cluster_tsv):
        print(f"[WARN] cluster_tsv not found: {cluster_tsv}")
        return member_to_cluster

    with open(cluster_tsv, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rep, mem = parts[0], parts[1]
            member_to_cluster[mem] = rep

    print(f"[cluster] loaded member_to_cluster: {len(member_to_cluster)}")
    return member_to_cluster


def infer_existing_cluster_id(seq, member_to_cluster):
    """
    Tries several common ID conventions:
      1. raw sequence as FASTA ID
      2. sha1 short ID
      3. no hit
    """
    if seq in member_to_cluster:
        return member_to_cluster[seq]
    h = seq_hash(seq)
    if h in member_to_cluster:
        return member_to_cluster[h]
    return ""


def write_fasta(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rid, seq in records:
            f.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def run_mmseqs_overlap(query_fasta, target_fasta, out_tsv, tmp_dir,
                       min_seq_id=0.40, cov=0.80, threads=8):
    """
    Uses easy-search, not reclustering.
    PDBBind query seqs are searched against existing train/valid/final seqs.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs not found in PATH")

    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "mmseqs", "easy-search",
        query_fasta,
        target_fasta,
        out_tsv,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(cov),
        "--cov-mode", "0",
        "--threads", str(threads),
        "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]

    print("[mmseqs]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_mmseqs_hits(path):
    """
    Returns query_id -> best hit.
    """
    best = {}
    if not os.path.exists(path):
        return best

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            q, t = parts[0], parts[1]
            try:
                pident = float(parts[2])
                bits = float(parts[7])
            except Exception:
                continue

            rec = {
                "target": t,
                "pident": pident,
                "bits": bits,
                "raw": parts,
            }
            if q not in best or bits > best[q]["bits"]:
                best[q] = rec

    return best


def mol_has_3d(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    xyz = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        xyz.append([p.x, p.y, p.z])
    xyz = np.asarray(xyz, dtype=float)
    return np.isfinite(xyz).all() and xyz.std() > 1e-3


def load_ligand_sdf(path):
    suppl = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
    mol = suppl[0] if len(suppl) > 0 else None
    if mol is None or not mol_has_3d(mol):
        return None
    return mol


def smiles_from_sdf_atom_order(mol):
    """
    Important:
    contact atom indices come from SDF atom order.
    canonical=False is used to reduce atom-order mismatch.
    """
    try:
        mol_noh = Chem.RemoveHs(mol, sanitize=False)
        return Chem.MolToSmiles(mol_noh, canonical=False)
    except Exception:
        return ""


def canonical_smiles_from_mol(mol):
    try:
        mol_noh = Chem.RemoveHs(mol, sanitize=False)
        return Chem.MolToSmiles(mol_noh, canonical=True)
    except Exception:
        return ""


def ligand_heavy_atom_count(mol):
    return sum(1 for a in mol.GetAtoms() if (a.GetSymbol() or "").upper() != "H")


def residue_seq(chain):
    seq = []
    res_ids = []
    for res in chain:
        if not is_aa(res, standard=False):
            continue
        aa = AA3_TO_1.get(res.get_resname().upper())
        if aa is None:
            continue
        seq.append(aa)
        res_ids.append(res.get_id())
    return "".join(seq), res_ids


def compute_contacts_from_complex(protein_pdb, ligand_mol, cutoff=4.5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", protein_pdb)

    lig_conf = ligand_mol.GetConformer()
    ligand_atoms = []
    ligand_atom_indices = []

    for i, atom in enumerate(ligand_mol.GetAtoms()):
        if (atom.GetSymbol() or "").upper() == "H":
            continue
        p = lig_conf.GetAtomPosition(i)
        ligand_atoms.append(np.array([p.x, p.y, p.z], dtype=float))
        ligand_atom_indices.append(i)

    best = None

    for model in structure:
        for chain in model:
            seq, res_ids = residue_seq(chain)
            if not seq:
                continue

            res_id_to_j = {rid: j for j, rid in enumerate(res_ids)}

            protein_atoms = []
            atom_to_res_j = {}

            for res in chain:
                if res.get_id() not in res_id_to_j:
                    continue
                j = res_id_to_j[res.get_id()]
                for atom in res.get_atoms():
                    if (atom.element or "").upper() == "H":
                        continue
                    protein_atoms.append(atom)
                    atom_to_res_j[atom] = j

            if not protein_atoms:
                continue

            ns = NeighborSearch(protein_atoms)

            contact_mask = [0] * len(seq)
            atom_contact_pairs = set()

            for sdf_i, xyz in zip(ligand_atom_indices, ligand_atoms):
                near_atoms = ns.search(xyz, cutoff, level="A")
                for pa in near_atoms:
                    j = atom_to_res_j.get(pa)
                    if j is None:
                        continue
                    contact_mask[j] = 1
                    atom_contact_pairs.add((int(sdf_i), int(j)))

            contact_n = int(sum(contact_mask))

            hit = {
                "seq": seq,
                "chain_id": chain.id,
                "contact_mask": "".join("1" if x else "0" for x in contact_mask),
                "contact_n": contact_n,
                "atom_contact_pairs": sorted(atom_contact_pairs),
            }

            if best is None or hit["contact_n"] > best["contact_n"]:
                best = hit

    return best


def affinity_to_pchem(s):
    s = s.strip()
    m = re.search(r"(Kd|Ki|IC50|EC50)\s*([<>=~]*)\s*([0-9.]+)\s*([munp]?M)", s, re.I)
    if not m:
        return None, None, None, None

    aff_type = m.group(1)
    op = m.group(2)
    value = float(m.group(3))
    unit = m.group(4).lower()

    factor_to_molar = {
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }

    molar = value * factor_to_molar[unit]
    pchem = -np.log10(molar)
    nm = molar / 1e-9

    return pchem, aff_type, nm, op


def read_pdbbind_index(index_path):
    aff = {}
    if not index_path or not os.path.exists(index_path):
        print(f"[WARN] index file not found: {index_path}")
        return aff

    n_data = 0
    n_fail = 0

    with open(index_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 4:
                continue

            pdb_id = parts[0].lower()
            raw_aff = parts[3]

            y, aff_type, aff_nm, aff_op = affinity_to_pchem(raw_aff)
            if y is None:
                n_fail += 1
                continue

            aff[pdb_id] = {
                "y": y,
                "aff_type": aff_type,
                "aff_nm": aff_nm,
                "aff_op": aff_op,
                "raw_affinity": raw_aff,
            }
            n_data += 1

    print(f"[index] parsed={n_data}, failed={n_fail}")
    return aff


def find_complex_dirs(root):
    items = []
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        base = os.path.basename(dirpath).lower()

        protein = None
        ligand = None

        candidates = [
            (f"{base}_protein.pdb", f"{base}_ligand.sdf"),
            (f"{base}_pocket.pdb", f"{base}_ligand.sdf"),
        ]

        for p, l in candidates:
            if p in files and l in files:
                protein = os.path.join(dirpath, p)
                ligand = os.path.join(dirpath, l)
                break

        if protein is None or ligand is None:
            for f in filenames:
                fl = f.lower()
                if fl.endswith("_protein.pdb"):
                    protein = os.path.join(dirpath, f)
                elif fl.endswith("_ligand.sdf"):
                    ligand = os.path.join(dirpath, f)

        if protein and ligand:
            items.append((base, protein, ligand))

    return sorted(items)


def encode_vqatom(smiles):
    try:
        from vqatom_module import encode_smiles_to_atom_tokens
        toks = encode_smiles_to_atom_tokens(smiles)
        return [int(x) for x in toks]
    except Exception:
        try:
            from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL
            if not _GLOBAL:
                init_tokenizer()
            out = infer_one(smiles)
            toks = out.get("tokens") or out.get("atom_tokens") or out.get("lig_tok")
            return [int(x) for x in toks]
        except Exception:
            return []


def build_pdbbind_contacts(args):
    aff = read_pdbbind_index(args.index_file)
    complexes = find_complex_dirs(args.pdbbind_root)

    print(f"[PDBBind] found complexes: {len(complexes)}")
    print(f"[PDBBind] affinity entries: {len(aff)}")

    rows = []
    stats = defaultdict(int)

    for pdb_id, protein_pdb, ligand_sdf in tqdm(complexes, desc="build contacts"):
        aff_rec = aff.get(pdb_id.lower())
        if args.index_file and aff_rec is None:
            stats["no_affinity"] += 1
            continue

        mol = load_ligand_sdf(ligand_sdf)
        if mol is None:
            stats["bad_ligand_sdf"] += 1
            continue

        smiles_atom_order = smiles_from_sdf_atom_order(mol)
        smiles_canonical = canonical_smiles_from_mol(mol)
        if not smiles_atom_order:
            stats["bad_smiles"] += 1
            continue

        hit = compute_contacts_from_complex(
            protein_pdb=protein_pdb,
            ligand_mol=mol,
            cutoff=args.contact_cutoff,
        )
        if hit is None:
            stats["no_contact_result"] += 1
            continue
        if hit["contact_n"] < args.min_contact_n:
            stats["too_few_contacts"] += 1
            continue

        lig_tok_list = encode_vqatom(smiles_atom_order)
        if not lig_tok_list:
            stats["token_fail"] += 1
            continue

        heavy_n = ligand_heavy_atom_count(mol)
        tok_n = len(lig_tok_list)
        atom_token_aligned = int(tok_n == heavy_n)
        if not atom_token_aligned and args.require_atom_token_aligned:
            stats["atom_token_mismatch"] += 1
            continue

        row = {
            "seq": hit["seq"],
            "smiles": smiles_atom_order,
            "smiles_canonical": smiles_canonical,
            "lig_tok": " ".join(map(str, lig_tok_list)),
            "lig_tok_list": json.dumps(lig_tok_list),
            "lig_heavy_atom_n": heavy_n,
            "lig_tok_n": tok_n,
            "atom_token_aligned": atom_token_aligned,
            "y": "" if aff_rec is None else f"{aff_rec['y']:.6f}",
            "aff_type": "" if aff_rec is None else aff_rec["aff_type"],
            "aff_nm": "" if aff_rec is None else aff_rec["aff_nm"],
            "aff_op": "" if aff_rec is None else aff_rec["aff_op"],
            "raw_affinity": "" if aff_rec is None else aff_rec["raw_affinity"],
            "pdb_id": pdb_id,
            "chain_id": hit["chain_id"],
            "contact_mask": hit["contact_mask"],
            "contact_n": hit["contact_n"],
            "atom_contact_pairs": json.dumps(hit["atom_contact_pairs"]),
            "protein_pdb": protein_pdb,
            "ligand_sdf": ligand_sdf,
            "pdbbind_seq_id": "PB_" + seq_hash(hit["seq"]),
        }
        rows.append(row)
        stats["kept"] += 1

    print("[contact stats]")
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))

    return rows, stats


def write_contacts_csv(rows, path):
    fields = [
        "seq", "smiles", "smiles_canonical", "lig_tok", "lig_tok_list",
        "lig_heavy_atom_n", "lig_tok_n", "atom_token_aligned",
        "y", "aff_type", "aff_nm", "aff_op", "raw_affinity",
        "pdb_id", "chain_id", "contact_mask", "contact_n", "atom_contact_pairs",
        "protein_pdb", "ligand_sdf", "pdbbind_seq_id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def build_mmseqs_overlap(args, pdbbind_rows):
    os.makedirs(args.mmseqs_tmp, exist_ok=True)

    split_to_seqs, _ = collect_existing_sequences(
        args.train_csv,
        args.valid_csv,
        args.final_csv,
    )
    member_to_cluster = load_mmseqs_cluster_tsv(args.cluster_tsv)

    query_records = []
    seen_q = set()
    qid_to_seq = {}
    for r in pdbbind_rows:
        qid = r["pdbbind_seq_id"]
        if qid in seen_q:
            continue
        seen_q.add(qid)
        qid_to_seq[qid] = r["seq"]
        query_records.append((qid, r["seq"]))

    query_fa = os.path.join(args.mmseqs_tmp, "pdbbind_query.fasta")
    write_fasta(query_records, query_fa)

    overlap = {qid: {} for qid, _ in query_records}
    target_id_to_info = {}

    for split in ["train", "valid", "final"]:
        target_records = []
        for seq in sorted(split_to_seqs[split]):
            tid = f"{split.upper()}_{seq_hash(seq)}"
            target_records.append((tid, seq))
            target_id_to_info[tid] = {
                "split": split,
                "seq": seq,
                "cluster_id": infer_existing_cluster_id(seq, member_to_cluster),
            }

        target_fa = os.path.join(args.mmseqs_tmp, f"{split}_target.fasta")
        out_tsv = os.path.join(args.mmseqs_tmp, f"pdbbind_vs_{split}.tsv")
        tmp = os.path.join(args.mmseqs_tmp, f"tmp_{split}")

        write_fasta(target_records, target_fa)

        if target_records:
            run_mmseqs_overlap(
                query_fasta=query_fa,
                target_fasta=target_fa,
                out_tsv=out_tsv,
                tmp_dir=tmp,
                min_seq_id=args.min_seq_id,
                cov=args.mmseqs_cov,
                threads=args.threads,
            )
            best_hits = parse_mmseqs_hits(out_tsv)
        else:
            best_hits = {}

        for qid, hit in best_hits.items():
            tinfo = target_id_to_info.get(hit["target"], {})
            overlap[qid][split] = {
                "hit": 1,
                "target_id": hit["target"],
                "pident": hit["pident"],
                "bits": hit["bits"],
                "target_cluster_id": tinfo.get("cluster_id", ""),
            }

    return overlap


def assign_guide_split(hit_train, hit_valid, hit_final):
    """
    guide_train:
      can be used as structure guide for train-side learning,
      but must not overlap with existing valid/final.

    guide_test:
      can be used as a held-out structure guide test,
      but must not overlap with existing train.

    If a protein overlaps neither train/valid/final, put it into guide_train by default.
    """
    if not hit_valid and not hit_final:
        return "guide_train"
    if not hit_train:
        return "guide_test"
    return "drop_overlap"


def write_guide_csv(args, pdbbind_rows, overlap):
    fields = [
        "split",
        "seq", "smiles", "smiles_canonical", "lig_tok", "lig_tok_list",
        "y", "aff_type", "aff_nm", "aff_op", "raw_affinity",
        "pdb_id", "chain_id",
        "contact_mask", "contact_n", "atom_contact_pairs",
        "protein_pdb", "ligand_sdf",
        "pdbbind_seq_id",
        "hit_train40", "hit_valid40", "hit_final40",
        "best_train_pident", "best_valid_pident", "best_final_pident",
        "best_train_cluster", "best_valid_cluster", "best_final_cluster",
        "lig_heavy_atom_n", "lig_tok_n", "atom_token_aligned",
    ]

    stats = defaultdict(int)

    with open(args.out_guide_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in pdbbind_rows:
            qid = r["pdbbind_seq_id"]
            ov = overlap.get(qid, {})

            ht = bool(ov.get("train", {}).get("hit", 0))
            hv = bool(ov.get("valid", {}).get("hit", 0))
            hf = bool(ov.get("final", {}).get("hit", 0))

            split = assign_guide_split(ht, hv, hf)
            stats[split] += 1

            if split == "drop_overlap":
                if not args.keep_dropped:
                    continue

            out = dict(r)
            out.update({
                "split": split,
                "hit_train40": int(ht),
                "hit_valid40": int(hv),
                "hit_final40": int(hf),
                "best_train_pident": ov.get("train", {}).get("pident", ""),
                "best_valid_pident": ov.get("valid", {}).get("pident", ""),
                "best_final_pident": ov.get("final", {}).get("pident", ""),
                "best_train_cluster": ov.get("train", {}).get("target_cluster_id", ""),
                "best_valid_cluster": ov.get("valid", {}).get("target_cluster_id", ""),
                "best_final_cluster": ov.get("final", {}).get("target_cluster_id", ""),
            })

            w.writerow({k: out.get(k, "") for k in fields})

    print("[guide stats]")
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pdbbind_root", default="/Users/taka/Documents/P-L")
    ap.add_argument("--index_file", default="/Users/taka/Documents/pdbbind_2020_index/INDEX_general_PL.2020R1.lst")

    ap.add_argument("--train_csv", default="/Users/taka/Downloads/train.csv")
    ap.add_argument("--valid_csv", default="/Users/taka/Downloads/valid.csv")
    ap.add_argument("--final_csv", default="/Users/taka/Downloads/test.csv")
    ap.add_argument("--cluster_tsv", default="/Users/taka/Desktop/kiba_fast_out/mmseqs40/seq40_cluster.tsv")

    ap.add_argument("--out_contacts_csv", default="/Users/taka/Documents/pdbbind_contacts.csv")
    ap.add_argument("--out_guide_csv", default="/Users/taka/Documents/pdbbind_guide_dti.csv")

    ap.add_argument("--contact_cutoff", type=float, default=4.5)
    ap.add_argument("--min_contact_n", type=int, default=3)
    ap.add_argument("--require_atom_token_aligned", action="store_true")

    ap.add_argument("--mmseqs_tmp", default="/Users/taka/Documents/pdbbind_mmseqs_tmp")
    ap.add_argument("--min_seq_id", type=float, default=0.40)
    ap.add_argument("--mmseqs_cov", type=float, default=0.80)
    ap.add_argument("--threads", type=int, default=8)

    ap.add_argument("--keep_dropped", action="store_true")

    args = ap.parse_args()

    pdbbind_rows, _ = build_pdbbind_contacts(args)
    write_contacts_csv(pdbbind_rows, args.out_contacts_csv)
    print(f"[DONE contacts] {args.out_contacts_csv}")

    overlap = build_mmseqs_overlap(args, pdbbind_rows)
    write_guide_csv(args, pdbbind_rows, overlap)
    print(f"[DONE guide] {args.out_guide_csv}")


if __name__ == "__main__":
    main()