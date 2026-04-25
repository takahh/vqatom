#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import csv
import math
import sqlite3
import gzip
import re
import subprocess
import shutil
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Tuple, List
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch

# adjust to your environment
from vqatom_module import encode_smiles_to_atom_tokens

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.debug")


# ============================================================
# PATHS
# ============================================================
GRAPHDTA_DIR = Path("/Users/taka/Documents/GraphDTA/data/kiba")
GRAPHDTA_LIGANDS = GRAPHDTA_DIR / "ligands_can.txt"
GRAPHDTA_PROTEINS = GRAPHDTA_DIR / "proteins.txt"
GRAPHDTA_Y = GRAPHDTA_DIR / "Y"

CORE_FINAL_EVAL_CSV = Path(
    "/Users/taka/Desktop/final_eval.csv"
)

OUT_DIR = Path(
    "/Users/taka/Desktop/bindingdb_fast_out"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MMCIF_DIR = Path("/Users/taka/Documents/mmcif")

PDB_CACHE_DB = OUT_DIR / "pdb_chain_cache.sqlite"
DEDUP_DB = OUT_DIR / "dedup.sqlite"

PASS1_CSV = OUT_DIR / "pass1_rows.csv"
UNIQUE_SMILES_CSV = OUT_DIR / "unique_smiles.csv"
SMILES_TOK_CSV = OUT_DIR / "smiles_tok_map.csv"
FINAL_CSV = OUT_DIR / "bindingdb_dedup_tok.csv"
ERROR_TSV = OUT_DIR / "errors.tsv"
PDBID_LIST_TXT = OUT_DIR / "bindingdb_unique_pdbids.txt"
PDB_CACHE_FAIL_TSV = OUT_DIR / "pdb_cache_failures.tsv"
SEQ_COL_CANDIDATES = [
    "BindingDB Target Chain Sequence",
    "Target Chain Sequence",
    "Target Sequence",
]
# split outputs
SPLIT_FASTA = OUT_DIR / "protein_unique_seqs.fasta"
MMSEQS_DIR = OUT_DIR / "mmseqs40"
MMSEQS_TMP = OUT_DIR / "mmseqs_tmp"
SEQ_TO_CLUSTER_CSV = OUT_DIR / "seq_to_cluster_40id.csv"
ROW_ANNOT_CSV = OUT_DIR / "bindingdb_dedup_tok_annotated.csv"
COMPONENT_SUMMARY_CSV = OUT_DIR / "split_components_summary.csv"
TRAIN_CSV = OUT_DIR / "train.csv"
VALID_CSV = OUT_DIR / "valid.csv"
TEST_CSV = OUT_DIR / "test.csv"


# ============================================================
# SETTINGS
# ============================================================
# balanced split settings
Y_THR = 12.1
balance_iterations = 0

LIG_MIN_PAIRS = 2
PROT_MIN_PAIRS = 6

POS_RATE_LOW = 0
POS_RATE_HIGH = 1

BALANCE_MAX_ITERS = 5

DROP_ROWS_WITH_MISSING_Y = True

MIN_SEQ_LEN = 50
MAX_SEQ_LEN = 1022
AFFINITY_PRIORITY = ["Ki (nM)", "Kd (nM)", "IC50 (nM)"]

CANON_CACHE_SIZE = 100_000
AA_SET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

PDB_COL = "PDB ID(s) for Ligand-Target Complex"
CONTACT_CUTOFF = 4.5
SECOND_RATIO_MAX = 0.5   # discard if second > 50% of first

PDB_PARSE_NPROC = max(1, min(12, cpu_count() - 1))
TOKENIZE_NPROC = max(1, min(12, cpu_count() - 1))
TOKENIZE_CHUNKSIZE = 100

PASS1_BATCH_SIZE = 2000
SQLITE_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA cache_size=-200000;",
]
USE_CORE_FILTER = False

# if True, exclude any seq appearing in core, even with different smiles
EXCLUDE_ANY_CORE_SEQ = False

# split settings
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
SPLIT_SEED = 123
MMSEQS_MIN_SEQ_ID = 0.40
MMSEQS_COV_MODE = 0
MMSEQS_C = 0.8
MMSEQS_CLUSTER_MODE = 2  # connected component style
MMSEQS_THREADS = max(1, min(24, cpu_count()))
USE_GENERIC_MURCKO = False

# graph-pruning settings for breaking giant connected components
MAX_SCAFFOLD_DEGREE = 8
MAX_PROTEIN_CLUSTER_DEGREE = 20
MIN_COMPONENT_ROWS = 20

# ============================================================
# RDKit / SMILES utils
# ============================================================
def load_graphdta_y(path: Path):
    import numpy as np
    import pickle

    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f, encoding="latin1")


def pass1_from_graphdta_processed():
    import json
    import numpy as np

    with GRAPHDTA_LIGANDS.open("r") as f:
        lig_dict = json.load(f)

    with GRAPHDTA_PROTEINS.open("r") as f:
        prot_dict = json.load(f)

    Y = np.asarray(load_graphdta_y(GRAPHDTA_Y), dtype=float)

    lig_items = list(lig_dict.items())
    prot_items = list(prot_dict.items())

    print("[graphdta] Y shape:", Y.shape)
    print("[graphdta] Y min/max/mean:", np.nanmin(Y), np.nanmax(Y), np.nanmean(Y))
    print("[graphdta] ligands:", len(lig_items))
    print("[graphdta] proteins:", len(prot_items))

    assert Y.shape[0] == len(lig_items)
    assert Y.shape[1] == len(prot_items)

    rows = []
    bad_smiles = 0
    bad_seq = 0
    missing_y = 0

    for i, (lig_id, smi_raw) in enumerate(lig_items):
        smi = canonicalize_smiles_cached(smi_raw)
        if not smi:
            bad_smiles += 1
            continue

        for j, (prot_id, seq_raw) in enumerate(prot_items):
            y = Y[i, j]

            if not np.isfinite(y):
                missing_y += 1
                continue

            seq = clean_protein_seq(seq_raw)
            if not seq:
                bad_seq += 1
                continue

            rows.append({
                "seq": seq,
                "smiles": smi,
                "aff_type": "KIBA",
                "aff_nm": 0.0,
                "y": float(y),
                "src_pdbid": "",
                "src_chain": "",
                "contact_mask": "",
                "contact_n": 0,
            })

    with PASS1_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "seq", "smiles", "aff_type", "aff_nm", "y",
            "src_pdbid", "src_chain", "contact_mask", "contact_n"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with UNIQUE_SMILES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for smi in sorted({r["smiles"] for r in rows}):
            w.writerow([smi])

    print(f"[graphdta] rows={len(rows):,}")
    print(f"[graphdta] bad_smiles={bad_smiles:,} bad_seq={bad_seq:,} missing_y={missing_y:,}")

def try_sanitize_rescue(mol):
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return mol
    except Exception:
        pass
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(mol)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return mol
    except Exception:
        pass
    return None


@lru_cache(maxsize=CANON_CACHE_SIZE)
def canonicalize_smiles_cached(smiles: str) -> Optional[str]:
    smiles = (smiles or "").strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            pass

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = try_sanitize_rescue(mol)
    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


@lru_cache(maxsize=CANON_CACHE_SIZE)
def murcko_scaffold_smiles_cached(smiles: str) -> Optional[str]:
    smi = canonicalize_smiles_cached(smiles)
    if not smi:
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None

    if scaf is None or scaf.GetNumAtoms() == 0:
        return None

    if USE_GENERIC_MURCKO:
        try:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        except Exception:
            return None

    try:
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# ============================================================
# Generic parsing helpers
# ============================================================

def clean_numeric_cell(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    while s and s[0] in ["<", ">", "~", "="]:
        s = s[1:].strip()
    s = s.replace(",", "")
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


def pick_affinity_nm(row: Dict[str, str]) -> Tuple[Optional[str], Optional[float]]:
    for col in AFFINITY_PRIORITY:
        v = clean_numeric_cell(row.get(col, ""))
        if v is not None:
            if col == "Ki (nM)":
                return "Ki", v
            if col == "Kd (nM)":
                return "Kd", v
            if col == "IC50 (nM)":
                return "IC50", v
    return None, None


def affinity_to_pvalue_from_nm(x_nm: float) -> Optional[float]:
    x_m = float(x_nm) * 1e-9
    if not math.isfinite(x_m) or x_m <= 0.0:
        return None
    return -math.log10(x_m)


def clean_protein_seq(seq: str) -> Optional[str]:
    seq = (seq or "").strip().upper()
    if not seq:
        return None
    seq = "".join(seq.split())
    if not seq:
        return None
    if any((not c.isalpha()) for c in seq):
        return None
    bad_frac = sum(c not in AA_SET for c in seq) / max(1, len(seq))
    if bad_frac > 0.1:
        return None
    if len(seq) < MIN_SEQ_LEN or len(seq) > MAX_SEQ_LEN:
        return None
    return seq

def get_bindingdb_target_seq(row: Dict[str, str]) -> Optional[str]:
    for col in SEQ_COL_CANDIDATES:
        seq_raw = row.get(col, "")
        seq = clean_protein_seq(seq_raw)
        if seq:
            return seq
    return None

def parse_pdb_ids(cell: str) -> List[str]:
    toks = re.split(r"[,\s;]+", (cell or "").strip())
    out = []
    seen = set()
    for t in toks:
        t = t.strip().upper()
        if re.fullmatch(r"[A-Z0-9]{4}", t) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def apply_sqlite_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    for q in SQLITE_PRAGMAS:
        cur.execute(q)
    con.commit()


# ============================================================
# mmCIF helpers
# ============================================================

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
    "ASX": "B", "GLX": "Z", "XLE": "J",
}


def residue_to_one_letter(resname: str) -> str:
    return AA3_TO_1.get((resname or "").upper(), "X")


def is_protein_residue(residue) -> bool:
    hetflag = residue.id[0]
    if hetflag != " ":
        return False
    resname = (residue.get_resname() or "").upper()
    return resname in AA3_TO_1


def residue_sequence_from_chain(chain) -> Optional[str]:
    seq_chars = []
    seen = set()
    for residue in chain.get_residues():
        if not is_protein_residue(residue):
            continue
        rid = residue.id
        if rid in seen:
            continue
        seen.add(rid)
        seq_chars.append(residue_to_one_letter(residue.get_resname()))
    seq = "".join(seq_chars)
    return clean_protein_seq(seq)


def get_mmcif_path(pdbid: str) -> Path:
    return MMCIF_DIR / f"{pdbid.upper()}.cif.gz"


def open_mmcif_structure(pdbid: str):
    path = get_mmcif_path(pdbid)
    if not path.exists():
        raise FileNotFoundError(f"mmcif not found: {path}")
    parser = MMCIFParser(QUIET=True)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return parser.get_structure(pdbid, fh)


def get_ligand_atoms(structure):
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag == " ":
                    continue
                resname = (residue.get_resname() or "").upper()
                if resname in {"HOH", "WAT", "DOD"}:
                    continue
                for atom in residue.get_atoms():
                    if atom.element != "H":
                        ligand_atoms.append(atom)
        break  # first model only
    return ligand_atoms


def build_contact_mask_for_chain(chain, ns, ligand_atoms, cutoff: float) -> Tuple[str, int]:
    """
    chain の protein residue 順に 0/1 mask を作る。
    seq を作る residue 順と同じ順序になるようにする。
    returns:
      contact_mask: str  e.g. "0010110..."
      contact_n: int
    """
    residues = []
    seen = set()

    for residue in chain.get_residues():
        if not is_protein_residue(residue):
            continue
        rid = residue.id
        if rid in seen:
            continue
        seen.add(rid)
        residues.append(residue)

    if not residues:
        return "", 0

    contact_res_ids = set()

    for latom in ligand_atoms:
        neighbors = ns.search(latom.coord, cutoff, level="A")
        for nb in neighbors:
            parent_res = nb.get_parent()
            parent_chain = parent_res.get_parent()

            if parent_chain.id != chain.id:
                continue
            if not is_protein_residue(parent_res):
                continue

            contact_res_ids.add(parent_res.id)

    bits = ["1" if r.id in contact_res_ids else "0" for r in residues]
    contact_mask = "".join(bits)
    contact_n = sum(1 for b in bits if b == "1")
    return contact_mask, contact_n


def choose_best_chain_from_pdb_impl(pdbid: str):
    """
    returns:
      (best_chain_id, best_seq, status, top_contacts, second_contacts, contact_mask, contact_n)
    """
    try:
        structure = open_mmcif_structure(pdbid)
    except FileNotFoundError:
        return "", "", "no_file", 0, 0, "", 0
    except Exception as e:
        return "", "", f"parse_error:{type(e).__name__}:{e}", 0, 0, "", 0

    try:
        model = next(structure.get_models())
    except StopIteration:
        return "", "", "no_model", 0, 0, "", 0

    ligand_atoms = get_ligand_atoms(structure)

    protein_chain_rows = []
    for chain in model:
        seq = residue_sequence_from_chain(chain)
        if not seq:
            continue

        protein_atoms = []
        for residue in chain:
            if not is_protein_residue(residue):
                continue
            for atom in residue.get_atoms():
                if atom.element != "H":
                    protein_atoms.append(atom)

        if not protein_atoms:
            continue

        protein_chain_rows.append({
            "chain": chain,
            "chain_id": chain.id,
            "seq": seq,
            "protein_atoms": protein_atoms,
        })

    if not protein_chain_rows:
        return "", "", "no_protein_chains", 0, 0, "", 0

    # single-chain case
    if len(protein_chain_rows) == 1:
        row = protein_chain_rows[0]
        best_seq = clean_protein_seq(row["seq"])
        if not best_seq:
            return "", "", "bad_top_seq", 0, 0, "", 0

        # ligand があるなら mask 作成、なければ空
        if ligand_atoms:
            all_atoms = [a for a in structure.get_atoms() if a.element != "H"]
            if all_atoms:
                ns = NeighborSearch(all_atoms)
                contact_mask, contact_n = build_contact_mask_for_chain(
                    row["chain"], ns, ligand_atoms, CONTACT_CUTOFF
                )
            else:
                contact_mask, contact_n = "", 0
        else:
            contact_mask, contact_n = "", 0

        return row["chain_id"], best_seq, "ok_single_chain", 0, 0, contact_mask, contact_n

    # multi-chain case
    if not ligand_atoms:
        return "", "", "no_ligand_atoms", 0, 0, "", 0

    all_atoms = [a for a in structure.get_atoms() if a.element != "H"]
    if not all_atoms:
        return "", "", "no_atoms", 0, 0, "", 0

    ns = NeighborSearch(all_atoms)

    chain_rows = []
    for row in protein_chain_rows:
        protein_atom_ids = {id(a) for a in row["protein_atoms"]}
        contact_count = 0

        for latom in ligand_atoms:
            neighbors = ns.search(latom.coord, CONTACT_CUTOFF, level="A")
            hit = False
            for nb in neighbors:
                if id(nb) in protein_atom_ids:
                    hit = True
                    break
            if hit:
                contact_count += 1

        contact_mask, contact_n = build_contact_mask_for_chain(
            row["chain"], ns, ligand_atoms, CONTACT_CUTOFF
        )

        chain_rows.append((
            row["chain_id"],
            row["seq"],
            contact_count,
            row["chain"],
            contact_mask,
            contact_n,
        ))

    chain_rows.sort(key=lambda x: (-x[2], x[0]))
    best_chain, best_seq, best_contacts, best_chain_obj, best_contact_mask, best_contact_n = chain_rows[0]
    second_contacts = chain_rows[1][2] if len(chain_rows) >= 2 else 0

    best_seq = clean_protein_seq(best_seq)
    if not best_seq:
        return "", "", "bad_top_seq", best_contacts, second_contacts, "", 0

    # 念のため長さ整合性チェック
    if best_contact_mask and len(best_contact_mask) != len(best_seq):
        best_contact_mask = ""
        best_contact_n = 0

    return best_chain, best_seq, "ok", best_contacts, second_contacts, best_contact_mask, best_contact_n

# ============================================================
# PDB cache stage
# ============================================================

def collect_unique_bindingdb_pdbids(tsv_path: Path, pdb_col: str) -> List[str]:
    ids = set()
    with tsv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="scan pdb ids"):
            for pdbid in parse_pdb_ids(row.get(pdb_col, "")):
                ids.add(pdbid)
    out = sorted(ids)
    PDBID_LIST_TXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[pdbids] unique={len(out):,}")
    print(f"[pdbids] wrote: {PDBID_LIST_TXT}")
    return out


def init_pdb_cache_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    apply_sqlite_pragmas(con)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE pdb_chain_cache (
            pdbid TEXT PRIMARY KEY,
            chain_id TEXT,
            seq TEXT,
            status TEXT NOT NULL,
            top_contacts INTEGER NOT NULL,
            second_contacts INTEGER NOT NULL,
            contact_mask TEXT,
            contact_n INTEGER NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_pdb_cache_status ON pdb_chain_cache(status)")
    con.commit()
    return con

def _worker_choose_best_chain(pdbid: str):
    try:
        chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n = choose_best_chain_from_pdb_impl(pdbid)
        return pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n
    except Exception as e:
        return pdbid, "", "", f"worker_error:{type(e).__name__}:{e}", 0, 0, "", 0

def build_pdb_chain_cache(unique_pdbids: List[str]) -> None:
    print(f"[pdb-cache] building for {len(unique_pdbids):,} unique pdb ids")
    con = init_pdb_cache_db(PDB_CACHE_DB)
    cur = con.cursor()

    failures = []
    done = 0

    with Pool(processes=PDB_PARSE_NPROC) as pool:
        for rec in tqdm(
            pool.imap_unordered(_worker_choose_best_chain, unique_pdbids, chunksize=1),
            total=len(unique_pdbids),
            desc=f"build pdb cache ({PDB_PARSE_NPROC} proc)"
        ):
            pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n = rec

            cur.execute("""
                INSERT INTO pdb_chain_cache(
                    pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n))

            if not status.startswith("ok"):
                failures.append((pdbid, status, top_contacts, second_contacts))

            done += 1
            if done % 1000 == 0:
                con.commit()
                print(f"[pdb-cache] committed {done:,}/{len(unique_pdbids):,}")

    con.commit()
    con.close()

def load_pdb_cache_map(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    out = {}
    for row in cur.execute("""
        SELECT pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n
        FROM pdb_chain_cache
    """):
        pdbid, chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n = row
        out[pdbid] = (
            chain_id or "",
            seq or "",
            status,
            int(top_contacts),
            int(second_contacts),
            contact_mask or "",
            int(contact_n or 0),
        )
    con.close()
    return out

def choose_best_chain_across_pdbids_cached(
    pdbids: List[str],
    pdb_cache_map: Dict[str, Tuple[str, str, str, int, int]],
):
    """
    Priority:
      1) ok_single_chain  -> accept directly
      2) ok               -> choose highest-contact multi-chain candidate
      3) otherwise fail
    returns:
      (best_pdbid, best_chain, best_seq, debug_status)
    """
    debug = []

    # first: any single-chain success
    for pdbid in pdbids:
        row = pdb_cache_map.get(pdbid)
        if row is None:
            debug.append(f"{pdbid}:missing_cache:0:0")
            continue

        chain_id, seq, status, top_contacts, second_contacts = row
        debug.append(f"{pdbid}:{status}:{top_contacts}:{second_contacts}")

        if status == "ok_single_chain":
            return pdbid, chain_id, seq, "ok_single_chain"

    # second: best multi-chain success
    best_pdbid = None
    best_chain = None
    best_seq = None
    best_contacts = -1
    best_second = 0

    for pdbid in pdbids:
        row = pdb_cache_map.get(pdbid)
        if row is None:
            continue

        chain_id, seq, status, top_contacts, second_contacts = row
        if status != "ok":
            continue

        if top_contacts > best_contacts:
            best_pdbid = pdbid
            best_chain = chain_id
            best_seq = seq
            best_contacts = top_contacts
            best_second = second_contacts

    if best_pdbid is not None:
        return best_pdbid, best_chain, best_seq, f"ok:{best_contacts}:{best_second}"

    return None, None, None, "all_failed|" + "|".join(debug)

def choose_best_chain_across_pdbids_cached_with_mode(
    pdbids: List[str],
    pdb_cache_map,
):
    """
    returns:
      (best_pdbid, best_chain, best_seq, debug_status, mode, contact_mask, contact_n)
    """
    debug = []

    for pdbid in pdbids:
        row = pdb_cache_map.get(pdbid)
        if row is None:
            debug.append(f"{pdbid}:missing_cache:0:0")
            continue

        chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n = row
        debug.append(f"{pdbid}:{status}:{top_contacts}:{second_contacts}")

        if status == "ok_single_chain":
            return pdbid, chain_id, seq, "ok_single_chain", "single", contact_mask, contact_n

    best_pdbid = None
    best_chain = None
    best_seq = None
    best_contacts = -1
    best_second = 0
    best_contact_mask = ""
    best_contact_n = 0

    for pdbid in pdbids:
        row = pdb_cache_map.get(pdbid)
        if row is None:
            continue

        chain_id, seq, status, top_contacts, second_contacts, contact_mask, contact_n = row
        if status != "ok":
            continue

        if top_contacts > best_contacts:
            best_pdbid = pdbid
            best_chain = chain_id
            best_seq = seq
            best_contacts = top_contacts
            best_second = second_contacts
            best_contact_mask = contact_mask
            best_contact_n = contact_n

    if best_pdbid is not None:
        return best_pdbid, best_chain, best_seq, f"ok:{best_contacts}:{best_second}", "multi", best_contact_mask, best_contact_n

    return None, None, None, "all_failed|" + "|".join(debug), "none", "", 0

def choose_best_chain_across_pdbids_no_cache(pdbids: List[str]):
    best_pdbid = None
    best_chain = None
    best_seq = None
    best_contacts = -1
    best_second = 0

    debug = []

    for pdbid in pdbids:
        chain_id, seq, status, top_contacts, second_contacts = choose_best_chain_from_pdb_impl(pdbid)
        debug.append(f"{pdbid}:{status}:{top_contacts}:{second_contacts}")

        if not status.startswith("ok"):
            continue

        if top_contacts > best_contacts:
            best_pdbid = pdbid
            best_chain = chain_id
            best_seq = seq
            best_contacts = top_contacts
            best_second = second_contacts

    if best_pdbid is None:
        return None, None, None, "all_failed|" + "|".join(debug)

    return best_pdbid, best_chain, best_seq, f"ok:{best_contacts}:{best_second}"

# ============================================================
# Core overlap
# ============================================================

def load_core_exact_pairs(core_csv: Path):
    core_pairs = set()
    core_seqs = set()

    if not USE_CORE_FILTER:
        return core_pairs, core_seqs

    if not core_csv.exists():
        return core_pairs, core_seqs

    with core_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seq = (row.get("seq") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            if seq and smiles:
                core_pairs.add((seq, smiles))
                core_seqs.add(seq)

    return core_pairs, core_seqs

# ============================================================
# Pass 1: clean + dedup by (seq, smiles) in SQLite
# ============================================================

def init_dedup_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    apply_sqlite_pragmas(con)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seq TEXT NOT NULL,
            smiles TEXT NOT NULL,
            aff_type TEXT NOT NULL,
            aff_nm REAL NOT NULL,
            y REAL NOT NULL,
            src_pdbid TEXT,
            src_chain TEXT,
            contact_mask TEXT,
            contact_n INTEGER NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_pairs_smiles ON pairs(smiles)")
    cur.execute("CREATE INDEX idx_pairs_seq ON pairs(seq)")
    con.commit()
    return con

def flush_pair_batch(cur: sqlite3.Cursor, con: sqlite3.Connection, batch: List[Tuple]):
    if not batch:
        return
    cur.executemany("""
        INSERT INTO pairs(
            seq, smiles, aff_type, aff_nm, y, src_pdbid, src_chain, contact_mask, contact_n
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)
    con.commit()
    batch.clear()

def analyze_pdb_chain_types():
    import sqlite3
    from collections import Counter

    con = sqlite3.connect(str(PDB_CACHE_DB))
    cur = con.cursor()

    counter = Counter()
    total = 0

    for (status,) in cur.execute("SELECT status FROM pdb_chain_cache"):
        total += 1

        if status.startswith("ok_single_chain"):
            counter["single_chain"] += 1
        elif status.startswith("ok"):
            counter["multi_chain"] += 1
        else:
            counter["failed"] += 1

    con.close()

    print("\n=== PDB chain type stats ===")
    print(f"total pdbids: {total:,}")

    for k, v in counter.items():
        frac = v / max(1, total)
        print(f"{k:15s}: {v:,} ({frac:.2%})")

def pass1_clean_and_dedup():
    core_pairs, core_seqs = load_core_exact_pairs(CORE_FINAL_EVAL_CSV)
    from collections import Counter

    drop_counter = Counter()
    keep_counter = Counter()

    # build/load pdb chain cache once
    if not PDB_CACHE_DB.exists():
        unique_pdbids = collect_unique_bindingdb_pdbids(BINDINGDB_TSV, PDB_COL)
        build_pdb_chain_cache(unique_pdbids)
    else:
        print(f"[pdb-cache] reuse existing: {PDB_CACHE_DB}")

    pdb_cache_map = load_pdb_cache_map(PDB_CACHE_DB)
    con = init_dedup_db(DEDUP_DB)
    cur = con.cursor()

    seen = 0
    kept = 0
    batch = []
    from collections import Counter

    lig_pair_counter_before = Counter()
    prot_pair_counter_before = Counter()
    pair_counter_before = Counter()

    debug_rows_before = []
    with ERROR_TSV.open("w", encoding="utf-8", newline="") as err_f:
        err_w = csv.writer(err_f, delimiter="\t")
        err_w.writerow(["row_id", "reason", "extra"])

        with BINDINGDB_TSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            pre_pdb_lig_counter = Counter()
            pre_pdb_prot_counter = Counter()
            pre_pdb_pair_counter = Counter()

            for row in tqdm(reader, desc="pass1 clean"):
                seen += 1
                row_id = row.get("BindingDB Reactant_set_id", str(seen))

                # --------------------------------------------------
                # 1) base sequence from BindingDB
                # --------------------------------------------------
                bindingdb_seq = get_bindingdb_target_seq(row)

                raw_smiles = row.get("Ligand SMILES", "")
                can_smiles = canonicalize_smiles_cached(raw_smiles)

                pair_key = (bindingdb_seq, can_smiles)

                pre_pdb_pair_counter[pair_key] += 1
                if pre_pdb_pair_counter[pair_key] == 1:
                    pre_pdb_lig_counter[can_smiles] += 1
                    pre_pdb_prot_counter[bindingdb_seq] += 1
                # --------------------------------------------------
                # 2) optional PDB-based disambiguation
                # --------------------------------------------------
                pdb_ids = parse_pdb_ids(row.get(PDB_COL, ""))
                best_pdbid, best_chain, best_pdb_seq, best_status, pdb_mode, contact_mask, contact_n = \
                    choose_best_chain_across_pdbids_cached_with_mode(pdb_ids, pdb_cache_map)

                # policy:
                # - no PDB usable candidate      -> use BindingDB seq
                # - single-chain PDB candidate   -> still prefer BindingDB seq
                # - multi-chain PDB candidate    -> use best contact chain seq
                seq = None
                src_pdbid = ""
                src_chain = ""
                row_contact_mask = ""
                row_contact_n = 0
                if pdb_mode == "multi" and best_pdb_seq:
                    seq = best_pdb_seq
                    src_pdbid = best_pdbid or ""
                    src_chain = best_chain or ""

                    row_contact_mask = contact_mask or ""
                    row_contact_n = int(contact_n or 0)

                    keep_counter["seq_from_pdb_multichain"] += 1


                elif pdb_mode == "single" and best_pdb_seq:
                    # single-chain は BindingDB seq 優先
                    seq = bindingdb_seq if bindingdb_seq else best_pdb_seq

                    src_pdbid = best_pdbid or ""
                    src_chain = best_chain or ""

                    # contact supervision は長さ一致時のみ使う
                    if contact_mask and len(contact_mask) == len(seq):
                        row_contact_mask = contact_mask
                        row_contact_n = int(contact_n or 0)
                    else:
                        row_contact_mask = ""
                        row_contact_n = 0

                    keep_counter["seq_from_bindingdb_single_chain"] += 1


                else:
                    seq = bindingdb_seq

                    row_contact_mask = ""
                    row_contact_n = 0

                    if pdb_mode == "none":
                        keep_counter["seq_from_bindingdb_no_pdb"] += 1
                    else:
                        keep_counter["seq_from_bindingdb_other"] += 1

                # ★ 最終安全弁（multi でも single でも必ず）
                if not seq:
                    drop_counter["no_valid_sequence"] += 1
                    err_w.writerow([row_id, "no_valid_sequence", best_status])
                    continue

                # ★ 最終安全弁
                if row_contact_mask and len(row_contact_mask) != len(seq):
                    row_contact_mask = ""
                    row_contact_n = 0
                    err_w.writerow([row_id, "no_valid_sequence", best_status])
                    continue

                # --------------------------------------------------
                # 3) smiles
                # --------------------------------------------------
                raw_smiles = row.get("Ligand SMILES", "")
                can_smiles = canonicalize_smiles_cached(raw_smiles)
                if not can_smiles:
                    drop_counter["bad_smiles"] += 1
                    err_w.writerow([row_id, "bad_smiles", raw_smiles])
                    continue

                if (seq, can_smiles) in core_pairs:
                    drop_counter["core_pair"] += 1
                    continue

                if EXCLUDE_ANY_CORE_SEQ and seq in core_seqs:
                    drop_counter["core_seq"] += 1
                    continue

                # --------------------------------------------------
                # 4) affinity
                # --------------------------------------------------
                aff_type, aff_nm = pick_affinity_nm(row)
                if aff_nm is None:
                    drop_counter["no_affinity"] += 1
                    err_w.writerow([row_id, "no_affinity", ""])
                    continue

                y = affinity_to_pvalue_from_nm(aff_nm)
                if y is None:
                    drop_counter["bad_affinity"] += 1
                    err_w.writerow([row_id, "bad_affinity", aff_nm])
                    continue
                pair_key = (seq, can_smiles)

                # raw rows count
                pair_counter_before[pair_key] += 1

                # unique pair existence for ligand / protein
                # 「同じ pair の assay repeat」はここでは1回として数えたい
                if pair_counter_before[pair_key] == 1:
                    lig_pair_counter_before[can_smiles] += 1
                    prot_pair_counter_before[seq] += 1
                batch.append((
                    seq, can_smiles, aff_type, aff_nm, y,
                    src_pdbid, src_chain, row_contact_mask, row_contact_n
                ))
                kept += 1
                if len(debug_rows_before) < 50000:
                    debug_rows_before.append({
                        "seq": seq,
                        "smiles": can_smiles,
                        "y": y
                    })
                if len(batch) >= PASS1_BATCH_SIZE:
                    flush_pair_batch(cur, con, batch)
                if seen % 10000 == 0:
                    err_f.flush()


    flush_pair_batch(cur, con, batch)

    # ---- dedup by (seq, smiles): median y ----

    rows_by_pair = defaultdict(list)

    for row in cur.execute("""
                           SELECT seq,
                                  smiles,
                                  aff_type,
                                  aff_nm,
                                  y,
                                  src_pdbid,
                                  src_chain,
                                  contact_mask,
                                  contact_n
                           FROM pairs
                           """):
        seq, smiles, aff_type, aff_nm, y, src_pdbid, src_chain, contact_mask, contact_n = row
        rows_by_pair[(seq, smiles)].append({
            "seq": seq,
            "smiles": smiles,
            "aff_type": aff_type,
            "aff_nm": aff_nm,
            "y": float(y),
            "src_pdbid": src_pdbid or "",
            "src_chain": src_chain or "",
            "contact_mask": contact_mask or "",
            "contact_n": int(contact_n or 0),
        })

    dedup_rows = []

    for (seq, smiles), group in rows_by_pair.items():
        group_sorted = sorted(group, key=lambda r: r["y"])
        mid = len(group_sorted) // 2

        if len(group_sorted) % 2 == 1:
            y_med = group_sorted[mid]["y"]
        else:
            y_med = 0.5 * (group_sorted[mid - 1]["y"] + group_sorted[mid]["y"])

        # median y に一番近い行を代表行にする
        rep = min(group, key=lambda r: abs(r["y"] - y_med))
        rep = dict(rep)
        rep["y"] = y_med

        # aff_nm は y_med から逆算しておく
        rep["aff_nm"] = 10 ** (9.0 - y_med)

        # 複数 assay が混ざるので aff_type は mixed にしてもよい
        aff_types = sorted({r["aff_type"] for r in group})
        rep["aff_type"] = aff_types[0] if len(aff_types) == 1 else "mixed"

        dedup_rows.append(rep)

    n_raw = cur.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    n_unique = len(dedup_rows)
    n_duplicates_collapsed = n_raw - n_unique

    with PASS1_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seq", "smiles", "aff_type", "aff_nm", "y",
            "src_pdbid", "src_chain", "contact_mask", "contact_n"
        ])

        for r in dedup_rows:
            w.writerow([
                r["seq"],
                r["smiles"],
                r["aff_type"],
                r["aff_nm"],
                r["y"],
                r["src_pdbid"],
                r["src_chain"],
                r["contact_mask"],
                r["contact_n"],
            ])

    with UNIQUE_SMILES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for smi in sorted({r["smiles"] for r in dedup_rows}):
            w.writerow([smi])

    n_unique_seq = len({r["seq"] for r in dedup_rows})
    n_unique_smiles = len({r["smiles"] for r in dedup_rows})

    con.close()
    import pandas as pd

    print("\n===== before dedup (true unique pair counts) =====")

    lig_vals = list(lig_pair_counter_before.values())
    prot_vals = list(prot_pair_counter_before.values())

    lig_s = pd.Series(lig_vals)
    prot_s = pd.Series(prot_vals)
    import pandas as pd

    print("\n===== BEFORE PDB FILTER =====")

    lig_vals = list(pre_pdb_lig_counter.values())
    prot_vals = list(pre_pdb_prot_counter.values())

    lig_s = pd.Series(lig_vals)
    prot_s = pd.Series(prot_vals)

    print("[ligand pairs before PDB]")
    print(lig_s.describe())
    print("single ratio:", (lig_s == 1).mean())

    print("\n[protein pairs before PDB]")
    print(prot_s.describe())
    print("single ratio:", (prot_s == 1).mean())

    print("[ligand unique pairs before dedup]")
    print(lig_s.describe())
    print("single ratio:", (lig_s == 1).mean())

    print("\n[protein unique pairs before dedup]")
    print(prot_s.describe())
    print("single ratio:", (prot_s == 1).mean())
    print(f"[pass1] unique seq={n_unique_seq:,}")
    print(f"[pass1] unique smiles={n_unique_smiles:,}")

    print("\n=== pass1 drop summary ===")
    for k, v in drop_counter.most_common():
        print(f"{k:30s}: {v:,}")

    print(f"\n[pass1] seen={seen:,}")
    print(f"[pass1] kept(before dedup)={kept:,}")
    print(f"[pass1] unique pairs(after dedup)={n_unique:,}")
    print(f"[pass1] duplicates collapsed={n_duplicates_collapsed:,}")
    print(f"[pass1] wrote: {PASS1_CSV}")
    print(f"[pass1] wrote: {UNIQUE_SMILES_CSV}")
    print(f"[pass1] wrote: {ERROR_TSV}")

    print("\n=== pass1 keep summary ===")
    for k, v in keep_counter.most_common():
        print(f"{k:30s}: {v:,}")

    debug_pair_stats(debug_rows_before, "pass1 before dedup (sample)")
    debug_pair_stats(dedup_rows, "pass1 after dedup")

# ============================================================
# Pass 2: tokenize unique smiles only once
# ============================================================

def _tokenize_one_smiles(smi: str):
    smi = (smi or "").strip()
    if not smi:
        return None, "empty_smiles"

    try:
        ids = encode_smiles_to_atom_tokens(smi)
    except Exception as e:
        return None, f"encode_exception:{type(e).__name__}:{e}"

    if ids is None:
        return None, "ids_is_none"

    try:
        ids_list = list(ids)
    except Exception as e:
        return None, f"ids_not_iterable:{type(e).__name__}:{e}"

    if len(ids_list) == 0:
        return None, "ids_empty"

    try:
        lig_tok = " ".join(str(int(x)) for x in ids_list)
    except Exception as e:
        return None, f"int_cast_exception:{type(e).__name__}:{e}"

    return (smi, lig_tok), None

from collections import Counter

def pass2_make_smiles_token_map_parallel():
    smiles_list = []

    with UNIQUE_SMILES_CSV.open("r", encoding="utf-8", newline="") as fin:
        r = csv.DictReader(fin)
        for row in r:
            smiles_list.append(row["smiles"])

    total = len(smiles_list)
    ok = 0
    fail_counter = Counter()

    fail_tsv = OUT_DIR / "tokenize_failures.tsv"

    with SMILES_TOK_CSV.open("w", encoding="utf-8", newline="") as fout, \
         fail_tsv.open("w", encoding="utf-8", newline="") as ferr:

        w = csv.writer(fout)
        w.writerow(["smiles", "lig_tok"])

        ew = csv.writer(ferr, delimiter="\t")
        ew.writerow(["smiles", "reason"])

        with Pool(processes=TOKENIZE_NPROC) as pool:
            for out, err in tqdm(
                pool.imap_unordered(_tokenize_one_smiles, smiles_list, chunksize=TOKENIZE_CHUNKSIZE),
                total=total,
                desc=f"pass2 tokenize unique smiles ({TOKENIZE_NPROC} proc)"
            ):
                if err is not None:
                    fail_counter[err] += 1
                    # out is None here, so write original smiles only if needed by restructuring more;
                    # for now just record reason count below
                    continue

                smi, lig_tok = out
                w.writerow([smi, lig_tok])
                ok += 1

                if ok % 10000 == 0:
                    fout.flush()

    print(f"[pass2] unique_smiles={total:,} tokenized={ok:,}")
    print(f"[pass2] wrote: {SMILES_TOK_CSV}")
    print(f"[pass2] wrote failures: {fail_tsv}")

    print("[pass2] failure summary:")
    for reason, n in fail_counter.most_common(20):
        print(f"  {reason}: {n}")


# ============================================================
# Pass 3: join pass1 rows with token map
# ============================================================

def pass3_join_tokens():
    tok_map = {}

    with SMILES_TOK_CSV.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tok_map[row["smiles"]] = row["lig_tok"]

    written = 0
    skipped = 0

    with PASS1_CSV.open("r", encoding="utf-8", newline="") as fin, \
         FINAL_CSV.open("w", encoding="utf-8", newline="") as fout:

        r = csv.DictReader(fin)
        w = csv.writer(fout)
        w.writerow([
            "seq", "smiles", "lig_tok", "aff_type", "aff_nm", "y",
            "src_pdbid", "src_chain", "contact_mask", "contact_n"
        ])
        for row in tqdm(r, desc="pass3 join tokens"):
            smi = row["smiles"]
            lig_tok = tok_map.get(smi)
            if not lig_tok:
                skipped += 1
                continue

            w.writerow([
                row["seq"],
                smi,
                lig_tok,
                row["aff_type"],
                row["aff_nm"],
                row["y"],
                row.get("src_pdbid", ""),
                row.get("src_chain", ""),
                row.get("contact_mask", ""),
                row.get("contact_n", "0"),
            ])
            written += 1

            if written % 10000 == 0:
                fout.flush()

    print(f"[pass3] wrote={written:,} skipped_no_tok={skipped:,}")
    print(f"[pass3] final: {FINAL_CSV}")


# ============================================================
# Pass 4: protein cluster split + Murcko scaffold split
# ============================================================

def stable_int_hash(text: str, seed: int = 0) -> int:
    h = hashlib.sha1(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def read_final_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_unique_seq_fasta(rows: List[Dict[str, str]], fasta_path: Path) -> List[str]:
    unique_seqs = sorted({row["seq"] for row in rows})
    with fasta_path.open("w", encoding="utf-8") as f:
        for i, seq in enumerate(unique_seqs):
            f.write(f">seq_{i}\n{seq}\n")
    print(f"[split] unique seqs={len(unique_seqs):,} wrote fasta={fasta_path}")
    return unique_seqs


def run_mmseqs_cluster(fasta_path: Path, out_dir: Path, tmp_dir: Path) -> Path:
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs was not found in PATH. Install MMseqs2 or provide a precomputed cluster map.")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / "seq40"
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id", str(MMSEQS_MIN_SEQ_ID),
        "-c", str(MMSEQS_C),
        "--cov-mode", str(MMSEQS_COV_MODE),
        "--cluster-mode", str(MMSEQS_CLUSTER_MODE),
        "--threads", str(MMSEQS_THREADS),
    ]
    print("[split] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster file not found: {cluster_tsv}")
    return cluster_tsv


def parse_fasta_headers(fasta_path: Path) -> Dict[str, str]:
    id_to_seq = {}
    cur_id = None
    cur_seq = []
    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    id_to_seq[cur_id] = "".join(cur_seq)
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line.strip())
    if cur_id is not None:
        id_to_seq[cur_id] = "".join(cur_seq)
    return id_to_seq


def build_seq_to_cluster_map(cluster_tsv: Path, fasta_path: Path) -> Dict[str, str]:
    id_to_seq = parse_fasta_headers(fasta_path)
    seq_to_cluster = {}
    rep_to_cluster_id = {}

    with cluster_tsv.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rep_id, member_id = line.rstrip("\n").split("\t")
            cluster_id = rep_to_cluster_id.setdefault(rep_id, f"protclust_{len(rep_to_cluster_id)}")
            member_seq = id_to_seq[member_id]
            seq_to_cluster[member_seq] = cluster_id

    missing = [seq for seq in id_to_seq.values() if seq not in seq_to_cluster]
    if missing:
        raise RuntimeError(f"{len(missing)} sequences missing from MMseqs cluster map")
    return seq_to_cluster


def save_seq_to_cluster_map(seq_to_cluster: Dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "protein_cluster"])
        for seq, cid in seq_to_cluster.items():
            w.writerow([seq, cid])
    print(f"[split] wrote seq->cluster map: {path}")


import random
from collections import defaultdict

import random
from collections import defaultdict

def write_split_csvs(rows: List[Dict[str, str]]) -> None:
    outs = {
        "train": TRAIN_CSV,
        "valid": VALID_CSV,
        "test": TEST_CSV,
    }
    fieldnames = [
        "seq", "smiles", "lig_tok", "aff_type", "aff_nm", "y",
        "src_pdbid", "src_chain", "contact_mask", "contact_n"
    ]
    # split -> protein_cluster -> rows
    split_cluster_rows = {
        "train": defaultdict(list),
        "valid": defaultdict(list),
        "test": defaultdict(list),
    }

    for row in rows:
        split_name = row["split"]
        prot_cluster = row["protein_cluster"]
        split_cluster_rows[split_name][prot_cluster].append(row)

    for split_name, path in outs.items():
        cluster_map = split_cluster_rows[split_name]

        # protein cluster order を再現可能に shuffle
        cluster_ids = list(cluster_map.keys())
        rng = random.Random(stable_int_hash(f"write_split::{split_name}", SPLIT_SEED))
        rng.shuffle(cluster_ids)

        n_written = 0
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for prot_cluster in cluster_ids:
                cluster_rows = list(cluster_map[prot_cluster])

                # cluster 内の row 順も再現可能に shuffle
                rng_in = random.Random(
                    stable_int_hash(f"write_split::{split_name}::cluster::{prot_cluster}", SPLIT_SEED)
                )
                rng_in.shuffle(cluster_rows)

                for row in cluster_rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
                    n_written += 1

        print(
            f"[split] wrote {split_name}: {path} "
            f"rows={n_written:,} protein_clusters={len(cluster_ids):,}"
        )
def summarize_splits(rows: List[Dict[str, str]]) -> None:
    stats = defaultdict(lambda: {"rows": 0, "seq": set(), "protclust": set(), "scaf": set()})
    for row in rows:
        s = stats[row["split"]]
        s["rows"] += 1
        s["seq"].add(row["seq"])
        s["protclust"].add(row["protein_cluster"])
        s["scaf"].add(row["ligand_scaffold"])

    print("\n=== split summary ===")
    for split_name in ["train", "valid", "test"]:
        s = stats[split_name]
        print(
            f"{split_name:>5} rows={s['rows']:,} "
            f"seq={len(s['seq']):,} "
            f"prot_clusters={len(s['protclust']):,} "
            f"scaffolds={len(s['scaf']):,}"
        )

def row_to_y_bin(row: Dict[str, str], y_thr: float = Y_THR) -> Optional[int]:
    y_raw = row.get("y", "")
    try:
        y = float(y_raw)
    except Exception:
        return None
    return 1 if y >= y_thr else 0


def compute_entity_stats(rows: List[Dict[str, str]], key: str):
    """
    returns:
      stats[key] = {
        "n": int,
        "pos": int,
        "neg": int,
        "pos_rate": float,
      }
    """
    counter = defaultdict(lambda: {"n": 0, "pos": 0, "neg": 0})
    for row in rows:
        ent = row.get(key, "")
        if not ent:
            continue
        yb = row.get("y_bin", None)
        if yb is None:
            continue
        counter[ent]["n"] += 1
        if int(yb) == 1:
            counter[ent]["pos"] += 1
        else:
            counter[ent]["neg"] += 1

    for ent, d in counter.items():
        d["pos_rate"] = d["pos"] / max(1, d["n"])
    return counter


def entity_is_good(stat: Dict[str, float], min_pairs: int, low: float, high: float) -> bool:
    if stat["n"] < min_pairs:
        return False
    r = stat["pos_rate"]
    return (low <= r <= high)


def filter_rows_by_entity_constraints(
    rows: List[Dict[str, str]],
    lig_min_pairs: int,
    prot_min_pairs: int,
    low: float,
    high: float,
):
    """
    Keep only rows whose ligand and protein both satisfy:
      - enough rows
      - pos_rate in [low, high]
    """
    lig_stats = compute_entity_stats(rows, "lig_tok")
    prot_stats = compute_entity_stats(rows, "seq")

    good_ligs = {
        lig for lig, st in lig_stats.items()
        if entity_is_good(st, lig_min_pairs, low, high)
    }
    good_prots = {
        prot for prot, st in prot_stats.items()
        if entity_is_good(st, prot_min_pairs, low, high)
    }

    out = [
        row for row in rows
        if row.get("lig_tok", "") in good_ligs and row.get("seq", "") in good_prots
    ]
    return out, lig_stats, prot_stats, good_ligs, good_prots


def balance_rows_within_entity(rows: List[Dict[str, str]], key: str, seed: int):
    """
    For each entity, keep min(pos, neg) positives and negatives.
    This pushes pos_rate toward 0.5 exactly.
    """
    by_ent = defaultdict(list)
    for row in rows:
        ent = row.get(key, "")
        if ent:
            by_ent[ent].append(row)

    out = []
    for ent, group in by_ent.items():
        pos = [r for r in group if int(r["y_bin"]) == 1]
        neg = [r for r in group if int(r["y_bin"]) == 0]

        n = min(len(pos), len(neg))
        if n == 0:
            continue

        rng_pos = random.Random(stable_int_hash(f"{seed}|{key}|{ent}|pos", SPLIT_SEED))
        rng_neg = random.Random(stable_int_hash(f"{seed}|{key}|{ent}|neg", SPLIT_SEED))

        if len(pos) > n:
            pos = rng_pos.sample(pos, n)
        if len(neg) > n:
            neg = rng_neg.sample(neg, n)

        out.extend(pos)
        out.extend(neg)

    return out


def summarize_binary_balance(rows: List[Dict[str, str]], name: str):
    split_rows = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)

    print(f"\n=== binary balance summary: {name} ===")
    for split in ["train", "valid", "test"]:
        rr = split_rows.get(split, [])
        n = len(rr)
        if n == 0:
            print(f"{split:>5}: rows=0")
            continue

        lig_stats = compute_entity_stats(rr, "lig_tok")
        prot_stats = compute_entity_stats(rr, "seq")

        lig_rates = [d["pos_rate"] for d in lig_stats.values() if d["n"] > 0]
        prot_rates = [d["pos_rate"] for d in prot_stats.values() if d["n"] > 0]

        n_pos = sum(int(r["y_bin"]) for r in rr)
        split_pos_rate = n_pos / max(1, n)

        def mean_or_zero(xs):
            return sum(xs) / len(xs) if xs else 0.0

        print(
            f"{split:>5}: rows={n:,} "
            f"split_pos_rate={split_pos_rate:.3f} "
            f"ligands={len(lig_stats):,} lig_pos_rate_mean={mean_or_zero(lig_rates):.3f} "
            f"proteins={len(prot_stats):,} prot_pos_rate_mean={mean_or_zero(prot_rates):.3f}"
        )


def write_annotated_rows(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        print(f"[split] no rows to write annotated csv: {path}")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[split] wrote annotated rows: {path}")

def filter_rows_by_protein_cluster_constraints(rows, min_pairs=8, low=0.1, high=0.9):
    cluster_stats = compute_entity_stats(rows, "protein_cluster")
    good_clusters = {
        c for c, st in cluster_stats.items()
        if st["n"] >= min_pairs and (low <= st["pos_rate"] <= high)
    }
    out = [row for row in rows if row.get("protein_cluster", "") in good_clusters]
    return out, cluster_stats, good_clusters

def print_entity_pair_stats(rows: List[Dict[str, str]], name: str):
    split_rows = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)

    print(f"\n=== entity pair stats: {name} ===")
    for split in ["train", "valid", "test"]:
        rr = split_rows.get(split, [])
        if not rr:
            print(f"{split:>5}: rows=0")
            continue

        lig_counts = defaultdict(int)
        prot_counts = defaultdict(int)
        for r in rr:
            lig_counts[r["lig_tok"]] += 1
            prot_counts[r["seq"]] += 1

        lig_vals = list(lig_counts.values())
        prot_vals = list(prot_counts.values())

        print(
            f"{split:>5}: "
            f"lig_pairs_mean={sum(lig_vals)/len(lig_vals):.2f} "
            f"lig_pairs_min={min(lig_vals)} "
            f"prot_pairs_mean={sum(prot_vals)/len(prot_vals):.2f} "
            f"prot_pairs_min={min(prot_vals)}"
        )


def drop_tiny_components(comp_rows, min_component_rows: int):
    kept = [c for c in comp_rows if c["weight"] >= min_component_rows]
    dropped = [c for c in comp_rows if c["weight"] < min_component_rows]

    n_drop_rows = sum(c["weight"] for c in dropped)
    print(
        f"[split] drop_tiny_components: kept={len(kept):,} dropped={len(dropped):,} "
        f"dropped_rows={n_drop_rows:,}"
    )
    return kept

def stage4_protein_cold_split():
    rows = read_final_rows(FINAL_CSV)
    if not rows:
        raise RuntimeError(f"No rows found in {FINAL_CSV}")

    # 0) attach y_bin
    rows0 = []
    n_missing_y = 0
    for row in rows:
        yb = row_to_y_bin(row, Y_THR)
        if yb is None:
            n_missing_y += 1
            if DROP_ROWS_WITH_MISSING_Y:
                continue
        row = dict(row)
        row["y_bin"] = yb
        rows0.append(row)

    print(f"[split] input rows={len(rows):,} usable_with_y={len(rows0):,} missing_y={n_missing_y:,}")
    if not rows0:
        raise RuntimeError("No usable rows after y parsing")

    # 1) protein clustering
    write_unique_seq_fasta(rows0, SPLIT_FASTA)
    cluster_tsv = run_mmseqs_cluster(SPLIT_FASTA, MMSEQS_DIR, MMSEQS_TMP)
    seq_to_cluster = build_seq_to_cluster_map(cluster_tsv, SPLIT_FASTA)
    save_seq_to_cluster_map(seq_to_cluster, SEQ_TO_CLUSTER_CSV)

    # 2) annotate protein_cluster + ligand_scaffold
    kept_rows = []
    dropped_no_cluster = 0
    dropped_no_scaffold = 0

    for row in rows0:
        prot_cluster = seq_to_cluster.get(row["seq"])
        if prot_cluster is None:
            dropped_no_cluster += 1
            continue

        scaffold = murcko_scaffold_smiles_cached(row["smiles"])
        if not scaffold:
            dropped_no_scaffold += 1
            continue

        row = dict(row)
        row["protein_cluster"] = prot_cluster
        row["ligand_scaffold"] = scaffold
        kept_rows.append(row)

    print(
        f"[split] annotated rows={len(kept_rows):,} "
        f"dropped_no_cluster={dropped_no_cluster:,} "
        f"dropped_no_scaffold={dropped_no_scaffold:,}"
    )
    if not kept_rows:
        raise RuntimeError("No usable rows after annotation")

    # 3) optional protein-cluster filtering
    filtered_rows, cluster_stats, good_clusters = filter_rows_by_protein_cluster_constraints(
        kept_rows,
        min_pairs=8,
        low=0.1,
        high=0.9,
    )
    print(f"[split] after protein-cluster filtering: {len(filtered_rows):,}")
    print(f"[split] kept protein clusters: {len(good_clusters):,}")

    if not filtered_rows:
        raise RuntimeError("No rows remain after protein-cluster filtering")

    kept_rows = filtered_rows

    # 4) stratified protein-cluster cold split
    #    protein_cluster を単位に保ったまま、
    #    row数とpositive数が train/valid/test で近くなるように greedy 割当する
    # after kept_rows is made and before protein-cluster filtering
    for r in kept_rows:
        r["y_bin"] = int(r["y_bin"])
    lig_stats = compute_entity_stats(kept_rows, "lig_tok")
    prot_stats = compute_entity_stats(kept_rows, "seq")
    clust_stats = compute_entity_stats(kept_rows, "protein_cluster")

    print("lig >=2:", sum(1 for s in lig_stats.values() if s["n"] >= 2))
    print("rows with lig>=2:", sum(s["n"] for s in lig_stats.values() if s["n"] >= 2))

    print("seq >=2:", sum(1 for s in prot_stats.values() if s["n"] >= 2))
    print("rows with seq>=2:", sum(s["n"] for s in prot_stats.values() if s["n"] >= 2))

    print("cluster >=2:", sum(1 for s in clust_stats.values() if s["n"] >= 2))
    print("rows with cluster>=2:", sum(s["n"] for s in clust_stats.values() if s["n"] >= 2))
    kept_rows, lig_stats, prot_stats, good_ligs, good_prots = filter_rows_by_entity_constraints(
        kept_rows,
        lig_min_pairs=2,
        prot_min_pairs=2,
        low=0.1,
        high=0.9,
    )

    print(f"[split] after ligand/protein entity filtering: {len(kept_rows):,}")
    print(f"[split] kept ligands: {len(good_ligs):,}")
    print(f"[split] kept proteins: {len(good_prots):,}")

    def make_cluster_stratified_split(rows, ratios, seed=123):
        split_names = ["train", "valid", "test"]

        by_cluster = defaultdict(list)
        for r in rows:
            by_cluster[r["protein_cluster"]].append(r)

        total_rows = len(rows)
        total_pos = sum(int(r["y_bin"]) for r in rows)
        global_pos_rate = total_pos / max(1, total_rows)

        cluster_stats = []
        for c, rs in by_cluster.items():
            n = len(rs)
            pos = sum(int(r["y_bin"]) for r in rs)
            pos_rate = pos / max(1, n)
            cluster_stats.append({
                "cluster": c,
                "n": n,
                "pos": pos,
                "pos_rate": pos_rate,
                "n_lig": len({r["lig_tok"] for r in rs}),
            })

        rng = random.Random(seed)
        rng.shuffle(cluster_stats)

        # 大きさとpos_rateで並べて、各binから8:1:1で取る
        cluster_stats.sort(
            key=lambda d: (
                -d["n"],
                -abs(d["pos_rate"] - global_pos_rate),
                -d["n_lig"],
            )
        )

        assigned = {s: set() for s in split_names}

        # 10個ずつのブロックで 8 train, 1 valid, 1 test
        # これで巨大clusterがvalid/testに偏りにくい
        for i in range(0, len(cluster_stats), 10):
            block = cluster_stats[i:i + 10]
            rng.shuffle(block)

            for j, st in enumerate(block):
                if j < 8:
                    s = "train"
                elif j == 8:
                    s = "valid"
                else:
                    s = "test"
                assigned[s].add(st["cluster"])

        # stats
        cur_rows = {s: 0 for s in split_names}
        cur_pos = {s: 0 for s in split_names}

        for st in cluster_stats:
            for s in split_names:
                if st["cluster"] in assigned[s]:
                    cur_rows[s] += st["n"]
                    cur_pos[s] += st["pos"]
                    break

        print("\n[split] rank-stratified protein-cluster assignment")
        for s in split_names:
            pr = cur_pos[s] / max(1, cur_rows[s])
            print(
                f"    {s}: clusters={len(assigned[s]):,} "
                f"rows={cur_rows[s]:,} "
                f"pos={cur_pos[s]:,} "
                f"pos_rate={pr:.3f}"
            )

        return assigned["train"], assigned["valid"], assigned["test"]

    train_clusters, valid_clusters, test_clusters = make_cluster_stratified_split(
        kept_rows,
        SPLIT_RATIOS,
        seed=SPLIT_SEED,
    )

    print(
        f"[split] protein clusters total="
        f"{len(train_clusters) + len(valid_clusters) + len(test_clusters):,} "
        f"train={len(train_clusters):,} "
        f"valid={len(valid_clusters):,} "
        f"test={len(test_clusters):,}"
    )

    n_total_clusters = (
            len(train_clusters)
            + len(valid_clusters)
            + len(test_clusters)
    )

    # 5) assign rows by protein cluster
    final_rows = []
    for row in kept_rows:
        pc = row["protein_cluster"]
        row2 = dict(row)
        if pc in train_clusters:
            row2["split"] = "train"
        elif pc in valid_clusters:
            row2["split"] = "valid"
        elif pc in test_clusters:
            row2["split"] = "test"
        else:
            raise RuntimeError(f"Protein cluster not assigned: {pc}")
        final_rows.append(row2)

    if not final_rows:
        raise RuntimeError("No rows assigned after protein-cold split")

    # 6) summaries
    after_counts = defaultdict(int)
    for row in final_rows:
        after_counts[row["split"]] += 1

    total_final = len(final_rows)
    print("\n[split] final row counts")
    for s in ["train", "valid", "test"]:
        frac = after_counts[s] / max(1, total_final)
        print(f"    {s}: rows={after_counts[s]:,} ({frac:.3%})")

    summarize_splits(final_rows)
    summarize_binary_balance(final_rows, "protein_cold_split")
    print_entity_pair_stats(final_rows, "protein_cold_split")

    # sanity checks
    train_prot = {r["protein_cluster"] for r in final_rows if r["split"] == "train"}
    valid_prot = {r["protein_cluster"] for r in final_rows if r["split"] == "valid"}
    test_prot  = {r["protein_cluster"] for r in final_rows if r["split"] == "test"}

    train_scaf = {r["ligand_scaffold"] for r in final_rows if r["split"] == "train"}
    valid_scaf = {r["ligand_scaffold"] for r in final_rows if r["split"] == "valid"}
    test_scaf  = {r["ligand_scaffold"] for r in final_rows if r["split"] == "test"}

    print("\n=== overlap sanity check ===")
    print(f"protein overlap train-valid: {len(train_prot & valid_prot)}")
    print(f"protein overlap train-test : {len(train_prot & test_prot)}")
    print(f"protein overlap valid-test : {len(valid_prot & test_prot)}")
    print(f"scaffold overlap train-valid: {len(train_scaf & valid_scaf)}")
    print(f"scaffold overlap train-test : {len(train_scaf & test_scaf)}")
    print(f"scaffold overlap valid-test : {len(valid_scaf & test_scaf)}")

    write_split_csvs(final_rows)
    write_annotated_rows(final_rows, ROW_ANNOT_CSV)

def debug_tokenizer_once():
    with UNIQUE_SMILES_CSV.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        first = next(r)["smiles"]


    try:
        ids = encode_smiles_to_atom_tokens(first)
    except Exception as e:
        raise

def debug_pair_stats(rows, tag):
    import pandas as pd
    df = pd.DataFrame(rows)

    print(f"\n===== {tag} =====")

    if "seq" in df.columns:
        g = df.groupby("seq").size()
        print("[protein pairs]")
        print(g.describe())
        print("single ratio:", (g == 1).mean())

    if "smiles" in df.columns:
        g2 = df.groupby("smiles").size()
        print("[ligand pairs]")
        print(g2.describe())
        print("single ratio:", (g2 == 1).mean())

def debug_tokenizer_once():
    with UNIQUE_SMILES_CSV.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        first = next(r)["smiles"]
    ids = encode_smiles_to_atom_tokens(first)
    ids_list = list(ids)

# ============================================================
# Main
# ============================================================

def main():
    print("\n=== stage 2: make KIBA rows ===")
    pass1_from_graphdta_processed()

    print("\n=== stage 3: tokenize unique smiles ===")
    pass2_make_smiles_token_map_parallel()

    print("\n=== stage 4: join tokens ===")
    pass3_join_tokens()

    print("\n=== stage 5: protein-cold split ===")
    stage4_protein_cold_split()


if __name__ == "__main__":
    main()
