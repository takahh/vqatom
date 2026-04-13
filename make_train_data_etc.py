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

BINDINGDB_TSV = Path("/Users/taka/Desktop/BindingDB_All.tsv")

CORE_FINAL_EVAL_CSV = Path(
    "/Users/taka/Desktop/final_eval.csv"
)

OUT_DIR = Path(
    "/Users/taka/Desktop/bindingdb_fast_out"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MMCIF_DIR = Path("/Volumes/Untitled/mmcif")

PDB_CACHE_DB = OUT_DIR / "pdb_chain_cache.sqlite"
DEDUP_DB = OUT_DIR / "dedup.sqlite"

PASS1_CSV = OUT_DIR / "pass1_rows.csv"
UNIQUE_SMILES_CSV = OUT_DIR / "unique_smiles.csv"
SMILES_TOK_CSV = OUT_DIR / "smiles_tok_map.csv"
FINAL_CSV = OUT_DIR / "bindingdb_dedup_tok.csv"
ERROR_TSV = OUT_DIR / "errors.tsv"
PDBID_LIST_TXT = OUT_DIR / "bindingdb_unique_pdbids.txt"
PDB_CACHE_FAIL_TSV = OUT_DIR / "pdb_cache_failures.tsv"

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
Y_THR = 7.0
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

# if True, exclude any seq appearing in core, even with different smiles
EXCLUDE_ANY_CORE_SEQ = True

# split settings
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
SPLIT_SEED = 123
MMSEQS_MIN_SEQ_ID = 0.40
MMSEQS_COV_MODE = 0
MMSEQS_C = 0.8
MMSEQS_CLUSTER_MODE = 2  # connected component style
MMSEQS_THREADS = max(1, min(24, cpu_count()))
USE_GENERIC_MURCKO = False


# ============================================================
# RDKit / SMILES utils
# ============================================================

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


def choose_best_chain_from_pdb_impl(pdbid: str) -> Tuple[str, str, str, int, int]:
    """
    chain selection rule:
      - protein chain が 0 本: fail
      - protein chain が 1 本: それを採用（接触 0 でも採用）
      - protein chain が 2 本以上: ligand との接触数が最大の chain を採用
    returns:
      (best_chain_id, best_seq, status, top_contacts, second_contacts)
    """
    try:
        structure = open_mmcif_structure(pdbid)
    except FileNotFoundError:
        return "", "", "no_file", 0, 0
    except Exception as e:
        return "", "", f"parse_error:{type(e).__name__}:{e}", 0, 0

    # first model only
    try:
        model = next(structure.get_models())
    except StopIteration:
        return "", "", "no_model", 0, 0

    ligand_atoms = get_ligand_atoms(structure)

    # protein chain 候補を全部集める
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
            "chain_id": chain.id,
            "seq": seq,
            "protein_atoms": protein_atoms,
        })

    if not protein_chain_rows:
        return "", "", "no_protein_chains", 0, 0

    # protein chain が1本しかなければそれを採用
    if len(protein_chain_rows) == 1:
        row = protein_chain_rows[0]
        best_seq = clean_protein_seq(row["seq"])
        if not best_seq:
            return "", "", "bad_top_seq", 0, 0
        return row["chain_id"], best_seq, "ok_single_chain", 0, 0

    # 2本以上ある場合は接触数で選ぶ
    if not ligand_atoms:
        return "", "", "no_ligand_atoms", 0, 0

    all_atoms = [a for a in structure.get_atoms() if a.element != "H"]
    if not all_atoms:
        return "", "", "no_atoms", 0, 0

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

        chain_rows.append((row["chain_id"], row["seq"], contact_count))

    chain_rows.sort(key=lambda x: (-x[2], x[0]))
    best_chain, best_seq, best_contacts = chain_rows[0]
    second_contacts = chain_rows[1][2] if len(chain_rows) >= 2 else 0

    best_seq = clean_protein_seq(best_seq)
    if not best_seq:
        return "", "", "bad_top_seq", best_contacts, second_contacts

    return best_chain, best_seq, "ok", best_contacts, second_contacts

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
            second_contacts INTEGER NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_pdb_cache_status ON pdb_chain_cache(status)")
    con.commit()
    return con


def _worker_choose_best_chain(pdbid: str):
    chain_id, seq, status, top_contacts, second_contacts = choose_best_chain_from_pdb_impl(pdbid)
    return pdbid, chain_id, seq, status, top_contacts, second_contacts


def build_pdb_chain_cache(unique_pdbids: List[str]) -> None:
    print(f"[pdb-cache] building for {len(unique_pdbids):,} unique pdb ids")
    con = init_pdb_cache_db(PDB_CACHE_DB)
    cur = con.cursor()

    failures = []

    with Pool(processes=PDB_PARSE_NPROC) as pool:
        for rec in tqdm(
            pool.imap_unordered(_worker_choose_best_chain, unique_pdbids, chunksize=20),
            total=len(unique_pdbids),
            desc=f"build pdb cache ({PDB_PARSE_NPROC} proc)"
        ):
            pdbid, chain_id, seq, status, top_contacts, second_contacts = rec
            cur.execute("""
                INSERT INTO pdb_chain_cache(pdbid, chain_id, seq, status, top_contacts, second_contacts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pdbid, chain_id, seq, status, top_contacts, second_contacts))

            if not status.startswith("ok"):
                failures.append((pdbid, status, top_contacts, second_contacts))

    con.commit()
    con.close()

    with PDB_CACHE_FAIL_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pdbid", "status", "top_contacts", "second_contacts"])
        for row in failures:
            w.writerow(row)

    print(f"[pdb-cache] wrote: {PDB_CACHE_DB}")
    print(f"[pdb-cache] failures: {len(failures):,}")
    print(f"[pdb-cache] wrote: {PDB_CACHE_FAIL_TSV}")

def load_pdb_cache_map(db_path: Path) -> Dict[str, Tuple[str, str, str, int, int]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    out = {}
    for row in cur.execute("""
        SELECT pdbid, chain_id, seq, status, top_contacts, second_contacts
        FROM pdb_chain_cache
    """):
        pdbid, chain_id, seq, status, top_contacts, second_contacts = row
        out[pdbid] = (
            chain_id or "",
            seq or "",
            status,
            int(top_contacts),
            int(second_contacts),
        )
    con.close()
    return out

def choose_best_chain_across_pdbids_cached(
    pdbids: List[str],
    pdb_cache_map: Dict[str, Tuple[str, str, str, int, int]],
):
    """
    複数PDB候補の中から、cache済み結果を使って最良 chain を選ぶ。
    ルール:
      - status が ok / ok_single_chain のものだけ候補
      - top_contacts が最大のものを採用
    returns:
      (best_pdbid, best_chain, best_seq, debug_status)
    """
    best_pdbid = None
    best_chain = None
    best_seq = None
    best_contacts = -1
    best_second = 0

    debug = []

    for pdbid in pdbids:
        row = pdb_cache_map.get(pdbid)
        if row is None:
            debug.append(f"{pdbid}:missing_cache:0:0")
            continue

        chain_id, seq, status, top_contacts, second_contacts = row
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
            seq TEXT NOT NULL,
            smiles TEXT NOT NULL,
            aff_type TEXT NOT NULL,
            aff_nm REAL NOT NULL,
            y REAL NOT NULL,
            src_pdbid TEXT,
            src_chain TEXT,
            PRIMARY KEY (seq, smiles)
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
        INSERT INTO pairs(seq, smiles, aff_type, aff_nm, y, src_pdbid, src_chain)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(seq, smiles) DO UPDATE SET
            aff_type=CASE WHEN excluded.aff_nm < pairs.aff_nm THEN excluded.aff_type ELSE pairs.aff_type END,
            aff_nm=MIN(pairs.aff_nm, excluded.aff_nm),
            y=CASE WHEN excluded.aff_nm < pairs.aff_nm THEN excluded.y ELSE pairs.y END,
            src_pdbid=CASE WHEN excluded.aff_nm < pairs.aff_nm THEN excluded.src_pdbid ELSE pairs.src_pdbid END,
            src_chain=CASE WHEN excluded.aff_nm < pairs.aff_nm THEN excluded.src_chain ELSE pairs.src_chain END
    """, batch)
    con.commit()
    batch.clear()

def pass1_clean_and_dedup():
    core_pairs, core_seqs = load_core_exact_pairs(CORE_FINAL_EVAL_CSV)

    # build/load pdb chain cache once
    if not PDB_CACHE_DB.exists():
        unique_pdbids = collect_unique_bindingdb_pdbids(BINDINGDB_TSV, PDB_COL)
        build_pdb_chain_cache(unique_pdbids)

    pdb_cache_map = load_pdb_cache_map(PDB_CACHE_DB)
    con = init_dedup_db(DEDUP_DB)
    cur = con.cursor()

    seen = 0
    kept = 0
    batch = []

    with ERROR_TSV.open("w", encoding="utf-8", newline="") as err_f:
        err_w = csv.writer(err_f, delimiter="\t")
        err_w.writerow(["row_id", "reason", "extra"])

        with BINDINGDB_TSV.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in tqdm(reader, desc="pass1 clean"):
                seen += 1
                row_id = row.get("BindingDB Reactant_set_id", str(seen))

                pdb_ids = parse_pdb_ids(row.get(PDB_COL, ""))

                best_pdbid, best_chain, best_seq, best_status = choose_best_chain_across_pdbids_cached(
                    pdb_ids,
                    pdb_cache_map,
                )
                if best_seq is None:
                    err_w.writerow([row_id, "no_valid_pdb_chain", best_status])
                    continue

                seq = best_seq
                src_pdbid = best_pdbid
                src_chain = best_chain

                if not seq:
                    err_w.writerow([row_id, "bad_bindingdb_seq1", ""])
                    continue

                raw_smiles = row.get("Ligand SMILES", "")
                can_smiles = canonicalize_smiles_cached(raw_smiles)
                if not can_smiles:
                    err_w.writerow([row_id, "bad_smiles", raw_smiles])
                    continue

                if (seq, can_smiles) in core_pairs:
                    continue
                if EXCLUDE_ANY_CORE_SEQ and seq in core_seqs:
                    continue

                aff_type, aff_nm = pick_affinity_nm(row)
                if aff_nm is None:
                    err_w.writerow([row_id, "no_affinity", ""])
                    continue

                y = affinity_to_pvalue_from_nm(aff_nm)
                if y is None:
                    err_w.writerow([row_id, "bad_affinity", aff_nm])
                    continue

                batch.append((seq, can_smiles, aff_type, aff_nm, y, src_pdbid, src_chain))
                kept += 1

                if len(batch) >= PASS1_BATCH_SIZE:
                    flush_pair_batch(cur, con, batch)

                if seen % 10000 == 0:
                    err_f.flush()

    flush_pair_batch(cur, con, batch)

    with PASS1_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "smiles", "aff_type", "aff_nm", "y", "src_pdbid", "src_chain"])
        for row in cur.execute("""
            SELECT seq, smiles, aff_type, aff_nm, y, src_pdbid, src_chain
            FROM pairs
        """):
            w.writerow(row)

    with UNIQUE_SMILES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for row in cur.execute("SELECT DISTINCT smiles FROM pairs"):
            w.writerow([row[0]])

    con.close()

    print(f"[pass1] seen={seen:,} pre_kept={kept:,}")
    print(f"[pass1] wrote: {PASS1_CSV}")
    print(f"[pass1] wrote: {UNIQUE_SMILES_CSV}")
    print(f"[pass1] wrote: {ERROR_TSV}")


# ============================================================
# Pass 2: tokenize unique smiles only once
# ============================================================

def _tokenize_one_smiles(smi: str):
    smi = (smi or "").strip()
    if not smi:
        return None
    try:
        ids = encode_smiles_to_atom_tokens(smi)
    except Exception:
        return None
    if not ids:
        return None
    try:
        lig_tok = " ".join(str(int(x)) for x in ids)
    except Exception:
        return None
    return smi, lig_tok


def pass2_make_smiles_token_map_parallel():
    smiles_list = []

    with UNIQUE_SMILES_CSV.open("r", encoding="utf-8", newline="") as fin:
        r = csv.DictReader(fin)
        for row in r:
            smiles_list.append(row["smiles"])

    total = len(smiles_list)
    ok = 0

    with SMILES_TOK_CSV.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["smiles", "lig_tok"])

        with Pool(processes=TOKENIZE_NPROC) as pool:
            for out in tqdm(
                pool.imap_unordered(_tokenize_one_smiles, smiles_list, chunksize=TOKENIZE_CHUNKSIZE),
                total=total,
                desc=f"pass2 tokenize unique smiles ({TOKENIZE_NPROC} proc)"
            ):
                if out is None:
                    continue
                smi, lig_tok = out
                w.writerow([smi, lig_tok])
                ok += 1

                if ok % 10000 == 0:
                    fout.flush()

    print(f"[pass2] unique_smiles={total:,} tokenized={ok:,}")
    print(f"[pass2] wrote: {SMILES_TOK_CSV}")


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
        w.writerow(["seq", "smiles", "lig_tok", "aff_type", "aff_nm", "y", "src_pdbid", "src_chain"])

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
    fieldnames = ["seq", "smiles", "lig_tok", "aff_type", "aff_nm", "y", "src_pdbid", "src_chain"]

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

def stage4_make_double_cold_splits_balanced():
    rows = read_final_rows(FINAL_CSV)
    if not rows:
        raise RuntimeError(f"No rows found in {FINAL_CSV}")

    # ------------------------------------------------------------
    # 0) attach y_bin first
    # ------------------------------------------------------------
    rows0 = []
    n_missing_y = 0
    for row in rows:
        yb = row_to_y_bin(row, Y_THR)
        if yb is None:
            n_missing_y += 1
            if DROP_ROWS_WITH_MISSING_Y:
                continue
        row["y_bin"] = yb
        rows0.append(row)

    print(f"[split] input rows={len(rows):,} usable_with_y={len(rows0):,} missing_y={n_missing_y:,}")
    if not rows0:
        raise RuntimeError("No usable rows after y parsing")

    # ------------------------------------------------------------
    # 1) protein clustering
    # ------------------------------------------------------------
    write_unique_seq_fasta(rows0, SPLIT_FASTA)
    cluster_tsv = run_mmseqs_cluster(SPLIT_FASTA, MMSEQS_DIR, MMSEQS_TMP)
    seq_to_cluster = build_seq_to_cluster_map(cluster_tsv, SPLIT_FASTA)
    save_seq_to_cluster_map(seq_to_cluster, SEQ_TO_CLUSTER_CSV)

    # ------------------------------------------------------------
    # 2) annotate protein_cluster + scaffold
    # ------------------------------------------------------------
    kept_rows = []
    dropped_no_cluster = 0
    dropped_no_scaffold = 0

    for row in rows0:
        seq = row["seq"]
        smi = row["smiles"]

        prot_cluster = seq_to_cluster.get(seq)
        if prot_cluster is None:
            dropped_no_cluster += 1
            continue

        scaffold = murcko_scaffold_smiles_cached(smi)
        if not scaffold:
            dropped_no_scaffold += 1
            continue

        row["protein_cluster"] = prot_cluster
        row["ligand_scaffold"] = scaffold
        kept_rows.append(row)

    print(
        f"[split] annotated rows={len(kept_rows):,} "
        f"dropped_no_cluster={dropped_no_cluster:,} "
        f"dropped_no_scaffold={dropped_no_scaffold:,}"
    )
    if not kept_rows:
        raise RuntimeError("No usable rows after protein-cluster / scaffold annotation")

    # ------------------------------------------------------------
    # helper: greedy balanced split by ROW COUNTS
    # ------------------------------------------------------------
    def greedy_split_by_weight(weight_map, seed):
        rng = random.Random(seed)

        items = list(weight_map.items())
        rng.shuffle(items)
        items.sort(key=lambda x: (-x[1], str(x[0])))

        total_weight = sum(w for _, w in items)
        targets = {
            "train": total_weight * SPLIT_RATIOS["train"],
            "valid": total_weight * SPLIT_RATIOS["valid"],
            "test":  total_weight * SPLIT_RATIOS["test"],
        }

        split_sets = {"train": set(), "valid": set(), "test": set()}
        split_weights = {"train": 0, "valid": 0, "test": 0}
        split_order = ["train", "valid", "test"]

        for item, w in items:
            deficits = {s: targets[s] - split_weights[s] for s in split_order}
            best_split = max(split_order, key=lambda s: (deficits[s], -split_weights[s]))
            split_sets[best_split].add(item)
            split_weights[best_split] += w

        print("[split] balanced assignment summary")
        for s in split_order:
            frac = split_weights[s] / max(1, total_weight)
            print(
                f"    {s}: target={targets[s]:,.1f} assigned={split_weights[s]:,} "
                f"({frac:.3%}) items={len(split_sets[s]):,}"
            )

        return split_sets, split_weights

    # ------------------------------------------------------------
    # 3) split protein clusters by row count
    # ------------------------------------------------------------
    prot_cluster_rowcount = defaultdict(int)
    for row in kept_rows:
        prot_cluster_rowcount[row["protein_cluster"]] += 1

    prot_split, _ = greedy_split_by_weight(
        prot_cluster_rowcount,
        SPLIT_SEED,
    )

    # ------------------------------------------------------------
    # 4) split scaffolds by row count
    # ------------------------------------------------------------
    scaffold_rowcount = defaultdict(int)
    for row in kept_rows:
        scaffold_rowcount[row["ligand_scaffold"]] += 1

    scaf_split, _ = greedy_split_by_weight(
        scaffold_rowcount,
        SPLIT_SEED + 1,
    )

    # ------------------------------------------------------------
    # 5) keep only rows where protein split == scaffold split
    # ------------------------------------------------------------
    final_rows = []
    dropped_conflict = 0

    for row in kept_rows:
        p = row["protein_cluster"]
        s = row["ligand_scaffold"]

        assigned = None
        for split in ["train", "valid", "test"]:
            if p in prot_split[split] and s in scaf_split[split]:
                assigned = split
                break

        if assigned is None:
            dropped_conflict += 1
            continue

        row["split"] = assigned
        final_rows.append(row)

    print(f"[split] after double-cold intersection kept={len(final_rows):,} dropped_conflict={dropped_conflict:,}")
    if not final_rows:
        raise RuntimeError("No rows survived protein+scaffold intersection")

    # ------------------------------------------------------------
    # 6) iterative filtering + balancing
    # ------------------------------------------------------------
    current_rows = list(final_rows)

    for it in range(BALANCE_MAX_ITERS):
        print(f"\n[balance] iteration {it+1}/{BALANCE_MAX_ITERS} start rows={len(current_rows):,}")

        # 6a) filter by ligand/protein constraints
        filtered_rows, lig_stats, prot_stats, good_ligs, good_prots = filter_rows_by_entity_constraints(
            current_rows,
            lig_min_pairs=LIG_MIN_PAIRS,
            prot_min_pairs=PROT_MIN_PAIRS,
            low=POS_RATE_LOW,
            high=POS_RATE_HIGH,
        )

        print(
            f"[balance] after filter rows={len(filtered_rows):,} "
            f"good_ligs={len(good_ligs):,} good_prots={len(good_prots):,}"
        )

        if not filtered_rows:
            print("[balance] no rows after filtering; stop")
            current_rows = filtered_rows
            break

        # 6b) balance by ligand first
        balanced_lig = balance_rows_within_entity(
            filtered_rows,
            key="lig_tok",
            seed=SPLIT_SEED + 100 + it,
        )
        print(f"[balance] after ligand balance rows={len(balanced_lig):,}")

        if not balanced_lig:
            print("[balance] no rows after ligand balancing; stop")
            current_rows = balanced_lig
            break

        # 6c) then balance by protein
        balanced_prot = balance_rows_within_entity(
            balanced_lig,
            key="seq",
            seed=SPLIT_SEED + 200 + it,
        )
        print(f"[balance] after protein balance rows={len(balanced_prot):,}")

        if not balanced_prot:
            print("[balance] no rows after protein balancing; stop")
            current_rows = balanced_prot
            break

        # 6d) stop if converged
        if len(balanced_prot) == len(current_rows):
            current_rows = balanced_prot
            print("[balance] converged")
            break

        current_rows = balanced_prot

    if not current_rows:
        raise RuntimeError("No rows remain after balancing")

    # ------------------------------------------------------------
    # 7) final summaries
    # ------------------------------------------------------------
    after_counts = defaultdict(int)
    for row in current_rows:
        after_counts[row["split"]] += 1

    total_final = len(current_rows)
    print("\n[split] final row counts after balancing")
    for s in ["train", "valid", "test"]:
        frac = after_counts[s] / max(1, total_final)
        print(f"    {s}: rows={after_counts[s]:,} ({frac:.3%})")

    summarize_splits(current_rows)
    summarize_binary_balance(current_rows, "final")
    print_entity_pair_stats(current_rows, "final")

    # ------------------------------------------------------------
    # 8) write outputs
    # ------------------------------------------------------------
    write_split_csvs(current_rows)
    write_annotated_rows(current_rows, ROW_ANNOT_CSV)

def debug_tokenizer_once():
    with UNIQUE_SMILES_CSV.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        first = next(r)["smiles"]

    print("[debug] first smiles:", first)

    try:
        ids = encode_smiles_to_atom_tokens(first)
        print("[debug] ids:", ids[:20] if ids else ids)
        print("[debug] n_ids:", 0 if ids is None else len(ids))
    except Exception as e:
        print("[debug] tokenizer exception:", repr(e))
        raise

# ============================================================
# Main
# ============================================================

def main():
    # print("\n=== stage 1a: collect unique pdb ids ===")
    # unique_pdbids = collect_unique_bindingdb_pdbids(BINDINGDB_TSV, PDB_COL)
    #
    # print("\n=== stage 1b: build pdb chain cache ===")
    # build_pdb_chain_cache(unique_pdbids)
    #
    print("\n=== stage 2: clean + dedup bindingdb rows ===")
    pass1_clean_and_dedup()

    print("\n=== stage 3: tokenize unique smiles ===")
    pass2_make_smiles_token_map_parallel()

    print("\n=== stage 4: join tokens ===")
    pass3_join_tokens()

    print("\n=== stage 5: balanced protein 40% cluster + Murcko scaffold double-cold split ===")
    stage4_make_double_cold_splits_balanced()


if __name__ == "__main__":
    main()