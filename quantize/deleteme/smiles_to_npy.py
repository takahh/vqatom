#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdchem

np.set_printoptions(threshold=np.inf)

# ============================================================
# Repro
# ============================================================
SEED_SAMPLE = 7
random.seed(SEED_SAMPLE)

# ============================================================
# INPUT (final discretize SMILES)
# ============================================================
DISCRETIZE_SMI = "/Users/mac/Documents/transformer_tape_dnabert/2024NEW/data_prep/final_splits_200k/discretize.noleak.final.smi"

# ============================================================
# OUTPUT
# ============================================================
SAVE_DIR_DISC = "/Volumes/Untitled/2024NEW/discret_final/"
os.makedirs(SAVE_DIR_DISC, exist_ok=True)

# ============================================================
# Params
# ============================================================
MAX_ATOMS = 100   # smiles stage filter: heavy atoms < MAX_ATOMS
MAX_SIZE  = 100   # padding size: must be >= max atoms used
BATCH_SIZE = 1000

# ============================================================
# SMILES loader (robust for .smi / csv-ish)
# ============================================================
def _extract_smiles(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None

    low = s.lower()
    # crude header skip
    if low.startswith("smiles") or low.startswith("chembl_id") or low.startswith("id,") or low.startswith("id\t"):
        return None

    # CSV-like: id,smiles or smiles,id
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        # heuristic: pick token that looks like smiles
        for tok in parts[:2]:
            if any(c in tok for c in "#[]=()@+\\/-") or any(ch.isalpha() for ch in tok):
                return tok
        return parts[0]

    # .smi: "SMILES [whitespace] name"
    return s.split()[0]

def get_smiles(ipath: str) -> List[str]:
    smiles = []
    with open(ipath, "r") as f:
        for line in f:
            smi = _extract_smiles(line)
            if smi:
                smiles.append(smi)
    return smiles

# ============================================================
# Size filter at SMILES stage
# ============================================================
def smiles_ok_size(smi: str, max_atoms: int) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return m.GetNumAtoms() < max_atoms  # heavy atoms

# ============================================================
# Label (atomic number)
# ============================================================
def assign_node_classes(atom: Chem.Atom) -> int:
    return atom.GetAtomicNum()

# ============================================================
# Functional group flags
# ============================================================
def extract_functional_groups(mol: Chem.Mol) -> np.ndarray:
    func_smarts = {
        "Hydroxyl": "[OX2H]",
        "Alcohol": "[CX4][OX2H]",
        "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
        "Amide": "C(=O)N",
        "Carboxyl": "C(=O)[OX2H1,O-]",
        "Carbonyl": "[CX3]=O",
        "Aldehyde": "[CX3H1](=O)[#6]",
        "Ketone": "[CX3](=O)[#6]",
        "Ester": "C(=O)O[#6]",
        "Ether": "[OD2]([#6])[#6]",
        "Phenol": "c[OH]",
        "Aromatic": "a",
        "HeteroAromaticN": "n",
        "Halogen": "[F,Cl,Br,I]",
        "Sulfonyl": "S(=O)(=O)[N,O]",
        "Thiol": "[SX2H]",
        "Disulfide": "S-S",
        "Phosphate": "P(=O)(O)(O)O",
    }

    flags = []
    n_atoms = mol.GetNumAtoms()
    for smarts in func_smarts.values():
        patt = Chem.MolFromSmarts(smarts)
        atom_flag = np.zeros(n_atoms, dtype=float)
        if patt is not None:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                for idx in match:
                    atom_flag[idx] = 1.0
        flags.append(atom_flag)

    return np.stack(flags, axis=1)  # (N, n_func)

# ============================================================
# H-bond donor/acceptor flags
# ============================================================
def compute_hbond_flags(mol: Chem.Mol) -> np.ndarray:
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    feats = factory.GetFeaturesForMol(mol)
    n = mol.GetNumAtoms()

    donor_flag = np.zeros(n, dtype=float)
    acceptor_flag = np.zeros(n, dtype=float)

    for f in feats:
        fam = f.GetFamily()
        if fam == "Donor":
            for aidx in f.GetAtomIds():
                donor_flag[aidx] = 1.0
        elif fam == "Acceptor":
            for aidx in f.GetAtomIds():
                acceptor_flag[aidx] = 1.0

    return np.stack([donor_flag, acceptor_flag], axis=1)  # (N, 2)

# ============================================================
# Ring-related features
# ============================================================
def compute_ring_features(mol: Chem.Mol):
    n_atoms = mol.GetNumAtoms()
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())

    ring_size = np.zeros(n_atoms, dtype=np.int32)
    for ring in atom_rings:
        size = len(ring)
        for aidx in ring:
            if ring_size[aidx] == 0 or size < ring_size[aidx]:
                ring_size[aidx] = size

    arom_nbrs = np.zeros(n_atoms, dtype=np.int32)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        arom_nbrs[i] = sum(1 for nbr in atom.GetNeighbors() if nbr.GetIsAromatic())

    fused_id = np.zeros(n_atoms, dtype=np.int32)
    if atom_rings:
        ring_sets = [set(r) for r in atom_rings]
        n_rings = len(ring_sets)
        ring_comp = [-1] * n_rings
        current_comp = 1

        for i in range(n_rings):
            if ring_comp[i] != -1:
                continue
            stack = [i]
            ring_comp[i] = current_comp
            while stack:
                k = stack.pop()
                for j in range(n_rings):
                    if ring_comp[j] != -1:
                        continue
                    if ring_sets[k] & ring_sets[j]:
                        ring_comp[j] = current_comp
                        stack.append(j)
            current_comp += 1

        for r_idx, ring in enumerate(atom_rings):
            comp_id = ring_comp[r_idx]
            for aidx in ring:
                if fused_id[aidx] == 0:
                    fused_id[aidx] = comp_id

    return ring_size, arom_nbrs, fused_id

# ============================================================
# 2-hop hetero code (het27)
# ============================================================
HETERO_TAG = {
    7:  "N",
    8:  "O",
    16: "R",
    9:  "R",
    17: "R",
    35: "R",
}

def _bucket2(x: int) -> int:
    if x <= 0:
        return 0
    if x == 1:
        return 1
    return 2

def compute_het27(adj_matrix: np.ndarray, Z_arr: np.ndarray) -> np.ndarray:
    N = adj_matrix.shape[0]
    het_codes = np.zeros(N, dtype=np.int32)

    for center in range(N):
        nbr1 = np.nonzero(adj_matrix[center])[0]

        nbr2 = set()
        for j in nbr1:
            for k in np.nonzero(adj_matrix[j])[0]:
                if k == center or k in nbr1:
                    continue
                nbr2.add(k)

        cntN = cntO = cntR = 0
        for k in nbr2:
            z = int(Z_arr[k])
            tag = HETERO_TAG.get(z, None)
            if tag == "N":
                cntN += 1
            elif tag == "O":
                cntO += 1
            elif tag == "R":
                cntR += 1

        bN = _bucket2(cntN)
        bO = _bucket2(cntO)
        bR = _bucket2(cntR)
        het_codes[center] = int(bN + 3 * bO + 9 * bR)

    return het_codes

# ============================================================
# Bond feature (24d)
# ============================================================
def encode_bond_feature(bond: rdchem.Bond) -> np.ndarray:
    feat = np.zeros(24, dtype=float)

    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE:
        feat[0] = 1.0
    elif bt == rdchem.BondType.DOUBLE:
        feat[1] = 1.0
    elif bt == rdchem.BondType.TRIPLE:
        feat[2] = 1.0
    elif bt == rdchem.BondType.AROMATIC:
        feat[3] = 1.0
    elif bt == rdchem.BondType.DATIVE:
        feat[4] = 1.0
    else:
        feat[5] = 1.0

    feat[6] = float(bond.GetIsAromatic())
    feat[7] = float(bond.GetIsConjugated())
    feat[8] = float(bond.IsInRing())

    ring_bucket = np.zeros(5, dtype=float)
    if bond.IsInRing():
        if bond.IsInRingSize(3):
            ring_bucket[0] = 1.0
        elif bond.IsInRingSize(4):
            ring_bucket[1] = 1.0
        elif bond.IsInRingSize(5):
            ring_bucket[2] = 1.0
        elif bond.IsInRingSize(6):
            ring_bucket[3] = 1.0
        else:
            ring_bucket[4] = 1.0
    feat[9:14] = ring_bucket

    stereo_oh = np.zeros(5, dtype=float)
    st = bond.GetStereo()
    if st == rdchem.BondStereo.STEREONONE:
        stereo_oh[0] = 1.0
    elif st == rdchem.BondStereo.STEREOCIS:
        stereo_oh[1] = 1.0
    elif st == rdchem.BondStereo.STEREOTRANS:
        stereo_oh[2] = 1.0
    elif st in (rdchem.BondStereo.STEREOE, rdchem.BondStereo.STEREOZ):
        stereo_oh[3] = 1.0
    else:
        stereo_oh[4] = 1.0
    feat[14:19] = stereo_oh

    dir_oh = np.zeros(5, dtype=float)
    bd = bond.GetBondDir()
    if bd == rdchem.BondDir.NONE:
        dir_oh[0] = 1.0
    elif bd in (rdchem.BondDir.BEGINWEDGE, rdchem.BondDir.ENDUPRIGHT):
        dir_oh[1] = 1.0
    elif bd in (rdchem.BondDir.BEGINDASH, rdchem.BondDir.ENDDOWNRIGHT):
        dir_oh[2] = 1.0
    elif bd == rdchem.BondDir.EITHERDOUBLE:
        dir_oh[3] = 1.0
    else:
        dir_oh[4] = 1.0
    feat[19:24] = dir_oh

    return feat

# ============================================================
# Bond env sum+max (48d)
# ============================================================
def compute_bond_env_raw(mol: Chem.Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    D = 24
    sum_feats = np.zeros((n_atoms, D), dtype=float)
    max_feats = np.zeros((n_atoms, D), dtype=float)

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        bond_feats = [encode_bond_feature(b) for b in atom.GetBonds()]
        if bond_feats:
            raw = np.stack(bond_feats, axis=0)
            sum_feats[i] = raw.sum(axis=0)
            max_feats[i] = raw.max(axis=0)

    return np.concatenate([sum_feats, max_feats], axis=1)  # (N, 48)

# ============================================================
# Feature vector (base + bondenv)
# ============================================================
def atom_to_feature_vector(atom: Chem.Atom,
                           func_flags: np.ndarray,
                           hbond_flags: np.ndarray,
                           ring_size: np.ndarray,
                           arom_nbrs: np.ndarray,
                           fused_id: np.ndarray,
                           hcount: int,
                           bond_env_raw: np.ndarray,
                           het27: np.ndarray) -> np.ndarray:
    idx = atom.GetIdx()

    base = np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        hcount,
        *func_flags[idx],
        *hbond_flags[idx],
        ring_size[idx],
        arom_nbrs[idx],
        fused_id[idx],
        het27[idx],
    ], dtype=float)

    return np.concatenate([base, bond_env_raw[idx]], axis=0)

# ============================================================
# SMILES -> (adj, features, labels)
# ============================================================
def smiles_to_graph_with_labels(smiles: str, idx: int):
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None:
        raise ValueError("Invalid SMILES at idx=%d: %s" % (idx, smiles))

    # Count Hs using AddHs (align heavy atom indices)
    mol_with_H = Chem.AddHs(base_mol)
    n_heavy = base_mol.GetNumAtoms()
    hcount_list = [mol_with_H.GetAtomWithIdx(i).GetTotalNumHs() for i in range(n_heavy)]

    mol = base_mol
    Chem.SanitizeMol(mol)
    num_atoms = mol.GetNumAtoms()

    # adjacency with bond order bucket
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            v = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            v = 2
        elif bt == Chem.rdchem.BondType.TRIPLE:
            v = 3
        elif bt == Chem.rdchem.BondType.AROMATIC:
            v = 4
        else:
            v = 1
        adj_matrix[i, j] = v
        adj_matrix[j, i] = v

    func_flags = extract_functional_groups(mol)
    hbond_flags = compute_hbond_flags(mol)
    ring_size, arom_nbrs, fused_id = compute_ring_features(mol)

    Z_arr = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    het27 = compute_het27(adj_matrix, Z_arr)

    bond_env_raw = compute_bond_env_raw(mol)

    feature_list = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        fv = atom_to_feature_vector(
            atom,
            func_flags,
            hbond_flags,
            ring_size,
            arom_nbrs,
            fused_id,
            hcount=hcount_list[i],
            bond_env_raw=bond_env_raw,
            het27=het27,
        )
        feature_list.append(fv)

    atom_features = np.vstack(feature_list).astype(np.float32)
    atom_labels = np.array([assign_node_classes(a) for a in mol.GetAtoms()], dtype=np.int32)

    return adj_matrix, atom_features, atom_labels

# ============================================================
# pad utils
# ============================================================
def pad_adj_matrix(matrix: np.ndarray, max_size: int) -> np.ndarray:
    padded = np.zeros((max_size, max_size), dtype=matrix.dtype)
    padded[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded

def pad_attr_matrix(matrix: np.ndarray, max_size: int, feat_dim: Optional[int] = None, pad_value: float = 0.0) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected 2D feature matrix")
    N, D = matrix.shape
    target_D = D if feat_dim is None else feat_dim
    padded = np.full((max_size, target_D), pad_value, dtype=matrix.dtype)
    padded[:N, :D] = matrix
    return padded

# ============================================================
# save batch
# ============================================================
def save_batch(adj_list, feat_list, smi_list, save_dir, idx):
    adj_p = [pad_adj_matrix(a, MAX_SIZE) for a in adj_list]
    feat_p = [pad_attr_matrix(f, MAX_SIZE) for f in feat_list]

    np.save(os.path.join(save_dir, "adj_%d.npy" % idx), np.stack(adj_p))
    np.save(os.path.join(save_dir, "attr_%d.npy" % idx), np.stack(feat_p))

    with open(os.path.join(save_dir, "smiles_%d.txt" % idx), "w") as f:
        for s in smi_list:
            f.write(s + "\n")

# ============================================================
# main batch runner (DISCRETIZE only)
# ============================================================
def run_discretize_only():
    rng = random.Random(SEED_SAMPLE)

    print("Loading SMILES:", DISCRETIZE_SMI)
    smiles_all = get_smiles(DISCRETIZE_SMI)
    print("raw lines:", len(smiles_all))

    # (optional) shuffle so batches are not ordered
    rng.shuffle(smiles_all)

    # size filter
    smiles_filt = [s for s in smiles_all if smiles_ok_size(s, MAX_ATOMS)]
    print("size filtered:", len(smiles_filt))

    # process & save
    adj_mats, attr_mats, smi_buf = [], [], []
    batch_idx, used, skipped = 0, 0, 0

    for idx, smi in enumerate(smiles_filt):
        try:
            adj, feat, _ = smiles_to_graph_with_labels(smi, idx)
        except Exception:
            skipped += 1
            continue

        if adj.shape[0] >= MAX_SIZE:
            skipped += 1
            continue

        used += 1
        adj_mats.append(adj)
        attr_mats.append(feat)
        smi_buf.append(smi)

        if used % BATCH_SIZE == 0:
            save_batch(adj_mats, attr_mats, smi_buf, SAVE_DIR_DISC, batch_idx)
            adj_mats, attr_mats, smi_buf = [], [], []
            batch_idx += 1
            print("saved batch", batch_idx, "used", used, "skipped", skipped)

    # remainder
    if adj_mats:
        save_batch(adj_mats, attr_mats, smi_buf, SAVE_DIR_DISC, batch_idx)
        batch_idx += 1

    print("\n===== SUMMARY [DISCRETIZE_FINAL] =====")
    print("input         :", len(smiles_all))
    print("size_filtered :", len(smiles_filt))
    print("used          :", used)
    print("skipped       :", skipped)
    print("batches       :", batch_idx)
    print("save_dir      :", SAVE_DIR_DISC)
    print("=====================================\n")

if __name__ == "__main__":
    run_discretize_only()
