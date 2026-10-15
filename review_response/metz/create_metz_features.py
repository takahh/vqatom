#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare Metz VQAtom, dense atom-wise Morgan and Continuous input CSVs.

VQAtom uses the existing vqatom_module. Morgan IDs are remapped globally
across Metz to 1..N (PAD=0, CLS=N+1). Save a shared vocabulary CSV.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

DEFAULT_INPUT = Path('/Users/taka/PycharmProjects/vqatom/review_response/metz/metz_dti_clean.csv')


def graph_features(mol):
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend((i, j)); dst.extend((j, i))
    rings = mol.GetRingInfo().AtomRings()
    features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        ring_size = next((len(r) for r in rings if idx in r), 0)
        features.append([atom.GetAtomicNum(), atom.GetFormalCharge(),
                         int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                         int(atom.IsInRing()), atom.GetDegree(), atom.GetTotalNumHs(),
                         atom.GetTotalValence(), ring_size, float(atom.GetMass()) / 200.0])
    return (json.dumps([src, dst], separators=(',', ':')),
            json.dumps(features, separators=(',', ':')))


def atomwise_morgan(mol, generator):
    """One sparse (unfolded) Morgan environment ID per atom, max radius <=2.

    For each center choose the radius-2 environment where available, otherwise
    the largest available radius. This avoids losing terminal/small-molecule atoms.
    """
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=ao)
    selected = [None] * mol.GetNumAtoms()
    for bit, centers in ao.GetBitInfoMap().items():
        for atom_idx, radius in centers:
            current = selected[atom_idx]
            candidate = (int(radius), int(bit))
            if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
                selected[atom_idx] = candidate
    if any(x is None for x in selected):
        raise ValueError('No Morgan environment for at least one atom')
    return [bit for _, bit in selected]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--module-dir', type=Path, default=None)
    ap.add_argument('--smiles-col', default=None)
    args = ap.parse_args()
    if args.module_dir:
        sys.path.insert(0, str(args.module_dir.expanduser().resolve()))
    try:
        from vqatom_module import encode_smiles_to_atom_tokens
    except ImportError as exc:
        raise SystemExit('Cannot import vqatom_module; specify --module-dir') from exc

    source = args.input.expanduser().resolve()
    out_dir = (args.out_dir or source.parent).expanduser().resolve()
    with source.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        smiles_col = args.smiles_col or next((x for x in ('smiles','canonical_smiles','Canonical_Smiles','SMILES') if x in fields), None)
        if smiles_col not in fields:
            raise SystemExit(f'SMILES column not found: {fields}')
        rows = list(reader)
    if not rows:
        raise SystemExit('Empty input CSV')
    unique = sorted({str(r[smiles_col]).strip() for r in rows})
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache, failures = {}, []
    for smiles in tqdm(unique, desc='VQAtom + Morgan + atom features'):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError('Invalid or empty molecule')
            vq = [int(v) for v in encode_smiles_to_atom_tokens(smiles)]
            mg = atomwise_morgan(mol, generator)
            n = mol.GetNumAtoms()
            if len(vq) != n or len(mg) != n:
                raise ValueError(f'Atom count {n}, VQ {len(vq)}, Morgan {len(mg)}')
            if any(v < 0 for v in vq):
                raise ValueError('Negative VQAtom ID')
            edge, feat = graph_features(mol)
            cache[smiles] = (' '.join(map(str,vq)), ' '.join(map(str,mg)), edge, feat)
        except Exception as exc:
            failures.append((smiles, f'{type(exc).__name__}: {exc}'))
    out_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        fail_path = out_dir / 'metz_feature_failures.csv'
        with fail_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['smiles','reason']); w.writerows(failures)
        raise SystemExit(f'{len(failures)} molecules failed; see {fail_path}. No output CSVs written.')

    raw_ids = sorted({int(mid) for _, mg, _, _ in cache.values() for mid in mg.split()})
    raw_to_dense = {raw: i + 1 for i, raw in enumerate(raw_ids)}
    vocab_path = out_dir / 'metz_morgan_vocab.csv'
    with vocab_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['raw_id', 'mapped_id'])
        w.writerows((raw, raw_to_dense[raw]) for raw in raw_ids)
    print(f'[saved] {vocab_path}: {len(raw_ids):,} IDs, PAD=0, CLS={len(raw_ids)+1}')

    extra = ['smiles', 'lig_tok', 'edge_index', 'atom_feat']
    output_fields = fields + [x for x in extra if x not in fields]
    for mode, filename in [('vqatom','metz_dti_vqatom.csv'), ('morgan','metz_dti_morgan.csv')]:
        path = out_dir / filename
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=output_fields, extrasaction='ignore'); w.writeheader()
            for row in rows:
                vq, mg, edge, feat = cache[str(row[smiles_col]).strip()]
                rr = dict(row)
                rr.update(smiles=str(row[smiles_col]).strip(), lig_tok=vq if mode=='vqatom' else ' '.join(str(raw_to_dense[int(x)]) for x in mg.split()),
                          edge_index=edge, atom_feat=feat)
                w.writerow(rr)
        print(f'[saved] {path} ({len(rows):,} rows)')
    print(f'[done] {len(unique):,} unique molecules; {len(rows):,} pairs; no failed molecules')
    print('[note] Morgan IDs are globally remapped across Metz; use the same vocabulary for all cold splits.')
    print('[note] Verify atomwise Morgan ID selection matches your original KIBA generator before model comparisons.')

if __name__ == '__main__':
    main()
