#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge existing Metz VQAtom, Morgan and continuous features into each cold split CSV.

No molecule inference or cold resplitting by default. Preserves original split
CSV in a .original.csv backup before replacing it. Can run repeatedly.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from rdkit import Chem

HERE = Path(__file__).resolve().parent
COLDS = ('protein_cold', 'ligand_cold', 'double_cold')
PARTS = ('train', 'valid', 'test')
FEATURE_COLS = ('lig_tok', 'morgan_tok', 'edge_index', 'atom_feat')


def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        raise ValueError(f'Invalid SMILES: {s}')
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def choose(row, *names):
    for name in names:
        val = row.get(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def load_features(out):
    sources = {
        'vqatom': out / 'metz_dti_vqatom.csv',
        'morgan': out / 'metz_dti_morgan.csv',
    }
    maps = {}
    for mode, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; run create_metz_features.py first')
        fields, rows = read_csv(path)
        if not rows:
            raise ValueError(f'Empty feature file: {path}')
        if not {'lig_tok', 'edge_index', 'atom_feat'}.issubset(fields):
            raise ValueError(f'Missing feature columns in {path}: {fields}')
        mapping = {}
        for row in rows:
            smi = choose(row, 'smiles', 'canonical_smiles', 'SMILES')
            if not smi:
                raise ValueError(f'Missing SMILES in {path}')
            key = canonical(smi)
            vals = (row['lig_tok'], row['edge_index'], row['atom_feat'])
            if key in mapping and mapping[key] != vals:
                raise ValueError(f'Conflicting features for {key} in {path}')
            mapping[key] = vals
        maps[mode] = mapping
    if set(maps['vqatom']) != set(maps['morgan']):
        raise ValueError('VQAtom and Morgan SMILES sets differ')
    combined = {}
    for smi in maps['vqatom']:
        vq, edge, feat = maps['vqatom'][smi]
        mg, mg_edge, mg_feat = maps['morgan'][smi]
        if edge != mg_edge or feat != mg_feat:
            raise ValueError(f'Graph/atom features disagree for {smi}')
        n = Chem.MolFromSmiles(smi).GetNumAtoms()
        if len(vq.split()) != n or len(mg.split()) != n or len(json.loads(feat)) != n:
            raise ValueError(f'Atom/token count mismatch for {smi}')
        combined[smi] = dict(lig_tok=vq, morgan_tok=mg, edge_index=edge, atom_feat=feat)
    return combined


def atomic_write(path, fields, rows):
    fd, tmp = tempfile.mkstemp(prefix='.metz_tmp_', suffix='.csv', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=HERE)
    ap.add_argument('--module-dir', type=Path, default=HERE.parents[1],
                    help='Used only with --generate-features')
    ap.add_argument('--input', type=Path, default=HERE / 'metz_dti_clean.csv')
    ap.add_argument('--generate-features', action='store_true',
                    help='Recompute feature CSVs first (normally unnecessary)')
    ap.add_argument('--remove-mode-dirs', action='store_true',
                    help='Delete old vqatom/morgan/continuous split folders after success')
    args = ap.parse_args()
    out = args.out_dir.expanduser().resolve()
    if args.generate_features:
        subprocess.run([sys.executable, str(out / 'create_metz_features.py'),
                        '--input', str(args.input), '--out-dir', str(out),
                        '--module-dir', str(args.module_dir)], check=True)
    features = load_features(out)
    print(f'[features] {len(features)} unique molecules loaded', flush=True)
    vocab = out / 'metz_morgan_vocab.csv'
    if not vocab.is_file():
        raise FileNotFoundError(f'Missing Morgan vocabulary: {vocab}')

    split_paths = [out / f'metz_{cold}_3splits' / f'split_{i}' /
                   f'metz_{cold}_split_{i}_{part}.csv'
                   for cold in COLDS for i in range(1, 4) for part in PARTS]
    missing = [p for p in split_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing original split CSV(s):\n' + '\n'.join(map(str, missing)))

    # Validate every split before changing any file.
    planned = []
    for path in split_paths:
        fields, rows = read_csv(path)
        if not rows:
            raise ValueError(f'Empty split: {path}')
        output_fields = list(dict.fromkeys(fields + ['smiles', 'seq', 'y'] + list(FEATURE_COLS)))
        prepared = []
        for row in rows:
            smi = choose(row, 'canonical_smiles', 'SMILES', 'smiles')
            if not smi:
                raise ValueError(f'Missing SMILES in {path}')
            key = canonical(smi)
            if key not in features:
                raise ValueError(f'Missing features for {key} in {path}')
            seq = choose(row, 'seq', 'protein_sequence', 'sequence', 'target_sequence')
            y = choose(row, 'y', 'pKi', 'pki', 'label')
            if not seq or y is None:
                raise ValueError(f'Missing seq/y in {path}; columns={fields}')
            merged = dict(row)
            merged.update(features[key])
            merged['smiles'] = key
            merged['seq'] = seq
            merged['y'] = y
            prepared.append(merged)
        planned.append((path, fields, output_fields, prepared))

    # Also create a combined full dataset from the VQAtom full CSV.
    full_fields, full_rows = read_csv(out / 'metz_dti_vqatom.csv')
    full_out_fields = list(dict.fromkeys(full_fields + list(FEATURE_COLS)))
    full_prepared = []
    for row in full_rows:
        smi = choose(row, 'smiles', 'canonical_smiles', 'SMILES')
        merged = dict(row)
        merged.update(features[canonical(smi)])
        full_prepared.append(merged)
    full_path = out / 'metz_dti_all_features.csv'
    atomic_write(full_path, full_out_fields, full_prepared)
    print(f'[saved] {full_path.name}: {len(full_prepared)} rows', flush=True)

    for path, original_fields, output_fields, prepared in planned:
        backup = path.with_name(path.stem + '.original.csv')
        if not backup.exists():
            if any(c in original_fields for c in FEATURE_COLS):
                print(f'[warning] {path} already has features; original unmodified CSV cannot be recovered automatically')
            else:
                shutil.copy2(path, backup)
        atomic_write(path, output_fields, prepared)
        print(f'[saved] {path.relative_to(out)}: {len(prepared)} rows', flush=True)

    if args.remove_mode_dirs:
        for cold in COLDS:
            for i in range(1, 4):
                parent = out / f'metz_{cold}_3splits' / f'split_{i}'
                for mode in ('vqatom', 'morgan', 'continuous'):
                    folder = parent / mode
                    if folder.is_dir():
                        shutil.rmtree(folder)
                        print(f'[removed] {folder.relative_to(out)}', flush=True)

    print('\n[done] 27 unified split CSVs + 1 full dataset CSV')
    print('[vocab]', vocab)
    print('[note] lig_tok=VQAtom; morgan_tok=Morgan; atom_feat=Continuous input')
    print('[note] Cold split membership and labels were preserved')
    print('[note] Existing training code must select morgan_tok for Morgan mode')


if __name__ == '__main__':
    main()
