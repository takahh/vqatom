#!/usr/bin/env python3
"""Audit Train/Test near-duplicate ligands and matching activity for KIBA ligand-cold splits.

Examples:
 python audit_kiba_ligand_similarity.py --root /path/kiba_ligand_cold_3splits --label-col label
 python audit_kiba_ligand_similarity.py --root /path/kiba_ligand_cold_3splits --label-col y --positive-threshold 7

Label semantics must be chosen to match the actual KIBA training pipeline.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

THRESHOLDS = (0.8, 0.9, 0.95)

def column(fields, requested, choices):
    if requested:
        if requested not in fields:
            raise ValueError(f'Column {requested!r} missing; available: {fields}')
        return requested
    for name in choices:
        if name in fields:
            return name
    raise ValueError(f'No matching column among {choices}; available: {fields}')

def load(path, args):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        smi = column(fields, args.smiles_col, ['smiles', 'SMILES', 'canonical_smiles'])
        prot = column(fields, args.protein_col, ['uniprot_id', 'protein_id', 'target_id', 'seq', 'protein_sequence'])
        label = column(fields, args.label_col, ['label', 'y', 'Y'])
        rows = []
        for i, row in enumerate(reader, 2):
            s = row[smi]
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f'Invalid SMILES at {path}:{i}: {s}')
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            value = float(row[label])
            if args.positive_threshold is None:
                if value not in (0.0, 1.0):
                    raise ValueError(f'{label} is not binary at {path}:{i} ({value}). Supply --positive-threshold matching your training definition.')
                activity = int(value)
            else:
                activity = int(value >= args.positive_threshold)
            rows.append({'smiles': canonical, 'protein': row[prot], 'label': activity, 'value': value})
    return rows, {'smiles': smi, 'protein': prot, 'label': label}

def find_splits(root):
    folders = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('split')])
    if not folders:
        folders = [root]
    for folder in folders:
        files = list(folder.glob('*.csv'))
        # Accept either train.csv or kiba_ligand_cold_split_1_train.csv.
        train = [p for p in files if p.stem == 'train' or p.stem.endswith('_train')]
        test = [p for p in files if p.stem == 'test' or p.stem.endswith('_test')]
        if len(train) != 1 or len(test) != 1:
            raise ValueError(f'Expected one train and test CSV in {folder}; found {train}, {test}')
        yield folder.name, train[0], test[0]

def audit(name, train_path, test_path, args, gen, out):
    train, train_cols = load(train_path, args)
    test, test_cols = load(test_path, args)
    fingerprints = {}
    def fp(s):
        if s not in fingerprints:
            fingerprints[s] = gen.GetFingerprint(Chem.MolFromSmiles(s))
        return fingerprints[s]
    # Group train by protein, deduplicating identical ligand/label records.
    by_protein = defaultdict(dict)
    for r in train:
        key = (r['smiles'], r['label'])
        by_protein[r['protein']][key] = r
    indexed = {}
    for protein, group in by_protein.items():
        entries = list(group.values())
        indexed[protein] = (entries, [fp(r['smiles']) for r in entries])
    counters = {t: {'similar': 0, 'same_label': 0, 'opposite_label': 0} for t in THRESHOLDS}
    same_ligand = 0
    nearest_rows = []
    for i, r in enumerate(test):
        candidates = indexed.get(r['protein'])
        if not candidates:
            nearest_rows.append({'test_index': i, 'protein': r['protein'], 'test_smiles': r['smiles'], 'test_label': r['label'], 'nearest_similarity': '', 'nearest_train_smiles': '', 'nearest_train_label': '', 'nearest_label_match': '', **{f'any_same_label_ge_{t}': 0 for t in THRESHOLDS}, **{f'any_opposite_label_ge_{t}': 0 for t in THRESHOLDS}})
            continue
        entries, fps = candidates
        sims = DataStructs.BulkTanimotoSimilarity(fp(r['smiles']), fps)
        best_idx = max(range(len(sims)), key=sims.__getitem__)
        best = entries[best_idx]
        if any(e['smiles'] == r['smiles'] for e in entries):
            same_ligand += 1
        record = {'test_index': i, 'protein': r['protein'], 'test_smiles': r['smiles'], 'test_label': r['label'], 'nearest_similarity': round(sims[best_idx], 6), 'nearest_train_smiles': best['smiles'], 'nearest_train_label': best['label'], 'nearest_label_match': int(best['label'] == r['label'])}
        for t in THRESHOLDS:
            matching = any(sim >= t and e['label'] == r['label'] for sim, e in zip(sims, entries))
            opposite = any(sim >= t and e['label'] != r['label'] for sim, e in zip(sims, entries))
            similar = matching or opposite
            counters[t]['similar'] += int(similar)
            counters[t]['same_label'] += int(matching)
            counters[t]['opposite_label'] += int(opposite)
            record[f'any_same_label_ge_{t}'] = int(matching)
            record[f'any_opposite_label_ge_{t}'] = int(opposite)
        nearest_rows.append(record)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'{name}_nearest_train_per_test.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(nearest_rows[0]) if nearest_rows else [])
        writer.writeheader()
        writer.writerows(nearest_rows)
    n = len(test)
    report = {'split': name, 'train_rows': len(train), 'test_rows': n, 'test_rows_with_same_ligand_and_protein': same_ligand, 'columns_train': train_cols, 'columns_test': test_cols, 'thresholds': {str(t): {**v, **{f'{k}_pct_of_test': round(100 * x / n, 3) for k, x in v.items()}} for t, v in counters.items()}}
    print(f'\n{name}: train={len(train)} test={n}, exact ligand+protein overlap={same_ligand}')
    for t, v in counters.items():
        print(f'  >= {t}: similar={v["similar"]} ({100*v["similar"]/n:.2f}% test), same label={v["same_label"]} ({100*v["same_label"]/n:.2f}%), opposite label={v["opposite_label"]} ({100*v["opposite_label"]/n:.2f}%)')
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--smiles-col')
    p.add_argument('--protein-col')
    p.add_argument('--label-col')
    p.add_argument('--positive-threshold', type=float, help='Required when target column contains continuous values. Must match training label definition.')
    args = p.parse_args()
    out = args.out or args.root / 'similarity_audit'
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reports = [audit(name, train, test, args, gen, out) for name, train, test in find_splits(args.root)]
    (out / 'summary.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print('\nSaved:', out / 'summary.json')

if __name__ == '__main__':
    main()
