#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command Metz features + 3 cold types x 3 splits x 3 model CSVs.

Place this file alongside create_metz_features.py and create_metz_cold_splits.py.
Existing cold split files are reused (never rewritten) by default.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'metz_dti_clean.csv'
MODES = ('vqatom', 'morgan', 'continuous')
COLDS = ('protein_cold', 'ligand_cold', 'double_cold')
PARTS = ('train', 'valid', 'test')


def canonical(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def run(cmd):
    print('[run]', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--module-dir', type=Path, default=HERE.parents[1])
    p.add_argument('--features-script', type=Path, default=HERE / 'create_metz_features.py')
    p.add_argument('--splits-script', type=Path, default=HERE / 'create_metz_cold_splits.py')
    p.add_argument('--regenerate-splits', action='store_true', help='Rerun cold splitting (requires MMseqs2)')
    p.add_argument('--positive-threshold', type=float, default=6.0)
    p.add_argument('--mmseqs', default='mmseqs')
    p.add_argument('--skip-feature-generation', action='store_true')
    a = p.parse_args()
    source = a.input.expanduser().resolve()
    out = (a.out_dir or source.parent).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_feature_generation:
        run([sys.executable, a.features_script, '--input', source,
             '--out-dir', out, '--module-dir', a.module_dir])

    feature_maps = {}
    for mode, fname in [('vqatom', 'metz_dti_vqatom.csv'), ('morgan', 'metz_dti_morgan.csv')]:
        _, rows = read_csv(out / fname)
        if not rows:
            raise ValueError(f'No rows in {fname}')
        mapping = {}
        for row in rows:
            key = canonical(row['smiles'])
            feat = {k: row[k] for k in ('lig_tok', 'edge_index', 'atom_feat')}
            if key in mapping and mapping[key] != feat:
                raise ValueError(f'Conflicting {mode} features for {key}')
            mapping[key] = feat
        feature_maps[mode] = mapping
    # Continuous uses the same graph/atom features; lig_tok is required by the
    # existing DTI dataset parser, even though continuous GIN uses atom_feat.
    feature_maps['continuous'] = feature_maps['vqatom']

    split_paths = [out / f'metz_{cold}_3splits' / f'split_{i}' /
                   f'metz_{cold}_split_{i}_{part}.csv'
                   for cold in COLDS for i in range(1, 4) for part in PARTS]
    missing = [path for path in split_paths if not path.is_file()]
    if a.regenerate_splits or missing:
        if missing:
            print(f'[split] {len(missing)} source split CSVs missing; generating all splits')
        run([sys.executable, a.splits_script, '--input', source,
             '--out', out, '--positive-threshold', a.positive_threshold,
             '--mmseqs', a.mmseqs])

    totals = {mode: 0 for mode in MODES}
    for cold in COLDS:
        for i in range(1, 4):
            for part in PARTS:
                source_split = out / f'metz_{cold}_3splits' / f'split_{i}' / f'metz_{cold}_split_{i}_{part}.csv'
                fields, rows = read_csv(source_split)
                if not rows:
                    raise ValueError(f'Empty split: {source_split}')
                for mode in MODES:
                    dest = source_split.parent / mode / source_split.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    output_fields = list(dict.fromkeys(fields + ['seq', 'smiles', 'y', 'lig_tok', 'edge_index', 'atom_feat']))
                    prepared = []
                    for row in rows:
                        smi = row.get('canonical_smiles') or row.get('SMILES') or row.get('smiles')
                        if not smi:
                            raise ValueError(f'Missing SMILES in {source_split}')
                        key = canonical(smi)
                        try:
                            features = feature_maps[mode][key]
                        except KeyError as e:
                            raise ValueError(f'No {mode} features for {key}') from e
                        r = dict(row)
                        r.update(features)
                        r['smiles'] = key
                        r['seq'] = row.get('seq') or row.get('protein_sequence')
                        r['y'] = row.get('y') or row.get('pKi')
                        if not r['seq'] or not r['y']:
                            raise ValueError(f'Missing seq/y in {source_split}')
                        n = Chem.MolFromSmiles(key).GetNumAtoms()
                        if len(r['lig_tok'].split()) != n or len(json.loads(r['atom_feat'])) != n:
                            raise ValueError(f'Atom/token count mismatch: {key}')
                        prepared.append(r)
                    with dest.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(prepared)
                    totals[mode] += len(prepared)
                    print(f'[saved] {dest.relative_to(out)} rows={len(prepared)}', flush=True)
    print('\n[done] 81 CSVs (3 feature modes x 3 cold modes x 3 splits x 3 partitions)')
    print('[vocab]', out / 'metz_morgan_vocab.csv')
    print('[note] Continuous lig_tok is VQAtom placeholder; graph/atom features are shared.')
    print('[note] Morgan uses global Metz vocabulary; no labels are used to construct it.')
    print('[note] Existing source cold splits were preserved unless --regenerate-splits was specified.')


if __name__ == '__main__':
    main()
