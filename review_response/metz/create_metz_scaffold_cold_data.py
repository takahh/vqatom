#!/usr/bin/env python3
"""Create three METZ Bemis-Murcko scaffold-cold splits with merged model features.

Inputs (same directory): metz_dti_clean.csv, metz_dti_vqatom.csv,
metz_dti_morgan.csv. Existing protein/ligand/double splits are untouched.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonical(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {s}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold(s):
    mol = Chem.MolFromSmiles(s)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def feature_table(path, kind):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'smiles' not in df or 'lig_tok' not in df:
        raise ValueError(f'{path}: missing smiles/lig_tok')
    df['_key'] = df.smiles.map(canonical)
    cols = ['lig_tok'] + [c for c in ('edge_index', 'atom_feat', 'morgan_atom_r2') if c in df]
    records = {}
    for row in df[['_key'] + cols].to_dict('records'):
        key = row.pop('_key')
        if key in records and records[key] != row:
            raise ValueError(f'{kind}: conflicting features for {key}')
        records[key] = row
    return records


def assign(df, seed):
    rng = np.random.default_rng(seed)
    counts = df.groupby('scaffold', sort=False).size()
    # Randomized tie order, largest scaffold groups first; group-disjoint split.
    order = rng.permutation(len(counts))
    order = sorted(order, key=lambda i: -int(counts.iloc[i]))
    loads = np.zeros(3, dtype=int)
    target = np.array([.8, .1, .1]) * len(df)
    mapping = {}
    for i in order:
        w = int(counts.iloc[i])
        # Select partition with largest deficit relative to its target.
        scores = (loads + w) / target
        candidates = np.flatnonzero(np.isclose(scores, scores.min()))
        j = int(rng.choice(candidates))
        mapping[counts.index[i]] = j
        loads[j] += w
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--positive-threshold', type=float, default=6.0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    clean = pd.read_csv(root / 'metz_dti_clean.csv')
    for c in ('SMILES', 'protein_sequence', 'pKi'):
        if c not in clean:
            raise ValueError(f'Clean METZ missing {c}')
    clean['_key'] = clean.SMILES.map(canonical)
    clean['scaffold'] = clean['_key'].map(scaffold)
    clean['seq'] = clean.protein_sequence
    clean['smiles'] = clean['_key']
    clean['y'] = clean.pKi
    clean['label'] = (clean.pKi >= args.positive_threshold).astype(int)
    vq = feature_table(root / 'metz_dti_vqatom.csv', 'VQAtom')
    mg = feature_table(root / 'metz_dti_morgan.csv', 'Morgan')
    for name, table in [('VQAtom', vq), ('Morgan', mg)]:
        missing = set(clean['_key']) - set(table)
        if missing:
            raise ValueError(f'{name}: missing {len(missing)} molecules, e.g. {next(iter(missing))}')
    clean['lig_tok'] = clean['_key'].map(lambda k: vq[k]['lig_tok'])
    clean['morgan_atom_r2_token'] = clean['_key'].map(lambda k: mg[k]['lig_tok'])
    for c in ('edge_index', 'atom_feat'):
        if c not in next(iter(vq.values())):
            raise ValueError(f'VQAtom features missing {c}')
        clean[c] = clean['_key'].map(lambda k: vq[k][c])
        if c in next(iter(mg.values())):
            bad = [k for k in set(clean['_key']) if vq[k][c] != mg[k][c]]
            if bad:
                raise ValueError(f'{c} differs between VQAtom/Morgan for {len(bad)} molecules')
    if 'morgan_atom_r2' in next(iter(mg.values())):
        clean['morgan_atom_r2'] = clean['_key'].map(lambda k: mg[k]['morgan_atom_r2'])
    outroot = root / 'metz_scaffold_cold_3splits'
    dests = [outroot / f'split_{i}' / f'metz_scaffold_cold_split_{i}_{part}.csv'
             for i in (1, 2, 3) for part in ('train', 'valid', 'test')]
    existing = [p for p in dests if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'{len(existing)} outputs already exist; use --overwrite only if intentional: {existing[0]}')
    columns = [c for c in clean.columns if c != '_key']
    summaries = []
    signatures = set()
    for split in (1, 2, 3):
        for trial in range(100):
            mapping = assign(clean, 1000 + split * 1000 + trial)
            signature = tuple(sorted(mapping.items()))
            if signature not in signatures:
                break
        else:
            raise RuntimeError('Could not produce 3 distinct scaffold splits')
        signatures.add(signature)
        clean['_partition'] = clean.scaffold.map(mapping)
        partitions = {}
        for idx, part in enumerate(('train', 'valid', 'test')):
            sub = clean.loc[clean['_partition'] == idx, columns].copy()
            if sub.empty:
                raise RuntimeError(f'Split {split}: empty {part}')
            partitions[part] = sub
        sets = [set(x.scaffold) for x in partitions.values()]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError('Scaffold leakage detected')
        for part, sub in partitions.items():
            dest = outroot / f'split_{split}' / f'metz_scaffold_cold_split_{split}_{part}.csv'
            dest.parent.mkdir(parents=True, exist_ok=True)
            sub.to_csv(dest, index=False)
            summaries.append({'split': split, 'partition': part, 'rows': len(sub),
                              'scaffolds': sub.scaffold.nunique(), 'positive_rate': float(sub.label.mean())})
            print(f'[saved] {dest} rows={len(sub)} scaffolds={sub.scaffold.nunique()}')
    pd.DataFrame(summaries).to_csv(outroot / 'scaffold_split_summary.csv', index=False)
    (outroot / 'scaffold_split_config.json').write_text(json.dumps({
        'method': 'Bemis-Murcko scaffold (RDKit), no chirality',
        'target_ratios': [0.8, 0.1, 0.1], 'splits': 3,
        'positive_threshold_pKi': args.positive_threshold,
        'features': 'VQAtom lig_tok, Morgan morgan_atom_r2_token, shared atom_feat/edge_index',
        'note': 'Whole scaffolds assigned to partitions; actual ratios may differ.'
    }, indent=2), encoding='utf-8')
    print('[done] 9 merged CSVs; scaffold leakage checks passed; existing cold splits untouched')


if __name__ == '__main__':
    main()
