#!/usr/bin/env python3
"""Create Metz ligand/protein/double-cold splits and comprehensive audits.

Requires: pandas, numpy, rdkit, mmseqs2 executable.
The default positive label is pKi >= 6.0; set --positive-threshold explicitly.
Protein clusters: connected components of MMseqs2 hits with sequence identity
>= 40% and aligned length / shorter sequence length >= 40%.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

DEFAULT_INPUT = Path('/Users/taka/PycharmProjects/vqatom/review_response/metz/metz_dti_clean.csv')
PARTS = ('train', 'valid', 'test')
RATIOS = np.array([.8, .1, .1])


class DSU:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}
    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def canonical_smiles(s):
    mol = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def make_protein_clusters(df, out, mmseqs, identity, coverage):
    proteins = df[['uniprot_id', 'protein_sequence']].drop_duplicates()
    if proteins.uniprot_id.duplicated().any():
        raise ValueError('One UniProt ID maps to multiple sequences')
    ids = proteins.uniprot_id.astype(str).tolist()
    seqs = dict(zip(ids, proteins.protein_sequence.astype(str)))
    dsu = DSU(ids)
    fasta = out / 'metz_proteins.fasta'
    with fasta.open('w') as f:
        for key, seq in seqs.items():
            if not seq or any(c not in 'ACDEFGHIKLMNPQRSTVWYUXOBZJ' for c in seq.upper()):
                raise ValueError(f'Unexpected protein sequence for {key}')
            f.write(f'>{key}\n{seq}\n')
    hits = out / 'mmseqs_all_vs_all.tsv'
    with tempfile.TemporaryDirectory(prefix='metz_mmseqs_') as tmp:
        cmd = [mmseqs, 'easy-search', str(fasta), str(fasta), str(hits), tmp,
               '--min-seq-id', str(identity), '-c', '0',
               '--max-seqs', str(max(1000, len(ids) * 5)),
               '--format-output', 'query,target,fident,alnlen,qlen,tlen',
               '--threads', '4']
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    edges = []
    with hits.open() as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 6:
                raise ValueError(f'Bad MMseqs row: {line[:200]}')
            q, t = parts[:2]
            ident, aln, qlen, tlen = map(float, parts[2:])
            if q not in seqs or t not in seqs:
                raise ValueError(f'Unexpected MMseqs identifier: {q} {t}')
            # fident from MMseqs may be a fraction or percent depending on version.
            if ident > 1:
                ident /= 100
            cov = aln / min(qlen, tlen)
            if ident >= identity and cov >= coverage:
                dsu.union(q, t)
                edges.append((q, t, ident, cov))
    components = {}
    for key in ids:
        components.setdefault(dsu.find(key), []).append(key)
    cluster = {k: min(members) for members in components.values() for k in members}
    pd.DataFrame([{'uniprot_id': k, 'protein_cluster': cluster[k],
                   'cluster_size': len(components[dsu.find(k)])} for k in ids]).to_csv(
        out / 'protein_clusters.csv', index=False)
    pd.DataFrame(edges, columns=['query', 'target', 'identity', 'shorter_coverage']).to_csv(
        out / 'mmseqs_passing_edges.csv', index=False)
    return cluster, len(components), len(edges)


def assign_groups(df, group_col, rng):
    """Greedy weighted partitioning, with seeded randomization of group order."""
    counts = df.groupby(group_col).size()
    groups = counts.index.to_numpy()
    weights = counts.to_numpy()
    order = np.arange(len(groups))
    rng.shuffle(order)
    order = sorted(order, key=lambda i: -weights[i])
    loads = np.zeros(3, dtype=float)
    assignments = {}
    for i in order:
        # normalized load makes target sizes 80/10/10 rather than equal.
        costs = (loads + weights[i]) / RATIOS
        p = int(np.argmin(costs))
        assignments[groups[i]] = p
        loads[p] += weights[i]
    return assignments


def assign_protein_groups(df, seed, used_signatures, attempts=300):
    """Random-restart balanced assignment, requiring genuinely distinct cold splits."""
    counts = df.groupby('protein_cluster').size()
    names = counts.index.to_numpy()
    weights = counts.to_numpy(dtype=float)
    total = weights.sum()
    target = RATIOS * total
    best = None
    best_score = float('inf')
    for trial in range(attempts):
        rng = np.random.default_rng(seed * 10000 + trial)
        # Random initialization prevents the fixed ordering from dominating all seeds.
        slots = rng.choice(3, size=len(names), p=RATIOS)
        for k in range(3):
            if not np.any(slots == k):
                slots[rng.integers(len(slots))] = k
        loads = np.bincount(slots, weights=weights, minlength=3)
        # Stochastic local improvement: vary update order and accept only improvements.
        for _ in range(15):
            changed = False
            for i in rng.permutation(len(names)):
                old = slots[i]
                current = np.sum(((loads - target) / target) ** 2)
                choices = rng.permutation(3)
                for new in choices:
                    if new == old or np.sum(slots == old) <= 1:
                        continue
                    candidate = loads.copy()
                    candidate[old] -= weights[i]
                    candidate[new] += weights[i]
                    score = np.sum(((candidate - target) / target) ** 2)
                    if score < current - 1e-12:
                        slots[i] = new
                        loads = candidate
                        changed = True
                        break
            if not changed:
                break
        signature = tuple(sorted((str(names[i]), int(slots[i])) for i in range(len(names))))
        if signature in used_signatures:
            continue
        score = float(np.sum(((loads - target) / target) ** 2))
        if score < best_score:
            best = (dict(zip(names, map(int, slots))), signature, loads.copy())
            best_score = score
    if best is None:
        raise RuntimeError('Could not generate distinct protein cold partitions')
    mapping, signature, loads = best
    used_signatures.add(signature)
    print(f'Protein Cold seed={seed}: pair loads={loads.astype(int).tolist()}, '
          f'imbalance_score={best_score:.4f}', flush=True)
    return mapping


def build_split(df, mode, seed, protein_map_override=None):
    rng = np.random.default_rng(seed)
    if mode == 'protein_cold':
        protein_map = (protein_map_override if protein_map_override is not None
                       else assign_groups(df, 'protein_cluster', rng))
        slot = df.protein_cluster.map(protein_map).to_numpy()
        keep = np.ones(len(df), dtype=bool)
    elif mode == 'ligand_cold':
        ligand_map = assign_groups(df, 'ligand_group', rng)
        slot = df.ligand_group.map(ligand_map).to_numpy()
        keep = np.ones(len(df), dtype=bool)
    else:
        ligand_map = assign_groups(df, 'ligand_group', rng)
        protein_map = assign_groups(df, 'protein_cluster', rng)
        ls = df.ligand_group.map(ligand_map).to_numpy()
        ps = df.protein_cluster.map(protein_map).to_numpy()
        slot, keep = ls, ls == ps
    return {name: df.loc[keep & (slot == i)].copy() for i, name in enumerate(PARTS)}


def audit(splits, mode):
    for p, frame in splits.items():
        if frame.empty:
            raise RuntimeError(f'{mode}: {p} empty')
    columns = ['protein_cluster'] if mode == 'protein_cold' else (
        ['ligand_group'] if mode == 'ligand_cold' else ['ligand_group', 'protein_cluster'])
    for col in columns:
        groups = [set(splits[p][col]) for p in PARTS]
        for i in range(3):
            for j in range(i + 1, 3):
                if groups[i] & groups[j]:
                    raise RuntimeError(f'{mode}: leakage in {col} between {PARTS[i]} and {PARTS[j]}')
    for p, frame in splits.items():
        if frame.duplicated(['ligand_group', 'uniprot_id']).any():
            raise RuntimeError(f'{mode}: duplicate pairs in {p}')


def metrics(frame, threshold):
    n = len(frame)
    if n == 0:
        return {}
    positive = frame['label'].astype(int)
    ligand = frame.groupby('ligand_group')['label'].agg(['min', 'max', 'size'])
    protein = frame.groupby('protein_cluster')['label'].agg(['min', 'max', 'size'])
    return {
        'pairs': n, 'positive_pairs': int(positive.sum()),
        'negative_pairs': int(n - positive.sum()),
        'positive_rate': float(positive.mean()),
        'ligands': int(frame.ligand_group.nunique()),
        'proteins': int(frame.uniprot_id.nunique()),
        'protein_clusters': int(frame.protein_cluster.nunique()),
        'ligands_positive_only': int((ligand['min'] == 1).sum()),
        'ligands_negative_only': int((ligand['max'] == 0).sum()),
        'ligands_mixed': int(((ligand['min'] == 0) & (ligand['max'] == 1)).sum()),
        'ligands_positive_only_fraction': float((ligand['min'] == 1).mean()),
        'ligands_negative_only_fraction': float((ligand['max'] == 0).mean()),
        'ligands_mixed_fraction': float(((ligand['min'] == 0) & (ligand['max'] == 1)).mean()),
        'proteins_positive_only_clusters': int((protein['min'] == 1).sum()),
        'proteins_negative_only_clusters': int((protein['max'] == 0).sum()),
        'proteins_mixed_clusters': int(((protein['min'] == 0) & (protein['max'] == 1)).sum()),
        'pKi_mean': float(frame.pKi.mean()),
        'pKi_std': float(frame.pKi.std()) if n > 1 else None,
        'pKi_min': float(frame.pKi.min()), 'pKi_max': float(frame.pKi.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--positive-threshold', type=float, default=6.0,
                        help='pKi >= threshold => positive; default 6.0 is provisional')
    parser.add_argument('--identity', type=float, default=.40)
    parser.add_argument('--coverage', type=float, default=.40)
    parser.add_argument('--mmseqs', default='mmseqs')
    args = parser.parse_args()
    print('Running script:', Path(__file__).resolve(), flush=True)
    out = args.out or args.input.parent
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = ['SMILES', 'uniprot_id', 'protein_sequence', 'pKi']
    if missing := (set(required) - set(df.columns)):
        raise ValueError(f'Missing columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('Missing values in required fields')
    if df.duplicated(['SMILES', 'uniprot_id']).any():
        raise ValueError('Input still has duplicate SMILES/UniProt pairs')
    df['ligand_group'] = df.SMILES.map(canonical_smiles)
    if df.ligand_group.isna().any():
        raise ValueError(f'{df.ligand_group.isna().sum()} invalid SMILES')
    # Canonicalizing can reveal new duplicate pairs; average the same way as input preprocessing.
    if df.duplicated(['ligand_group', 'uniprot_id']).any():
        duplicates = df.loc[df.duplicated(['ligand_group', 'uniprot_id'], keep=False)]
        duplicates.to_csv(out / 'canonical_duplicate_pairs_review.csv', index=False)
        raise ValueError('Canonicalization revealed duplicate ligand/protein pairs. '
                         'Inspect canonical_duplicate_pairs_review.csv and consolidate first.')
    df['label'] = (df.pKi >= args.positive_threshold).astype(int)
    clusters, cluster_count, edges = make_protein_clusters(
        df, out, args.mmseqs, args.identity, args.coverage)
    df['protein_cluster'] = df.uniprot_id.map(clusters)
    df['canonical_smiles'] = df.ligand_group
    df.to_csv(out / 'metz_labeled_and_grouped.csv', index=False)
    all_rows = []
    global_metrics = metrics(df, args.positive_threshold)
    global_metrics.update({'mode': 'all', 'split': 0, 'partition': 'all',
                           'retained_pairs': len(df), 'discarded_pairs': 0,
                           'retained_fraction': 1.0})
    all_rows.append(global_metrics)
    protein_signatures = set()
    for mode in ['protein_cold', 'ligand_cold', 'double_cold']:
        folder = out / f'metz_{mode}_3splits'
        folder.mkdir(parents=True, exist_ok=True)
        # Remove only legacy flat split CSVs from this generated output folder.
        legacy_files = [folder / f'{part}_split{i}.csv'
                        for part in PARTS for i in range(1, 4)]
        for legacy in legacy_files:
            if legacy.is_file():
                legacy.unlink()
                print('Removed legacy output:', legacy, flush=True)
        for split_num in range(1, 4):
            split_folder = folder / f'split_{split_num}'
            split_folder.mkdir(parents=True, exist_ok=True)
            # Multiple distinct random seeds, same seeds across methods.
            seed = 1000 + split_num
            protein_override = (assign_protein_groups(df, seed, protein_signatures)
                                if mode == 'protein_cold' else None)
            splits = build_split(df, mode, seed, protein_override)
            audit(splits, mode)
            retained = sum(map(len, splits.values()))
            for part, frame in splits.items():
                # Keep useful metadata and both pKi and binary label.
                frame.to_csv(
                    split_folder / f'metz_{mode}_split_{split_num}_{part}.csv',
                    index=False
                )
                row = metrics(frame, args.positive_threshold)
                row.update({'mode': mode, 'split': split_num, 'partition': part,
                            'retained_pairs': retained,
                            'discarded_pairs': len(df) - retained,
                            'retained_fraction': retained / len(df),
                            'fraction_of_retained': len(frame) / retained})
                all_rows.append(row)
            print(f'{mode} split{split_num}: '
                  + ', '.join(f'{p}={len(splits[p])}' for p in PARTS)
                  + f' | retained={retained} discarded={len(df)-retained}', flush=True)
    # Independently check that protein-cold split memberships differ.
    protein_folder = out / 'metz_protein_cold_3splits'
    membership = []
    for split_num in range(1, 4):
        membership.append({part: set(pd.read_csv(
            protein_folder / f'split_{split_num}' /
            f'metz_protein_cold_split_{split_num}_{part}.csv',
            usecols=['protein_cluster'])['protein_cluster'].astype(str))
            for part in PARTS})
    for i in range(3):
        for j in range(i + 1, 3):
            if all(membership[i][part] == membership[j][part] for part in PARTS):
                raise RuntimeError(f'Protein cold splits {i+1} and {j+1} are identical')
    print('Protein Cold: all 3 partitions have distinct cluster membership.')
    summary = pd.DataFrame(all_rows)
    summary.to_csv(out / 'metz_split_summary.csv', index=False)
    config = {'input': str(args.input), 'identity': args.identity,
              'coverage': args.coverage,
              'coverage_definition': 'alignment length / shorter sequence length',
              'positive_threshold': args.positive_threshold,
              'label_definition': 'positive if pKi >= threshold',
              'protein_clusters': cluster_count, 'passing_mmseqs_hits': edges,
              'double_cold': 'independent entity partitions; cross-partition pairs discarded',
              'note': 'Split ratios are targets, not guaranteed for imbalanced cold groups.'}
    (out / 'metz_split_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('\nGLOBAL:', global_metrics)
    print('Saved:', out / 'metz_split_summary.csv')
    print('All cold-leakage audits passed.')


if __name__ == '__main__':
    main()#!/usr/bin/env python3
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

