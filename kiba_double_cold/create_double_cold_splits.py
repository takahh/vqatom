#!/usr/bin/env python3
"""Generate two KIBA double-cold splits using MMseqs2 protein similarity components.

Protein rule: no cross-split MMseqs2 hits at >=40% identity and >=80% coverage
(using MMseqs2 easy-search, cov-mode 0). Connected components of the hit graph
are indivisible. Ligands are grouped by canonical isomeric SMILES.

Cross-partition protein-ligand rows are excluded. A heuristic candidate search
balances retained rows and positive rates; no exact 80/10/10 guarantee.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

NAMES = np.array(['train', 'valid', 'test'])


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {str(smiles)[:100]}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def choose_assignment(n, rng, valid_frac, test_frac):
    if n < 3:
        raise ValueError(f'Only {n} independent groups; need at least 3')
    perm = rng.permutation(n)
    nv = max(1, int(round(n * valid_frac)))
    nt = max(1, int(round(n * test_frac)))
    if nv + nt >= n:
        nv, nt = 1, 1
    assignment = np.zeros(n, dtype=np.int8)
    assignment[perm[:nv]] = 1
    assignment[perm[nv:nv + nt]] = 2
    return assignment


def summary(frame, name, threshold):
    label = pd.to_numeric(frame['y']) >= threshold
    return {'split': name, 'rows': len(frame),
            'proteins': frame['seq'].nunique(), 'ligands': frame['_ligand'].nunique(),
            'positive_rows': int(label.sum()),
            'positive_rate': float(label.mean()) if len(frame) else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in NAMES:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('kiba_double_cold_mmseqs40'))
    parser.add_argument('--candidates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--threshold', type=float, default=12.1)
    parser.add_argument('--min-seq-id', type=float, default=0.4)
    parser.add_argument('--coverage', type=float, default=0.8)
    parser.add_argument('--sensitivity', type=float, default=7.5)
    parser.add_argument('--valid-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--min-eval-rows', type=int, default=100)
    parser.add_argument('--mmseqs', default='mmseqs')
    args = parser.parse_args()
    if not (0 < args.min_seq_id <= 1 and 0 < args.coverage <= 1):
        parser.error('Identity and coverage must be in (0, 1]')
    if not (0 < args.valid_fraction and 0 < args.test_fraction and args.valid_fraction + args.test_fraction < 1):
        parser.error('Invalid partition fractions')
    if args.candidates < 1:
        parser.error('--candidates must be >= 1')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in NAMES:
        frame = pd.read_csv(getattr(args, name), dtype=str, keep_default_na=False)
        missing = {'seq', 'smiles', 'y'} - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {missing}')
        frame['_original_split'] = name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if df[['seq', 'smiles', 'y']].eq('').any().any():
        raise ValueError('Empty seq, smiles, or y')
    df['_ligand'] = df['smiles'].map(canonicalize)
    df['_numeric_y'] = pd.to_numeric(df['y'], errors='coerce')
    if df['_numeric_y'].isna().any():
        raise ValueError('Non-numeric y')
    df['_label'] = (df['_numeric_y'] >= args.threshold).astype(np.int8)

    grouped = df.groupby(['seq', '_ligand'], sort=False, dropna=False)
    audit = []
    for (seq, lig), part in grouped:
        if len(part) > 1:
            audit.append({'seq': seq, 'canonical_smiles': lig, 'n_rows': len(part),
                          'y_values': ';'.join(part['y']),
                          'decision': 'drop_conflicting_labels' if part['_label'].nunique() > 1 else 'median_y'})
    pd.DataFrame(audit, columns=['seq', 'canonical_smiles', 'n_rows', 'y_values', 'decision']).to_csv(
        out / 'duplicate_pair_audit.csv', index=False)
    unique_labels = grouped['_label'].nunique()
    medians = grouped['_numeric_y'].median()
    dedup = grouped.head(1).copy()
    idx = pd.MultiIndex.from_frame(dedup[['seq', '_ligand']])
    dedup['y'] = [format(v, '.12g') for v in medians.reindex(idx).to_numpy()]
    df = dedup.loc[unique_labels.reindex(idx).to_numpy() == 1].reset_index(drop=True)
    df = df.drop(columns=['_numeric_y', '_label'])
    proteins = np.sort(df['seq'].unique())
    ligands = np.sort(df['_ligand'].unique())
    seq_to_idx = {seq: i for i, seq in enumerate(proteins)}
    protein_names = [f'protein_{i:06d}' for i in range(len(proteins))]
    name_to_idx = {name: i for i, name in enumerate(protein_names)}
    fasta = out / 'proteins.fasta'
    with fasta.open('w') as handle:
        for name, seq in zip(protein_names, proteins):
            handle.write(f'>{name}\n{seq}\n')

    hits = out / 'mmseqs_all_vs_all.tsv'
    with tempfile.TemporaryDirectory(prefix='mmseqs_double_cold_') as tmp:
        cmd = [args.mmseqs, 'easy-search', str(fasta), str(fasta), str(hits), tmp,
               '--min-seq-id', str(args.min_seq_id), '-c', str(args.coverage),
               '--cov-mode', '0', '-s', str(args.sensitivity),
               '--format-output', 'query,target,fident,alnlen,qlen,tlen', '--threads', '0']
        print('[mmseqs]', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    dsu = DSU(len(proteins))
    edges = 0
    with hits.open() as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2:
                continue
            a, b = name_to_idx[fields[0]], name_to_idx[fields[1]]
            if a != b:
                dsu.union(a, b)
                edges += 1
    roots = np.array([dsu.find(i) for i in range(len(proteins))])
    _, protein_components = np.unique(roots, return_inverse=True)
    n_components = int(protein_components.max()) + 1
    print(f'[proteins] {len(proteins)} unique, {n_components} MMseqs-connected components, {edges} directed non-self hits', flush=True)
    if n_components < 3:
        raise RuntimeError('Too few independent protein similarity components for three splits')
    pd.DataFrame({'protein_id': protein_names, 'seq': proteins,
                  'component': protein_components}).to_csv(out / 'protein_components.csv', index=False)

    p_idx = pd.Categorical(df['seq'], categories=proteins).codes
    l_idx = pd.Categorical(df['_ligand'], categories=ligands).codes
    row_component = protein_components[p_idx]
    labels = (pd.to_numeric(df['y']).to_numpy() >= args.threshold)
    global_rate = labels.mean()
    rng = np.random.default_rng(args.seed)
    summaries = [summary(df, 'combined_cleaned', args.threshold)]
    prior_test_proteins, prior_test_ligands = set(), set()
    keep_cols = [col for col in df.columns if col not in ('_original_split', '_ligand')]

    for run in (2, 3):
        best, best_score = None, np.inf
        for _ in range(args.candidates):
            pc = choose_assignment(n_components, rng, args.valid_fraction, args.test_fraction)
            la = choose_assignment(len(ligands), rng, args.valid_fraction, args.test_fraction)
            pcode = pc[row_component]
            lcode = la[l_idx]
            kept = pcode == lcode
            sizes = np.bincount(pcode[kept], minlength=3)
            if sizes.min() == 0 or sizes[1] < args.min_eval_rows or sizes[2] < args.min_eval_rows:
                continue
            positive = np.bincount(pcode[kept], weights=labels[kept].astype(float), minlength=3)
            rates = positive / sizes
            retained = sizes.sum() / len(df)
            score = 2 * (1 - retained) + 4 * np.sum((rates - global_rate) ** 2)
            score += 0.1 * abs(np.log(sizes[1] / sizes[2]))
            if prior_test_proteins:
                score += 0.3 * np.mean(np.isin(proteins[pc[protein_components] == 2], list(prior_test_proteins)))
            if prior_test_ligands:
                score += 0.3 * np.mean(np.isin(ligands[la == 2], list(prior_test_ligands)))
            if score < best_score:
                best_score = score
                best = (pc.copy(), la.copy(), kept.copy(), sizes.copy(), rates.copy())
        if best is None:
            raise RuntimeError(f'No feasible split_{run}: reduce --min-eval-rows, adjust fractions, or inspect protein components')
        pc, la, kept, sizes, rates = best
        pcode = pc[row_component]
        folder = out / f'split_{run}'
        folder.mkdir(exist_ok=True)
        for code, name in enumerate(NAMES):
            part = df.loc[kept & (pcode == code)]
            part[keep_cols].to_csv(folder / f'{name}.csv', index=False)
            summaries.append(summary(part, f'split_{run}_{name}', args.threshold))
        df.loc[~kept, keep_cols].to_csv(folder / 'excluded_cross_partition.csv', index=False)
        pd.DataFrame({'seq': proteins, 'component': protein_components,
                      'split': NAMES[pc[protein_components]]}).to_csv(folder / 'protein_assignment.csv', index=False)
        pd.DataFrame({'canonical_smiles': ligands, 'split': NAMES[la]}).to_csv(folder / 'ligand_assignment.csv', index=False)
        # Validate *every* MMseqs similarity edge, including pairs not retained as rows.
        with hits.open() as handle:
            for line in handle:
                a, b = line.rstrip('\n').split('\t')[:2]
                if pc[protein_components[name_to_idx[a]]] != pc[protein_components[name_to_idx[b]]]:
                    raise AssertionError('MMseqs cross-split similarity violation')
        for a in range(3):
            for b in range(a + 1, 3):
                left = df.loc[kept & (pcode == a)]
                right = df.loc[kept & (pcode == b)]
                assert not (set(left['seq']) & set(right['seq'])), 'Protein-cold violation'
                assert not (set(left['_ligand']) & set(right['_ligand'])), 'Ligand-cold violation'
        prior_test_proteins = set(proteins[pc[protein_components] == 2])
        prior_test_ligands = set(ligands[la == 2])
        print(f'[split_{run}] retained={int(kept.sum())}/{len(df)} ({100*kept.mean():.1f}%), '
              f'excluded={int((~kept).sum())}, rows={sizes.tolist()}, '
              f'positive_rates={np.round(rates, 4).tolist()}, score={best_score:.5f}', flush=True)
    pd.DataFrame(summaries).to_csv(out / 'split_entity_analysis.csv', index=False)
    print('[done]', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
