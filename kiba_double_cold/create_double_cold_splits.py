#!/usr/bin/env python3
"""Generate three KIBA double-cold splits using MMseqs2 protein similarity components.

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