#!/usr/bin/env python3
"""Add raw and globally indexed atom-wise radius-2 Morgan IDs to KIBA CSVs.

Input:  train.csv, valid.csv, test.csv under --root
Output: train_morgan.csv, valid_morgan.csv, test_morgan.csv,
        morgan_vocab_global.csv

Existing lig_tok (VQAtom) is preserved. Vocabulary is shared across all
three input files. For strict ligand-cold/double-cold experiments, map
Morgan IDs absent from that split's training set to UNK at split time.
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def atom_morgan_ids(smiles, generator):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"Invalid SMILES: {str(smiles)[:120]}")
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=info)
    by_center = [[] for _ in range(mol.GetNumAtoms())]
    for identifier, occurrences in info.GetBitInfoMap().items():
        for center, radius in occurrences:
            if radius == 2:
                by_center[center].append(int(identifier))
    if any(len(ids) != 1 for ids in by_center):
        raise ValueError(f"Radius-2 identifiers missing or ambiguous: {[len(x) for x in by_center]}; SMILES={smiles}")
    return [ids[0] for ids in by_center]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    root = args.root.expanduser()
    parts = ('train', 'valid', 'test')

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, includeRedundantEnvironments=True, countSimulation=False
    )

    frames = {}
    cache = {}
    for part in parts:
        path = root / f'{part}.csv'
        print(f'[read] {path}', flush=True)
        df = pd.read_csv(path, dtype={'lig_tok': 'string'})
        if 'smiles' not in df.columns:
            raise ValueError(f'Missing smiles column: {path}')
        if df['smiles'].isna().any():
            raise ValueError(f'Missing SMILES in {path}')
        frames[part] = df
        for smiles in df['smiles'].unique():
            if smiles not in cache:
                cache[smiles] = atom_morgan_ids(smiles, generator)
        print(f'[{part}] rows={len(df):,}; unique ligands so far={len(cache):,}', flush=True)

    vocab = sorted({v for ids in cache.values() for v in ids})
    # PAD=0; reserve UNK=1; known environments start at 2.
    mapping = {v: i + 2 for i, v in enumerate(vocab)}
    vocab_path = root / 'morgan_vocab_global.csv'
    pd.DataFrame({
        'original_id': [-1] + vocab,
        'mapped_id': [1] + list(range(2, len(vocab) + 2)),
    }).to_csv(vocab_path, index=False)
    print(f'[vocab] {vocab_path}; known={len(vocab):,}; PAD=0, UNK=1', flush=True)

    for part, df in frames.items():
        df['morgan_atom_r2'] = df['smiles'].map(
            lambda s: ' '.join(map(str, cache[s]))
        )
        df['morgan_atom_r2_token'] = df['smiles'].map(
            lambda s: ' '.join(str(mapping[v]) for v in cache[s])
        )
        for i, (smiles, raw, tokens) in enumerate(zip(
            df['smiles'], df['morgan_atom_r2'], df['morgan_atom_r2_token']
        )):
            n = len(cache[smiles])
            if len(raw.split()) != n or len(tokens.split()) != n:
                raise AssertionError(f'{part} row {i}: atom/token count mismatch')
        output = root / f'{part}_morgan.csv'
        df.to_csv(output, index=False)
        print(f'[saved] {output}; rows={len(df):,}', flush=True)

    print('Done. Existing lig_tok was preserved.', flush=True)


if __name__ == '__main__':
    main()
