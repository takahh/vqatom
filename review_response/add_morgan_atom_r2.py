#!/usr/bin/env python3
"""Create atom-wise radius-2 Morgan tokens and remap train vocabulary to dense IDs.

Usage:
 python add_morgan_atom_r2_combined.py --data-dir /Users/taka/Downloads
 python add_morgan_atom_r2_combined.py --train /path/train.csv --valid /path/valid.csv --test /path/test.csv
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
    n = mol.GetNumAtoms()
    # Obtain the radius-2 environment identifier for each atom. The generator
    # may suppress redundant environments; includeRedundantEnvironments=True
    # ensures that each center is represented at radius 2.
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=info)
    by_center = [[] for _ in range(n)]
    for identifier, occurrences in info.GetBitInfoMap().items():
        for center, radius in occurrences:
            if radius == 2:
                by_center[center].append(int(identifier))
    if any(len(x) != 1 for x in by_center):
        counts = [len(x) for x in by_center]
        raise ValueError(f"Radius-2 identifiers missing or ambiguous: {counts}, SMILES={smiles}")
    return [x[0] for x in by_center]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('/Users/taka/Downloads'))
    for split in ('train', 'valid', 'test'):
        p.add_argument('--' + split, type=Path, default=None)
    args = p.parse_args()
    paths = {s: getattr(args, s) or (args.data_dir / f'{s}.csv') for s in ('train','valid','test')}
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, includeRedundantEnvironments=True, countSimulation=False
    )
    frames = {}
    ids_by_split = {}
    for split, path in paths.items():
        print(f'[{split}] reading {path}', flush=True)
        df = pd.read_csv(path, dtype={'lig_tok': 'string'})
        if 'smiles' not in df.columns:
            raise ValueError(f'Missing smiles column in {path}')
        raw_ids = []
        for i, smiles in enumerate(df['smiles']):
            try:
                raw_ids.append(atom_morgan_ids(smiles, generator))
            except Exception as e:
                raise ValueError(f'{path}, row {i + 2}: {e}') from e
        frames[split] = df
        ids_by_split[split] = raw_ids
        unique = len({v for row in raw_ids for v in row})
        print(f'[{split}] rows={len(df):,} atoms={sum(map(len,raw_ids)):,} unique_raw_IDs={unique:,}', flush=True)
    vocab = sorted({v for row in ids_by_split['train'] for v in row})
    mapping = {v: i + 1 for i, v in enumerate(vocab)}  # PAD=0
    print(f'[vocab] train vocabulary={len(vocab):,}, embedding num_embeddings={len(vocab)+1:,}', flush=True)
    # Validate everything before writing any output.
    for split, rows in ids_by_split.items():
        unseen = {v for row in rows for v in row if v not in mapping}
        if unseen:
            raise ValueError(f'{split}: {len(unseen)} IDs absent from train vocabulary; refusing to write. Add UNK mapping explicitly if desired.')
    out_dir = args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'original_id': vocab, 'mapped_id': range(1, len(vocab)+1)}).to_csv(out_dir / 'morgan_vocab.csv', index=False)
    for split, df in frames.items():
        raw = ids_by_split[split]
        df['morgan_atom_r2'] = [' '.join(map(str, row)) for row in raw]
        df['lig_tok'] = [' '.join(str(mapping[v]) for v in row) for row in raw]
        output = out_dir / f'{split}_morgan.csv'
        df.to_csv(output, index=False)
        print(f'[saved] {output} rows={len(df):,}', flush=True)
    print('Done. PAD=0; dense Morgan IDs start at 1. Original Morgan IDs retained in morgan_atom_r2.')

if __name__ == '__main__':
    main()
