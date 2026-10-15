#!/usr/bin/env python3
"""Convert train/valid/test for split_2 and split_3 to atom-wise radius-2 Morgan IDs."""
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
        raise ValueError(f"Radius-2 identifiers missing or ambiguous: {[len(x) for x in by_center]}, SMILES={smiles}")
    return [ids[0] for ids in by_center]


def convert_split(root, split, generator):
    folder = root
    frames, ids_by_part = {}, {}
    for part in ('train', 'valid', 'test'):
        path = folder / f"{part}_split{split}.csv"
        print(f"[{split}/{part}] reading {path}", flush=True)
        df = pd.read_csv(path, dtype={'lig_tok': 'string'})
        if 'smiles' not in df.columns:
            raise ValueError(f"Missing smiles column in {path}")
        rows = []
        for i, smiles in enumerate(df['smiles']):
            try:
                rows.append(atom_morgan_ids(smiles, generator))
            except Exception as exc:
                raise ValueError(f"{path}, row {i+2}: {exc}") from exc
        frames[part], ids_by_part[part] = df, rows
        print(f"[{split}/{part}] rows={len(df):,}, atoms={sum(map(len, rows)):,}, unique={len({x for row in rows for x in row}):,}", flush=True)

    vocab = sorted({v for row in ids_by_part['train'] for v in row})
    mapping = {v: i + 1 for i, v in enumerate(vocab)}
    for part, rows in ids_by_part.items():
        unseen = {v for row in rows for v in row if v not in mapping}
        if unseen:
            raise ValueError(f"split_{split}/{part}: {len(unseen)} Morgan IDs absent from training vocabulary; no files written for this split")

    pd.DataFrame({'original_id': vocab, 'mapped_id': range(1, len(vocab) + 1)}).to_csv(folder / f'morgan_vocab_split{split}.csv', index=False)
    for part, df in frames.items():
        rows = ids_by_part[part]
        df['morgan_atom_r2'] = [' '.join(map(str, row)) for row in rows]
        df['lig_tok'] = [' '.join(str(mapping[v]) for v in row) for row in rows]
        output = folder / f'{part}_split{split}_morgan.csv'
        df.to_csv(output, index=False)
        print(f'[saved] {output}', flush=True)
    print(f'[split_{split}] vocab={len(vocab):,}; PAD=0, token IDs=1..{len(vocab)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/Users/taka/Desktop/kiba_extra_splits'))
    parser.add_argument('--splits', type=int, nargs='+', default=[2, 3])
    args = parser.parse_args()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeRedundantEnvironments=True, countSimulation=False)
    for split in args.splits:
        convert_split(args.root, split, generator)
    print('Done.', flush=True)

if __name__ == '__main__':
    main()
