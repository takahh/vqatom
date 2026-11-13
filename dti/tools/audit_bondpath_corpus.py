"""Reconstruct the SMILES shards recorded by a trusted bond-path checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from rdkit import Chem
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shuffle-seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Checkpoints contain project metadata, so load only a trusted project file.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reference = checkpoint['reference_sources']
    original = args.smiles.read_text().splitlines()
    smiles = list(original)
    random.Random(args.shuffle_seed).shuffle(smiles)
    filtered = []
    counts = Counter()
    atom_counts = {}
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            molecule = Chem.MolFromSmiles(text, sanitize=False)
            if molecule is not None:
                molecule = Chem.RemoveHs(molecule, sanitize=False)
                molecule.UpdatePropertyCache(strict=False)
                counts['permissive_parser_records'] += 1
        if molecule is None:
            raise ValueError('Reference SMILES could not be parsed')
        atoms = molecule.GetNumAtoms()
        if atoms >= 100:
            counts['size_excluded'] += 1
            continue
        filtered.append(text)
        atom_counts[text] = atoms
    batches = {
        f'smiles_{start // args.batch_size}.txt': filtered[start:start + args.batch_size]
        for start in range(0, len(filtered), args.batch_size)
    }
    selected = []
    shard_records = []
    for entry in reference:
        batch = batches[entry['smiles']]
        digest = hashlib.sha256(('\n'.join(batch) + '\n').encode()).hexdigest()
        if digest != entry['smiles_sha256']:
            raise ValueError('Reference SMILES SHA256 mismatch for ' + entry['smiles'])
        shard_records.append(dict(file=entry['smiles'],sha256=digest,molecules=len(batch)))
        selected.extend(batch)
    fitted_atoms = sum(stat['points'] for stat in checkpoint['fit_stats'].values())
    selected_atoms = sum(atom_counts[text] for text in selected)
    if selected_atoms != fitted_atoms:
        raise ValueError('Reconstructed atom count differs from checkpoint fitting metadata')
    report = dict(
        input_sha256=sha256(args.smiles), checkpoint_sha256=sha256(args.checkpoint),
        input_records=len(original), input_unique_strings=len(set(original)),
        filtered_records=len(filtered), selected_records=len(selected),
        selected_unique_strings=len(set(selected)), selected_atom_records=selected_atoms,
        shuffle_seed=args.shuffle_seed, batch_size=args.batch_size,
        selection='checkpoint reference_sources order; lexicographically selected adjacency shards',
        parser_counts=dict(counts), shards=shard_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key:report[key] for key in ('selected_records','selected_atom_records')}))


if __name__ == '__main__':
    main()
