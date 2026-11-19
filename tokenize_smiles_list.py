"""Convert a plain-text or CSV SMILES list to one JSON record per molecule."""
import argparse
import csv
import json
from pathlib import Path
import sys

from infer_one_smiles_bondpath import BondpathTokenizer


def records(path, column, id_column):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if path.suffix.lower() == '.csv':
            reader = csv.DictReader(stream)
            for name in (column, id_column):
                if name and name not in (reader.fieldnames or []):
                    raise ValueError(f'CSV column not found: {name}')
            for row in reader:
                yield reader.line_num, (row.get(column) or '').strip(), row.get(id_column) if id_column else None
        else:
            for number, line in enumerate(stream, 1):
                yield number, line.strip(), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One SMILES per line, or a .csv file')
    parser.add_argument('--output', required=True, type=Path, help='New JSONL file; existing files are not overwritten')
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--smiles-column', default='smiles')
    parser.add_argument('--id-column', help='Optional CSV identifier column to preserve')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must differ')
    tokenizer = BondpathTokenizer(args.ckpt)
    # Validate the input header before creating output.
    source = iter(records(args.input, args.smiles_column, args.id_column))
    first = next(source, None)
    import itertools
    count = failed = 0
    with args.output.open('x', encoding='utf-8') as output:
        for line, smiles, identifier in itertools.chain(() if first is None else (first,), source):
            record = {'line': line, 'smiles': smiles}
            if args.id_column:
                record['id'] = identifier
            try:
                if not smiles:
                    raise ValueError('Empty SMILES')
                record['tokens'] = tokenizer.encode(smiles)
                record['n_atoms'] = len(record['tokens'])
                record['status'] = 'ok'
            except (ValueError, RuntimeError, OverflowError) as error:
                record.update(status='error', error=str(error))
                failed += 1
            record['checkpoint_sha256'] = tokenizer.checkpoint_sha256
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    print(json.dumps({'records': count, 'converted': count-failed, 'errors': failed,
                      'output': str(args.output), 'tokenizer': tokenizer.identity}), file=sys.stderr)


if __name__ == '__main__':
    main()
