import argparse
import ast
from pathlib import Path

import pandas as pd
from rdkit import Chem

from infer_one_smiles import init_tokenizer, encode_smiles_to_atom_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    init_tokenizer(
        ckpt_path="/vqatom/data/model_epoch_6_5k.pt",
        device=0,
        hidden_dim=16,
        codebook_size=5000,
        strict=True,
        edge_emb_dim=32,
        ema_decay=0.8,
        maxn=100,
        d_attr=79,
    )

    df = pd.read_csv(args.input)
    assert "smiles" in df.columns
    assert "lig_tok" in df.columns

    cache = {}
    for smiles in df["smiles"].dropna().unique():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        tokens = encode_smiles_to_atom_tokens(smiles)

        if len(tokens) != mol.GetNumAtoms():
            raise ValueError(
                f"Atom/token mismatch: {smiles}, "
                f"atoms={mol.GetNumAtoms()}, tokens={len(tokens)}"
            )

        if not all(0 <= t < 5021 for t in tokens):
            raise ValueError(f"Token ID out of range: {smiles}: {tokens}")

        cache[smiles] = tokens

    df["lig_tok"] = df["smiles"].map(lambda s: str(cache[s]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Rows: {len(df)}")
    print(f"Unique SMILES: {len(cache)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
