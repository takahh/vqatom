import ast
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem

from infer_one_smiles import init_tokenizer, encode_smiles_to_atom_tokens

ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "review_response/kiba_protein_scaffold_cold_v2"
))

CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", "/vqatom/data"))

MODELS = [
    ("cb3k", 3000, "model_epoch_2_3k.pt"),
    ("cb4k", 4000, "model_epoch_3_4k.pt"),
]

files = [
    ROOT / f"split_{i}" /
    f"kiba_protein_scaffold_cold_split_{i}_{part}_morgan_cb5k_cb10k_cb15k.csv"
    for i in (1, 2, 3)
    for part in ("train", "valid", "test")
]

for path in files:
    if not path.is_file():
        raise FileNotFoundError(path)

smiles_set = set()

for path in files:
    df = pd.read_csv(path, usecols=["smiles"])
    smiles_set.update(df["smiles"].dropna().unique())

print(f"Unique SMILES: {len(smiles_set)}", flush=True)

caches = {}

for column, size, filename in MODELS:
    ckpt = CKPT_ROOT / filename
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    print(f"\n[START] {column}: {ckpt}", flush=True)

    init_tokenizer(
        ckpt_path=str(ckpt),
        device=-1,
        hidden_dim=16,
        codebook_size=size,
        strict=True,
        edge_emb_dim=32,
        ema_decay=0.8,
        maxn=100,
        d_attr=79,
    )

    cache = {}

    for n, smiles in enumerate(sorted(smiles_set), 1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        tokens = encode_smiles_to_atom_tokens(smiles)

        if len(tokens) != mol.GetNumAtoms():
            raise ValueError(
                f"Atom/token mismatch: {smiles}: "
                f"atoms={mol.GetNumAtoms()}, tokens={len(tokens)}"
            )

        if not all(isinstance(t, int) and 0 <= t < size + 21 for t in tokens):
            raise ValueError(
                f"Token ID out of range: {smiles}: {tokens}"
            )

        cache[smiles] = str(tokens)

        if n % 100 == 0:
            print(
                f"[{column}] {n}/{len(smiles_set)}",
                flush=True
            )

    caches[column] = cache
    print(f"[DONE] {column}", flush=True)

# 元のCSVは上書きせず、新しいファイルに保存する
for path in files:
    df = pd.read_csv(path)

    for column, cache in caches.items():
        df[f"lig_tok_{column}"] = df["smiles"].map(cache)

        if df[f"lig_tok_{column}"].isna().any():
            raise ValueError(f"Missing {column} tokens: {path}")

        for smiles, value in zip(df["smiles"], df[f"lig_tok_{column}"]):
            tokens = ast.literal_eval(value)
            mol = Chem.MolFromSmiles(smiles)
            if len(tokens) != mol.GetNumAtoms():
                raise ValueError(f"Atom mismatch in output: {path}")

    output = path.with_name(
        path.name.replace(
            "_morgan_cb5k_cb10k_cb15k.csv",
            "_morgan_cb5k_cb10k_cb15k_cb3k_cb4k.csv"
        )
    )

    df.to_csv(output, index=False)
    print(f"[OK] {output}: {len(df)} rows", flush=True)

print("\n[done] CB3k and CB4k added to all 9 files", flush=True)
