import os, re, tarfile, io
import numpy as np
import torch

path0 = "/Users/taka/Documents/discret_50k.tar,gz"
path1 = "/Users/taka/Documents/infer_token_ids.pt"
out_dir = "/Users/taka/Documents/pretrain_ragged"
os.makedirs(out_dir, exist_ok=True)

MAX_ATOMS = 100
ATTR_DIM = 79

d = torch.load(path1, map_location="cpu")
cluster_stream = d["cluster_id"].to(torch.int64)  # already global IDs
base_vocab = int(cluster_stream.max().item()) + 1
print("cluster_stream:", len(cluster_stream), "base_vocab:", base_vocab)

def read_bytes_from_tar(tar, name: str) -> bytes:
    f = tar.extractfile(name)
    if f is None:
        raise FileNotFoundError(name)
    return f.read()

# list batches
with tarfile.open(path0, "r:gz") as tar:
    names = tar.getnames()
pat = re.compile(r"both_mono/concatenated_attr_batch_(\d+)\.npy$")
batches = sorted({int(pat.search(n).group(1)) for n in names if pat.search(n)})
print("num batches:", len(batches), "first/last:", batches[:3], batches[-3:])

pos = 0

with tarfile.open(path0, "r:gz") as tar:
    for b in batches:
        attr_name = f"both_mono/concatenated_attr_batch_{b}.npy"
        smiles_name = f"both_mono/smiles_{b}.txt"

        attr = np.load(io.BytesIO(read_bytes_from_tar(tar, attr_name)), allow_pickle=True)
        if attr.ndim != 2 or attr.shape[1] != ATTR_DIM:
            raise RuntimeError(f"attr shape unexpected at batch {b}: {attr.shape}")

        smiles_txt = read_bytes_from_tar(tar, smiles_name).decode("utf-8", errors="replace")
        smiles = [x for x in smiles_txt.splitlines() if x.strip()]

        n_mols = len(smiles)
        n_nodes = attr.shape[0]

        if n_nodes != n_mols * MAX_ATOMS:
            raise RuntimeError(
                f"batch {b}: nodes!=mols*MAX_ATOMS : nodes={n_nodes} mols={n_mols} MAX_ATOMS={MAX_ATOMS}"
            )

        row_is_real = (np.abs(attr).sum(axis=1) != 0).reshape(n_mols, MAX_ATOMS)
        lengths = row_is_real.sum(axis=1).astype(np.int64)  # (n_mols,)
        need = int(lengths.sum())

        if pos + need > len(cluster_stream):
            raise RuntimeError(f"token stream exhausted at batch {b}: need {need}, have {len(cluster_stream)-pos}")

        tokens_flat = cluster_stream[pos:pos+need].clone()
        pos += need

        offsets = np.zeros(n_mols + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        offsets = torch.from_numpy(offsets)

        out_path = os.path.join(out_dir, f"pretrain_ragged_batch{b:03d}.pt")
        torch.save({
            "batch": b,
            "tokens_flat": tokens_flat,
            "offsets": offsets,
            "lengths": torch.from_numpy(lengths),
            "base_vocab": base_vocab,
            "smiles": smiles,
        }, out_path)

        print("saved", out_path, "mols", n_mols, "need", need, "pos", pos)

print("DONE consumed:", pos, "/", len(cluster_stream))
if pos != len(cluster_stream):
    print("WARNING: consumed != stream_len (ordering/padding rule mismatch?)")
