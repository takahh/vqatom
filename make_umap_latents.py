import torch
import matplotlib.pyplot as plt
from rdkit import Chem

from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL

smiles_list = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Imatinib", "CC1=CC(=NC(=N1)NC2=CC(=C(C=C2)Cl)C(=O)NCC3=CN=CC=C3)C"),
    ("Nirmatrelvir", "CC(C)(C)[C@H](NC(=O)[C@H](NC(=O)C(F)(F)F)C(C)C)C(=O)N[C@H](C(=O)N1CC2(CC1)C[C@H](C(=O)NC3=NC=NC=C3)C2)C(C)(C)C"),
]

init_tokenizer(
    ckpt_path="/Users/taka/Documents/DTI/model_epoch_3.pt",
    device=-1,
)

all_z = []
all_tokens = []
all_names = []
all_atom_symbols = []

for name, smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("bad smiles:", name)
        continue

    tokens, kid, cid, id2safe, payload = infer_one(
        _GLOBAL["model"],
        _GLOBAL["dev"],
        smi,
        maxn=_GLOBAL["maxn"],
        d_attr=_GLOBAL["d_attr"],
        return_latents=True,
    )

    z = payload["h_vq"]  # (N_atoms, dim)
    all_z.append(z)

    all_tokens.extend(tokens)
    all_names.extend([name] * len(tokens))
    all_atom_symbols.extend([a.GetSymbol() for a in mol.GetAtoms()])

Z = torch.cat(all_z, dim=0)

torch.save(
    {
        "Z": Z,
        "tokens": all_tokens,
        "names": all_names,
        "atom_symbols": all_atom_symbols,
    },
    "vqatom_umap_input.pt",
)

print("saved: vqatom_umap_input.pt")
print("Z shape:", Z.shape)

# ===== UMAP =====
import umap
import numpy as np

emb = umap.UMAP(
    n_neighbors=10,
    min_dist=0.15,
    random_state=0,
).fit_transform(Z.numpy())

plt.figure(figsize=(7, 6))
plt.scatter(
    emb[:, 0],
    emb[:, 1],
    c=np.array(all_tokens),
    s=35,
    cmap="tab20",
)
plt.colorbar(label="VQ-Atom token ID")
plt.title("UMAP of atom-level VQ-Atom latent representations")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("vqatom_umap_by_token.png", dpi=300)
print("saved: vqatom_umap_by_token.png")