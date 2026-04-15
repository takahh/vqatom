from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem

from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL

# ----------------------------
# 1) init
# ----------------------------
init_tokenizer(
    ckpt_path="/Users/taka/Documents/DTI/model_epoch_3.pt",
    device=-1,
)

smiles = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

tokens, kid_list, cid_list, id2safe = infer_one(
    _GLOBAL["model"],
    _GLOBAL["dev"],
    smiles,
    maxn=_GLOBAL["maxn"],
    d_attr=_GLOBAL["d_attr"],
)

# ----------------------------
# 2) mol
# ----------------------------
mol = Chem.MolFromSmiles(smiles)
AllChem.Compute2DCoords(mol)

# ============================
# ① アノテーション付き
# ============================
mol_annot = Chem.Mol(mol)

for i, atom in enumerate(mol_annot.GetAtoms()):
    # atom.SetProp("atomLabel", atom.GetSymbol())
    atom.SetProp("atomLabel", "")
    atom.SetProp("atomNote", str(tokens[i]))

drawer = rdMolDraw2D.MolDraw2DCairo(1400, 900)
opts = drawer.drawOptions()
opts.annotationFontScale = 1.1
opts.additionalAtomLabelPadding = 0.3
opts.baseFontSize = 1.3
opts.bondLineWidth = 2
opts.padding = 0.08

drawer.DrawMolecule(mol_annot)
drawer.FinishDrawing()

with open("caffeine_with_tokens.png", "wb") as f:
    f.write(drawer.GetDrawingText())

print("saved: caffeine_with_tokens.png")

# ============================
# ② アノテーションなし
# ============================
mol_plain = Chem.Mol(mol)

# ←これ重要：ラベルやノートを一切つけない

drawer2 = rdMolDraw2D.MolDraw2DCairo(700, 300)
opts2 = drawer2.drawOptions()

opts2.baseFontSize = 1.25
opts2.bondLineWidth = 2
opts2.padding = 0.08

drawer2.DrawMolecule(mol_plain)
drawer2.FinishDrawing()

with open("caffeine_plain.png", "wb") as f:
    f.write(drawer2.GetDrawingText())

print("saved: caffeine_plain.png")

# ============================
# ② アノテーションなし・元素なし
# ============================
mol_plain = Chem.Mol(mol)

for atom in mol_plain.GetAtoms():
    atom.SetProp("atomLabel", "")   # ← 元素記号を消す

drawer2 = rdMolDraw2D.MolDraw2DCairo(1400, 900)
opts2 = drawer2.drawOptions()

opts2.baseFontSize = 1.25
opts2.bondLineWidth = 2.5
opts2.padding = 0.08

drawer2.DrawMolecule(mol_plain)
drawer2.FinishDrawing()

with open("caffeine_plain_no_elements.png", "wb") as f:
    f.write(drawer2.GetDrawingText())

print("saved: caffeine_plain_no_elements.png")