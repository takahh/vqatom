from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from PIL import Image, ImageDraw, ImageFont
import io

from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL

# ===============================
# 入力
# ===============================
example_smiles = {
    "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib": "COCCOc1ccc2ncnc(Nc3cccc(C#C)c3)c2c1",
}

init_tokenizer(
    ckpt_path="/Users/taka/Documents/DTI/model_epoch_3.pt",
    device=-1,
)

# ===============================
# レイアウト固定（ここ重要）
# ===============================
LEFT_W = 1450
RIGHT_W = 1450
ARROW_W = 140
PAIR_W = LEFT_W + ARROW_W + RIGHT_W
OUTER_GAP = 100


# ===============================
# 描画関数
# ===============================
def draw_single_mol(mol, width, height, token_mode=False):
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()

    opts.padding = 0.003 if token_mode else 0.08
    opts.bondLineWidth = 1.0 if token_mode else 2.4
    opts.baseFontSize = 1.0

    if token_mode:
        opts.annotationFontScale = 1.35
        opts.additionalAtomLabelPadding = 0.12
        opts.setSymbolColour((0.90, 0.90, 0.90))

    if not token_mode:
        for atom in mol.GetAtoms():
            atom.SetProp("atomLabel", "")

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 68)
    except:
        font = ImageFont.load_default()

    for atom in mol.GetAtoms():
        if token_mode:
            continue

        sym = atom.GetSymbol()
        if sym == "C":
            continue

        pt = drawer.GetDrawCoords(atom.GetIdx())
        x, y = int(pt.x), int(pt.y)

        bbox = draw.textbbox((0, 0), sym, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        pad = 20
        draw.rectangle(
            [x - tw//2 - pad, y - th//2 - pad,
             x + tw//2 + pad, y + th//2 + pad],
            fill="white",
        )

        draw.text((x - tw//2, y - th//2), sym, fill="black", font=font)

    return img


def add_title(img, title, font_size=100, title_height=120):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + title_height), "white")
    canvas.paste(img, (0, title_height))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text(
        ((w - tw)//2, (title_height - th)//2),
        title,
        fill="black",
        font=font,
    )

    return canvas


# ===============================
# メイン処理
# ===============================
pair_images = []

for mol_name, smiles in example_smiles.items():
    print(f"processing: {mol_name}")

    tokens, *_ = infer_one(
        _GLOBAL["model"],
        _GLOBAL["dev"],
        smiles,
        maxn=_GLOBAL["maxn"],
        d_attr=_GLOBAL["d_attr"],
    )

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    # --- 左（通常）
    mol_plain = Chem.Mol(mol)
    img_plain = draw_single_mol(mol_plain, LEFT_W, 1320, token_mode=False)
    img_plain = add_title(img_plain, mol_name)

    # --- 右（token）
    mol_token = Chem.Mol(mol)
    for i, atom in enumerate(mol_token.GetAtoms()):
        atom.SetProp("atomLabel", "")
        atom.SetProp("atomNote", str(tokens[i]))

    img_token = draw_single_mol(mol_token, RIGHT_W, 1320, token_mode=True)
    img_token = add_title(img_token, f"{mol_name} with VQ-Atom IDs")

    # ===============================
    # ペア生成（完全整列）
    # ===============================
    pair_h = max(img_plain.height, img_token.height)
    pair_canvas = Image.new("RGB", (PAIR_W, pair_h), "white")

    # 左固定
    pair_canvas.paste(
        img_plain,
        (0, (pair_h - img_plain.height)//2)
    )

    # 右固定
    pair_canvas.paste(
        img_token,
        (LEFT_W + ARROW_W, (pair_h - img_token.height)//2)
    )

    # 矢印（中央固定）
    draw = ImageDraw.Draw(pair_canvas)
    y = pair_h // 2
    x1 = LEFT_W + 20
    x2 = LEFT_W + ARROW_W - 20

    draw.line([(x1, y), (x2, y)], fill="black", width=6)
    draw.polygon([(x2, y), (x2-25, y-15), (x2-25, y+15)], fill="black")

    pair_images.append(pair_canvas)


# ===============================
# 縦連結（中央揃え）
# ===============================
final_w = PAIR_W
final_h = sum(img.height for img in pair_images) + OUTER_GAP * (len(pair_images)-1)

final_canvas = Image.new("RGB", (final_w, final_h), "white")

y = 0
for img in pair_images:
    final_canvas.paste(img, (0, y))
    y += img.height + OUTER_GAP


# ===============================
# ファイル名自動生成
# ===============================
name_str = "_".join(example_smiles.keys())
out_path = f"{name_str}_VQAtom.png"

final_canvas.save(out_path, dpi=(300, 300))

print(f"saved: {out_path}")