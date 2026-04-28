from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from PIL import Image, ImageDraw, ImageFont
import io

from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL

example_smiles = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Imatinib": "CC1=CC(=NC(=N1)NC2=CC(=C(C=C2)Cl)C(=O)NCC3=CN=CC=C3)C",
}

init_tokenizer(
    ckpt_path="/Users/taka/Documents/DTI/model_epoch_3.pt",
    device=-1,
)

def draw_single_mol(
    mol,
    width=1350,
    height=1020,
    token_mode=False,
):
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()

    opts.padding = 0.0775
    if token_mode:
        opts.padding = 0.003
    opts.bondLineWidth = 2.4
    opts.baseFontSize = 1.0   # ← ここは大きくしすぎない

    if token_mode:
        opts.annotationFontScale = 1.35
        opts.additionalAtomLabelPadding = 0.12

        # 構造線と元素記号を薄く
        opts.setSymbolColour((0.90, 0.90, 0.90))
        opts.bondLineWidth = 1.0

        try:
            opts.setAnnotationColour((0.0, 0.0, 0.0))
        except Exception:
            pass
    # =========================================
    # RDKit側の元素記号を消す（PILで描き直すため）
    # =========================================
    if not token_mode:
        for atom in mol.GetAtoms():
            atom.SetProp("atomLabel", "")

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png)).convert("RGB")

    # =========================================
    # 元素記号だけ後から大きく上書きする
    # =========================================
    draw = ImageDraw.Draw(img)

    try:
        elem_font = ImageFont.truetype("Arial.ttf", 68)
    except Exception:
        elem_font = ImageFont.load_default()

    for atom in mol.GetAtoms():

        if token_mode:
            continue
        sym = atom.GetSymbol()

        # 炭素は通常省略。必要ならこの if を消す
        if sym == "C":
            continue

        # RDKit の描画座標を取得
        pt = drawer.GetDrawCoords(atom.GetIdx())
        x, y = int(pt.x), int(pt.y)

        color = "black" if not token_mode else (120, 120, 120)

        bbox = draw.textbbox((0, 0), sym, font=elem_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 背景を少し白抜きして見やすくする
        pad = 20
        draw.rectangle(
            [x - tw // 2 - pad, y - th // 2 - pad,
             x + tw // 2 + pad, y + th // 2 + pad],
            fill="white",
        )

        draw.text(
            (x - tw // 2, y - th // 2),
            sym,
            fill=color,
            font=elem_font,
        )

    return img

def add_title(
    img,
    title,
    font_size=142,
    title_height=220,
):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + title_height), "white")
    canvas.paste(img, (0, title_height))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        ((w - tw) // 2, (title_height - th) // 2),
        title,
        fill="black",
        font=font,
    )

    return canvas

pair_images = []

for mol_name, smiles in example_smiles.items():
    print(f"processing: {mol_name}")

    tokens, kid_list, cid_list, id2safe = infer_one(
        _GLOBAL["model"],
        _GLOBAL["dev"],
        smiles,
        maxn=_GLOBAL["maxn"],
        d_attr=_GLOBAL["d_attr"],
    )

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    # 左：通常の分子図
    mol_plain = Chem.Mol(mol)

    img_plain = draw_single_mol(
        mol_plain,
        width=1150,
        height=1220,
        token_mode=False,
    )

    img_plain = add_title(
        img_plain,
        mol_name,
        font_size=100,
        title_height=120,
    )

    # 右：VQ-Atom ID付き
    mol_token = Chem.Mol(mol)

    for i, atom in enumerate(mol_token.GetAtoms()):
        atom.SetProp("atomLabel", "")
        atom.SetProp("atomNote", str(tokens[i]))

    img_token = draw_single_mol(
        mol_token,
        width=1450,
        height=1320,
        token_mode=True,
    )

    img_token = add_title(
        img_token,
        f"{mol_name} with VQ-Atom IDs",
        font_size=102,
        title_height=100,
    )

    # =================================================
    # 1分子分：plain → token を横ペアにする（矢印付き）
    # =================================================

    arrow_width = 140

    pair_w = img_plain.width + arrow_width + img_token.width
    pair_h = max(img_plain.height, img_token.height)

    pair_canvas = Image.new("RGB", (pair_w, pair_h), "white")

    # 左：通常分子図
    pair_canvas.paste(
        img_plain,
        (0, (pair_h - img_plain.height) // 2)
    )

    # 右：VQ-Atom ID付き
    right_x = img_plain.width + arrow_width
    pair_canvas.paste(
        img_token,
        (right_x, (pair_h - img_token.height) // 2)
    )

    # =================================================
    # 中央の矢印
    # =================================================
    draw = ImageDraw.Draw(pair_canvas)

    y_center = pair_h // 2
    x1 = img_plain.width + 20
    x2 = img_plain.width + arrow_width - 20

    # 矢印の線
    draw.line(
        [(x1, y_center), (x2, y_center)],
        fill="black",
        width=8,
    )

    # 矢印の先端
    draw.polygon(
        [
            (x2, y_center),
            (x2 - 30, y_center - 20),
            (x2 - 30, y_center + 20),
        ],
        fill="black",
    )

    pair_images.append(pair_canvas)


# =================================================
# Aspirin ペア + Imatinib ペアを横に連結
# =================================================
outer_gap = 2

final_w = sum(img.width for img in pair_images) + outer_gap * (len(pair_images) - 1)
final_h = max(img.height for img in pair_images)

final_canvas = Image.new("RGB", (final_w, final_h), "white")

x = 0
for img in pair_images:
    final_canvas.paste(img, (x, (final_h - img.height) // 2))
    x += img.width + outer_gap

out_path = "Aspirin_Imatinib_horizontal_pair.png"
final_canvas.save(out_path, dpi=(300, 300))

print(f"saved: {out_path}")
print("done.")