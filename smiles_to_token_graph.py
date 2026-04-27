from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from PIL import Image, ImageDraw, ImageFont
import io

from infer_one_smiles import init_tokenizer, infer_one, _GLOBAL

example_smiles = {
    "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib": "COCCOc1ccc2ncnc(Nc3cccc(C#C)c3)c2c1",
    "Imatinib": "CC1=CC(=NC(=N1)NC2=CC(=C(C=C2)Cl)C(=O)NCC3=CN=CC=C3)C",
    "Sorafenib": "CNC(=O)Nc1ccc(Oc2ccnc(n2)Nc3ccc(Cl)c(C(F)(F)F)c3)cc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}

init_tokenizer(
    ckpt_path="/Users/taka/Documents/DTI/model_epoch_3.pt",
    device=-1,
)


def draw_single_mol(
    mol,
    width=1800,
    height=800,
    token_mode=False,
):
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()

    opts.padding = 0.06
    opts.bondLineWidth = 2.4
    opts.baseFontSize = 1.0

    if token_mode:
        # Token ID を少し小さめに
        opts.annotationFontScale = 0.70
        opts.additionalAtomLabelPadding = 0.16

        # 構造線と元素記号をかなり薄く
        opts.setSymbolColour((0.94, 0.94, 0.94))
        opts.bondLineWidth = 1

        # Token ID は黒
        try:
            opts.setAnnotationColour((0.0, 0.0, 0.0))
        except Exception:
            pass

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png)).convert("RGB")


def add_title(
    img,
    title,
    font_size=72,
    title_height=120,
):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + title_height), "white")
    canvas.paste(img, (0, title_height))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
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
        width=1600,
        height=850,
        token_mode=False,
    )

    img_plain = add_title(
        img_plain,
        mol_name,
        font_size=72,
        title_height=120,
    )

    # 右：VQ-Atom ID付き
    mol_token = Chem.Mol(mol)

    for i, atom in enumerate(mol_token.GetAtoms()):
        atom.SetProp("atomLabel", "")
        atom.SetProp("atomNote", str(tokens[i]))

    img_token = draw_single_mol(
        mol_token,
        width=1600,
        height=850,
        token_mode=True,
    )

    img_token = add_title(
        img_token,
        f"{mol_name} with VQ-Atom IDs",
        font_size=72,
        title_height=120,
    )

    # 横に連結
    gap = 100

    w = img_plain.width + gap + img_token.width
    h = max(img_plain.height, img_token.height)

    canvas = Image.new("RGB", (w, h), "white")

    canvas.paste(
        img_plain,
        (0, (h - img_plain.height) // 2),
    )

    canvas.paste(
        img_token,
        (img_plain.width + gap, (h - img_token.height) // 2),
    )

    out_path = f"{mol_name}_horizontal_pair.png"
    canvas.save(out_path, dpi=(300, 300))

    print(f"saved: {out_path}")

print("done.")