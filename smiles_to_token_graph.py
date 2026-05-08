from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, rdDepictor
from PIL import Image, ImageDraw, ImageFont, ImageChops
import io
import math

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

PLAIN_W = 1450
PLAIN_H = 1320
TOKEN_W = 1700
TOKEN_H = 1400


# ===============================
# token取得
# ===============================
token_dict = {}

for mol_name, smiles in example_smiles.items():
    tokens, *_ = infer_one(
        _GLOBAL["model"],
        _GLOBAL["dev"],
        smiles,
        maxn=_GLOBAL["maxn"],
        d_attr=_GLOBAL["d_attr"],
    )
    token_dict[mol_name] = list(tokens)

common_tokens = (
    set(token_dict["Gefitinib"])
    & set(token_dict["Erlotinib"])
)


# ===============================
# コア揃え
# ===============================
template = Chem.MolFromSmiles(example_smiles["Gefitinib"])
AllChem.Compute2DCoords(template)

def align(mol):
    try:
        rdDepictor.GenerateDepictionMatching2DStructure(
            mol,
            template,
            acceptFailure=True,
        )
    except Exception:
        AllChem.Compute2DCoords(mol)

    return mol


# ===============================
# 幾何変換
# ===============================
def flip_mol_x(mol):
    conf = mol.GetConformer()

    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (-p.x, p.y, p.z))

    return mol


def rotate_mol_2d(mol, angle_deg):
    conf = mol.GetConformer()

    xs = [
        conf.GetAtomPosition(i).x
        for i in range(mol.GetNumAtoms())
    ]

    ys = [
        conf.GetAtomPosition(i).y
        for i in range(mol.GetNumAtoms())
    ]

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    th = math.radians(angle_deg)

    c = math.cos(th)
    s = math.sin(th)

    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)

        x = p.x - cx
        y = p.y - cy

        nx = cx + c * x - s * y
        ny = cy + s * x + c * y

        conf.SetAtomPosition(i, (nx, ny, p.z))

    return mol


# ===============================
# 画像処理
# ===============================
def crop_white_margin(img, pad=20):
    img_rgb = img.convert("RGB")

    bg = Image.new("RGB", img_rgb.size, "white")

    diff = ImageChops.difference(img_rgb, bg)

    bbox = diff.getbbox()

    if bbox is None:
        return img_rgb

    l, t, r, b = bbox

    l = max(l - pad, 0)
    t = max(t - pad, 0)

    r = min(r + pad, img_rgb.width)
    b = min(b + pad, img_rgb.height)

    return img_rgb.crop((l, t, r, b))


def fit_height(img, target_h):
    w, h = img.size

    s = target_h / h

    return img.resize(
        (int(w * s), target_h),
        Image.BICUBIC,
    )


def expand_canvas_x(img, scale=1.15):
    w, h = img.size

    new_w = int(w * scale)

    canvas = Image.new("RGB", (new_w, h), "white")

    off = (new_w - w) // 2

    canvas.paste(img, (off, 0))

    return canvas


# ===============================
# タイトル重ね描画
# ===============================
def overlay_title(
    img,
    title,
    x=30,
    y=20,
    font_size=90,
):
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        "Arial.ttf",
        font_size,
    )

    # 白縁
    for dx, dy in [
        (-3,0),
        (3,0),
        (0,-3),
        (0,3),
        (-2,-2),
        (2,2),
    ]:
        draw.text(
            (x + dx, y + dy),
            title,
            fill="white",
            font=font,
        )

    draw.text(
        (x, y),
        title,
        fill="black",
        font=font,
    )

    return img


# ===============================
# 描画
# ===============================
def draw_single_mol(
    mol,
    token_mode=False,
    highlight_tokens=None,
    width=1450,
    height=1320,
):
    drawer = rdMolDraw2D.MolDraw2DCairo(
        width,
        height,
    )

    opts = drawer.drawOptions()

    opts.fixedBondLength = 90

    if token_mode:
        opts.padding = 0.02
        opts.bondLineWidth = 1.0
        opts.setSymbolColour((0.78, 0.78, 0.78))
    else:
        opts.padding = 0.03
        opts.bondLineWidth = 2.4

    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", "")

    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    img = Image.open(
        io.BytesIO(drawer.GetDrawingText())
    ).convert("RGB")

    draw = ImageDraw.Draw(img)

    elem_font = ImageFont.truetype(
        "Arial.ttf",
        72,
    )

    token_font = ImageFont.truetype(
        "Arial.ttf",
        44,
    )

    bold_font = ImageFont.truetype(
        "Arial Bold.ttf",
        48,
    )

    for atom in mol.GetAtoms():

        pt = drawer.GetDrawCoords(atom.GetIdx())

        x = int(pt.x)
        y = int(pt.y)

        if token_mode:

            tok = int(atom.GetProp("vq_token"))

            is_common = (
                highlight_tokens
                and tok in highlight_tokens
            )

            font = (
                bold_font
                if is_common
                else token_font
            )

            color = (
                (0,0,0)
                if is_common
                else (80,80,80)
            )

            text = str(tok)

            bbox = draw.textbbox(
                (0,0),
                text,
                font=font,
            )

            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            for dx,dy in [
                (-2,0),
                (2,0),
                (0,-2),
                (0,2),
            ]:
                draw.text(
                    (
                        x - tw//2 + dx,
                        y - th//2 + dy
                    ),
                    text,
                    fill=(255,255,255),
                    font=font,
                )

            draw.text(
                (
                    x - tw//2,
                    y - th//2
                ),
                text,
                fill=color,
                font=font,
            )

            continue

        sym = atom.GetSymbol()

        if sym == "C":
            continue

        bbox = draw.textbbox(
            (0,0),
            sym,
            font=elem_font,
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle(
            [
                x - tw//2 - 14,
                y - th//2 - 14,
                x + tw//2 + 14,
                y + th//2 + 14,
            ],
            fill="white",
        )

        draw.text(
            (
                x - tw//2,
                y - th//2,
            ),
            sym,
            fill="black",
            font=elem_font,
        )

    return img


# ===============================
# メイン
# ===============================
pair_images = []

for mol_name, smiles in example_smiles.items():

    mol = Chem.MolFromSmiles(smiles)

    mol = align(mol)

    if mol_name == "Erlotinib":
        mol = flip_mol_x(mol)
        mol = rotate_mol_2d(mol, -12)

    tokens = token_dict[mol_name]

    # -------------------------
    # 左画像
    # -------------------------
    img_plain = draw_single_mol(
        mol,
        False,
        width=PLAIN_W,
        height=PLAIN_H,
    )

    img_plain = crop_white_margin(
        img_plain,
        pad=20,
    )

    img_plain = overlay_title(
        img_plain,
        mol_name,
        x=325,
        y=110,
        font_size=85,
    )

    # -------------------------
    # 右画像
    # -------------------------
    mol_token = Chem.Mol(mol)

    for i, a in enumerate(mol_token.GetAtoms()):
        a.SetProp(
            "vq_token",
            str(tokens[i]),
        )

    img_token = draw_single_mol(
        mol_token,
        token_mode=True,
        highlight_tokens=common_tokens,
        width=TOKEN_W,
        height=TOKEN_H,
    )

    img_token = crop_white_margin(
        img_token,
        pad=10,
    )

    img_token = overlay_title(
        img_token,
        f"{mol_name} with VQ-Atom IDs",
        x=25,
        y=5,  # 10 → 65
        font_size=70,
    )

    img_token = expand_canvas_x(
        img_token,
        scale=1.12,
    )

    img_token = fit_height(
        img_token,
        int(img_plain.height * 0.98),
    )

    # -------------------------
    # ペア結合
    # -------------------------
    pair_w = (
        img_plain.width
        + 90
        + img_token.width
    )

    pair_h = max(
        img_plain.height,
        img_token.height,
    )

    canvas = Image.new(
        "RGB",
        (pair_w, pair_h),
        "white",
    )

    canvas.paste(
        img_plain,
        (
            0,
            (pair_h - img_plain.height)//2,
        ),
    )

    canvas.paste(
        img_token,
        (
            img_plain.width + 90,
            (pair_h - img_token.height)//2,
        ),
    )

    draw = ImageDraw.Draw(canvas)

    y = pair_h // 2

    x1 = img_plain.width + 20
    x2 = img_plain.width + 250

    draw.line(
        [(x1, y), (x2, y)],
        fill="black",
        width=10,
    )

    draw.polygon(
        [
            (x2, y),
            (x2 - 35, y - 28),
            (x2 - 35, y + 28),
        ],
        fill="black",
    )

    pair_images.append(canvas)


# ===============================
# 縦連結
# ===============================
gap = 80

final_w = max(i.width for i in pair_images)

final_h = (
    sum(i.height for i in pair_images)
    + gap * (len(pair_images) - 1)
)

final = Image.new(
    "RGB",
    (final_w, final_h),
    "white",
)

y = 0

for img in pair_images:

    final.paste(
        img,
        (
            (final_w - img.width)//2,
            y,
        ),
    )

    y += img.height + gap


# ===============================
# 最終トリム
# ===============================
final = crop_white_margin(
    final,
    pad=10,
)

final.save(
    "FINAL.png",
    dpi=(300,300),
)

print("saved FINAL.png")