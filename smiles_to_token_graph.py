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

common_tokens = set(token_dict["Gefitinib"]) & set(token_dict["Erlotinib"])


# ===============================
# コア揃え
# ===============================
template = Chem.MolFromSmiles(example_smiles["Gefitinib"])
AllChem.Compute2DCoords(template)

def align(mol):
    try:
        rdDepictor.GenerateDepictionMatching2DStructure(
            mol, template, acceptFailure=True
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
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)

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
    return img.resize((int(w * s), target_h), Image.BICUBIC)

def expand_canvas_x(img, scale=1.2):
    w, h = img.size
    new_w = int(w * scale)
    canvas = Image.new("RGB", (new_w, h), "white")
    off = (new_w - w) // 2
    canvas.paste(img, (off, 0))
    return canvas


# ===============================
# 描画
# ===============================
def draw_single_mol(mol, token_mode=False, highlight_tokens=None, width=1450, height=1320):
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
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

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    draw = ImageDraw.Draw(img)

    elem_font = ImageFont.truetype("Arial.ttf", 72)
    token_font = ImageFont.truetype("Arial.ttf", 44)
    bold_font  = ImageFont.truetype("Arial Bold.ttf", 48)

    for atom in mol.GetAtoms():
        pt = drawer.GetDrawCoords(atom.GetIdx())
        x, y = int(pt.x), int(pt.y)

        if token_mode:
            tok = int(atom.GetProp("vq_token"))
            is_common = highlight_tokens and tok in highlight_tokens

            font = bold_font if is_common else token_font
            color = (0,0,0) if is_common else (80,80,80)

            text = str(tok)
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

            # 白縁のみ
            for dx,dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                draw.text((x-tw//2+dx, y-th//2+dy), text, fill=(255,255,255), font=font)

            draw.text((x-tw//2, y-th//2), text, fill=color, font=font)
            continue

        sym = atom.GetSymbol()
        if sym == "C":
            continue

        bbox = draw.textbbox((0,0), sym, font=elem_font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

        draw.rectangle([x-tw//2-14, y-th//2-14, x+tw//2+14, y+th//2+14], fill="white")
        draw.text((x-tw//2, y-th//2), sym, fill="black", font=elem_font)

    return img


def add_title(img, title):
    w, h = img.size
    title_h = 180  # ←詰める

    canvas = Image.new("RGB", (w, h + title_h), "white")
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("Arial.ttf", 120)

    bbox = draw.textbbox((0,0), title, font=font)
    tw = bbox[2] - bbox[0]

    draw.text(
        ((w - tw) // 2, 13),  # ←上に寄せる
        title,
        fill="black",
        font=font,
    )

    return canvas


# ===============================
# メイン
# ===============================
pair_images = []

for mol_name, smiles in example_smiles.items():
    mol = Chem.MolFromSmiles(smiles)
    mol = align(mol)

    # Erlotinib: 反転＋回転
    if mol_name == "Erlotinib":
        mol = flip_mol_x(mol)
        mol = rotate_mol_2d(mol, -12)

    tokens = token_dict[mol_name]

    img_plain = add_title(
        draw_single_mol(mol, False, width=PLAIN_W, height=PLAIN_H),
        mol_name
    )

    mol_token = Chem.Mol(mol)
    for i,a in enumerate(mol_token.GetAtoms()):
        a.SetProp("vq_token", str(tokens[i]))

    img_token = add_title(
        draw_single_mol(mol_token, True, common_tokens, TOKEN_W, TOKEN_H),
        f"{mol_name} with VQ-Atom IDs"
    )

    # ★ 右側だけ最適化
    # ① 先に分子だけ描く
    img_token_raw = draw_single_mol(
        mol_token,
        token_mode=True,
        highlight_tokens=common_tokens,
        width=TOKEN_W,
        height=TOKEN_H,
    )

    # ② 余白削除
    img_token_raw = crop_white_margin(img_token_raw, pad=10)

    # ③ 高さフィット
    img_token_raw = fit_height(img_token_raw, int(img_plain.height * 0.92))

    # ④ タイトル追加（最後！）
    img_token = add_title(
        img_token_raw,
        f"{mol_name} with VQ-Atom IDs"
    )

    # ⑤ 横キャンバス拡張
    img_token = expand_canvas_x(img_token, scale=1.2)
    img_token = fit_height(img_token, int(img_plain.height * 0.97))

    pair_w = img_plain.width + 110 + img_token.width
    pair_h = max(img_plain.height, img_token.height)

    canvas = Image.new("RGB", (pair_w, pair_h), "white")

    canvas.paste(img_plain, (0, (pair_h - img_plain.height)//2))
    y_offset = (pair_h - img_token.height) // 2 - 40  # ←ここで上に調整
    canvas.paste(img_token, (img_plain.width + 110, y_offset))
    pair_images.append(canvas)
    draw = ImageDraw.Draw(canvas)

    y = pair_h // 2
    x1 = img_plain.width + 25
    x2 = img_plain.width + 325

    # 線
    draw.line([(x1, y), (x2, y)], fill="black", width=12)

    # 矢印ヘッド
    draw.polygon(
        [(x2, y), (x2 - 42, y - 38), (x2 - 42, y + 38)],
        fill="black"
    )

# ===============================
# 縦連結
# ===============================
gap = 120  # ←詰める

final_w = max(i.width for i in pair_images)
final_h = sum(i.height for i in pair_images) + gap * (len(pair_images) - 1)

final = Image.new("RGB", (final_w, final_h), "white")

y = 0
for img in pair_images:
    final.paste(img, ((final_w - img.width)//2, y))
    y += img.height + gap


# ===============================
# 保存
# ===============================
final.save("FINAL.png", dpi=(300,300))
print("saved FINAL.png")