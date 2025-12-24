from train_teacher import get_args
import torch.nn as nn
import torch


class BondWeightLayer(nn.Module):
    def __init__(self, bond_types=4, hidden_dim=64):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)  # Learnable bond representation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.edge_mlp = self.edge_mlp.to(device)  # Move edge MLP to correct device

        nn.init.xavier_uniform_(self.bond_embedding.weight)  # Xavier for embeddings
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, edge_types):
        bond_feats = self.bond_embedding(edge_types)  # Convert bond type to learnable vector
        edge_weight = self.edge_mlp(bond_feats).squeeze()  # Compute edge weight
        edge_weight = edge_weight.squeeze()  # Compute edge weight
        return edge_weight

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


import torch.nn.functional as F
from torch_geometric.nn import GINEConv

import torch.nn as nn
from vq import VectorQuantize  # Ensure correct import


class BondWeightLayer(nn.Module):
    def __init__(self, bond_types=4, hidden_dim=64):
        import torch
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)  # Learnable bond representation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output weight in range (0,1)
        )
        self.edge_mlp = self.edge_mlp.to(device)  # Move edge MLP to correct device

    def forward(self, edge_types):
        bond_feats = self.bond_embedding(edge_types)  # Convert bond type to learnable vector
        edge_weight = self.edge_mlp(bond_feats).squeeze()  # Compute edge weight
        return edge_weight

import torch

def _zeros_like(x, device=None, dtype=None):
    return torch.zeros((), device=device or getattr(x, "device", None), dtype=dtype or getattr(x, "dtype", None))
import torch

def _normalize_quantize_output(qo, logger, device=None, dtype=None):
    """
    Normalize quantizer outputs to 7-tuple:
      (loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss)

    Assumption (ALWAYS):
      qo == (total_loss, (commit_loss, codebook_loss, repel_loss, cb_repel_loss))

    Rules:
      - Never detach.
      - Never wrap existing tensors into new tensors.
      - Only convert python/np scalars/None to tensors.
    """

    def _infer_device_dtype(*xs):
        nonlocal device, dtype
        if device is None or dtype is None:
            for x in xs:
                if torch.is_tensor(x):
                    if device is None:
                        device = x.device
                    if dtype is None:
                        dtype = x.dtype
                    break
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

    def _scalar_to_tensor(x):
        if torch.is_tensor(x):
            return x
        _infer_device_dtype()
        if x is None:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    logger.info("norm q 0")

    def _extract_loss(qo_):
        # case1: already scalar/tensor loss
        if torch.is_tensor(qo_) or isinstance(qo_, (float, int)):
            return qo_, (0.0, 0.0, 0.0, 0.0)

        # case2: (loss, inner)
        if isinstance(qo_, (tuple, list)) and len(qo_) == 2:
            loss_, inner_ = qo_
            if isinstance(inner_, (tuple, list)) and len(inner_) == 4:
                return loss_, tuple(inner_)
            # inner malformed -> treat as zeros
            return loss_, (0.0, 0.0, 0.0, 0.0)

        # case3: model forward style (quantize, embed_ind_dict, embed)
        if isinstance(qo_, (tuple, list)) and len(qo_) == 3:
            raise TypeError(
                "qo looks like model forward output (quantize, embed_ind_dict, embed), "
                "not (loss, (commit, cb, rep, cb_rep)). "
                "You probably passed the wrong variable into this normalization."
            )

        raise TypeError(f"Unrecognized qo format: type={type(qo_)} value={repr(qo_)[:500]}")

    loss, inner = _extract_loss(qo)
    logger.info("norm q 1")

    # inner is guaranteed 4-tuple here (or zeros)
    if not isinstance(inner, (tuple, list)) or len(inner) != 4:
        raise TypeError(
            f"Expected inner=(commit, cb, rep, cb_rep), "
            f"got type={type(inner)} len={len(inner) if hasattr(inner, '__len__') else 'NA'} value={inner}"
        )

    commit, cb, rep, cb_rep = inner
    logger.info("norm q 2")

    _infer_device_dtype(loss, commit, cb, rep, cb_rep)

    embed = None  # qoにはembedが無い前提
    sil = 0.0     # silhouette_loss はここでは 0 埋め

    logger.info("norm q 3")

    return (
        _scalar_to_tensor(loss),
        embed,
        _scalar_to_tensor(commit),
        _scalar_to_tensor(cb),
        _scalar_to_tensor(sil),
        _scalar_to_tensor(rep),
        _scalar_to_tensor(cb_rep),
    )

import torch.nn as nn
import torch
import torch.nn as nn
from utils import CORE_ELEMENTS
import torch
import torch.nn as nn


class AtomEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- 埋め込み定義（離散 0-29 部分）----
        self.degree_embed   = nn.Embedding(num_embeddings=7, embedding_dim=4)  # 0..6
        self.ring_embed     = nn.Embedding(num_embeddings=2, embedding_dim=4)  # ring flag 0/1
        self.charge_embed   = nn.Embedding(num_embeddings=8, embedding_dim=4)  # 0..7 にクリップ
        self.aromatic_embed = nn.Embedding(num_embeddings=2, embedding_dim=4)  # 0/1
        self.hybrid_embed   = nn.Embedding(num_embeddings=6, embedding_dim=4)  # 0..5
        self.hydrogen_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)  # 0..4

        # ---- element (Z) ----
        # 既知元素 + UNK を 1 つ追加
        ELEMENTS = [5, 6, 7, 8, 14, 15, 16]
        self.ELEMENTS = ELEMENTS
        self.UNK_ELEM_INDEX = len(ELEMENTS)  # 最後をUNK

        self.register_buffer("element_lut", self._build_element_lut(ELEMENTS, unk_index=self.UNK_ELEM_INDEX))
        self.element_embed = nn.Embedding(num_embeddings=len(ELEMENTS) + 1, embedding_dim=4)  # +1 for UNK

        # 0/1 フラグ系は全部 2 クラス想定
        def flag_emb():
            return nn.Embedding(num_embeddings=2, embedding_dim=2)

        # 官能基フラグ 18 個 (7-24)
        self.func_embeds = nn.ModuleList([flag_emb() for _ in range(18)])

        # H-bond Donor / Acceptor (25, 26)
        self.h_don_embed = flag_emb()
        self.h_acc_embed = flag_emb()

        # ring size / aromatic neighbors / fused-id (27, 28, 29)
        # ringSize のユニーク値: [0,3,4,5,6,7,8] → 7種類
        self.ringsize_embed  = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.aroma_num_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)  # 0..4
        self.fused_id_embed  = nn.Embedding(num_embeddings=8, embedding_dim=4)  # 0..7

        # 官能基フラグ 18 個 (各 2 次元) → 36 次元を 4 次元に圧縮
        self.func_reduce = nn.Linear(18 * 2, 4)

        # ringSize の LUT（高速化）
        # raw: 0,3,4,5,6,7,8 を 0..6 に写像。その他は最後(=6)に落とす。
        ring_vals = [0, 3, 4, 5, 6, 7, 8]
        self.register_buffer("ringsize_lut", self._build_value_lut(ring_vals, unk_index=len(ring_vals) - 1))

        # ---- bond_env_raw (30-77, 48 dims) 用の射影 ----
        self.bond_env_proj = nn.Linear(48, 16)

        # このクラスの最終出力次元:
        #   離散部: 7*4 + 4 + 2*2 + 3*4 = 48
        #   bond_env_proj: 16
        #   合計 = 64
        self.out_dim = 48 + 16

    @staticmethod
    def _build_element_lut(elements, unk_index: int):
        """
        Z -> index (0..len(elements)) の LUT。
        - elements に含まれない Z は unk_index に落とす
        - Z が範囲外 (>=lut_size) になった場合は forward 側で clamp する
        """
        max_z = max(elements)
        lut = torch.full((max_z + 1,), fill_value=unk_index, dtype=torch.long)
        for i, z in enumerate(elements):
            lut[z] = i
        return lut

    @staticmethod
    def _build_value_lut(allowed_values, unk_index: int):
        """
        value -> index LUT を作る（負や大きい値は forward 側で処理）。
        allowed_values の max まで LUT を作り、存在しない値は unk_index。
        """
        mx = max(allowed_values)
        lut = torch.full((mx + 1,), fill_value=unk_index, dtype=torch.long)
        for i, v in enumerate(allowed_values):
            lut[v] = i
        return lut

    def forward(self, atom_inputs: torch.Tensor) -> torch.Tensor:
        """
        atom_inputs: Tensor [N, 78] を想定

        0 : Z
        1 : degree
        2 : charge
        3 : hyb
        4 : arom
        5 : ring
        6 : hcount
        7-24  : func_flags (18)
        25    : H-donor flag
        26    : H-acceptor flag
        27    : ringSize (0,3,4,5,6,7,8,...)
        28    : #aromatic neighbors
        29    : fused ring id (0..7)
        30-77 : bond_env_raw (48)  ← sum+max bond features
        """
        # device は buffer 側（LUT）に合わせるのが安全
        device = self.element_lut.device
        atom_inputs = atom_inputs.to(device, non_blocking=True)

        # 0: element (Z) -> known index or UNK
        z_raw = atom_inputs[:, 0].long()

        # z_raw を LUT 範囲に入れる（負や >max_z は一旦 0 にして後で mask で UNK にする）
        max_z = self.element_lut.shape[0] - 1
        z_safe = z_raw.clamp(0, max_z)
        idx0 = self.element_lut[z_safe]

        # ただし、範囲外 (z_raw<0 or z_raw>max_z) は UNK に落とす
        out_of_range = (z_raw < 0) | (z_raw > max_z)
        if out_of_range.any():
            idx0 = idx0.clone()
            idx0[out_of_range] = self.UNK_ELEM_INDEX

        x0 = self.element_embed(idx0.clamp(0, self.element_embed.num_embeddings - 1))

        # 1: degree
        idx1 = atom_inputs[:, 1].long().clamp(0, self.degree_embed.num_embeddings - 1)
        x1 = self.degree_embed(idx1)

        # 2: ring flag (0/1)  ※ここが修正点（+1しない）
        ring_idx = atom_inputs[:, 5].long().clamp(0, 1)
        x2 = self.ring_embed(ring_idx)

        # 3: charge
        chg_idx = atom_inputs[:, 2].long().clamp(0, self.charge_embed.num_embeddings - 1)
        x3 = self.charge_embed(chg_idx)

        # 4: aromatic (0/1)
        arom_idx = atom_inputs[:, 4].long().clamp(0, 1)
        x4 = self.aromatic_embed(arom_idx)

        # 5: hybrid
        hyb_idx = atom_inputs[:, 3].long().clamp(0, self.hybrid_embed.num_embeddings - 1)
        x5 = self.hybrid_embed(hyb_idx)

        # 6: #hydrogens
        h_idx = atom_inputs[:, 6].long().clamp(0, self.hydrogen_embed.num_embeddings - 1)
        x6 = self.hydrogen_embed(h_idx)

        # ---- functional flags (7～24) / H-don / H-acc を 0/1 に clamp ----
        def flag_idx(col: int) -> torch.Tensor:
            return atom_inputs[:, col].long().clamp(0, 1)

        # 7–24: 18 functional flags → ModuleList でまとめて処理
        func_embs = [emb(flag_idx(7 + i)) for i, emb in enumerate(self.func_embeds)]
        flags = torch.cat(func_embs, dim=-1)  # [N, 36]
        flags4 = self.func_reduce(flags)      # [N, 4]

        # 25, 26: H-bond donor / acceptor
        x25 = self.h_don_embed(flag_idx(25))
        x26 = self.h_acc_embed(flag_idx(26))

        # 27: ringSize (0,3,4,5,6,7,8 → 0..6)
        raw27 = atom_inputs[:, 27].long()
        # 負は 0 に寄せる。max より大きい値は LUT 参照できないので clamp。
        raw27_safe = raw27.clamp(0, self.ringsize_lut.shape[0] - 1)
        mapped27 = self.ringsize_lut[raw27_safe]
        # 元が範囲外 (raw27 > max_supported) も最後(=UNK扱い)に落とす
        too_large = raw27 > (self.ringsize_lut.shape[0] - 1)
        if too_large.any():
            mapped27 = mapped27.clone()
            mapped27[too_large] = self.ringsize_embed.num_embeddings - 1
        x27 = self.ringsize_embed(mapped27.clamp(0, self.ringsize_embed.num_embeddings - 1))

        # 28: #aromatic neighbors
        idx28 = atom_inputs[:, 28].long().clamp(0, self.aroma_num_embed.num_embeddings - 1)
        x28 = self.aroma_num_embed(idx28)

        # 29: fused ring id
        idx29 = atom_inputs[:, 29].long().clamp(0, self.fused_id_embed.num_embeddings - 1)
        x29 = self.fused_id_embed(idx29)

        # 30-77: bond_env_raw (48 dims, float)
        bond_env = atom_inputs[:, 30:].to(torch.float32)  # [N, 48]
        bond_env_emb = self.bond_env_proj(bond_env)       # [N, 16]

        # 離散部分の埋め込み (48 dims) ＋ bond_env_emb (16 dims) = 64 dims
        out = torch.cat(
            [
                x0, x1, x2, x3, x4, x5, x6,   # 7*4 = 28
                flags4,                       # 4
                x25, x26,                     # 2+2 = 4
                x27, x28, x29,                # 3*4 = 12
                bond_env_emb,                 # 16
            ],
            dim=-1,
        )
        return out

class EquivariantThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super().__init__()
        import torch
        if args is None:
            args = get_args()

        self.feat_embed = AtomEmbedding()
        # AtomEmbedding → 64 dims (48 from discrete + 16 from bond_env)
        self.linear_0 = nn.Linear(64, args.hidden_dim)  # h0

        edge_emb_dim = getattr(args, "edge_emb_dim", 32)
        self.bond_emb = nn.Embedding(5, edge_emb_dim, padding_idx=0)

        def mlp(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f), nn.ReLU(),
                nn.Linear(out_f, out_f), nn.ReLU()
            )

        nn1 = mlp(args.hidden_dim, hidden_feats)
        nn2 = mlp(hidden_feats, hidden_feats)
        nn3 = mlp(hidden_feats, hidden_feats)

        self.gine1 = GINEConv(nn1, edge_dim=edge_emb_dim)
        self.gine2 = GINEConv(nn2, edge_dim=edge_emb_dim)
        self.gine3 = GINEConv(nn3, edge_dim=edge_emb_dim)

        self.ln_in = nn.LayerNorm(args.hidden_dim)
        self.ln1   = nn.LayerNorm(hidden_feats)
        self.ln2   = nn.LayerNorm(hidden_feats)
        self.ln3   = nn.LayerNorm(hidden_feats)

        # Residual scales
        self.res1 = nn.Parameter(torch.tensor(0.5))
        self.res2 = nn.Parameter(torch.tensor(0.5))
        self.res3 = nn.Parameter(torch.tensor(0.5))

        # If args.hidden_dim != hidden_feats, project skip for hop1
        self.skip0 = None
        if args.hidden_dim != hidden_feats:
            self.skip0 = nn.Linear(args.hidden_dim, hidden_feats, bias=False)

        # JK: concat h0 + h1 + h2 + h3
        jk_dim = args.hidden_dim + 3 * hidden_feats
        self.mix = nn.Sequential(
            nn.Linear(jk_dim, 2 * hidden_feats), nn.ReLU(),
            nn.Linear(2 * hidden_feats, hidden_feats), nn.ReLU(),
        )

        # Project to VQ dim
        self.out_proj   = nn.Linear(hidden_feats, args.hidden_dim)
        self.pre_vq_ln  = nn.LayerNorm(args.hidden_dim)

        self.vq = VectorQuantize(
            dim=args.hidden_dim,
            codebook_size=args.codebook_size,
            decay=getattr(args, "ema_decay", 0.8),
            threshold_ema_dead_code=2,
        )
        edge_emb_dim = getattr(args, "edge_emb_dim", 32)
        self.bond_emb = nn.Embedding(5, edge_emb_dim, padding_idx=0)  # 0..4
        self.hop_emb = nn.Embedding(4, edge_emb_dim, padding_idx=0)  # 0..3

    def reset_kmeans(self):
        self.vq._codebook.reset_kmeans()

    def forward(self, data, features, chunk_i, mask_dict=None, logger=None, epoch=None,
                batched_graph_base=None, mode=None, attr_list=None):
        import torch
        dev = next(self.parameters()).device

        # ----------------------------
        # init_kmeans_final
        # ----------------------------
        if mode == "init_kmeans_final":
            if hasattr(data, "to"):
                data = data.to(dev)
            if torch.is_tensor(features):
                features = features.to(dev, non_blocking=True)
            self.vq(data, attr_list, mask_dict, logger, chunk_i, epoch, mode)
            return 0

        # ----------------------------
        # Edge build (mirror for undirected)
        # ----------------------------
        s1, d1 = data.edges()
        s1 = s1.to(dev, non_blocking=True)
        d1 = d1.to(dev, non_blocking=True)
        src = torch.cat([s1, d1], dim=0)
        dst = torch.cat([d1, s1], dim=0)
        edge_index = torch.stack([src, dst], dim=0)   # [2, 2E]

        # ----------------------------
        # Edge attributes:
        #   bond_type: 0..4 (0=none/self, 1..4=bond)
        #   edge_type: 0..3 (0=self/other, 1=1hop, 2=2hop, 3=3hop)
        # ----------------------------
        bt = data.edata.get("bond_type", None)
        if bt is None:
            # fallback (旧グラフ互換) - できれば convert_to_dgl 側で bond_type を必ず付ける
            bt = data.edata.get("weight", torch.zeros(data.num_edges(), dtype=torch.long, device=dev)).long()
        bt = bt.to(dev, non_blocking=True).long()
        bt = torch.where((bt >= 1) & (bt <= 4), bt, torch.zeros_like(bt))
        bt = torch.cat([bt, bt], dim=0)  # mirror -> [2E]

        et = data.edata.get("edge_type", None)
        if et is None:
            et = torch.zeros(data.num_edges(), dtype=torch.long, device=dev)
        et = et.to(dev, non_blocking=True).long().clamp(0, 3)
        et = torch.cat([et, et], dim=0)  # mirror -> [2E]

        # bond + hop
        edge_attr = self.bond_emb(bt) + self.hop_emb(et)   # [2E, edge_emb_dim]

        if logger is not None:
            # NOTE: bt/et は mirror 後なので 2E 個。unique/ratio の確認には十分。
            logger.info(f"[EDGE] bond_type unique={torch.unique(bt)[:10].tolist()}")
            logger.info(f"[EDGE] edge_type unique={torch.unique(et)[:10].tolist()}")
            logger.info(f"[EDGE] bond_nonzero_ratio={(bt != 0).float().mean().item():.4f}")
            logger.info(f"[EDGE] hop_nonzero_ratio={(et != 0).float().mean().item():.4f}")

        # ----------------------------
        # Node features -> h0
        # ----------------------------
        features = features.to(dev, non_blocking=True)
        h0_raw = self.feat_embed(features)             # [N, 64]
        h0 = self.ln_in(self.linear_0(h0_raw))         # [N, args.hidden_dim]

        # ----------------------------
        # 3-hop GINE + residual + LN
        # ----------------------------
        h0_for1 = self.skip0(h0) if self.skip0 is not None else h0

        h1 = self.gine1(h0, edge_index, edge_attr)
        h1 = self.ln1(h1 * self.res1 + h0_for1)

        h2 = self.gine2(h1, edge_index, edge_attr)
        h2 = self.ln2(h2 * self.res2 + h1)

        h3 = self.gine3(h2, edge_index, edge_attr)
        h3 = self.ln3(h3 * self.res3 + h2)

        # ----------------------------
        # JK concat -> mix -> out_proj
        # ----------------------------
        h_cat = torch.cat([h0, h1, h2, h3], dim=-1)
        h_mid = self.mix(h_cat)
        h_out = self.out_proj(h_mid)

        # ----------------------------
        # init_kmeans_loop: return pre-VQ vector
        # ----------------------------
        if mode == "init_kmeans_loop":
            return h_out

        # ----------------------------
        # VQ
        # ----------------------------
        h_vq = self.pre_vq_ln(h_out)

        quantize_output = self.vq(
            h_vq,
            attr_list,
            mask_dict,
            logger,
            chunk_i,
            epoch,
            mode,
        )

        loss, _embed_ignored, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss = _normalize_quantize_output(
            quantize_output,
            logger,
            device=h_vq.device,
            dtype=h_vq.dtype,
        )

        # embed は現状 h_vq を返す（「量子化した埋め込み」を返す版にするならここを差し替え）
        embed = h_vq

        return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"],
                lamb_div_ele=conf["lamb_div_ele"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])

    def forward(self, data, feats, epoch, logger):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats, epoch, logger)

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats):
        if "SAGE" in self.model_name:
            # return self.forward(data, feats)

            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)
