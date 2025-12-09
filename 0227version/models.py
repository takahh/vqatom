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


import torch.nn as nn
import torch
import torch.nn as nn
from utils import CORE_ELEMENTS

class AtomEmbedding(nn.Module):
    def __init__(self):
        super(AtomEmbedding, self).__init__()

        # ---- 埋め込み定義（離散 0-29 部分）----
        self.degree_embed   = nn.Embedding(num_embeddings=7,   embedding_dim=4)   # 0..6
        self.ring_embed     = nn.Embedding(num_embeddings=2,   embedding_dim=4)   # ring flag+1 など
        self.charge_embed   = nn.Embedding(num_embeddings=8,   embedding_dim=4)   # 0..7 にクリップ
        self.aromatic_embed = nn.Embedding(num_embeddings=2,   embedding_dim=4)   # 0/1
        self.hybrid_embed   = nn.Embedding(num_embeddings=6,   embedding_dim=4)   # 0..5
        self.hydrogen_embed = nn.Embedding(num_embeddings=5,   embedding_dim=4)   # 0..4

        ELEMENTS = [5, 6, 7, 8, 14, 15, 16]
        self.register_buffer(
            "element_lut",
            self._build_element_lut(ELEMENTS)
        )
        self.element_embed = nn.Embedding(num_embeddings=len(ELEMENTS), embedding_dim=4)

        # 0/1 フラグ系は全部 2 クラス想定
        def flag_emb():
            return nn.Embedding(num_embeddings=2, embedding_dim=2)

        # 官能基フラグ 18 個 (7-24)
        self.func_embed_0  = flag_emb()
        self.func_embed_1  = flag_emb()
        self.func_embed_2  = flag_emb()
        self.func_embed_3  = flag_emb()
        self.func_embed_4  = flag_emb()
        self.func_embed_5  = flag_emb()
        self.func_embed_6  = flag_emb()
        self.func_embed_7  = flag_emb()
        self.func_embed_8  = flag_emb()
        self.func_embed_9  = flag_emb()
        self.func_embed_10 = flag_emb()
        self.func_embed_11 = flag_emb()
        self.func_embed_12 = flag_emb()
        self.func_embed_13 = flag_emb()
        self.func_embed_14 = flag_emb()
        self.func_embed_15 = flag_emb()
        self.func_embed_16 = flag_emb()
        self.func_embed_17 = flag_emb()

        # H-bond Donor / Acceptor (25, 26)
        self.h_don_embed   = flag_emb()
        self.h_acc_embed   = flag_emb()

        # ring size / aromatic neighbors / fused-id (27, 28, 29)
        # ringSize のユニーク値: ['0', '3', '4', '5', '6', '7', '8'] → 7種類
        self.ringsize_embed  = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.aroma_num_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)  # 0..4
        self.fused_if_embed  = nn.Embedding(num_embeddings=8, embedding_dim=4)  # 0..7

        # 官能基フラグ 18 個 (各 2 次元) → 36 次元を 4 次元に圧縮
        self.func_reduce = nn.Linear(18 * 2, 4)

        # ringSize の元の値 → index の対応（0,3,4,5,6,7,8 → 0..6）
        uniq = ['0', '3', '4', '5', '6', '7', '8']
        uniq_int = sorted(int(x) for x in uniq)  # [0,3,4,5,6,7,8]
        mapping = {v: i for i, v in enumerate(uniq_int)}
        self.register_buffer(
            "ring_values_tensor",
            torch.tensor(uniq_int, dtype=torch.long)
        )
        self.ring_value_to_index = mapping

        # ---- bond_env_raw (30-77, 48 dims) 用の射影 ----
        # 48 次元 → 16 次元に圧縮して concat
        self.bond_env_proj = nn.Linear(48, 16)

        # このクラスの最終出力次元:
        #   離散部: 7*4 + 4 + 2*2 + 3*4 = 48
        #   bond_env_proj: 16
        #   合計 = 64
        self.out_dim = 48 + 16

    @staticmethod
    def _build_element_lut(ELEMENTS):
        max_z = max(ELEMENTS)
        lut = torch.zeros(max_z + 1, dtype=torch.long)
        for i, z in enumerate(ELEMENTS):
            lut[z] = i
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
        device = next(self.parameters()).device
        atom_inputs = atom_inputs.to(device, non_blocking=True)

        # 0: element (Z)
        z_raw = atom_inputs[:, 0].long()
        max_z = self.element_lut.shape[0] - 1
        mask = (z_raw >= 0) & (z_raw <= max_z)
        z = torch.where(mask, z_raw, torch.zeros_like(z_raw))
        idx0 = self.element_lut[z]
        idx0 = idx0.clamp(0, self.element_embed.num_embeddings - 1)
        x0 = self.element_embed(idx0)

        # 1: degree
        idx1 = atom_inputs[:, 1].long().clamp(0, self.degree_embed.num_embeddings - 1)
        x1 = self.degree_embed(idx1)

        # 2: ring_embed 用スカラー（ここでは ring flag + 1）
        val_raw = atom_inputs[:, 5].long() + 1
        val_idx = val_raw.clamp(0, self.ring_embed.num_embeddings - 1)
        x2 = self.ring_embed(val_idx)

        # 3: charge
        chg_raw = atom_inputs[:, 2].long()
        chg_idx = chg_raw.clamp(0, self.charge_embed.num_embeddings - 1)
        x3 = self.charge_embed(chg_idx)

        # 4: aromatic (0/1)
        arom_idx = atom_inputs[:, 4].long().clamp(0, self.aromatic_embed.num_embeddings - 1)
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

        # 7–24: 18 functional flags
        x7  = self.func_embed_0(flag_idx(7))
        x8  = self.func_embed_1(flag_idx(8))
        x9  = self.func_embed_2(flag_idx(9))
        x10 = self.func_embed_3(flag_idx(10))
        x11 = self.func_embed_4(flag_idx(11))
        x12 = self.func_embed_5(flag_idx(12))
        x13 = self.func_embed_6(flag_idx(13))
        x14 = self.func_embed_7(flag_idx(14))
        x15 = self.func_embed_8(flag_idx(15))
        x16 = self.func_embed_9(flag_idx(16))
        x17 = self.func_embed_10(flag_idx(17))
        x18 = self.func_embed_11(flag_idx(18))
        x19 = self.func_embed_12(flag_idx(19))
        x20 = self.func_embed_13(flag_idx(20))
        x21 = self.func_embed_14(flag_idx(21))
        x22 = self.func_embed_15(flag_idx(22))
        x23 = self.func_embed_16(flag_idx(23))
        x24 = self.func_embed_17(flag_idx(24))

        # merge functional flags: 18 flags * 2 dim = 36 dim → 4 dim
        flags = torch.cat(
            [x7, x8, x9, x10, x11, x12, x13, x14, x15,
             x16, x17, x18, x19, x20, x21, x22, x23, x24],
            dim=-1
        )  # [N, 36]
        flags4 = self.func_reduce(flags)  # [N, 4]

        # 25, 26: H-bond donor / acceptor
        x25 = self.h_don_embed(flag_idx(25))
        x26 = self.h_acc_embed(flag_idx(26))

        # 27: ringSize (0,3,4,5,6,7,8 → 0..6 に写像)
        raw27 = atom_inputs[:, 27].long()
        mapped27 = torch.full_like(raw27, fill_value=self.ringsize_embed.num_embeddings - 1)
        for v, idx in self.ring_value_to_index.items():
            mapped27[raw27 == v] = idx
        mapped27 = mapped27.clamp(0, self.ringsize_embed.num_embeddings - 1)
        x27 = self.ringsize_embed(mapped27)

        # 28: #aromatic neighbors
        raw28 = atom_inputs[:, 28].long()
        idx28 = raw28.clamp(0, self.aroma_num_embed.num_embeddings - 1)
        x28 = self.aroma_num_embed(idx28)

        # 29: fused ring id
        raw29 = atom_inputs[:, 29].long()
        idx29 = raw29.clamp(0, self.fused_if_embed.num_embeddings - 1)
        x29 = self.fused_if_embed(idx29)

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

    def reset_kmeans(self):
        self.vq._codebook.reset_kmeans()

    def forward(self, data, features, chunk_i, mask_dict=None, logger=None, epoch=None,
                batched_graph_base=None, mode=None, attr_list=None):
        import torch
        dev = next(self.parameters()).device

        if mode == "init_kmeans_final":
            if hasattr(data, "to"):
                data = data.to(dev)
            if torch.is_tensor(features):
                features = features.to(dev, non_blocking=True)
            self.vq(data, attr_list, mask_dict, logger, chunk_i, epoch, mode)
            return 0

        # Edges (mirror for undirected) -> dev
        s1, d1 = data.edges()
        s1 = s1.to(dev, non_blocking=True); d1 = d1.to(dev, non_blocking=True)
        src = torch.cat([s1, d1], 0); dst = torch.cat([d1, s1], 0)
        edge_index = torch.stack([src, dst], 0)

        # Edge attributes (bond types {1..4}, 0 otherwise)
        eb = data.edata.get("weight", torch.zeros(data.num_edges(), dtype=torch.long, device=s1.device))
        eb = eb.to(dev, non_blocking=True)
        e  = torch.cat([eb, eb], 0)
        e  = torch.where((e >= 1) & (e <= 4), e, torch.zeros_like(e))
        edge_attr = self.bond_emb(e.long())

        # Node features -> h0
        features = features.to(dev, non_blocking=True)
        h0_raw = self.feat_embed(features)              # [N, 64]
        h0 = self.ln_in(self.linear_0(h0_raw))          # [N, args.hidden_dim]

        # Hop 1
        h0_for1 = self.skip0(h0) if self.skip0 is not None else h0
        h1 = self.gine1(h0, edge_index, edge_attr)
        h1 = self.ln1(h1 * self.res1 + h0_for1)

        # Hop 2
        h2 = self.gine2(h1, edge_index, edge_attr)
        h2 = self.ln2(h2 * self.res2 + h1)

        # Hop 3
        h3 = self.gine3(h2, edge_index, edge_attr)
        h3 = self.ln3(h3 * self.res3 + h2)

        # JK concat
        h_cat = torch.cat([h0, h1, h2, h3], dim=-1)
        h_mid = self.mix(h_cat)
        h_out = self.out_proj(h_mid)

        if mode == "init_kmeans_loop":
            return h_out

        h_vq = self.pre_vq_ln(h_out)
        quantize_output = self.vq(h_vq, attr_list, mask_dict, logger, chunk_i, epoch, mode)
        (loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss) = quantize_output
        return loss, embed, [commit_loss, repel_loss, cb_repel_loss]


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
