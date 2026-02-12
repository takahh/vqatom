from args import get_args
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
                    return
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

    def _scalar_to_tensor(x):
        if torch.is_tensor(x):
            return x
        if x is None:
            _infer_device_dtype()
            return torch.zeros((), device=device, dtype=dtype)
        _infer_device_dtype()
        return torch.as_tensor(x, device=device, dtype=dtype)

    # ---------------------------
    # only supported shape:
    # (loss, (commit, cb, rep, cb_rep))
    # ---------------------------
    if not (isinstance(qo, (tuple, list)) and len(qo) == 2 and isinstance(qo[1], (tuple, list))):
        raise TypeError(f"Expected (loss, (commit, cb, rep, cb_rep)) but got {type(qo)}: {qo}")

    loss, inner = qo
    if len(inner) != 5:
        raise TypeError(f"Expected inner tuple length 5: (commit, cb, rep, cb_rep, ent), got {len(inner)}: {inner}")

    commit, cb, rep, cb_rep, ent = inner[0], inner[1], inner[2], inner[3], inner[4]

    _infer_device_dtype(loss, commit, cb, rep, cb_rep, ent)

    embed = None  # qoにはembedが無い前提
    sil = None    # qoにはsilが無い前提（0で埋める）

    return (
        _scalar_to_tensor(loss),
        embed,
        _scalar_to_tensor(commit),
        _scalar_to_tensor(cb),
        _scalar_to_tensor(sil),      # silhouette_loss = 0
        _scalar_to_tensor(rep),
        _scalar_to_tensor(cb_rep),
        _scalar_to_tensor(ent),
    )

import torch.nn as nn
import torch
import torch.nn as nn
from utils import CORE_ELEMENTS

class AtomEmbedding(nn.Module):
    def __init__(self):
        super(AtomEmbedding, self).__init__()

        # ---- 埋め込み定義（離散 0-30 部分）----
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

        # NEW: het27 (30) 用の embedding (0..26 → 27クラス)
        self.het27_embed = nn.Embedding(num_embeddings=27, embedding_dim=4)

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

        # ---- bond_env_raw (31-78, 48 dims) 用の射影 ----
        # 48 次元 → 16 次元に圧縮して concat
        self.bond_env_proj = nn.Linear(48, 16)

        # ---- 離散部分（x0..x6 + flags4 + x25,x26 + x27,x28,x29,x30）を 52→48 に圧縮 ----
        #   x0..x6: 7*4 = 28
        #   flags4: 4
        #   x25,x26: 2+2 = 4
        #   x27,x28,x29,x30: 4*4 = 16
        #   合計 = 28 + 4 + 4 + 16 = 52
        self.disc_proj = nn.Linear(52, 48)

        # このクラスの最終出力次元:
        #   離散部 (disc_proj 後) : 48
        #   bond_env_proj         : 16
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
        Encode per-atom features into a dense embedding.

        Parameters
        ----------
        atom_inputs : torch.Tensor
            Shape [N, 79].

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
            30    : het27 (0..26)
            31-78 : bond_env_raw (48)  float
        Returns
        -------
        out : torch.Tensor
            Shape [N, 64] = disc_emb(48) + bond_env_emb(16)
        """
        device = next(self.parameters()).device
        x = atom_inputs.to(device=device, non_blocking=True)

        if x.ndim != 2 or x.size(1) != 79:
            raise ValueError(f"atom_inputs must be [N,79], got {tuple(x.shape)}")

        # ----------------------------
        # helpers
        # ----------------------------
        def clamp_idx(col: int, n_embed: int, lo: int = 0) -> torch.Tensor:
            """Read integer column and clamp to valid embedding range."""
            return x[:, col].long().clamp(lo, n_embed - 1)

        def flag01(col: int) -> torch.Tensor:
            """Read 0/1 flag column."""
            return x[:, col].long().clamp(0, 1)

        # ----------------------------
        # core discrete features
        # ----------------------------

        # 0: Z -> element_lut -> element_embed
        z_raw = x[:, 0].long()
        max_z = int(self.element_lut.shape[0] - 1)
        z_safe = torch.where((0 <= z_raw) & (z_raw <= max_z), z_raw, torch.zeros_like(z_raw))
        idx0 = self.element_lut[z_safe].clamp(0, self.element_embed.num_embeddings - 1)
        x0 = self.element_embed(idx0)

        # 1: degree
        x1 = self.degree_embed(clamp_idx(1, self.degree_embed.num_embeddings))

        # 2: ring flag (+1) for ring_embed (0..)
        ring_plus1 = (x[:, 5].long() + 1).clamp(0, self.ring_embed.num_embeddings - 1)
        x2 = self.ring_embed(ring_plus1)

        # 3: charge
        x3 = self.charge_embed(clamp_idx(2, self.charge_embed.num_embeddings))

        # 4: aromatic (0/1)
        x4 = self.aromatic_embed(clamp_idx(4, self.aromatic_embed.num_embeddings))

        # 5: hybrid
        x5 = self.hybrid_embed(clamp_idx(3, self.hybrid_embed.num_embeddings))

        # 6: #hydrogens
        x6 = self.hydrogen_embed(clamp_idx(6, self.hydrogen_embed.num_embeddings))

        # ----------------------------
        # functional flags (7..24): 18 flags -> per-flag embed -> reduce
        # ----------------------------
        # assumes each func_embed_i is an Embedding(2, d_flag) (you used 2 dims each before)
        func_embeds = [
            self.func_embed_0,  self.func_embed_1,  self.func_embed_2,  self.func_embed_3,
            self.func_embed_4,  self.func_embed_5,  self.func_embed_6,  self.func_embed_7,
            self.func_embed_8,  self.func_embed_9,  self.func_embed_10, self.func_embed_11,
            self.func_embed_12, self.func_embed_13, self.func_embed_14, self.func_embed_15,
            self.func_embed_16, self.func_embed_17,
        ]
        flags = torch.cat([emb(flag01(7 + i)) for i, emb in enumerate(func_embeds)], dim=-1)  # [N, 18*d_flag]
        flags4 = self.func_reduce(flags)  # [N, 4]

        # 25,26: H-bond donor/acceptor flags
        x25 = self.h_don_embed(flag01(25))
        x26 = self.h_acc_embed(flag01(26))

        # ----------------------------
        # ring size mapping (27)
        # ----------------------------
        raw27 = x[:, 27].long()

        # default: last index = "other/unknown"
        mapped27 = torch.full_like(raw27, fill_value=self.ringsize_embed.num_embeddings - 1)

        # map known values via dict {value:int_index}
        # NOTE: this loop is small (few keys), so it's fine
        for v, idx in self.ring_value_to_index.items():
            mapped27[raw27 == int(v)] = int(idx)

        mapped27 = mapped27.clamp(0, self.ringsize_embed.num_embeddings - 1)
        x27 = self.ringsize_embed(mapped27)

        # 28: #aromatic neighbors
        x28 = self.aroma_num_embed(clamp_idx(28, self.aroma_num_embed.num_embeddings))

        # 29: fused ring id
        x29 = self.fused_if_embed(clamp_idx(29, self.fused_if_embed.num_embeddings))

        # 30: het27
        x30 = self.het27_embed(clamp_idx(30, self.het27_embed.num_embeddings))

        # ----------------------------
        # bond_env_raw (31..78): 48 floats -> proj -> 16
        # ----------------------------
        bond_env = x[:, 31:].to(dtype=torch.float32)
        if bond_env.size(1) != 48:
            raise ValueError(f"bond_env_raw must be 48 dims, got {bond_env.size(1)}")
        bond_env_emb = self.bond_env_proj(bond_env)  # [N, 16]

        # ----------------------------
        # concatenate discrete, project, then concat with bond env
        # ----------------------------
        disc_cat = torch.cat(
            [
                x0, x1, x2, x3, x4, x5, x6,   # base discrete embeds
                flags4,                       # [N,4]
                x25, x26,                     # donor/acceptor
                x27, x28, x29, x30,           # ringSize, aromNbrs, fusedId, het27
            ],
            dim=-1,
        )  # expected [N, 52] in your original design

        disc_emb = self.disc_proj(disc_cat)          # [N, 48]
        out = torch.cat([disc_emb, bond_env_emb], dim=-1)  # [N, 64]
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

        # K-means 用のループ時は VQ を通さずにそのまま返す
        if mode == "init_kmeans_loop":
            return h_out
        import torch.nn.functional as F
        h_vq = self.pre_vq_ln(h_out)
        h_vq = F.normalize(h_vq, p=2, dim=-1, eps=1e-12)

        quantize_output = self.vq(
            h_vq,
            attr_list,
            mask_dict,
            logger,
            chunk_i,
            epoch,
            mode,
        )
        # -------------------------
        # INFER: return token IDs
        # -------------------------
        if mode == "infer":
            # vq は infer のとき (key_id_full, cluster_id_full, id2safe) を返す設計にする
            # ここで normalize は使わない（loss群とは別物なので）
            if not (isinstance(quantize_output, (tuple, list)) and len(quantize_output) == 3):
                raise TypeError(
                    f"[model.forward] mode='infer' expects vq() -> (key_id_full, cluster_id_full, id2safe), "
                    f"got {type(quantize_output)} with len={len(quantize_output) if isinstance(quantize_output,(tuple,list)) else 'NA'}"
                )

            key_id_full, cluster_id_full, id2safe = quantize_output

            # shape guarantee
            key_id_full = key_id_full.reshape(-1).long()
            cluster_id_full = cluster_id_full.reshape(-1).long()

            return key_id_full, cluster_id_full, id2safe

        # quantizer の生の出力（2タプルなど）を正規化して loss 群だけ取り出す
        loss, _embed_ignored, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss, ent_loss = _normalize_quantize_output(
            quantize_output, logger,
            device=h_vq.device if torch.is_tensor(h_vq) else None,
            dtype=getattr(h_vq, "dtype", None)
        )

        # embed は今回は h_vq をそのまま使う（まだ本当の量子化はしていないバージョン）
        embed = h_vq

        # モデルが返すのは (total_loss, embed, loss_list)
        # return loss, embed, [commit_loss, repel_loss, cb_repel_loss]
        #         loss, cb, loss_list3 = outputs
        return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss, ent_loss]


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
