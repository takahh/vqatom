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
class AtomEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 既存の embedding 定義はそのまま ...

        # 今のバッチで観測されている ringSize 値に対応する index
        # 値:  0,3,4,5,6,7,8  → idx: 0,1,2,3,4,5,6
        ring_vals = [0, 3, 4, 5, 6, 7, 8]
        self.register_buffer(
            "ringsize_known_vals",
            torch.tensor(ring_vals, dtype=torch.long)
        )
        self.register_buffer(
            "ringsize_indices",
            torch.arange(len(ring_vals), dtype=torch.long)
        )

    def forward(self, atom_inputs):
        import torch
        device = next(self.parameters()).device
        atom_inputs = atom_inputs.to(device, non_blocking=True)

        # ... x0〜x26 まではそのまま ...

        # ------- ringSize (col 27) -------
        raw27 = atom_inputs[:, 27].long()

        # デフォルト 0 にしておいて、既知の値だけ上書き
        mapped27 = torch.zeros_like(raw27)
        for v, idx in zip(self.ringsize_known_vals, self.ringsize_indices):
            mapped27[raw27 == v] = idx

        # 念のため安全側に clamp（未知値が来ても 0〜6 に収める）
        mapped27 = mapped27.clamp_(0, self.ringsize_embed.num_embeddings - 1)
        x27 = self.ringsize_embed(mapped27)

        # ------- arom_nbrs (col 28) -------
        raw28 = atom_inputs[:, 28].long()
        # 0〜4 だけなら、そのままでもいいけど、保険で clamp
        raw28 = raw28.clamp_(0, self.aroma_num_embed.num_embeddings - 1)
        x28 = self.aroma_num_embed(raw28)

        # ------- fused_id (col 29) -------
        raw29 = atom_inputs[:, 29].long()
        raw29 = raw29.clamp_(0, self.fused_if_embed.num_embeddings - 1)
        x29 = self.fused_if_embed(raw29)

        out = torch.cat(
            [x0, x1, x2, x3, x4, x5, x6,
             x7, x8, x9, x10, x11, x12, x13, x14,
             x15, x16, x17, x18, x19, x20, x21, x22, x23, x24,
             x25, x26, x27, x28, x29],
            dim=-1
        )
        return out


class EquivariantThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super().__init__()
        import torch
        if args is None:
            args = get_args()

        self.feat_embed = AtomEmbedding()
        self.linear_0 = nn.Linear(92, args.hidden_dim)  # h0

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

        # JK: concat h0 + h1 + h2 + h3  -> correct dim
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
            # Only move if tensors (features may be None)
            if hasattr(data, "to"):  # dgl graph has .to()
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
        h0 = self.ln_in(self.linear_0(self.feat_embed(features)))  # [N, args.hidden_dim]

        # Hop 1 (dimension-safe residual)
        h0_for1 = self.skip0(h0) if self.skip0 is not None else h0
        h1 = self.gine1(h0, edge_index, edge_attr)
        h1 = self.ln1(h1 * self.res1 + h0_for1)  # [N, hidden_feats]

        # Hop 2
        h2 = self.gine2(h1, edge_index, edge_attr)
        h2 = self.ln2(h2 * self.res2 + h1)       # [N, hidden_feats]

        # Hop 3
        h3 = self.gine3(h2, edge_index, edge_attr)
        h3 = self.ln3(h3 * self.res3 + h2)       # [N, hidden_feats]

        # JK concat (include h0)
        h_cat = torch.cat([h0, h1, h2, h3], dim=-1)  # [N, args.hidden_dim + 3*hidden_feats]
        h_mid = self.mix(h_cat)
        h_out = self.out_proj(h_mid)                 # [N, args.hidden_dim]

        if mode == "init_kmeans_loop":
            return h_out

        h_vq = self.pre_vq_ln(h_out)
        quantize_output = self.vq(h_vq, attr_list, mask_dict, logger, chunk_i, epoch, mode)
        (loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss) = quantize_output
        if logger is not None:
            logger.info(f"weighted avg : commit {commit_loss}, lat_repel {repel_loss}, co_repel {cb_repel_loss}")
        return loss, embed, [commit_loss.item(), repel_loss.item(), cb_repel_loss.item()]

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
