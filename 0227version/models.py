from train_teacher import get_args
import torch.nn as nn


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
        super(AtomEmbedding, self).__init__()
        self.element_embed = nn.Embedding(num_embeddings=100, embedding_dim=64)   # element
        self.degree_embed = nn.Embedding(num_embeddings=7, embedding_dim=4)       # degree
        self.valence_embed = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.charge_embed = nn.Embedding(num_embeddings=8, embedding_dim=4)
        self.aromatic_embed = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.hybrid_embed = nn.Embedding(num_embeddings=6, embedding_dim=4)
        self.hydrogen_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)

    def forward(self, atom_inputs):
        """
        atom_inputs: LongTensor of shape [num_atoms, 7]
        Each column is an integer feature:
        [element, degree, valence, charge, aromaticity, hybridization, num_hydrogens]
        """
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        atom_inputs = atom_inputs.to(device)

        x0 = self.element_embed(atom_inputs[:, 0].long())
        x1 = self.degree_embed(atom_inputs[:, 1].long())
        x2 = self.valence_embed(atom_inputs[:, 2].long() + 1)
        x3 = self.charge_embed(atom_inputs[:, 3].long())
        x4 = self.aromatic_embed(atom_inputs[:, 4].long())
        x5 = self.hybrid_embed(atom_inputs[:, 5].long())
        x6 = self.hydrogen_embed(atom_inputs[:, 6].long())

        out = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=-1)  # shape: [num_atoms, total_embedding_dim]
        return out


class EquivariantThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super(EquivariantThreeHopGINE, self).__init__()
        if args is None:
            args = get_args()  # Ensure this function is defined elsewhere
        self.feat_embed = AtomEmbedding()
        self.linear_0 = nn.Linear(88, args.hidden_dim)
        # GINEConv layers with specified edge_dim
        nn1 = nn.Sequential(
            nn.Linear(args.hidden_dim, hidden_feats),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
        )
        nn4 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
        )
        self.gine1 = GINEConv(nn1, edge_dim=1)
        self.gine2 = GINEConv(nn2, edge_dim=1)
        self.gine3 = GINEConv(nn3, edge_dim=1)
        self.gine4 = GINEConv(nn4, edge_dim=1)
        # Vector quantization layer
        self.vq = VectorQuantize(
            dim=args.hidden_dim,
            codebook_size=args.codebook_size,
            decay=0.8,
            use_cosine_sim=False,
            threshold_ema_dead_code=2,
        )
        # Bond weight layer
        self.bond_weight = BondWeightLayer(bond_types=4, hidden_dim=args.hidden_dim)
        # Activation and normalization layers
        self.leaky_relu0 = nn.LeakyReLU(0.2)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.ln0 = nn.LayerNorm(args.hidden_dim)
        self.ln1 = nn.LayerNorm(args.hidden_dim)
        self.ln2 = nn.LayerNorm(args.hidden_dim)
        self.ln3 = nn.LayerNorm(args.hidden_dim)
        self.linear_1 = nn.Linear(hidden_feats, hidden_feats)
        self.train_or_infer = args.train_or_infer

    def reset_kmeans(self):
        """Reset k-means clustering for vector quantization."""
        self.vq._codebook.reset_kmeans()

    def is_bidirectional(self, src, dst):
        # Create a set of all edges as tuples
        edges = set([(int(u), int(v)) for u, v in zip(src, dst)])        # Check if the reverse of each edge exists
        for u, v in edges:
            if u == v:
                continue
            elif (v, u) not in edges:
                print(f"Edge ({u}, {v}) doesn't have corresponding reverse edge ({v}, {u})")
                return False
        return True

    def forward(self, data, features, chunk_i, logger=None, epoch=None, batched_graph_base=None, mode=None):
        import torch
        torch.set_printoptions(threshold=10_000)  # Set a high threshold to print all elements
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        if mode != "init_kmeans_final":
            src_one_way, dst_one_way = data.edges()
            src = torch.cat([src_one_way, dst_one_way])
            dst = torch.cat([dst_one_way, src_one_way])
            src_output = src.detach().clone()
            dst_output = dst.detach().clone()
            num_nodes = data.num_nodes()
            sample_adj = torch.zeros((num_nodes, num_nodes), device=src.device)
            self.gine1 = self.gine1.to(device)
            self.gine2 = self.gine2.to(device)
            self.gine3 = self.gine3.to(device)
            self.gine4 = self.gine4.to(device)
            self.vq = self.vq.to(device)
            self.bond_weight = self.bond_weight.to(device)
            feat_before_transform = features.detach()
            features = self.feat_embed(features).to(device)
            features = features.to(device)
            h = self.linear_0(features)
            init_feat = h
            edge_weight = data.edata.get(
                'weight', torch.zeros(data.num_edges(), dtype=torch.long, device=device)
            )
            edge_weight = torch.cat([edge_weight, edge_weight])
            mapped_indices = torch.where(
                (edge_weight >= 1) & (edge_weight <= 4),
                edge_weight - 1,
                torch.zeros_like(edge_weight)
            )
            mapped_indices = mapped_indices.long()
            transformed_edge_weight = self.bond_weight(mapped_indices).squeeze(-1)  # [num_edges]
            transformed_edge_weight = transformed_edge_weight.unsqueeze(
                -1) if transformed_edge_weight.dim() == 1 else transformed_edge_weight
            edge_index = torch.stack([src, dst], dim=0)  #　隣接情報
            edge_attr = transformed_edge_weight
            edge_attr = torch.ones(edge_attr.shape).to(device)
            h = self.gine1(h, edge_index=edge_index, edge_attr=edge_attr)
            h = self.ln0(h)
            h = self.gine2(h, edge_index=edge_index, edge_attr=edge_attr)
            h = self.ln1(h)
            h = self.gine3(h, edge_index=edge_index, edge_attr=edge_attr)
            h = self.ln2(h)
            h = self.gine4(h, edge_index=edge_index, edge_attr=edge_attr)
            h = self.ln3(h)
            h = self.linear_1(h)
            # h = F.normalize(h, p=2, dim=1)  # e.g. scaling_factor = 1.0 ~ 2.0
            # norms = h.norm(dim=1)
            # if chunk_i % 50 == 0:
            #     print("###### ===  h norm stats:", norms.min().item(), norms.mean().item(), norms.max().item())
        if mode == "init_kmeans_loop":
            return h
        if mode == None:
            quantize_output = self.vq(
                h, init_feat, logger, chunk_i, epoch, mode
            )
        elif mode == "init_kmeans_final":
            self.vq(
                data, None, logger, chunk_i, epoch, mode
            )
            return 0
        (quantize, emb_ind, loss, dist, embed, commit_loss, latents, div_nega_loss,
         x, cb_loss, sil_loss, num_unique, repel_loss, cb_repel_loss) = quantize_output
        detached_quantize = quantize.detach()
        losslist = [0, commit_loss.item(), cb_loss.item(), sil_loss.item(),
                    repel_loss.item(), cb_repel_loss.item()]
        if batched_graph_base:  # from evaluate
            latents = h
            sample_adj_base = batched_graph_base.adj().to_dense()
            sample_bond_info = batched_graph_base.edata["weight"]
            # print(f"in model py, if batched_graph_base, latent shape is {latents.shape}")
            sample_list = [emb_ind, feat_before_transform, latents, sample_bond_info, src_output, dst_output, sample_adj_base]
        else:   # from train_sage
            sample_bond_info = data.edata["weight"]
            sample_list = [emb_ind, feat_before_transform, sample_adj, sample_bond_info, src_output, dst_output]
        sample_list = [t.clone().detach() if t is not None else torch.zeros_like(sample_list[0]) for t in sample_list]
        return [], h, loss, dist, embed, losslist, x, detached_quantize, latents, sample_list, num_unique


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
