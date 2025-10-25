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
        self.element_embed = nn.Embedding(num_embeddings=120, embedding_dim=16)   # element
        self.degree_embed = nn.Embedding(num_embeddings=7, embedding_dim=4)       # degree
        self.valence_embed = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.charge_embed = nn.Embedding(num_embeddings=8, embedding_dim=4)
        self.aromatic_embed = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.hybrid_embed = nn.Embedding(num_embeddings=6, embedding_dim=4)
        self.hydrogen_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)
        self.func_embed_0 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_1 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_2 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_3 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_4 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_5 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_6 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_7 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_8 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_9 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_10 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_11 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_12 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_13 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_14 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_15 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_16 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.func_embed_17 = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.h_don_embed = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.h_acc_embed = nn.Embedding(num_embeddings=2, embedding_dim=4)

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

        # functional group flags
        x7 = self.func_embed_0(atom_inputs[:, 7].long())
        x8 = self.func_embed_1(atom_inputs[:, 8].long())
        x9 = self.func_embed_2(atom_inputs[:, 9].long())
        x10 = self.func_embed_3(atom_inputs[:, 10].long())
        x11 = self.func_embed_4(atom_inputs[:, 11].long())
        x12 = self.func_embed_5(atom_inputs[:, 12].long())
        x13 = self.func_embed_6(atom_inputs[:, 13].long())
        x14 = self.func_embed_7(atom_inputs[:, 14].long())
        x15 = self.func_embed_8(atom_inputs[:, 15].long())
        x16 = self.func_embed_9(atom_inputs[:, 16].long())
        x17 = self.func_embed_10(atom_inputs[:, 17].long())
        x18 = self.func_embed_11(atom_inputs[:, 18].long())
        x19 = self.func_embed_12(atom_inputs[:, 19].long())
        x20 = self.func_embed_13(atom_inputs[:, 20].long())
        x21 = self.func_embed_14(atom_inputs[:, 21].long())
        x22 = self.func_embed_15(atom_inputs[:, 22].long())
        x23 = self.func_embed_16(atom_inputs[:, 23].long())
        x24 = self.func_embed_17(atom_inputs[:, 24].long())

        x25 = self.h_don_embed(atom_inputs[:, 25].long())  # 2 numbers
        x26 = self.h_acc_embed(atom_inputs[:, 26].long())  # 2 numbers

        out = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
                         x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26], dim=-1)  # shape: [num_atoms, total_embedding_dim]
        return out


class EquivariantThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super(EquivariantThreeHopGINE, self).__init__()
        if args is None:
            args = get_args()  # Ensure this function is defined elsewhere
        self.feat_embed = AtomEmbedding()
        self.linear_0 = nn.Linear(120, args.hidden_dim)
        # GINEConv layers with specified edge_dim
        nn1 = nn.Sequential(
            nn.Linear(args.hidden_dim, hidden_feats),
            nn.ReLU(),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
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
        # self.linear_1 = nn.Linear(hidden_feats, hidden_feats)
        self.linear_1 = nn.Sequential(
            nn.Linear(hidden_feats * 3, 2 * hidden_feats),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(2 * hidden_feats, 2 * hidden_feats),
            nn.ReLU(),
            nn.Linear(2 * hidden_feats, hidden_feats),
            nn.ReLU()
        )
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

    def forward(self, data, features, chunk_i, mask_dict=None, logger=None, epoch=None, batched_graph_base=None, mode=None):
        import torch
        torch.set_printoptions(threshold=10_000)  # Set a high threshold to print all elements
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        if mode != "init_kmeans_final":
            src_one_way, dst_one_way = data.edges()
            src = torch.cat([src_one_way, dst_one_way])
            dst = torch.cat([dst_one_way, src_one_way])
            self.gine1 = self.gine1.to(device)
            self.gine2 = self.gine2.to(device)
            self.gine3 = self.gine3.to(device)
            self.gine4 = self.gine4.to(device)
            self.vq = self.vq.to(device)
            self.bond_weight = self.bond_weight.to(device)
            features_first = features.clone()
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
            # 2 Three GNN Layers
            h_list = []
            h1 = self.ln0(self.gine1(h, edge_index, edge_attr))
            h_list.append(h1)
            h2 = self.ln1(self.gine2(h1, edge_index, edge_attr))
            h_list.append(h2)
            h3 = self.ln2(self.gine3(h2, edge_index, edge_attr))
            h_list.append(h3)
            h = torch.cat(h_list, dim=-1)  # concat mode
            # h = sum(h_list)              # sum mode
            h = self.linear_1(h)
            # h = F.normalize(h, p=2, dim=1)  # e.g. scaling_factor = 1.0 ~ 2.0
            # norms = h.norm(dim=1)
            # if chunk_i % 50 == 0:
            #     print("###### ===  h norm stats:", norms.min().item(), norms.mean().item(), norms.max().item())
        if mode == "init_kmeans_loop":
            return h
        if mode == None:  # train or test
            quantize_output = self.vq(
                h, features_first, mask_dict, logger, chunk_i, epoch, mode
            )
        elif mode == "init_kmeans_final":
            self.vq(
                data, features, mask_dict, logger, chunk_i, epoch, mode
            )
            return 0
        (loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss) = quantize_output
        # print(f"commit_loss {commit_loss}")
        # print(f"cb_loss {cb_loss}")
        # print(f"sil_loss {sil_loss}")
        # print(f"repel_loss {repel_loss}")
        # print(f"cb_repel_loss {cb_repel_loss}")
        logger.info(f"weighted avg : commit {commit_loss}, lat_repel {repel_loss}, co_repel {cb_repel_loss}")
        losslist = [commit_loss.item(), repel_loss.item(), cb_repel_loss.item()]
        return loss, embed, losslist


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
