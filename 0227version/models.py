from dgl.nn import GraphConv, APPNPConv, GATConv

from train_teacher import get_args
from old_train_and_eval import transform_node_feats
# from train_and_eval import filter_small_graphs_from_blocks
import torch.nn as nn


class BondWeightLayer(nn.Module):
    def __init__(self, bond_types=4, hidden_dim=64):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)  # Learnable bond representation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()  # Output weight in range (0,1)
            # nn.Softplus()  # Smooth and maintains gradient flow
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        atom_inputs = atom_inputs.to(device)
        # print("valence input min:", atom_inputs[:, 3].min().item())
        # print("valence input max:", atom_inputs[:, 3].max().item())

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
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(hidden_feats, hidden_feats)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(hidden_feats, hidden_feats)
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(hidden_feats, out_feats)
        )
        nn4 = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(hidden_feats, out_feats)
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

    def forward(self, data, features, chunk_i, logger=None, epoch=None, batched_graph_base=None):
        import torch
        torch.set_printoptions(threshold=10_000)  # Set a high threshold to print all elements
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
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
        quantize_output = self.vq(
            h, init_feat, logger, chunk_i, epoch
        )
        (quantize, emb_ind, loss, dist, embed, commit_loss, latents, div_nega_loss,
         x, cb_loss, sil_loss, num_unique, repel_loss, cb_repel_loss) = quantize_output
        detached_quantize = quantize.detach()
        losslist = [0, commit_loss.item(), cb_loss.item(), sil_loss.item(),
                    repel_loss.item(), cb_repel_loss.item()]
        if batched_graph_base:
            latents = h
            sample_adj_base = batched_graph_base.adj().to_dense()
            sample_bond_info = batched_graph_base.edata["weight"]
            sample_list = [emb_ind, feat_before_transform, latents, sample_bond_info, src_output, dst_output, sample_adj_base]
        else:
            sample_bond_info = data.edata["weight"]
            sample_list = [emb_ind, feat_before_transform, sample_adj, sample_bond_info, src_output, dst_output]
        sample_list = [t.clone().detach() if t is not None else torch.zeros_like(sample_list[0]) for t in sample_list]
        return [], h, loss, dist, embed, losslist, x, detached_quantize, latents, sample_list, num_unique


class MLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linear = nn.Linear(hidden_dim, input_dim)
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
                vq = self.linear(h)
                h_list.append(vq)
        return h_list, h


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""


class GCN(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type,
            codebook_size,
            lamb_edge,
            lamb_node
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.decoder_1 = nn.Linear(input_dim, input_dim)
        self.decoder_2 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25,
                                 use_cosine_sim=True)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node

    def forward(self, g, feats):
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        h_list = []
        h = self.graph_layer_1(g, feats)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)
        quantized, _, commit_loss, dist, codebook = self.vq(h)
        quantized_edge = self.decoder_1(quantized)
        quantized_node = self.decoder_2(quantized)

        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))

        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h = self.graph_layer_2(g, quantized_edge)
        h_list.append(h)
        h = self.linear(h)
        loss = feature_rec_loss + edge_rec_loss + commit_loss
        return h_list, h, loss, dist, codebook, [feature_rec_loss, edge_rec_loss, commit_loss]


# def feat_elem_divergence_loss(embed_ind, atom_types, num_codebooks=1500, temperature=0.02):
#     device = embed_ind.device
#
#     # Ensure embed_ind is within valid range
#     embed_ind = torch.clamp(embed_ind, min=0, max=num_codebooks - 1).long()
#
#     # Map atom_types to sequential indices
#     unique_atom_numbers = torch.unique(atom_types, sorted=True)
#     atom_types_mapped = torch.searchsorted(unique_atom_numbers, atom_types)
#
#     # Create one-hot representations
#     embed_one_hot = torch.nn.functional.one_hot(embed_ind, num_classes=num_codebooks).float()
#     atom_type_one_hot = torch.nn.functional.one_hot(atom_types_mapped, num_classes=len(unique_atom_numbers)).float()
#
#     # Compute soft assignments
#     soft_assignments = torch.softmax(embed_one_hot / temperature, dim=-1)
#
#     # Compute co-occurrence matrix
#     co_occurrence = torch.einsum("ni,nj->ij", [soft_assignments, atom_type_one_hot])
#
#     # Normalize co-occurrence
#     co_occurrence_normalized = co_occurrence / (co_occurrence.sum(dim=1, keepdim=True) + 1e-6)
#
#     # Compute row-wise entropy
#     row_entropy = -torch.sum(co_occurrence_normalized * torch.log(co_occurrence_normalized + 1e-6), dim=1)
#
#     # Compute sparsity loss
#     sparsity_loss = row_entropy.mean()
#
#     # Debug connection to the graph
#     print(f"sparsity_loss.requires_grad: {sparsity_loss.requires_grad}")
#     print(f"sparsity_loss.grad_fn: {sparsity_loss.grad_fn}")
#
#     return sparsity_loss

import dgl
import torch


def filetr_src_and_dst(src, dst):
    new_src = []
    new_dst = []
    for src_value, dst_value in zip(src, dst):
        if src_value > 9999 or dst_value > 9999:
            continue
        else:
            new_src.append(src_value)
            new_dst.append(dst_value)
    return new_src, new_dst


def remove_bond_with_other_blocks(src, dst):
    args = get_args()
    mask = torch.abs(src - dst) < args.batch_size * 0.5
    filtered_src = src[mask]
    filtered_dst = dst[mask]
    return filtered_src, filtered_dst  # Return count of unique values


class SAGE(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type,
            codebook_size,
            lamb_edge,
            lamb_node,
            lamb_div_ele
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        # self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        # self.decoder_1 = nn.Linear(input_dim, input_dim)
        # self.decoder_2 = nn.Linear(input_dim, input_dim)
        # self.linear = nn.Linear(hidden_dim, output_dim)
        self.linear_2 = nn.Linear(7, hidden_dim)  # added to change 7 dim feat vecs to the larger dim
        self.codebook_size = codebook_size
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, decay=0.8, use_cosine_sim=False)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.lamb_div_ele = lamb_div_ele

    def reset_kmeans(self):
        self.vq._codebook.reset_kmeans()

    def forward(self, blocks, feats, epoch):
        # feats is batch_feats !!!!!!!!!!!
        h = feats.clone() if not feats.requires_grad else feats
        init_feat = h.clone()  # Store initial features (for later use)
        torch.save(init_feat, "/h.pt")  # Save for reference
        device = h.device
        global_node_ids = set()

        # print([g.etypes for g in blocks])  # Check edge types of all graphs
        for block in blocks:
            src, dst = block.all_edges()
            global_node_ids.update(src.tolist())
            global_node_ids.update(dst.tolist())
        global_node_ids = sorted(global_node_ids)
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_node_ids)}
        idx_tensor = torch.tensor(global_node_ids, dtype=torch.int64, device=device)
        h = h[idx_tensor]
        init_feat = init_feat[idx_tensor]  # Important: reindex init_feat as well!
        remapped_edge_list = []
        remapped_bond_orders = []  # List to hold bond orders, if available
        for block in blocks:
            src, dst = block.all_edges()
            src = src.to(torch.int64)
            dst = dst.to(torch.int64)
            local_src = torch.tensor([global_to_local[i.item()] for i in src],
                                     dtype=torch.int64, device=device)
            local_dst = torch.tensor([global_to_local[i.item()] for i in dst],
                                     dtype=torch.int64, device=device)
            remapped_edge_list.append((local_src, local_dst))
            remapped_edge_list.append((local_dst, local_src))
            if "bond_order" in block.edata:
                bond_order = block.edata["bond_order"].to(torch.float32).to(device)
                remapped_bond_orders.append(bond_order)
                remapped_bond_orders.append(bond_order)  # For the reverse edge
        g = dgl.DGLGraph().to(device)
        g.add_nodes(len(global_node_ids))
        if remapped_bond_orders:
            for (src, dst), bond_order in zip(remapped_edge_list, remapped_bond_orders):
                g.add_edges(src, dst, data={"bond_order": bond_order})
        else:
            for src, dst in remapped_edge_list:
                g.add_edges(src, dst)
        h_list = []  # To store intermediate node representations
        h = self.linear_2(h)
        h = self.graph_layer_1(g, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h_list.append(h)  # Store the latent representation

        (quantized, emb_ind, loss, dist, codebook, raw_commit_loss, latents, margin_loss,
         spread_loss, pair_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss,
         aroma_div_loss, ringy_div_loss, h_num_div_loss, sil_loss, charge_div_loss, elec_state_div_loss) = \
            self.vq(h, init_feat, epoch)

        return (h_list, h, loss, dist, codebook,
                [div_ele_loss, bond_num_div_loss, aroma_div_loss, ringy_div_loss,
                 h_num_div_loss, charge_div_loss, elec_state_div_loss, spread_loss, pair_loss, sil_loss],
                x, detached_quantize, latents)

    def inference(self, dataloader, feats):
        device = feats.device
        dist_all = torch.zeros(feats.shape[0], self.codebook_size, device=device)
        y = torch.zeros(feats.shape[0], self.output_dim, device=device)
        latent_list = []
        input_node_list = []
        embed_ind_list = []
        div_ele_loss_list = []
        bond_num_div_loss_list = []
        aroma_div_loss_list = []
        ringy_div_loss_list = []
        h_num_div_loss_list = []
        sil_loss_list = []
        elec_state_div_loss_list = []
        charge_div_loss_list = []
        for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [blk.int().to(device) for blk in blocks]  # Convert blocks to device
            batch_feats = feats[input_nodes]
            batch_feats = transform_node_feats(batch_feats)
            # ------------------------------
            # ここから SAGE.forward と同じにする
            # ------------------------------
            h = batch_feats.clone()
            # print("h.shape +++++++++++++++++++")
            # print(h.shape)
            # print(f"input_nodes {input_nodes[:10]}, {input_nodes[-10:]}")
            init_feat = h.clone()
            device = h.device
            # print("----INFER ------")
            global_node_ids = set()
            for block in blocks:
                src, dst = block.all_edges()
                # print(f"src {src[:20]} len {src.shape}")
                global_node_ids.update(src.tolist())  # Converting to a list is okay here for set operations
                global_node_ids.update(dst.tolist())

            global_node_ids = sorted(global_node_ids)
            global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_node_ids)}
            # print("global_to_local (first 20 items):")
            # print(list(global_to_local.items())[:20])
            local_to_global = {local_id: global_id for global_id, local_id in global_to_local.items()}
            # print("local_to_global (first 20 items):")
            # print(list(local_to_global.items())[:20])

            idx_tensor = torch.tensor(global_node_ids, dtype=torch.int64, device=device)
            # h = h[idx_tensor]
            init_feat = init_feat[idx_tensor]
            remapped_edge_list = []
            remapped_bond_orders = []
            edge_list = []
            new_node_count_total = 0
            # ----------------------------------------------
            # obtain edge source and destination from block
            # ----------------------------------------------
            total_src = torch.empty(0, dtype=torch.int64, device=device)
            total_dst = torch.empty(0, dtype=torch.int64, device=device)

            for idex, block in enumerate(blocks):
                src, dst = block.all_edges()
                src, dst = src.to(torch.int64), dst.to(torch.int64)
                local_src = torch.tensor([global_to_local[i.item()] for i in src], dtype=torch.int64, device=device)
                local_dst = torch.tensor([global_to_local[i.item()] for i in dst], dtype=torch.int64, device=device)
                local_src, local_dst = remove_bond_with_other_blocks(local_src, local_dst)
                total_src = torch.cat((total_src, local_src))
                total_dst = torch.cat((total_dst, local_dst))
                remapped_edge_list.append((local_src, local_dst))
                remapped_edge_list.append((local_dst, local_src))
                edge_list.append((local_src, local_dst))
                edge_list.append((local_dst, local_src))
                if "bond_order" in block.edata:
                    bond_order = block.edata["bond_order"].to(torch.float32).to(device)
                    remapped_bond_orders.append(bond_order)
                    remapped_bond_orders.append(bond_order)  # Bidirectional bond orders
                if idex == 0 and idx == 0:
                    sample_bond_to_edge = [local_src, local_dst]
                    sample_bond_order = block.edata["bond_order"].to(torch.float32).to(device)
            new_nodes = torch.unique(torch.cat([total_src, total_dst]))
            # print(f"new_nodes {new_nodes[:10]}, {new_nodes[-10:]}")
            # print(f"total new nodes count {new_nodes.shape}")
            new_nodes_global = torch.tensor([local_to_global[i.item()] for i in new_nodes], dtype=torch.int64, device=device)
            # print(f"new_nodes_global {new_nodes_global[:10]}, {new_nodes_global[-10:]}")
            # print(f"total new nodes count {new_nodes_global.shape}")
            g = dgl.DGLGraph().to(device)
            g.add_nodes(new_node_count_total)

            num_nodes_in_g = g.num_nodes()
            h_shape = h.shape[0]

            # print(f"🔹 Number of nodes in g: {num_nodes_in_g}")
            # print(f"🔹 Shape of h: {h_shape}")

            if remapped_bond_orders:
                for (src, dst), bond_order in zip(remapped_edge_list, remapped_bond_orders):
                    g.add_edges(src, dst, data={"bond_order": bond_order})
            else:
                for src, dst in edge_list:
                    g.add_edges(src, dst)
            if idx == 0:
                sample_feat = h.clone().detach()
                adj_matrix = g.adjacency_matrix().to_dense()
                sample_adj = adj_matrix.to_dense()
            # --- Graph Layer Processing ---
            h_list = []
            h = self.linear_2(h)
            h = self.graph_layer_1(g, h)
            if self.norm_type != "none":
                h = self.norms[0](h)
            h_list.append(h)

            # --- Quantization ---
            (quantized, embed_ind, loss, dist, codebook, raw_commit_loss, latent_vectors, margin_loss,
             spread_loss, pair_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss, aroma_div_loss,
             ringy_div_loss, h_num_div_loss, sil_loss, charge_div_loss, elec_state_div_loss) = self.vq(h, init_feat)

            # Store computed values
            embed_ind_list.append(embed_ind)
            input_node_list.append(input_nodes)
            div_ele_loss_list.append(div_ele_loss)
            bond_num_div_loss_list.append(bond_num_div_loss)
            aroma_div_loss_list.append(aroma_div_loss)
            ringy_div_loss_list.append(ringy_div_loss)
            h_num_div_loss_list.append(h_num_div_loss)
            elec_state_div_loss_list.append(elec_state_div_loss)
            charge_div_loss_list.append(charge_div_loss)
            sil_loss_list.append(sil_loss)

            if idx == 0:
                sample_ind = embed_ind
                sample_list = [sample_ind, sample_feat, sample_adj, sample_bond_order, sample_bond_to_edge]

        return h_list, y, loss, dist_all, codebook, [
            div_ele_loss_list, bond_num_div_loss_list, aroma_div_loss_list, ringy_div_loss_list,
            h_num_div_loss_list, charge_div_loss_list, elec_state_div_loss_list, spread_loss, pair_loss, sil_loss_list
        ], latent_list, sample_list


class GAT(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            num_heads=8,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        return h_list, h


class APPNP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            edge_drop=0.5,
            alpha=0.1,
            k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h


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
