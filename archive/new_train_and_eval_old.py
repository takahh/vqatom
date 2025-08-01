import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from models import WeightedThreeHopGCN
import copy
from utils import set_seed
import dgl.dataloading
from train_teacher import get_args
import dgl
import logging
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
DATAPATH = "data/both_mono"


def transform_node_feats(a):
    transformed = torch.empty_like(a)
    transformed[:, 0] = torch.where(a[:, 0] == 6, 1,
                        torch.where(a[:, 0] == 8, 20, torch.where(a[:, 0] == 7, 10,
                        torch.where(a[:, 0] == 17, 5, torch.where(a[:, 0] == 9, 15,
                        torch.where(a[:, 0] == 35, 8, torch.where(a[:, 0] == 16, 3,
                        torch.where(a[:, 0] == 15, 12, torch.where(a[:, 0] == 1, 18,
                        torch.where(a[:, 0] == 5, 2, torch.where(a[:, 0] == 53, 16,
                        torch.where(a[:, 0] == 14, 4, torch.where(a[:, 0] == 34, 6,
                        torch.where(a[:, 0] == 19, 7, torch.where(a[:, 0] == 11, 9,
                        torch.where(a[:, 0] == 3, 11, torch.where(a[:, 0] == 30, 13,
                        torch.where(a[:, 0] == 33, 14, torch.where(a[:, 0] == 12, 17,
                        torch.where(a[:, 0] == 52, 19, -2))))))))))))))))))))

    transformed[:, 1] = torch.where(a[:, 1] == 1, 1,
    torch.where(a[:, 1] == 2, 20, torch.where(a[:, 1] == 3, 10,
    torch.where(a[:, 1] == 0, 15, torch.where(a[:, 1] == 4, 5,
    torch.where(a[:, 1] == 6, 7,
    torch.where(a[:, 1] == 5, 12, -2)))))))

    transformed[:, 2] = torch.where(a[:, 2] == 0, 1,
    torch.where(a[:, 2] == 1, 20, torch.where(a[:, 2] == -1, 10,
    torch.where(a[:, 2] == 3, 5,
    torch.where(a[:, 2] == 2, 15, -2)))))

    transformed[:, 3] = torch.where(a[:, 3] == 4, 1,
    torch.where(a[:, 3] == 3, 20, torch.where(a[:, 3] == 1, 10,
    torch.where(a[:, 3] == 2, 5, torch.where(a[:, 3] == 7, 15,
    torch.where(a[:, 3] == 6, 18, -2))))))

    transformed[:, 4] = torch.where(a[:, 4] == 0, 1,
    torch.where(a[:, 4] == 1, 20, -2))

    transformed[:, 5] = torch.where(a[:, 5] == 0, 1,
    torch.where(a[:, 5] == 1, 20, -2))

    transformed[:, 6] = torch.where(a[:, 6] == 3, 1,
    torch.where(a[:, 6] == 0, 20, torch.where(a[:, 6] == 1, 10,
    torch.where(a[:, 6] == 2, 15, torch.where(a[:, 6] == 4, 5, -2)))))
    return transformed

#            # model, batched_graph, batched_feats, optimizer, epoch, logger)
import time
import torch
def train_sage(model, g, feats, optimizer, epoch, logger):
    device = torch.device("cuda")
    feats = feats.to(device, non_blocking=True)
    g = g.to(device, non_blocking=True)
    model = model.to(device, non_blocking=True)

    model.train()
    loss_list, latent_list, cb_list = [], [], []

    # ✅ Use AMP with Gradient Scaler to reduce memory usage
    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits, loss, _, cb, loss_list3, latent_train, quantized, latents, sample_list_train = model(g, feats, epoch, logger)

    # ✅ Zero out gradients (set_to_none=True is more memory-efficient)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    loss = loss.double()  # Converts loss to float64

    # print(f"train_sage quantized {quantized}")
    # ✅ Scale loss before backward to avoid overflow issues
    scaler.scale(loss).backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"after model forward {name}: {param.grad.abs().mean()}")  # Mean absolute activation
        else:
            print(f"after model forward {name}: param.grad is None")  # Mean absolute activation
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"after model forward {name}: {param.grad.abs().mean()}")
    #     else:
    #         print(f"after model forward {name}: param.grad is None")
    # print(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # ✅ Step optimizer with scaler and update gradients
    scaler.step(optimizer)
    scaler.update()

    # ✅ Detach loss after backprop to avoid holding graph
    loss = loss.detach()

    # ✅ Append latents with detach() to avoid accumulating history
    latent_list.append(latent_train.detach())
    cb_list.append(cb.detach())

    # ✅ Detach unused tensors to avoid holding references
    del logits, quantized, sample_list_train
    torch.cuda.empty_cache()
    return loss, loss_list3, latent_list, latents.detach()
#
#
# def train_sage(model, g, feats, optimizer, epoch, logger):
#
#     device = torch.device("cuda")
#     feats = feats.to(device)
#     g = g.to(device)
#     model = model.to(device)
#     # for name, param in model.named_parameters():
#     #     in_optimizer = any(param in group["params"] for group in optimizer.param_groups)
#     #     if not in_optimizer:
#     #         print(f"{name} is missing from optimizer!")
#
#     model.train()
#     loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
#
#     with torch.cuda.amp.autocast(dtype=torch.float16):
#         _, logits, loss, _, cb, loss_list3, latent_train, quantized, latents, sample_list_train = model(g, feats, epoch,
#                                                                                                         logger)  # g is blocks
#
#     # del logits, quantized
#     torch.cuda.empty_cache()
#
#     # optimizer.zero_grad()
#     optimizer.zero_grad(set_to_none=False)  # Ensure it resets to zero instead of None
#
#     loss.backward()
#     loss = loss.detach()
#
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             print(f"after model forward {name}: {param.grad.abs().mean()}")  # Mean absolute activation
#         else:
#             print(f"after model forward {name}: param.grad is None")  # Mean absolute activation
#     optimizer.step()
#
#     latent_list.append(latent_train)
#     cb_list.append(cb)
#     # print(f"loss_list {loss_list3}")
#
#     return loss, loss_list3, latent_list, latents
#

def evaluate(model, g, feats, epoch, logger, g_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feats = feats.to(device)  # Ensure feats are on GPU
    model.eval()
    loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
    # with torch.no_grad(), autocast():
    with torch.no_grad():
        #   logits, loss, _, cb, loss_list3, latent_train, quantized, latents, sample_list_train = model(g, feats, epoch, logger)
        logits, test_loss, _, cb, test_loss_list3, latent_train, quantized, test_latents, sample_list_test = model(g, feats, epoch, logger, g_base)  # g is blocks
    latent_list.append(latent_train.detach().cpu())
    cb_list.append(cb.detach().cpu())
    test_latents = test_latents.detach().cpu()
    return test_loss, test_loss_list3, latent_list, test_latents, sample_list_test


class MoleculeGraphDataset(Dataset):
    def __init__(self, adj_dir, attr_dir):
        self.adj_files = sorted(glob.glob(f"{adj_dir}/concatenated_adj_batch_*.npy"))
        self.attr_files = sorted(glob.glob(f"{attr_dir}/concatenated_attr_batch_*.npy"))
        assert len(self.adj_files) == len(self.attr_files), "Mismatch in adjacency and attribute files"

    def __len__(self):
        return len(self.adj_files)

    def __getitem__(self, idx):
        attr = []
        adj = []
        adj_matrix = torch.tensor(np.load(self.adj_files[idx]))  # Load adjacency matrix

        attr_matrix = torch.tensor(np.load(self.attr_files[idx]))  # Load atom features
    #     # print(f"attr_matrix.shape {attr_matrix.shape}")
    #     # pad_size = 100 - attr_matrix.shape[0]
    #     attr.append(attr_matrix)  # Pad rows only
    #     # print(f"padded_attr.shape {padded_attr.shape}")
    #
        return torch.tensor(adj_matrix, dtype=torch.float32), torch.tensor(attr_matrix, dtype=torch.float32)


def collate_fn(batch):
    """Pads adjacency matrices and attributes while handling size mismatches."""
    adj_matrices, attr_matrices = zip(*batch)

    # Find max number of nodes in this batch
    # max_nodes = max(adj.shape[0] for adj in adj_matrices)

    # # Pad adjacency matrices to ensure square shape (max_nodes, max_nodes)
    # padded_adj = []
    # for adj in adj_matrices:
    #     pad_size = max_nodes - adj.shape[0]
    #     padded_adj.append(torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size)))  # Pad both dimensions

    # padded_adj = torch.stack(padded_adj)  # Now safely stack

    # Pad attribute matrices (features) to (max_nodes, num_features)
    # num_features = attr_matrices[0].shape[1]  # Keep number of features same
    # padded_attr = []
    # for attr in attr_matrices:
    #     pad_size = max_nodes - attr.shape[0]
    #     padded_attr.append(torch.nn.functional.pad(attr, (0, 0, 0, pad_size)))  # Pad rows only
    #
    # padded_attr = torch.stack(padded_attr)  # Now safely stack

    return adj_matrices, attr_matrices


import dgl
import torch

import torch
import time
import dgl

import dgl
import torch
import time

def convert_to_dgl(adj_batch, attr_batch):
    """
    Converts a batch of adjacency matrices and attributes to two lists of DGLGraphs.

    - Base graphs: Only 1-hop binary edges.
    - Extended graphs: Includes 2-hop and 3-hop weighted edges.

    Optimized to avoid unnecessary copies and handle different edge types correctly.
    """
    start_total = time.time()  # Track execution time

    base_graphs = []
    extended_graphs = []

    for i in range(len(adj_batch)):  # Loop over each molecule
        adj_matrices = adj_batch[i].view(1000, 100, 100)  # Reshape adjacency matrices
        attr_matrices = attr_batch[i].view(1000, 100, 7)  # Reshape node features

        for j in range(len(attr_matrices)):  # Process each molecule separately
            adj_matrix = adj_matrices[j]
            attr_matrix = attr_matrices[j]

            # ------------------------------------------
            # Remove padding: keep only non-zero attribute rows
            # ------------------------------------------
            nonzero_mask = (attr_matrix.abs().sum(dim=1) > 0)
            num_total_nodes = nonzero_mask.sum().item()
            filtered_attr_matrix = attr_matrix[nonzero_mask]
            filtered_adj_matrix = adj_matrix[:num_total_nodes, :num_total_nodes]

            # ------------------------------------------
            # Create the base graph (only 1-hop edges)
            # ------------------------------------------
            src, dst = filtered_adj_matrix.nonzero(as_tuple=True)

            # Remove duplicate edges (only keep src > dst to avoid bidirection)
            mask = src > dst
            src = src[mask]
            dst = dst[mask]

            # Extract weights for 1-hop edges
            edge_weights = filtered_adj_matrix[src, dst]

            base_g = dgl.graph((src, dst), num_nodes=num_total_nodes)
            base_g.ndata["feat"] = filtered_attr_matrix
            base_g.edata["weight"] = edge_weights.float()

            # Assign edge types for 1-hop edges
            base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), dtype=torch.int)

            # Add self-loops
            base_g = dgl.add_self_loop(base_g)
            base_graphs.append(base_g)

            # ------------------------------------------
            # Generate 2-hop and 3-hop adjacency matrices
            # ------------------------------------------
            adj_2hop = dgl.khop_adj(base_g, 2).to_dense()  # Convert sparse to dense
            adj_3hop = dgl.khop_adj(base_g, 3).to_dense()

            # ------------------------------------------
            # Construct the full adjacency matrix
            # ------------------------------------------
            full_adj_matrix = filtered_adj_matrix.clone()

            # Add 2-hop and 3-hop connections but avoid overwriting 1-hop edges
            full_adj_matrix += (adj_2hop * 0.5) * (filtered_adj_matrix == 0)  # Add 2-hop only where 1-hop is absent
            full_adj_matrix += (adj_3hop * 0.3) * (filtered_adj_matrix == 0)  # Add 3-hop only where 1-hop is absent

            # Ensure diagonal values (self-connections) remain 1.0
            torch.diagonal(full_adj_matrix).fill_(1.0)

            # ------------------------------------------
            # Create the extended graph from the full adjacency matrix
            # ------------------------------------------
            src_full, dst_full = full_adj_matrix.nonzero(as_tuple=True)
            extended_g = dgl.graph((src_full, dst_full), num_nodes=num_total_nodes)

            # Extract edge weights from the full adjacency matrix
            edge_weights = full_adj_matrix[src_full, dst_full]
            extended_g.edata["weight"] = edge_weights.float()

            # ------------------------------------------
            # Assign edge types (1-hop, 2-hop, 3-hop)
            # ------------------------------------------
            one_hop = filtered_adj_matrix[src_full, dst_full] > 0
            two_hop = (adj_2hop[src_full, dst_full] > 0) & ~one_hop
            three_hop = (adj_3hop[src_full, dst_full] > 0) & ~(one_hop | two_hop)

            edge_types = torch.zeros_like(src_full, dtype=torch.int)
            edge_types[one_hop] = 1
            edge_types[two_hop] = 2
            edge_types[three_hop] = 3

            # Assign self-loop type explicitly
            self_loop_mask = src_full == dst_full
            edge_types[self_loop_mask] = 4  # Self-loops are assigned type 4

            extended_g.edata["edge_type"] = edge_types

            # ------------------------------------------
            # Assign node features to the extended graph
            # ------------------------------------------
            extended_g.ndata["feat"] = filtered_attr_matrix
            extended_g = dgl.add_self_loop(extended_g)

            # ------------------------------------------
            # Debug: Validate that remaining features are zero
            # ------------------------------------------
            remaining_features = attr_matrix[base_g.num_nodes():]
            if not torch.all(remaining_features == 0):
                print("⚠️ WARNING: Non-zero values found in remaining features!")

            extended_graphs.append(extended_g)

    # return base_graphs, extended_graphs

    return base_graphs, base_graphs


from torch.utils.data import Dataset
import dgl
class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs  # List of DGLGraphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]


def print_large_tensors(threshold=10):  # Only print tensors larger than 10MB
    count = 0
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size_MB = obj.numel() * obj.element_size() / 1024 ** 2  # Convert to MB
                if size_MB > threshold:  # Print only large tensors
                    print(f"Tensor {count}: Shape={obj.shape}, Size={size_MB:.2f}MB, Requires Grad={obj.requires_grad}")
                    count += 1
        except:
            pass
    print(f"Total Large Tensors (> {threshold}MB): {count}")


def run_inductive(
        conf,
        model,
        optimizer,
        accumulation_steps,
        logger
):
    import gc
    import torch
    import itertools
    # ----------------------------
    # define train and test list
    # ----------------------------
    import time
    # Initialize dataset and dataloader
    start_dataload = time.time()

    dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    dataload_time = time.time() - start_dataload
    print(f"dataload_time: {dataload_time:.6f} sec")

    for epoch in range(1, conf["max_epoch"] + 1):
        loss_list_list_train = [[] for _ in range(11)]
        loss_list_list_test = [[] for _ in range(11)]
        loss_list = []
        print(f"epoch {epoch} ------------------------------")
        # --------------------------------
        # Train
        # --------------------------------
        if conf["train_or_infer"] == "train":
            print("train ============")
            # Iterate through batches
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 5:
                    break
                start_convert_to_dgl = time.time()
                glist_base, glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
                chunk_size = conf["chunk_size"]  # in 10,000 molecules
                convert_to_dgl_time = time.time() - start_convert_to_dgl
                for i in range(0, len(glist), chunk_size):
                    import gc
                    import torch
                    chunk = glist[i:i + chunk_size]    # including 2-hop and 3-hop
                    batched_graph = dgl.batch(chunk)
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]
                    loss, loss_list_train, latent_train, latents = train_sage(
                        model, batched_graph, batched_feats, optimizer, epoch, logger)
                    print(loss)
                    # -----------------------------------------
                    # Total allocated memory by tensors (bytes)
                    # -----------------------------------------
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                    # Total reserved memory by PyTorch (bytes)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
                    print(f"Allocated Memory: {allocated:.2f} MB")
                    print(f"Reserved Memory: {reserved:.2f} MB")
                    # print(torch.cuda.memory_summary(device=0, abbreviated=False))

                    # model.reset_kmeans()
                    # latent_train = torch.stack(latent_train).detach() if isinstance(latent_train,
                    #                                                                 list) else latent_train.detach()
                    loss_list.append(loss.detach().cpu().item())  # Ensure loss is detached
                    torch.cuda.synchronize()
                    args = get_args()
                    if args.get_umap_data:
                        cb_new = model.vq._codebook.emb.detach()
                        np.savez(f"./init_codebook_{epoch}", cb_new.cpu().numpy())  # Already detached

                        # ------------------
                        # memory release
                        # ------------------
                        latents = latents.detach().cpu().numpy()  # Detach before saving
                        np.savez(f"./latents_{epoch}", latents)
                        del cb_new, latents  # Explicitly delete tensors
                        gc.collect()
                        torch.cuda.empty_cache()

                    loss_list_list_train = [x + [y] for x, y in
                                            zip(loss_list_list_train, loss_list_train)]
                    # ------------------
                    # memory release
                    # ------------------
                    del loss_list_train  # Explicitly delete it
                    latents.detach()
                    del batched_graph, batched_feats, chunk, latent_train, latents
                    gc.collect()
                    torch.cuda.empty_cache()

        # --------------------------------
        # Save model
        # --------------------------------
        state = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        model.load_state_dict(state)
        # --------------------------------
        # Test
        # --------------------------------
        test_loss_list = []
        print("eval ============")
        for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, 5, None), start=5):
            if idx == 6:
                break
            glist_base, glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
            chunk_size = conf["chunk_size"]  # in 10,000 molecules
            for i in range(0, len(glist), chunk_size):
                chunk = glist[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]   # only 1-hop
                batched_graph = dgl.batch(chunk)
                batched_graph_base = dgl.batch(chunk_base)
                # Ensure node features are correctly extracted
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]
                # model, g, feats, epoch, logger, g_base
                test_loss, loss_list_test, latent_train, latents, sample_list_test = evaluate(
                    model, batched_graph, batched_feats, epoch, logger, batched_graph_base)
                model.reset_kmeans()
                test_loss_list.append(test_loss.cpu().item())  # Ensures loss does not retain computation graph
                torch.cuda.synchronize()
                # ----------------
                # memory release
                # ----------------
                loss_list_list_test = [x + [y] for x, y in zip(loss_list_list_test, loss_list_test)]

                # ------------------
                # memory release
                # ------------------
                latents.detach()
                del batched_graph, batched_feats, chunk, latent_train, latents
                gc.collect()
                torch.cuda.empty_cache()

            """div_ele_loss.item(), bond_num_div_loss.item(), aroma_div_loss.item(),
            ringy_div_loss.item(), h_num_div_loss.item(), charge_div_loss.item(),
            elec_state_div_loss.item(), spread_loss, pair_loss,
            sil_loss, equivalent_atom_loss"""

        print(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.7f}, test_loss {sum(test_loss_list)/len(test_loss_list):.7f}")
        logger.info(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.7f}, test_loss {sum(test_loss_list)/len(test_loss_list):.7f}")
        print(f"train - div_element_loss: {sum(loss_list_list_train[0]) / len(loss_list_list_train[0]): 7f}, "
              f"train - bond_num_div_loss: {sum(loss_list_list_train[1]) / len(loss_list_list_train[1]): 7f}, "
              f"train - aroma_div_loss: {sum(loss_list_list_train[2]) / len(loss_list_list_train[2]): 7f}, "
              f"train - ringy_div_loss: {sum(loss_list_list_train[3]) / len(loss_list_list_train[3]): 7f}, "
              f"train - h_num_div_loss: {sum(loss_list_list_train[4]) / len(loss_list_list_train[4]): 7f}, "
              f"train - elec_state_div_loss: {sum(loss_list_list_train[6]) / len(loss_list_list_train[6]): 7f}, "
              f"train - charge_div_loss: {sum(loss_list_list_train[5]) / len(loss_list_list_train[5]): 7f}, "
              f"train - sil_loss: {sum(loss_list_list_train[9]) / len(loss_list_list_train[9]): 7f},"
              f"train - equiv_atom_loss: {sum(loss_list_list_train[10]) / len(loss_list_list_train[10]): 7f},"
              )

        print(f"test - div_element_loss: {sum(loss_list_list_test[0]) / len(loss_list_list_test[0]): 7f}, "
              f"test - bond_num_div_loss: {sum(loss_list_list_test[1]) / len(loss_list_list_test[1]): 7f}, "
              f"test - aroma_div_loss: {sum(loss_list_list_test[2]) / len(loss_list_list_test[2]): 7f}, "
              f"test - ringy_div_loss: {sum(loss_list_list_test[3]) / len(loss_list_list_test[3]): 7f}, "
              f"test - h_num_div_loss: {sum(loss_list_list_test[4]) / len(loss_list_list_test[4]): 7f}, "
              f"test - elec_state_div_loss: {sum(loss_list_list_test[6]) / len(loss_list_list_test[6]): 7f}, "
              f"test - charge_div_loss: {sum(loss_list_list_test[5]) / len(loss_list_list_test[5]): 7f}, "
              f"test - sil_loss: {sum(loss_list_list_test[9]) / len(loss_list_list_test[9]): 7f}",
              f"test - equiv_atom_loss: {sum(loss_list_list_test[10]) / len(loss_list_list_test[10]): 7f}",
              )

        # Log training losses
        logger.info(
            f"train - div_element_loss: {sum(loss_list_list_train[0]) / len(loss_list_list_train[0]):7f}, "
            f"train - bond_num_div_loss: {sum(loss_list_list_train[1]) / len(loss_list_list_train[1]):7f}, "
            f"train - aroma_div_loss: {sum(loss_list_list_train[2]) / len(loss_list_list_train[2]):7f}, "
            f"train - ringy_div_loss: {sum(loss_list_list_train[3]) / len(loss_list_list_train[3]):7f}, "
            f"train - h_num_div_loss: {sum(loss_list_list_train[4]) / len(loss_list_list_train[4]):7f}, "
            f"train - elec_state_div_loss: {sum(loss_list_list_train[6]) / len(loss_list_list_train[6]):7f}, "
            f"train - charge_div_loss: {sum(loss_list_list_train[5]) / len(loss_list_list_train[5]):7f}, "
            f"train - sil_loss: {sum(loss_list_list_train[9]) / len(loss_list_list_train[9]):7f}, "
            f"train - equiv_atom_loss: {sum(loss_list_list_train[10]) / len(loss_list_list_train[10]):7f}, "
        )

        # Log testing losses
        logger.info(
            f"test - div_element_loss: {sum(loss_list_list_test[0]) / len(loss_list_list_test[0]):7f}, "
            f"test - bond_num_div_loss: {sum(loss_list_list_test[1]) / len(loss_list_list_test[1]):7f}, "
            f"test - aroma_div_loss: {sum(loss_list_list_test[2]) / len(loss_list_list_test[2]):7f}, "
            f"test - ringy_div_loss: {sum(loss_list_list_test[3]) / len(loss_list_list_test[3]):7f}, "
            f"test - h_num_div_loss: {sum(loss_list_list_test[4]) / len(loss_list_list_test[4]):7f}, "
            f"test - elec_state_div_loss: {sum(loss_list_list_test[6]) / len(loss_list_list_test[6]):7f}, "
            f"test - charge_div_loss: {sum(loss_list_list_test[5]) / len(loss_list_list_test[5]):7f}, "
            f"test - sil_loss: {sum(loss_list_list_test[9]) / len(loss_list_list_test[9]):7f}, "
            f"test - equiv_atom_loss: {sum(loss_list_list_test[10]) / len(loss_list_list_test[10]):7f}, "
        )
        np.savez(f"./sample_emb_ind_{epoch}", sample_list_test[0].cpu())
        np.savez(f"./sample_node_feat_{epoch}", sample_list_test[1].cpu())
        np.savez(f"./sample_adj_{epoch}", sample_list_test[2].cpu()[:500, :500])
        np.savez(f"./sample_bond_num_{epoch}", sample_list_test[3].cpu()[:500])
        np.savez(f"./sample_src_{epoch}", sample_list_test[4].cpu()[:500])
        np.savez(f"./sample_dst_{epoch}", sample_list_test[5].cpu()[:500])
        # np.savez(f"./sample_hop_type_{epoch}", None)
        np.savez(f"./sample_adj_base_{epoch}", sample_list_test[7].cpu()[:500])

