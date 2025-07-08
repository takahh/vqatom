from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from train_teacher import get_args
from collections import Counter

DATAPATH = "../data/both_mono"
DATAPATH_INFER = "../data/additional_data_for_analysis"


def train_sage(model, g, feats, optimizer, chunk_i, logger, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    g.to(device)
    feats.to(device)
    loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
    scaler = torch.cuda.amp.GradScaler(init_scale=1e2)  # Try lower scale like 1e1
    optimizer.zero_grad()
    with (torch.cuda.amp.autocast()):
        _, logits, loss, _, cb, loss_list3, latent_train, quantized, latents, sample_list_train, num_unique \
        = model(g, feats, chunk_i, logger, epoch)  # g is blocks
    model.vq._codebook.embed.data.copy_(cb)
    loss = loss.to(device)
    del logits, quantized
    torch.cuda.empty_cache()
    scaler.scale(loss).backward(retain_graph=False)  # Ensure this is False unless needed
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    latent_list.append(latent_train.detach().cpu())
    cb_list.append(cb.detach().cpu())
    return loss, loss_list3, latent_list, latents, num_unique


def evaluate(model, g, feats, epoch, logger, g_base, chunk_i, mode=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
    # with torch.no_grad(), autocast():
    with (torch.no_grad()):  # data, features, chunk_i, logger=None, epoch=None, batched
        if mode == "init_kmeans_loop":
            latents = model(g, feats, chunk_i, logger, epoch, g_base, mode)
            return latents
        elif mode == "init_kmeans_final":
            model(g, feats, chunk_i, logger, epoch, g_base, mode)
            return 0
        else:
            _, logits, test_loss, _, cb, test_loss_list3, latent_train, quantized, test_latents, sample_list_test,\
            num_unique = model(g, feats, chunk_i, logger, epoch, g_base, mode)  # g is blocks
    latent_list.append(latent_train.detach().cpu())
    cb_list.append(cb.detach().cpu())
    test_latents = test_latents.detach().cpu()
    test_loss = test_loss.to(device)
    del logits
    torch.cuda.empty_cache()
    return test_loss, test_loss_list3, latent_list, test_latents, sample_list_test, quantized, num_unique


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
        return torch.tensor(adj_matrix, dtype=torch.float32), torch.tensor(attr_matrix, dtype=torch.float32)


def collate_fn(batch):
    """Pads adjacency matrices and attributes while handling size mismatches."""
    adj_matrices, attr_matrices = zip(*batch)
    # for item in adj_matrices:
    #     print(item.shape)
    return adj_matrices, attr_matrices


import dgl
import torch


def convert_to_dgl(adj_batch, attr_batch):
    """
    Converts a batch of adjacency matrices and attributes to two lists of DGLGraphs.

    This version includes optimizations such as vectorized edge-type assignment,
    and avoids unnecessary copies where possible.
    """
    base_graphs = []
    extended_graphs = []
    for i in range(len(adj_batch)):  # Loop over each molecule set
        # Reshape the current batch
        args = get_args()
        if args.train_or_infer == 'analysis':
            adj_matrices = adj_batch[i].view(-1, 100, 100)
            attr_matrices = attr_batch[i].view(-1, 100, 7)
        else:
            adj_matrices = adj_batch[i].view(-1, 100, 100)
            attr_matrices = attr_batch[i].view(-1, 100, 7)

        for j in range(len(attr_matrices)):
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
            # Only consider one direction to avoid duplicate edges
            mask = src > dst
            src = src[mask]
            dst = dst[mask]
            edge_weights = filtered_adj_matrix[src, dst]  # Extract weights for 1-hop edges

            base_g = dgl.graph((src, dst), num_nodes=num_total_nodes)
            base_g.ndata["feat"] = filtered_attr_matrix
            base_g.edata["weight"] = edge_weights.float()
            # You can optionally customize edge types for the base graph; here we assign all 1-hop edges.
            base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), dtype=torch.int)
            base_g = dgl.add_self_loop(base_g)

            base_graphs.append(base_g)

            # ------------------------------------------
            # Generate 2-hop and 3-hop adjacency matrices
            # ------------------------------------------
            adj_2hop = dgl.khop_adj(base_g, 2)
            adj_3hop = dgl.khop_adj(base_g, 3)

            # ------------------------------------------
            # Combine adjacency matrices into one
            # ------------------------------------------
            full_adj_matrix = filtered_adj_matrix.clone()
            full_adj_matrix += (adj_2hop * 0.5)  # Incorporate 2-hop connections
            full_adj_matrix += (adj_3hop * 0.3)  # Incorporate 3-hop connections

            # Ensure diagonal values are set to 1.0 (self-connections)
            torch.diagonal(full_adj_matrix).fill_(1.0)

            # ------------------------------------------
            # Create the extended graph from the full adjacency matrix
            # ------------------------------------------
            src_full, dst_full = filtered_adj_matrix.nonzero(as_tuple=True)
            extended_g = dgl.graph((src_full, dst_full), num_nodes=num_total_nodes)
            new_src, new_dst = extended_g.edges()

            # Assign edge weights from the full adjacency matrix
            edge_weights = filtered_adj_matrix[new_src, new_dst]
            extended_g.edata["weight"] = edge_weights.float()

            # ------------------------------------------
            # Vectorized assignment of edge types
            # ------------------------------------------
            one_hop = filtered_adj_matrix[new_src, new_dst] > 0
            two_hop = (adj_2hop[new_src, new_dst] > 0) & ~one_hop
            three_hop = (adj_3hop[new_src, new_dst] > 0) & ~(one_hop | two_hop)
            edge_types = torch.zeros_like(new_src, dtype=torch.int)
            edge_types[one_hop] = 1
            edge_types[two_hop] = 2
            edge_types[three_hop] = 3

            extended_g.edata["edge_type"] = edge_types

            # ------------------------------------------
            # Assign node features to the extended graph
            # ------------------------------------------
            extended_g.ndata["feat"] = filtered_attr_matrix
            extended_g = dgl.add_self_loop(extended_g)
            # ------------------------------------------
            # Validate that remaining features are zero (if applicable)
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

import psutil
import os

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[{tag}] Memory Usage: {mem:.2f} MB")


def run_inductive(
        conf,
        model,
        optimizer,
        accumulation_steps,
        logger):
    import gc
    import torch
    import itertools
    # ----------------------------
    # define train and test list
    # ----------------------------
    # Initialize dataset and dataloader
    if conf['train_or_infer'] == "hptune" or conf['train_or_infer'] == "infer" or conf['train_or_infer'] == "use_nonredun_cb_infer":
        datapath = DATAPATH
    else:
        datapath = DATAPATH_INFER
    dataset = MoleculeGraphDataset(adj_dir=datapath, attr_dir=datapath)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(1, conf["max_epoch"] + 1):
        loss_list_list_train = [[]] * 11
        loss_list_list_test = [[]] * 11
        loss_list = []
        cb_unique_num_list = []
        cb_unique_num_list_test = []

        # ------------------------------------------
        # Infer 時、対象データ全体で kmeans, CB 確定
        # ------------------------------------------
        if conf["train_or_infer"] == "infer":
            kmeans_start_num = 6
            # kmeans_end_num = 18
            kmeans_end_num = 7
            # ------------------------------------------
            # Collect latent vectors (goes to model.py)
            # ------------------------------------------
            all_latents = []
            for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, kmeans_start_num, kmeans_end_num),
                                                          start=kmeans_start_num):
                glist_base, glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
                chunk_size = conf["chunk_size"]  # in 10,000 molecules
                for i in range(0, len(glist), chunk_size):
                    # print(f"init kmeans idx {i}/{len(glist) - 1}")
                    chunk = glist[i:i + chunk_size]
                    chunk_base = glist_base[i:i + chunk_size]   # only 1-hop
                    batched_graph = dgl.batch(chunk)
                    batched_graph_base = dgl.batch(chunk_base)
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]
                    latents \
                        = evaluate(model, batched_graph, batched_feats, epoch, logger, batched_graph_base, idx, "init_kmeans_loop")
                    all_latents.append(latents.cpu())  # move to CPU if needed to save memory
            all_latents_tensor = torch.cat(all_latents, dim=0)  # Shape: [total_atoms_across_all_batches, latent_dim]
            # -------------------------------------------------------------------
            # Run k-means on the collected latent vectors (goes to the deepest)
            # -------------------------------------------------------------------
            evaluate(model, all_latents_tensor, batched_feats, epoch, logger, None, None, "init_kmeans_final")
            print("initial kmeans done")

        print(f"epoch {epoch} ------------------------------")
        # --------------------------------
        # Train
        # --------------------------------
        if conf["train_or_infer"] == "hptune" or conf["train_or_infer"] == "train":
            # make initted FALSE to run kmeans at the beginning in every epoch in train
            model.vq._codebook.initted.data.copy_(torch.Tensor([False]))
            print("TRAIN ---------------")
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 5:
                    break
                # # --------------- delete soon !!!! ----------------
                # if idx == 1:
                #     break
                # --------------- delete soon !!!! ----------------
                print(f"idx {idx}")

                glist_base, glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
                chunk_size = conf["chunk_size"]  # in 10,000 molecules
                for i in range(0, len(glist), chunk_size):
                    # print_memory_usage(f"idx {idx}")
                    chunk = glist[i:i + chunk_size]  # including 2-hop and 3-hop
                    # all_chunks = glist
                    batched_graph = dgl.batch(chunk)
                    # batched_graph_all = dgl.batch(all_chunks)
                    # Ensure node features are correctly extracted
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]
                    loss, loss_list_train, latent_train, latents, cb_num_unique = train_sage(
                        model, batched_graph, batched_feats, optimizer, int(i/chunk_size), logger, idx)
                    cb_unique_num_list.append(cb_num_unique)
                    loss_list.append(loss.detach().cpu().item())  # Ensures loss does not retain computation graph
                    torch.cuda.synchronize()
                    del batched_graph, batched_feats, chunk
                    gc.collect()
                    torch.cuda.empty_cache()
                    args = get_args()
                    if args.get_umap_data:
                        cb_new = model.vq._codebook.embed
                        np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())
                        latents = torch.squeeze(latents)
                        # random_indices = np.random.choice(latent_train.shape[0], 20000, replace=False)
                        np.savez(f"./latents_{epoch}", latents.cpu().detach().numpy())
                    loss_list_list_train = [x + [y] for x, y in zip(loss_list_list_train, loss_list_train)]
        # --------------------------------
        # Save model
        # --------------------------------
        if conf["train_or_infer"] == "analysis":
            pass
        else:
            state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            torch.save(model.__dict__, f"model_buffers_{epoch}.pth")
            model.load_state_dict(state)
        # --------------------------------
        # Test
        # --------------------------------
        test_loss_list = []
        ind_list = []
        latent_list = []
        quantized = None
        if conf['train_or_infer'] == "hptune":
            start_num = 5
            end_num = 6
            # end_num = 31
        elif conf['train_or_infer'] == "analysis":
            start_num = 0
            end_num = 1
        elif conf['train_or_infer'] == "infer":
            start_num = 6
            end_num = 18
            end_num = 8
        print(f"start num {start_num}, end num {end_num}")

        # ------------------------------------------
        # Infer 時、対象データ全体で
        # ------------------------------------------
        for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, start_num, end_num), start=start_num):
            print(f"TEST --------------- {idx}")
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
                test_loss, loss_list_test, latent_train, latents, sample_list_test, quantized, cb_num_unique \
                    = evaluate(model, batched_graph, batched_feats, epoch, logger, batched_graph_base, idx)
                cb_unique_num_list_test.append(cb_num_unique)
                # model.reset_kmeans()
                test_loss_list.append(test_loss.cpu().item())  # Ensures loss does not retain computation graph
                torch.cuda.synchronize()
                del batched_graph, batched_feats, chunk
                gc.collect()
                torch.cuda.empty_cache()
                loss_list_list_test = [x + [y] for x, y in zip(loss_list_list_test, loss_list_test)]
                try:
                    ind_chunk = sample_list_test[0].cpu().tolist()
                    # print(f"len(ind_chunk) = {len(ind_chunk)}")
                    ind_list.append(ind_chunk)
                    latent_chunk = sample_list_test[2].cpu().tolist()
                    # print(f"sample_list_test[2].shape {sample_list_test[2].shape}")
                    latent_list.append(latent_chunk)
                except IndexError:
                    print("INDEX ERROR in collecting ind_list !!!!!!!")
                    pass

            args = get_args()
            if args.get_umap_data:
                cb_new = model.vq._codebook.embed
                np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())
        # -------------------------------------------
        # Count Codebook Frequency and used Codebook
        # -------------------------------------------
        args = get_args()
        if args.train_or_infer == 'use_nonredun_cb_infer':
            ind_list = [item for sublist in ind_list for item in sublist]
            ind_list = [item for sublist in ind_list for item in sublist]
        flat = [item for sublist in ind_list for item in sublist]
        print(f"len(flat) = {len(flat)}")
        unique_flat = list(set(flat))
        print(f"len(unique_flat) = {len(unique_flat)}")
        count = Counter(flat)
        import json
        flat = torch.tensor(flat)  # or torch.tensor(flat, dtype=torch.long)
        unique_sorted_indices = torch.unique(flat, sorted=True).long()
        used_cb_vectors_all_epochs = model.vq._codebook.embed[0][unique_sorted_indices]
        num_zero_keys = 1 if 0.0 in count else 0
        VOCAB_SIZE = conf["codebook_size"]
        all_keys = set(range(VOCAB_SIZE))
        observed_keys = set(count.keys())
        missing_keys = all_keys - observed_keys
        num_missing_keys = len(missing_keys)
        num_observed = len(observed_keys)
        unique_vectors = torch.unique(model.vq._codebook.embed[0], dim=0)
        num_unique_vectors = unique_vectors.size(0)
        logger.info(f"{num_unique_vectors} is num_unique_vectors")
        print(f"Number of observed keys in the test set: {num_observed}")
        print(f"Number of zero keys: {num_zero_keys}")
        print(f"Number of unused keys: {num_missing_keys}")
        logger.info(f"Number of observed keys: {num_observed}")
        logger.info(f"Number of zero keys: {num_zero_keys}")
        logger.info(f"Number of unused keys: {num_missing_keys}")

        # -------------------------------
        # Save loss information and else
        # -------------------------------
        import os
        kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        os.makedirs(kw, exist_ok=True)
        if conf['train_or_infer'] == "hptune" or conf['train_or_infer'] == "train":
            # ---------------------------
            # unique cb per minibatch
            # ---------------------------
            print(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.9f}, test_loss {sum(test_loss_list)/len(test_loss_list):.9f}, "
                f"unique_cb_vecs mean: {sum(cb_unique_num_list) / len(cb_unique_num_list): 9f},"
                f"unique_cb_vecs min: {min(cb_unique_num_list): 9f},"
                f"unique_cb_vecs max: {max(cb_unique_num_list): 9f},")
            logger.info(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.9f}, test_loss {sum(test_loss_list)/len(test_loss_list):.9f}, "
                f"unique_cb_vecs mean: {sum(cb_unique_num_list) / len(cb_unique_num_list): 9f},"
                f"unique_cb_vecs min: {min(cb_unique_num_list): 9f},"
                f"unique_cb_vecs max: {max(cb_unique_num_list): 9f},")
            # ---------------------------
            # losses of train
            # ---------------------------
            print(
                f"train - commit_loss: {sum(loss_list_list_train[1]) / len(loss_list_list_train[1]): 9f}, "
                f"train - cb_loss: {sum(loss_list_list_train[2]) / len(loss_list_list_train[2]): 9f},"
                f"train - sil_loss: {sum(loss_list_list_train[3]) / len(loss_list_list_train[3]): 9f},"
                f"train - unique_cb_vecs: {sum(cb_unique_num_list) / len(cb_unique_num_list): 9f},"
                f"train - repel_loss: {sum(loss_list_list_train[4]) / len(loss_list_list_train[4]): 9f},"
                f"train - attract_loss: {sum(loss_list_list_train[5]) / len(loss_list_list_train[5]): 9f},")
            # Log training losses
            #
            # [0, commit_loss.item(), cb_loss.item(), sil_loss.item(),
            #                     repel_loss.item(), attract_loss.item()]
            logger.info(
                f"train - commit_loss: {sum(loss_list_list_train[1]) / len(loss_list_list_train[1]): 9f}, "
                f"train - cb_loss: {sum(loss_list_list_train[2]) / len(loss_list_list_train[2]): 9f},"
                f"train - sil_loss: {sum(loss_list_list_train[3]) / len(loss_list_list_train[3]): 9f},"
                f"train - repel_loss: {sum(loss_list_list_train[4]) / len(loss_list_list_train[4]): 9f},"
                f"train - attract_loss: {sum(loss_list_list_train[5]) / len(loss_list_list_train[5]): 9f},")
            print("used_cb_vectors_all_epochs.shape")
            print(used_cb_vectors_all_epochs.shape)
            # -----------------------------
            # write latents and cb vectors
            # -----------------------------
            np.savez(f"./{kw}/latents_all_{epoch}.npz", **{f"arr_{i}": arr for i, arr in enumerate(latent_list)})
            np.savez(f"./{kw}/used_cb_vectors_{epoch}", used_cb_vectors_all_epochs.detach().cpu().numpy())

        elif conf['train_or_infer'] == "infer" or conf['train_or_infer'] == "analysis":
            logger.info(f"epoch {epoch}:"
                f"unique_cb_vecs mean: {sum(cb_unique_num_list_test) / len(cb_unique_num_list_test): 9f},"
                f"unique_cb_vecs min: {min(cb_unique_num_list_test): 9f},"
                f"unique_cb_vecs max: {max(cb_unique_num_list_test): 9f},")
            print("used_cb_vectors_all_epochs.shape")
            print(used_cb_vectors_all_epochs.shape)
            np.savez(f"./{kw}/sample_emb_ind_{epoch}", sample_list_test[0].cpu())
            np.savez(f"./{kw}/sample_node_feat_{epoch}", sample_list_test[1].cpu())
            np.savez(f"./{kw}/latents_mol_{epoch}", sample_list_test[2].cpu())
            np.savez(f"./{kw}/sample_bond_num_{epoch}", sample_list_test[3].cpu())
            np.savez(f"./{kw}/sample_src_{epoch}", sample_list_test[4].cpu())
            np.savez(f"./{kw}/latents_all_{epoch}.npz", **{f"arr_{i}": arr for i, arr in enumerate(latent_list)})
            np.savez(f"./{kw}/sample_dst_{epoch}", sample_list_test[5].cpu())
            np.savez(f"./{kw}/used_cb_vectors_{epoch}", used_cb_vectors_all_epochs.detach().cpu().numpy())
            np.savez(f"./{kw}/quantized_{epoch}", quantized.detach().cpu().numpy())
            kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
            with open(f"./{kw}/ind_frequencies.json", 'w') as f:
                json.dump(dict(count), f)
            np.savez(f"./{kw}/sample_adj_base_{epoch}", sample_list_test[6].cpu())

        # ---------------------------
        # losses of evaluation
        # ---------------------------
        print(
              f"test - commit_loss: {sum(loss_list_list_test[1]) / len(loss_list_list_test[1]): 9f}, "
              f"test - cb_loss: {sum(loss_list_list_test[2]) / len(loss_list_list_test[2]): 9f},"
              f"test - sil_loss: {sum(loss_list_list_test[3]) / len(loss_list_list_test[3]): 9f},"
              f"test - repel_loss: {sum(loss_list_list_test[4]) / len(loss_list_list_test[4]): 9f},"
              f"test - attract_loss: {sum(loss_list_list_test[5]) / len(loss_list_list_test[5]): 9f},")
        # Log testing losses
        logger.info(
            f"test - commit_loss: {sum(loss_list_list_test[1]) / len(loss_list_list_test[1]): 9f}, "
            f"test - cb_loss: {sum(loss_list_list_test[2]) / len(loss_list_list_test[2]): 9f},"
            f"test - sil_loss: {sum(loss_list_list_test[3]) / len(loss_list_list_test[3]): 9f},"
            f"test - repel_loss: {sum(loss_list_list_test[4]) / len(loss_list_list_test[4]): 9f},"
            f"test - attract_loss: {sum(loss_list_list_test[5]) / len(loss_list_list_test[5]): 9f},")
