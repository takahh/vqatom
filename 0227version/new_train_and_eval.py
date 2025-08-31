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
    # Ensure model is on device
    model = model.to(device)
    model.train()

    # Move graph and features to device
    g = g.to(device)
    g.ndata['feat'] = g.ndata['feat'].to(device) if 'feat' in g.ndata else g.ndata['feat']
    feats = feats.to(device)

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(init_scale=1e2)
    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    with torch.cuda.amp.autocast():
        # data, features, chunk_i, logger=None, epoch=None, batched_graph_base=None, mode=None):
        outputs = model(g, feats, chunk_i, logger, epoch)
        (_, logits, loss, _, cb, loss_list3,
         latent_train, quantized, latents,
         sample_list_train, num_unique) = outputs

    # Sync codebook weights
    model.vq._codebook.embed.data.copy_(cb.to(device))

    # Free some unused outputs early
    del logits, quantized
    # torch.cuda.empty_cache()  # remove in production

    # Backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Return only what’s needed
    return (
        loss.detach(),                     # keep as tensor if you want
        [l.item() if hasattr(l, 'item') else l for l in loss_list3],
        latent_train.detach().cpu(),       # safe on CPU
        latents.detach().cpu() if torch.is_tensor(latents) else latents,
        num_unique
    )

def evaluate(model, g, feats, epoch, logger, g_base, chunk_i, mode=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if mode == "init_kmeans_loop":
            latents = model(g, feats, chunk_i, logger, epoch, g_base, mode)
            return latents
        elif mode == "init_kmeans_final":
            model(g, feats, chunk_i, logger, epoch, g_base, mode)
            return 0
        else:
            (
                _,
                logits,
                test_loss,
                _,
                cb,
                test_loss_list3,
                latent_train,
                quantized,
                test_latents,
                sample_list_test,
                num_unique,
            ) = model(g, feats, chunk_i, logger, epoch, g_base, mode)

    # detach and move to cpu
    latent_train_cpu = latent_train.detach().cpu()
    cb_cpu = cb.detach().cpu()
    test_latents_cpu = test_latents.detach().cpu()
    test_loss = test_loss.to(device)

    # cleanup
    del logits, latent_train, cb, test_latents
    torch.cuda.empty_cache()

    # return single tensors, not lists
    return test_loss, test_loss_list3, latent_train_cpu, test_latents_cpu, sample_list_test, quantized, num_unique


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
            attr_matrices = attr_batch[i].view(-1, 100, 27)
        else:
            adj_matrices = adj_batch[i].view(-1, 100, 100)
            attr_matrices = attr_batch[i].view(-1, 100, 27)

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
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
    print(f"[{tag}] GPU Allocated: {allocated:.2f} MB | GPU Reserved: {reserved:.2f} MB")

def run_inductive(conf, model, optimizer, accumulation_steps, logger):
    import gc, itertools, torch, os
    from collections import Counter

    # ----------------------------
    # dataset and dataloader
    # ----------------------------
    datapath = DATAPATH if conf['train_or_infer'] in ("hptune", "infer", "use_nonredun_cb_infer") else DATAPATH_INFER
    dataset = MoleculeGraphDataset(adj_dir=datapath, attr_dir=datapath)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(1, conf["max_epoch"] + 1):
        print(f"epoch {epoch} ------------------------------")

        # fresh containers per epoch
        loss_list_list_train = [[] for _ in range(11)]
        loss_list_list_test = [[] for _ in range(11)]
        loss_list = []
        cb_unique_num_list = []
        cb_unique_num_list_test = []
        # ------------------------------------------
        # 2 batch data で kmeans, CB 確定
        # ------------------------------------------
        if conf["train_or_infer"] == "infer" or conf["train_or_infer"] == "hptune":
            kmeans_start_num = 6
            # kmeans_end_num = 18
            kmeans_end_num = 7
        if conf["train_or_infer"] == "analysis":
            kmeans_start_num = 0
            kmeans_end_num = 1
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
        print(f"all_latents_tensor.shape {all_latents_tensor.shape}")
        # -------------------------------------------------------------------
        # Run k-means on the collected latent vectors (goes to the deepest)
        # -------------------------------------------------------------------
        evaluate(model, all_latents_tensor, batched_feats, epoch, logger, None, None, "init_kmeans_final")
        print("initial kmeans done")

        # ---------------------------
        # TRAIN
        # ---------------------------
        if conf["train_or_infer"] in ("hptune", "train"):
            # re-init codebook
            model.vq._codebook.initted.data.copy_(torch.tensor([False], device=model.vq._codebook.initted.device))
            print("TRAIN ---------------")

            for idx, (adj_batch, attr_batch) in enumerate(dataloader):

                # # ------------- remove thi soon --------------
                # if idx == 1:
                #     break
                # # ------------- remove thi soon --------------
                if idx == 5:
                    break
                print(f"idx {idx}")

                glist_base, glist = convert_to_dgl(adj_batch, attr_batch)
                chunk_size = conf["chunk_size"] if epoch < 2 else conf["chunk_size2"]
                print(f"CHUNK SIZE {chunk_size} ============================")
                for i in range(0, len(glist), chunk_size):
                    print_memory_usage(f"idx {idx}")
                    # # ------------- remove thi soon --------------
                    # if i == chunk_size:
                    #     break
                    # # ------------- remove thi soon --------------
                    chunk = glist[i:i + chunk_size]
                    batched_graph = dgl.batch(chunk)
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]

                    # train step
                    # (model, g, feats, optimizer, chunk_i, logger, epoch):
                    loss, loss_list_train, latent_train_cpu, latents, cb_num_unique = train_sage(
                        model, batched_graph, batched_feats, optimizer, int(i / chunk_size), logger, epoch
                    )

                    # record scalar losses
                    clean_losses = [(l.detach().cpu().item() if hasattr(l, "detach") else float(l))
                                    for l in loss_list_train]
                    for j, val in enumerate(clean_losses):
                        loss_list_list_train[j].append(val)
                    loss_list.append(loss.detach().cpu().item())
                    cb_unique_num_list.append(int(cb_num_unique) if torch.is_tensor(cb_num_unique) else cb_num_unique)

                    # cleanup
                    del batched_graph, batched_feats, chunk, latents, loss, loss_list_train, cb_num_unique
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # cleanup glist
                for g in glist:
                    g.ndata.clear()
                    g.edata.clear()
                del glist, glist_base
                gc.collect()

        # ---------------------------
        # SAVE MODEL
        # ---------------------------
        if conf["train_or_infer"] != "analysis":
            state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            # torch.save(model.__dict__, f"model_buffers_{epoch}.pth")  # ⚠️ removed to avoid leaks
            model.load_state_dict(state)

        # ---------------------------
        # TEST
        # ---------------------------
        test_loss_list = []
        quantized = None
        if conf['train_or_infer'] == "hptune":
            start_num, end_num = 5, 6
        elif conf['train_or_infer'] == "analysis":
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
            start_num, end_num = 0, 1
        else:  # infer
            start_num, end_num = 6, 10
        print(f"start num {start_num}, end num {end_num}")

        ind_counts = Counter()

        for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, start_num, end_num), start=start_num):
            print(f"TEST --------------- {idx}")
            glist_base, glist = convert_to_dgl(adj_batch, attr_batch)
            chunk_size = conf["chunk_size"]

            for i in range(0, len(glist), chunk_size):

                # # ------------- remove thi soon --------------
                # if i == chunk_size:
                #     break
                # # ------------- remove thi soon --------------
                chunk = glist[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]
                batched_graph = dgl.batch(chunk)
                batched_graph_base = dgl.batch(chunk_base)
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]

                test_loss, loss_list_test, latent_train_cpu, latents_cpu, sample_list_test, quantized, cb_num_unique = \
                    evaluate(model, batched_graph, batched_feats, epoch, logger, batched_graph_base, idx)

                test_loss_list.append(test_loss.cpu().item())
                cb_unique_num_list_test.append(int(cb_num_unique) if torch.is_tensor(cb_num_unique) else cb_num_unique)
                loss_list_list_test = [x + [y] for x, y in zip(loss_list_list_test, loss_list_test)]

                # optionally save small parts per chunk instead of keeping in lists
                try:
                    ind_chunk = sample_list_test[0].cpu().numpy()
                    latent_chunk = sample_list_test[2].cpu().numpy()
                    # save per chunk to avoid huge accumulation
                    np.savez(f"./tmp_ind_{epoch}_{idx}_{i}.npz", ind_chunk=ind_chunk)
                    np.savez(f"./tmp_latent_{epoch}_{idx}_{i}.npz", latent_chunk=latent_chunk)
                    del latent_chunk
                    # count indices
                    ind_counts.update(ind_chunk.flatten().tolist())
                except Exception as e:
                    print(f"[WARN] collecting chunk failed: {e}")

                del batched_graph, batched_graph_base, batched_feats, chunk, chunk_base
                gc.collect()
                torch.cuda.empty_cache()

            # cleanup graphs after idx
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base
            gc.collect()

            args = get_args()
            if args.get_umap_data:
                cb_new = model.vq._codebook.embed
                np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())

        # ---------------------------
        # stats and save
        # ---------------------------
        flat = list(ind_counts.elements())
        print(f"len(flat) = {len(flat)}, unique = {len(set(flat))}")
        used_cb_vectors_all_epochs = model.vq._codebook.embed[0][torch.unique(torch.tensor(flat), sorted=True).long()]

        kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        os.makedirs(kw, exist_ok=True)

        # log stats
        print(f"test - commit_loss: {sum(loss_list_list_test[1]) / max(1,len(loss_list_list_test[1])):.6f}, "
              f"test - cb_loss: {sum(loss_list_list_test[2]) / max(1,len(loss_list_list_test[2])):.6f}")
        logger.info(f"test - commit_loss: {sum(loss_list_list_test[1]) / max(1,len(loss_list_list_test[1])):.6f}, "
                    f"test - cb_loss: {sum(loss_list_list_test[2]) / max(1,len(loss_list_list_test[2])):.6f}")

        if conf['train_or_infer'] in ("hptune", "train"):
            print(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.9f}, test_loss {sum(test_loss_list)/len(test_loss_list):.9f}")
            np.savez(f"./{kw}/used_cb_vectors_{epoch}", used_cb_vectors_all_epochs.detach().cpu().numpy())

        # cleanup big lists
        loss_list_list_train.clear()
        loss_list_list_test.clear()
        loss_list.clear()
        cb_unique_num_list.clear()
        cb_unique_num_list_test.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # debug growth
        # import objgraph
        # objgraph.show_growth(limit=10)
        # del model
        gc.collect()
        torch.cuda.empty_cache()
        print(next(model.parameters()).device)  # should say cuda:0

        # for name, buf in model.named_buffers():
        #     print(f"{name}: {buf.shape} {buf.device}")
        # for name, p in model.named_parameters():
        #     print(f"{name}: {p.shape} {p.device}")

        # state = copy.deepcopy(model.state_dict())
        # del model
        # gc.collect()
        # torch.cuda.empty_cache()
        # from models import EquivariantThreeHopGINE
        # # Recreate fresh model and load weights
        # model = EquivariantThreeHopGINE(in_feats=args.hidden_dim, hidden_feats=args.hidden_dim,
        #                                 out_feats=args.hidden_dim, args=args)
        # device = torch.device("cuda")
        # model.load_state_dict(state)
        # model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'], weight_decay=1e-4)


