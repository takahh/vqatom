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
    Faster conversion:
      - PyTorch-only (no NumPy)
      - Vectorized mask building
      - Dense boolean matmul for 2/3-hop (N<=100)
    """
    import torch
    import dgl
    from collections import defaultdict

    args = get_args()                       # move out of inner loops
    B = len(adj_batch)

    base_graphs = []
    extended_graphs = []

    # elem -> list[torch.BoolTensor], later cat into one mask per elem
    masks_dict = defaultdict(list)

    # --- which column carries element IDs? adjust if needed ---
    ELEMENT_COL = 0

    for i in range(B):
        # reshape once (use .reshape to be safe with non-contiguous tensors)
        adj_mats  = adj_batch[i].reshape(-1, 100, 100)   # [M, N, N]
        attr_mats = attr_batch[i].reshape(-1, 100, 27)   # [M, N, F]

        # ----- per-molecule in this set -----
        M = adj_mats.shape[0]
        for j in range(M):
            A  = adj_mats[j]                  # [N, N], (float or int)
            X  = attr_mats[j]                 # [N, F]

            # keep only non-padded rows
            nonzero_row = (X.abs().sum(dim=1) > 0)       # [N]
            if not torch.any(nonzero_row):
                continue

            Xf = X[nonzero_row]                            # [n, F]
            Af = A[nonzero_row][:, nonzero_row]            # [n, n]
            n  = Xf.shape[0]

            # --------- element masks (vectorized, compressed length n) ----------
            elem_vec = Xf[:, ELEMENT_COL].to(torch.long)   # [n]
            nz = elem_vec[elem_vec != 0]
            if nz.numel() > 0:
                uniq = torch.unique(nz)                    # [U]
                # broadcast eq: [U, n_valid], but we need only positions where elem!=0
                # Build full-length mask over n (not just nz) to keep alignment per graph
                # Make a dense per-elem mask over n (fast for n<=100)
                for e in uniq.tolist():
                    masks_dict[int(e)].append((elem_vec == e))  # [n] bool

            # --------- BASE GRAPH: 1-hop (upper triangle to avoid dup) ----------
            # one-way edges (we'll let DGL keep them directed)
            src, dst = (Af > 0).nonzero(as_tuple=True)
            keep = src > dst
            src, dst = src[keep], dst[keep]
            w1 = Af[src, dst].float()

            g1 = dgl.graph((src, dst), num_nodes=n)
            g1.ndata["feat"] = Xf
            g1.edata["weight"] = w1
            g1.edata["edge_type"] = torch.ones(g1.num_edges(), dtype=torch.int)
            g1 = dgl.add_self_loop(g1)  # optional
            base_graphs.append(g1)
            # A1: 1-hop reachability
            A1_bool = (Af > 0)  # bool [n, n]
            A1 = A1_bool.to(torch.int8)  # int for matmul

            # 2/3-hop via dense matmul (n<=100 → fast)
            A2 = (A1 @ A1) > 0  # bool
            A3 = (A2.to(torch.int8) @ A1) > 0  # bool

            # Remove self for hop>1 (we’ll add self loops later)
            A2.fill_diagonal_(False)
            A3.fill_diagonal_(False)

            # Union of edges
            U = A1_bool | A2 | A3  # bool

            es, ed = U.nonzero(as_tuple=True)

            g_ext = dgl.graph((es, ed), num_nodes=n)

            # edge weights: take from Af (1-hop weights); else set hop-specific weights
            # Build weights by rule: 1-hop: Af, 2-hop: 0.5, 3-hop: 0.3
            one_mask   = A1[es, ed]
            two_mask   = (~one_mask) & A2[es, ed]
            three_mask = (~one_mask) & (~two_mask) & A3[es, ed]

            w = torch.zeros_like(es, dtype=Af.dtype)
            if one_mask.any():
                w[one_mask] = Af[es[one_mask], ed[one_mask]]
            if two_mask.any():
                w[two_mask] = 0.5
            if three_mask.any():
                w[three_mask] = 0.3

            # edge_type by priority (1 > 2 > 3)
            et = torch.zeros_like(es, dtype=torch.int)
            et[three_mask] = 3
            et[two_mask]   = 2
            et[one_mask]   = 1

            g_ext.edata["weight"] = w.float()
            g_ext.edata["edge_type"] = et
            g_ext.ndata["feat"] = Xf
            g_ext = dgl.add_self_loop(g_ext)

            extended_graphs.append(g_ext)

    # -------- finalize masks: make one tensor per element (concat per graph) --------
    for e in list(masks_dict.keys()):
        masks_dict[e] = torch.cat(masks_dict[e], dim=0)   # 1-D bool tensor

    return base_graphs, extended_graphs, masks_dict

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
        hmask_list = []

        from collections import defaultdict
        import numpy as np
        # Initialize a dict of lists to collect masks per atom type
        all_masks_dict = defaultdict(list)

        for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, kmeans_start_num, kmeans_end_num),
                                                      start=kmeans_start_num):
            glist_base, glist, mask_dict = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
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
        print("mask_dict")
        print(mask_dict)
        # Flatten the list of lists into a single list of [h_mask, c_mask, n_mask, o_mask]
        # flattened = [masks_per_sample for batch in all_masks for masks_per_sample in batch]

        # # Initialize 4 lists
        # h_masks = []
        # c_masks = []
        # n_masks = []
        # o_masks = []
        #
        # # Fill them
        # for h, c, n, o in flattened:
        #     h_masks.append(h.cpu().numpy())
        #     c_masks.append(c.cpu().numpy())
        #     n_masks.append(n.cpu().numpy())
        #     o_masks.append(o.cpu().numpy())

        # Now convert to object arrays (since lengths of masks may vary)
        h_masks_np = np.array(hmask_list, dtype=object)
        # c_masks_np = np.array(c_masks, dtype=object)
        # n_masks_np = np.array(n_masks, dtype=object)
        # o_masks_np = np.array(o_masks, dtype=object)

        # Save to file (optional)
        np.save("all_masks_dict.npy", all_masks_dict)
        # np.save(f"c_masks_{epoch}.npy", c_masks_np)
        # np.save(f"n_masks_{epoch}.npy", n_masks_np)
        # np.save(f"o_masks_{epoch}.npy", o_masks_np)
        # np.savez(f"./naked_embed_{epoch}.npz", embed=embed.cpu().detach().numpy())
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

                glist_base, glist, masks_2 = convert_to_dgl(adj_batch, attr_batch)
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
            glist_base, glist, masks_3 = convert_to_dgl(adj_batch, attr_batch)
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


