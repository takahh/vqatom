from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from train_teacher import get_args
from collections import Counter

DATAPATH = "../data/both_mono"
DATAPATH_INFER = "../data/additional_data_for_analysis"

def train_sage(model, g, feats, optimizer, chunk_i, mask_dict, logger, epoch, chunk_size=None):
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
        #          data, features, chunk_i, mask_dict=None, logger=None, epoch=None, batched_graph_base=None, mode=None):
        outputs = model(g, feats, chunk_i, mask_dict, logger, epoch)
        (loss, cb, loss_list3) = outputs
    #
    # # Sync codebook weights
    # model.vq._codebook.embed.data.copy_(cb.to(device))
    # -----------------
    # update variables
    # -----------------
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Return only what’s needed
    return (
        loss.detach(),                     # keep as tensor if you want
        [l.item() if hasattr(l, 'item') else l for l in loss_list3]
    )

# evaluate(model, all_latents_tensor, first_batch_feat, epoch, all_masks_dict, logger, None, None, "init_kmeans_final")
def evaluate(model, g, feats, epoch, mask_dict, logger, g_base, chunk_i, mode=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if mode == "init_kmeans_loop":
            latents = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode)
            return latents
        elif mode == "init_kmeans_final":
            model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode)
            return 0
        else:  # test
            outputs = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode)
            (loss, cb, loss_list3) = outputs

    # Return only what’s needed
    return (
        loss.detach(),                     # keep as tensor if you want
        [l.item() if hasattr(l, 'item') else l for l in loss_list3]
    )

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
from collections import defaultdict
import numpy as np

# 許容パターン（表のルール）
ALLOWED_Z      = {5, 6, 7, 8, 14, 15, 16}
ALLOWED_CHARGE = {-1, 0, 1}
ALLOWED_HYB    = {2, 3, 4}
ALLOWED_BOOL   = {0, 1}   # aromatic / ring 共通

def collect_global_indices_compact(adj_batch, attr_batch,
                                   start_atom_id=0,
                                   start_mol_id=0):
    """
    戻り値:
      masks_dict: {(Z, charge, hyb, aromatic, ring): [global_idx, ...], ...}
      atom_offset: 次の開始ID
      mol_id:     処理した分子数
    """
    masks_dict = defaultdict(list)

    B = len(attr_batch)
    atom_offset = int(start_atom_id)
    mol_id = int(start_mol_id)

    for i in range(B):
        # (M, 100, 27)
        attr_mats = attr_batch[i].view(-1, 100, 27)

        # 必要列だけ CPU へ: [Z, charge, hyb, aromatic, ring]
        cols = [0, 2, 3, 4, 5]
        Aall = (
            attr_mats[..., cols]
            .detach()
            .to("cpu", non_blocking=True)
            .numpy()
            .astype(np.int32)      # 以降の isin 判定を安定化
        )
        # Z だけ別に参照（実在原子判定）
        Zall = Aall[..., 0]

        M = Aall.shape[0]
        for j in range(M):
            A = Aall[j]            # (100, 5) = [Z,chg,hyb,aro,ring]
            Z = Zall[j]            # (100,)

            # 実在原子（Z!=0）
            exist = (Z != 0)
            if not exist.any():
                mol_id += 1
                continue

            # 実在原子だけ切り出し
            rows = A[exist]        # (n_atoms, 5) int32
            Ze   = rows[:, 0]

            # ---- 元素で回す（np.uniqueはここだけに使用）----
            for elem in np.unique(Ze):
                # 元素フィルタ（まずは元素だけ許容）
                if elem not in ALLOWED_Z:
                    continue

                mask_e = (Ze == elem)
                local_idxs = np.flatnonzero(mask_e).astype(np.int64)   # 分子内 index
                rows_e = rows[mask_e]                                  # (n_e,5)

                # 各原子を個別にチェックして登録（ユニーク化しない）
                for k_local, r in zip(local_idxs, rows_e):
                    Zv, chg, hyb, aro, ring = map(int, r)

                    # ルール適合チェック
                    if (chg in ALLOWED_CHARGE and
                        hyb in ALLOWED_HYB and
                        aro in ALLOWED_BOOL and
                        ring in ALLOWED_BOOL):
                        key = f"{Zv}_{chg}_{hyb}_{aro}_{ring}"               # 5属性キー
                        global_idx = atom_offset + int(k_local)
                        masks_dict[key].append(global_idx)
                    # ルール外は鍵を作らずスキップ

            # 実在原子ぶん進める（ルール外でも offset は進める）
            atom_offset += rows.shape[0]
            mol_id += 1

    return masks_dict, atom_offset, mol_id



def convert_to_dgl(adj_batch, attr_batch, start_atom_id=0, start_mol_id=0):
    from collections import defaultdict
    masks_dict, start_atom_id, start_mol_id = collect_global_indices_compact(adj_batch, attr_batch, start_atom_id, start_mol_id)   # ✅ unpack

    base_graphs = []
    extended_graphs = []
    attr_matrices_all = []

    for i in range(len(adj_batch)):
        args = get_args()
        # both branches identical; keep one
        adj_matrices  = adj_batch[i].view(-1, 100, 100)
        attr_matrices = attr_batch[i].view(-1, 100, 27)

        for j in range(len(attr_matrices)):
            adj_matrix  = adj_matrices[j]
            attr_matrix = attr_matrices[j]

            # ---- depad ----
            nonzero_mask      = (attr_matrix.abs().sum(dim=1) > 0)
            num_total_nodes   = int(nonzero_mask.sum().item())
            filtered_attr     = attr_matrix[nonzero_mask]
            filtered_adj      = adj_matrix[:num_total_nodes, :num_total_nodes]

            # ---- base graph (make it bidirected before khop) ----
            src, dst = filtered_adj.nonzero(as_tuple=True)
            # keep both directions to ensure khop works symmetrically
            # (if filtered_adj is symmetric, this already includes both directions)
            base_g = dgl.graph((src, dst), num_nodes=num_total_nodes)
            # ensure simple & bidirected
            base_g = dgl.to_simple(dgl.to_bidirected(base_g))
            base_g.ndata["feat"] = filtered_attr
            base_g.edata["weight"] = filtered_adj[base_g.edges()[0], base_g.edges()[1]].float()
            base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), dtype=torch.int)
            base_g = dgl.add_self_loop(base_g)
            base_graphs.append(base_g)

            # ---- k-hop ----
            adj_2hop = dgl.khop_adj(base_g, 2)  # [N,N], 0/1
            adj_3hop = dgl.khop_adj(base_g, 3)

            # ---- combine ----
            full_adj = filtered_adj.clone()
            full_adj += (adj_2hop * 0.5)
            full_adj += (adj_3hop * 0.3)
            torch.diagonal(full_adj).fill_(1.0)

            # ---- extended graph should come from full_adj ----
            sf, df = full_adj.nonzero(as_tuple=True)                 # ✅ use full_adj
            extended_g = dgl.graph((sf, df), num_nodes=num_total_nodes)
            e_src, e_dst = extended_g.edges()
            extended_g.edata["weight"] = full_adj[e_src, e_dst].float()  # ✅ use full_adj

            # edge types: classify by source matrices
            one_hop   = (filtered_adj[e_src, e_dst] > 0)
            two_hop   = (adj_2hop[e_src, e_dst] > 0) & ~one_hop
            three_hop = (adj_3hop[e_src, e_dst] > 0) & ~(one_hop | two_hop)
            edge_types = torch.zeros_like(e_src, dtype=torch.int)
            edge_types[one_hop]   = 1
            edge_types[two_hop]   = 2
            edge_types[three_hop] = 3
            extended_g.edata["edge_type"] = edge_types

            extended_g.ndata["feat"] = filtered_attr
            extended_g = dgl.add_self_loop(extended_g)

            # optional sanity check on padding tail
            remaining_features = attr_matrix[num_total_nodes:]
            if remaining_features.numel() and not torch.all(remaining_features == 0):
                print("⚠️ WARNING: Non-zero values found in remaining features!")
            attr_matrices_all.append(filtered_attr)  # store per-molecule attributes
            extended_graphs.append(extended_g)

    return base_graphs, extended_graphs, masks_dict, attr_matrices_all, start_atom_id, start_mol_id  # ✅ fixed


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
        print("initial kmeans start ....")
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
        all_attr = []

        from collections import defaultdict
        import numpy as np
        # Initialize a dict of lists to collect masks per atom type
        all_masks_dict = defaultdict(list)
        first_batch_feat = None
        start_atom_id = 0
        start_mol_id = 0
        # for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, kmeans_start_num, kmeans_end_num),
        #                                               start=kmeans_start_num):
        for idx, (adj_batch, attr_batch) in enumerate(dataloader):
            print(f"idx {idx}")
            # ======== Delete this soon ==============
            if idx == 5:
                break
            # ========================================
            glist_base, glist, masks_dict, attr_matrices, start_atom_id, start_mol_id = convert_to_dgl(adj_batch, attr_batch, start_atom_id, start_mol_id)  # 10000 molecules per glist
            all_attr.append(attr_matrices)
            chunk_size = conf["chunk_size"]  # in 10,000 molecules
            # Aggregate masks into all_masks_dict
            for atom_type, masks in masks_dict.items():
                all_masks_dict[atom_type].extend(masks)

            for i in range(0, len(glist), chunk_size):
                # print(f"init kmeans idx {i}/{len(glist) - 1}")
                chunk = glist[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]   # only 1-hop
                batched_graph = dgl.batch(chunk)
                batched_graph_base = dgl.batch(chunk_base)
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]
                # model, g, feats, epoch, mask_dict, logger, g_base, chunk_i, mode=None
                latents \
                    = evaluate(model, batched_graph, batched_feats, epoch, all_masks_dict, logger, batched_graph_base, idx, "init_kmeans_loop")
                all_latents.append(latents.cpu())  # move to CPU if needed to save memory
                if i == 0 and idx == 0:
                    first_batch_feat = batched_feats.clone()

        all_latents_tensor = torch.cat(all_latents, dim=0)  # Shape: [total_atoms_across_all_batches, latent_dim]

        freq = {k: len(v) for k, v in all_masks_dict.items()}
        # 多い順に表示
        for k, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            print(k, c)

        print("------")

        # Flatten the list of lists into a single list of [h_mask, c_mask, n_mask, o_mask]
        # flattened = [masks_per_sample for batch in all_masks for masks_per_sample in batch]
        # for key, value in all_masks_dict.items():
        #     value = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in value]
        #     all_masks_dict[key] = torch.cat(value)
        # Save to file (optional)
        # np.save("all_masks_dict.npy", all_masks_dict)
        # np.savez_compressed("all_masks_dict.npz", all_masks_dict)
        print(f"all_latents_tensor.shape {all_latents_tensor.shape}")
        # --------------------------------------------------------------------------------------
        # Run k-means on the collected latent vectors (goes to the deepest) and Silhuette score
        # --------------------------------------------------------------------------------------
        evaluate(model, all_latents_tensor, first_batch_feat, epoch, all_masks_dict, logger, None, None, "init_kmeans_final")
        print("initial kmeans done....")
        model.vq._codebook.latent_size_sum = 0

        # ---------------------------
        # TRAIN
        # ---------------------------
        if conf["train_or_infer"] in ("hptune", "train"):
            # re-init codebook
            model.vq._codebook.initted.data.copy_(torch.tensor([False], device=model.vq._codebook.initted.device))
            model.latent_size_sum = 0
            print("TRAIN ---------------")

            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                # ======== Delete this soon ==============
                if idx == 1:
                    break
                # ========================================
                if idx == 5:
                    break
                print(f"idx {idx}")
                glist_base, glist, masks_2, attr_matrices_all = convert_to_dgl(adj_batch, attr_batch)
                chunk_size = conf["chunk_size"]
                for i in range(0, len(glist), chunk_size):
                    print(f"chunk {i}")
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
                    loss, loss_list_train = train_sage(
                        model, batched_graph, batched_feats, optimizer, i, masks_2, logger, epoch, chunk_size
                    )

                    # record scalar losses
                    clean_losses = [(l.detach().cpu().item() if hasattr(l, "detach") else float(l))
                                    for l in loss_list_train]
                    for j, val in enumerate(clean_losses):
                        loss_list_list_train[j].append(val)
                    loss_list.append(loss.detach().cpu().item())
                    # cleanup
                    del batched_graph, batched_feats, chunk, loss, loss_list_train
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
                test_loss, loss_list_test = evaluate(model, batched_graph, batched_feats, epoch, masks_3, logger, batched_graph_base, idx)
                # record scalar losses
                clean_losses = [(l.detach().cpu().item() if hasattr(l, "detach") else float(l))
                                for l in loss_list_test]
                for j, val in enumerate(clean_losses):
                    loss_list_list_test[j].append(val)
                test_loss_list.append(test_loss.detach().cpu().item())
                # cleanup
                del batched_graph, batched_feats, chunk, test_loss, loss_list_test
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                #
                # # rethink codes below
                # test_loss_list.append(test_loss.cpu().item())
                # # cb_unique_num_list_test.append(int(cb_num_unique) if torch.is_tensor(cb_num_unique) else cb_num_unique)
                # loss_list_list_test = [x + [y] for x, y in zip(loss_list_list_test, loss_list_test)]

                # optionally save small parts per chunk instead of keeping in lists
                # try:
                #     ind_chunk = sample_list_test[0].cpu().numpy()
                #     latent_chunk = sample_list_test[2].cpu().numpy()
                #     # save per chunk to avoid huge accumulation
                #     np.savez(f"./tmp_ind_{epoch}_{idx}_{i}.npz", ind_chunk=ind_chunk)
                #     np.savez(f"./tmp_latent_{epoch}_{idx}_{i}.npz", latent_chunk=latent_chunk)
                #     del latent_chunk
                #     # count indices
                #     ind_counts.update(ind_chunk.flatten().tolist())
                # except Exception as e:
                #     print(f"[WARN] collecting chunk failed: {e}")

            # cleanup graphs after idx
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base
            gc.collect()

            # args = get_args()
            # if args.get_umap_data:
            #     cb_new = model.vq._codebook.embed
            #     np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())

        # ---------------------------
        # stats and save
        # ---------------------------
        flat = list(ind_counts.elements())
        print(f"len(flat) = {len(flat)}, unique = {len(set(flat))}")
        # used_cb_vectors_all_epochs = model.vq._codebook.embed[0][torch.unique(torch.tensor(flat), sorted=True).long()]

        kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        os.makedirs(kw, exist_ok=True)
        # ----------------------------------------------
        # log train raw losses
        # ----------------------------------------------
        print(f"train - commit_loss: {sum(loss_list_list_train[0]) / max(1,len(loss_list_list_train[0])):.6f}, "
              f"train - lat_repel_loss: {sum(loss_list_list_train[1]) / max(1,len(loss_list_list_train[1])):.6f},"
              f"train - cb_repel_loss: {sum(loss_list_list_train[2]) / max(1,len(loss_list_list_train[2])):.6f}")
        logger.info(f"train - commit_loss: {sum(loss_list_list_train[0]) / max(1,len(loss_list_list_train[0])):.6f}, "
                    f"train - lat_repel_loss: {sum(loss_list_list_train[1]) / max(1,len(loss_list_list_train[1])):.6f},"
                    f"train - cb_repel_loss: {sum(loss_list_list_train[2]) / max(1,len(loss_list_list_train[2])):.6f}")
        print(f"train - total_loss: {sum(loss_list) / max(1,len(loss_list)):.6f}")
        logger.info(f"train - total_loss: {sum(loss_list) / max(1,len(loss_list)):.6f}")
        # ----------------------------------------------
        # log test raw losses
        # ----------------------------------------------
        print(f"test - commit_loss: {sum(loss_list_list_test[0]) / max(1,len(loss_list_list_test[0])):.6f}, "
              f"test - lat_repel_loss: {sum(loss_list_list_test[1]) / max(1,len(loss_list_list_test[1])):.6f},"
              f"test - cb_repel_loss: {sum(loss_list_list_test[2]) / max(1,len(loss_list_list_test[2])):.6f}")
        logger.info(f"test - commit_loss: {sum(loss_list_list_test[0]) / max(1,len(loss_list_list_test[0])):.6f}, "
                    f"test - lat_repel_loss: {sum(loss_list_list_test[1]) / max(1,len(loss_list_list_test[1])):.6f},"
                    f"test - cb_repel_loss: {sum(loss_list_list_test[2]) / max(1,len(loss_list_list_test[2])):.6f}")
        print(f"test - total_loss: {sum(test_loss_list) / max(1,len(test_loss_list)):.6f}")
        logger.info(f"test - total_loss: {sum(test_loss_list) / max(1,len(test_loss_list)):.6f}")

        # if conf['train_or_infer'] in ("hptune", "train"):
        #     print(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.9f}, test_loss {sum(test_loss_list)/len(test_loss_list):.9f}")
        #     np.savez(f"./{kw}/used_cb_vectors_{epoch}", used_cb_vectors_all_epochs.detach().cpu().numpy())

        model.vq._codebook.latent_size_sum = 0
        # cleanup big lists
        loss_list_list_train.clear()
        loss_list_list_test.clear()
        loss_list.clear()
        cb_unique_num_list.clear()
        cb_unique_num_list_test.clear()
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


