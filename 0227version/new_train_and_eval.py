from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from train_teacher import get_args
from collections import Counter

DATAPATH = "../data/both_mono"
DATAPATH_INFER = "../data/additional_data_for_analysis"

def train_sage(model, g, feats, optimizer, chunk_i, mask_dict, logger, epoch, chunk_size=None, attr=None):
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
        # data, features, chunk_i, mask_dict=None, logger=None, epoch=None,
        #                 batched_graph_base=None, mode=None, attr_list=None):
        outputs = model(g, feats, chunk_i, mask_dict, logger, epoch, g, "train", attr)
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
def evaluate(model, g, feats, epoch, mask_dict, logger, g_base, chunk_i, mode=None, attr_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if mode == "init_kmeans_loop":
            latents = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
            return latents
        elif mode == "init_kmeans_final":
            model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
            return 0
        else:  # test
            outputs = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
            return outputs

    # # Return only what’s needed
    # return (
    #     loss.detach(),                     # keep as tensor if you want
    #     [l.item() if hasattr(l, 'item') else l for l in loss_list3]
    # )

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

def collect_global_indices_compact(
    adj_batch,
    attr_batch,
    start_atom_id=0,
    start_mol_id=0,
    degree_cap=None,
    ring_size_cap=8,
    arom_nbrs_cap=6,
    fused_id_cap=255,
    # If your new features are already packed in attr_batch, give their column indices:
    ring_size_col=None,
    arom_nbrs_col=None,
    fused_id_col=None,
    # Or pass them separately (same batching/shape as attr_batch[...,0]):
    ring_size_batch=None,         # list[Tensor] with shape (M,100) per batch item, or a Tensor viewable to (-1,100)
    arom_nbrs_batch=None,         # ditto
    fused_ring_id_batch=None,     # ditto
    # Which fields to include in the string key and in what order:
    include_keys=("Z","charge","hyb","arom","ring","deg","ringSize","aromNbrs","fusedId"),
    debug=False,
    debug_max_print=10,
):
    """
    Returns:
      masks_dict: { 'Z_q_h_a_r_deg_ringSize_aromNbrs_fusedId' : [global_idx, ...], ... }
      atom_offset: next start atom id (global)
      mol_id:      number of processed molecules

    Notes:
      - attr columns assumed base: [Z, charge, hyb, arom, ring] at indices [0,2,3,4,5]
      - Degree computed from adjacency excluding self-loops
      - ringSize/aromNbrs/fusedId can come from attr columns or side-channel tensors
      - No CB_DICT/CBDICT filtering is applied
    """
    from collections import defaultdict
    import numpy as np
    import torch

    # Base columns in attr: [Z, charge, hyb, arom, ring]
    COL_Z, COL_CHARGE, COL_HYB, COL_AROM, COL_RING = 0, 2, 3, 4, 5
    BASE_COLS = [COL_Z, COL_CHARGE, COL_HYB, COL_AROM, COL_RING]

    def _to_cpu_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", non_blocking=True).numpy()
        return x

    def _fetch_side_features(i, M):
        """Returns (ring_size_np, arom_nbrs_np, fused_id_np) as np.int32 arrays of shape (M, 100), or None."""
        rs = an = fid = None
        if ring_size_batch is not None:
            rs = _to_cpu_np(ring_size_batch[i]).astype(np.int32).reshape(-1, 100)
        if arom_nbrs_batch is not None:
            an = _to_cpu_np(arom_nbrs_batch[i]).astype(np.int32).reshape(-1, 100)
        if fused_ring_id_batch is not None:
            fid = _to_cpu_np(fused_ring_id_batch[i]).astype(np.int32).reshape(-1, 100)
        return rs, an, fid

    masks_dict = defaultdict(list)
    atom_offset = int(start_atom_id)
    mol_id = int(start_mol_id)

    B = len(attr_batch)

    # ---------- main loop over batch items ----------
    for i in range(B):
        # Shapes: (M,100,27) / (M,100,100)
        attr_mats = attr_batch[i].view(-1, 100, 27)
        adj_mats  = adj_batch[i].view(-1, 100, 100)

        attr_np = _to_cpu_np(attr_mats)  # (M,100,27)
        adj_np  = _to_cpu_np(adj_mats)   # (M,100,100)

        M = attr_np.shape[0]

        # Base attributes (Z, charge, hyb, arom, ring)
        A_sel = attr_np[..., BASE_COLS].astype(np.int32)         # (M,100,5)

        # Valid (unpadded) node mask
        node_mask = (np.abs(attr_np).sum(axis=2) > 0)            # (M,100) bool

        # Degree from adjacency (exclude self-loops)
        deg_total = (adj_np != 0).sum(axis=2).astype(np.int32)   # (M,100)
        diag_nz   = (np.abs(np.diagonal(adj_np, axis1=1, axis2=2)) != 0).astype(np.int32)
        degrees   = deg_total - diag_nz
        if degree_cap is not None:
            degrees = np.minimum(degrees, int(degree_cap))

        # ---- ringSize / aromNbrs / fusedId ----
        # Prefer columns inside attr if specified; else fall back to side-channel tensors; else zeros
        ring_size_np = attr_np[..., ring_size_col].astype(np.int32) if ring_size_col is not None else None
        arom_nbrs_np = attr_np[..., arom_nbrs_col].astype(np.int32) if arom_nbrs_col is not None else None
        fused_id_np  = attr_np[..., fused_id_col ].astype(np.int32) if fused_id_col  is not None else None

        if (ring_size_np is None) or (arom_nbrs_np is None) or (fused_id_np is None):
            rs_side, an_side, fid_side = _fetch_side_features(i, M)
            if ring_size_np is None:
                ring_size_np = rs_side if rs_side is not None else np.zeros((M,100), np.int32)
            if arom_nbrs_np is None:
                arom_nbrs_np = an_side if an_side is not None else np.zeros((M,100), np.int32)
            if fused_id_np is None:
                fused_id_np = fid_side if fid_side is not None else np.zeros((M,100), np.int32)

        # Caps
        if ring_size_cap is not None:
            ring_size_np = np.minimum(ring_size_np, int(ring_size_cap))
        if arom_nbrs_cap is not None:
            arom_nbrs_np = np.minimum(arom_nbrs_np, int(arom_nbrs_cap))
        if fused_id_cap is not None:
            fused_id_np = np.minimum(fused_id_np, int(fused_id_cap))

        # ---- per-molecule pass (vectorized within the molecule) ----
        for m in range(M):
            nm = node_mask[m]  # (100,)
            if not nm.any():
                mol_id += 1
                continue

            # Extract base cols
            z      = A_sel[m, :, 0][nm]
            charge = A_sel[m, :, 1][nm]
            hyb    = A_sel[m, :, 2][nm]
            arom   = A_sel[m, :, 3][nm]
            ring   = A_sel[m, :, 4][nm]
            deg    = degrees[m][nm]

            # New features
            rs     = ring_size_np[m][nm]
            an     = arom_nbrs_np[m][nm]
            fid    = fused_id_np[m][nm]

            # Optional: position flag for sp2 aromatic ring carbons with degree==2 (outer rim)
            pos = (((z == 6) & (hyb == 3) & (arom == 1) & (ring == 1) & (deg == 2))).astype(np.int32)

            # Choose which fields to include and stack in that order
            fields = {
                "Z": z, "charge": charge, "hyb": hyb, "arom": arom, "ring": ring, "deg": deg,
                "ringSize": rs, "aromNbrs": an, "fusedId": fid, "pos": pos
            }
            cols_to_stack = [fields[name] for name in include_keys]
            keys = np.stack(cols_to_stack, axis=1).astype(np.int32)   # (N, K)
            if keys.ndim == 1:
                keys = keys.reshape(1, -1)
            N = int(keys.shape[0])

            # Global ids for these atoms
            global_ids = np.arange(atom_offset, atom_offset + N, dtype=np.int64)

            # Build string keys fast with np.char
            ks = keys.astype(str)
            key_strings = ks[:, 0]
            for c in range(1, ks.shape[1]):
                key_strings = np.char.add(np.char.add(key_strings, "_"), ks[:, c])

            if debug and i == 0 and m == 0:
                peek = key_strings[:min(debug_max_print, len(key_strings))].tolist()
                print("[collect][peek] first keys:", peek)

            # Group by unique key and extend once per key
            uniq_keys, inv = np.unique(key_strings, return_inverse=True)
            # Bucket append (Python list-of-lists for minimal overhead)
            buckets = [[] for _ in range(len(uniq_keys))]
            inv_list = inv.tolist()
            for row_idx, bucket_id in enumerate(inv_list):
                buckets[bucket_id].append(int(global_ids[row_idx]))
            for uk, ids in zip(uniq_keys.tolist(), buckets):
                masks_dict[uk].extend(ids)

            # advance offsets per molecule
            atom_offset += N
            mol_id += 1

    if debug:
        print(f"[collect] total buckets: {len(masks_dict)}")

    return masks_dict, atom_offset, mol_id




def convert_to_dgl(adj_batch, attr_batch, start_atom_id=0, start_mol_id=0):
    from collections import defaultdict
    masks_dict, start_atom_id, start_mol_id = collect_global_indices_compact(adj_batch, attr_batch, start_atom_id, start_mol_id)   # ✅ unpack
    print("masks_dict.keys()")
    print(masks_dict.keys())
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
        # if conf["train_or_infer"] == "infer" or conf["train_or_infer"] == "hptune":
        #     kmeans_start_num = 6
        #     # kmeans_end_num = 18
        #     kmeans_end_num = 7
        # if conf["train_or_infer"] == "analysis":
        #     kmeans_start_num = 0
        #     kmeans_end_num = 1
        # ------------------------------------------
        # Collect latent vectors (goes to model.py)
        # ------------------------------------------
        all_latents = []
        all_attr = []

        from collections import defaultdict
        import numpy as np
        # Initialize a dict of lists to collect masks per atom type
        all_masks_dict = defaultdict(list)
        masks_count = defaultdict(int)
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
            # print(f"len(attr_matrices) {len(attr_matrices)}")
            chunk_size = conf["chunk_size"]  # in 10,000 molecules
            # Aggregate masks into all_masks_dict
            for atom_type, masks in masks_dict.items():
                all_masks_dict[atom_type].extend(masks)
            # collect counts
            for atom_type, masks in masks_dict.items():
                masks_count[atom_type] += len(masks)
            for i in range(0, len(glist), chunk_size):
                # print(f"init kmeans idx {i}/{len(glist) - 1}")
                chunk = glist[i:i + chunk_size]
                attr_chunk = attr_matrices[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]   # only 1-hop
                batched_graph = dgl.batch(chunk)
                batched_graph_base = dgl.batch(chunk_base)
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]
                # model, g, feats, epoch, mask_dict, logger, g_base, chunk_i, mode=None
                latents \
                    = evaluate(model, batched_graph, batched_feats, epoch, all_masks_dict, logger, batched_graph_base, idx, "init_kmeans_loop", attr_chunk)
                all_latents.append(latents.cpu())  # move to CPU if needed to save memory
                if i == 0 and idx == 0:
                    first_batch_feat = batched_feats.clone()

        all_latents_tensor = torch.cat(all_latents, dim=0)  # Shape: [total_atoms_across_all_batches, latent_dim]

        freq = {k: v for k, v in masks_count.items()}
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
        # print(f"all_latents_tensor.shape {all_latents_tensor.shape}")
        print(f"init_kmeans_final start ")
        # --------------------------------------------------------------------------------------
        # Run k-means on the collected latent vectors (goes to the deepest) and Silhuette score
        # --------------------------------------------------------------------------------------

        # first reshape all_attr to [atom count, 27]
        flat_list = [t for batch in all_attr for t in batch]
        # Now concat along the atom dimension
        all_attr_tensor = torch.cat(flat_list, dim=0)  # (N, 27)
        # print("all_attr_tensor shape:", all_attr_tensor.shape)

        evaluate(model, all_latents_tensor, first_batch_feat, epoch, all_masks_dict, logger, None, None, "init_kmeans_final", all_attr_tensor)
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
                # # ======== Delete this soon ==============
                # if idx == 1:
                #     break
                # # ========================================
                if idx == 5:
                    break
                print(f"idx {idx}")
                # base_graphs, extended_graphs, masks_dict, attr_matrices_all, start_atom_id, start_mol_id  # ✅ fixed
                glist_base, glist, masks_2, attr_matrices_all, _, _ = convert_to_dgl(adj_batch, attr_batch)
                chunk_size = conf["chunk_size"]
                for i in range(0, len(glist), chunk_size):
                    print(f"chunk {i}")
                    # # ------------- remove thi soon --------------
                    # if i == chunk_size:
                    #     break
                    # # ------------- remove thi soon --------------
                    chunk = glist[i:i + chunk_size]
                    batched_graph = dgl.batch(chunk)
                    attr_chunk = attr_matrices_all[i:i + chunk_size]
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]

                    # train step
                    # (model, g, feats, optimizer, chunk_i, logger, epoch):
                    loss, loss_list_train = train_sage(
                        # model, g, feats, optimizer, chunk_i, mask_dict, logger, epoch, chunk_size=None, attr=None
                        model, batched_graph, batched_feats, optimizer, i, masks_2, logger, epoch, chunk_size, attr_chunk
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
            glist_base, glist, masks_3, attr_matrices_all_test, _, _ = convert_to_dgl(adj_batch, attr_batch)
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
                attr_chunk_test = attr_matrices_all_test[i:i + chunk_size]
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]
                #  loss, embed, [commit_loss.item(), repel_loss.item(), cb_repel_loss.item()]
                test_loss, test_emb, loss_list_test = evaluate(model, batched_graph, batched_feats, epoch, masks_3, logger, batched_graph_base, idx, "test", attr_chunk_test)
                # record scalar losses
                clean_losses = [(l.detach().cpu().item() if hasattr(l, "detach") else float(l))
                                for l in loss_list_test]
                for j, val in enumerate(clean_losses):
                    loss_list_list_test[j].append(val)
                test_loss_list.append(test_loss.detach().cpu().item())
                # cleanup
                del batched_graph, batched_feats, chunk, test_loss, loss_list_test, attr_chunk_test
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


