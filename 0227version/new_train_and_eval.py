from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from train_teacher import get_args
from collections import Counter

DATAPATH = "../data/both_mono"
DATAPATH_INFER = "../data/additional_data_for_analysis"

def train_sage(model, g, feats, optimizer, chunk_i, mask_dict, logger, epoch,
               chunk_size=None, attr=None):
    import torch
    from contextlib import nullcontext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    g = g.to(device)
    if "feat" in g.ndata:
        g.ndata["feat"] = g.ndata["feat"].to(device)
    feats = feats.to(device)

    if device.type == "cuda":
        ctx = torch.amp.autocast("cuda")
        scaler = torch.amp.GradScaler("cuda", init_scale=1e2)
    else:
        ctx = nullcontext()
        scaler = None

    optimizer.zero_grad(set_to_none=True)

    with ctx:
        outputs = model(g, feats, chunk_i, mask_dict, logger, epoch, g, "train", attr)
        loss, cb, loss_list3 = outputs

    # ---- sanity checks on loss ----
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError(
            f"[train_sage] model returned non-Tensor loss: {type(loss)}. "
            "Do not use .item() / float() for the training loss."
        )

    if not loss.requires_grad:
        raise RuntimeError(
            f"[train_sage] loss.requires_grad=False (shape={loss.shape}, "
            f"device={loss.device}). "
            "Likely .detach() or .item() used on loss inside the model."
        )

    # ---- backward & step ----
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # ---- convert outputs for logging ----
    loss_scalar = float(loss.detach().cpu())

    loss_list_out = []
    for l in loss_list3:  # loss_list3 = [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        # handle both Tensor and float / numpy
        if isinstance(l, torch.Tensor):
            loss_list_out.append(float(l.detach().cpu()))
        else:
            loss_list_out.append(float(l))
    return loss_scalar, loss_list_out

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
            #  return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
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
    logger,
    start_atom_id=0,
    start_mol_id=0,
    debug=True,
    debug_max_print=1000,
):
    """
    Runtime version:
      - Keys are intrinsic only: (Z, charge, hyb, arom, ring, hnum)
      - Environmental context (deg, ringSize, aromNbrs, fusedId, pos, func, het27)
        is NOT part of the key anymore.
      - Assumes env features are already present in attr_batch and will be used
        by the GNN as node features, not for grouping.
    """
    from collections import defaultdict
    import numpy as np
    import torch

    # columns in attr: must match your data layout
    COL_Z, COL_DEG, COL_CHARGE, COL_HYB, COL_AROM, COL_RING, COL_HNUM = 0, 1, 2, 3, 4, 5, 6

    def _to_cpu_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    masks_dict = defaultdict(list)
    atom_offset = int(start_atom_id)
    mol_id = int(start_mol_id)

    B = len(attr_batch)

    for i in range(B):
        D = attr_batch[i].shape[-1]
        attr_mats = attr_batch[i].view(-1, 100, D)
        adj_mats  = adj_batch[i].view(-1, 100, 100)

        attr_np = _to_cpu_np(attr_mats)
        adj_np  = _to_cpu_np(adj_mats)

        M = attr_np.shape[0]

        # just select intrinsic cols once
        A_sel = attr_np[..., [COL_Z, COL_CHARGE, COL_HYB, COL_AROM, COL_RING, COL_HNUM]].astype(np.int32)

        # node existence mask
        node_mask = (np.abs(attr_np).sum(axis=2) > 0)

        for m in range(M):
            nm = node_mask[m]
            if not nm.any():
                mol_id += 1
                continue

            z      = A_sel[m, :, 0][nm]
            charge = A_sel[m, :, 1][nm]
            hyb    = A_sel[m, :, 2][nm]
            arom   = A_sel[m, :, 3][nm]
            ring   = A_sel[m, :, 4][nm]
            hnum   = A_sel[m, :, 5][nm]

            # intrinsic fields only
            fields = {
                "Z": z,
                "charge": charge,
                "hyb": hyb,
                "arom": arom,
                "ring": ring,
                "hnum": hnum,
            }

            include_keys = ("Z", "charge", "hyb", "arom", "ring", "hnum")
            cols_to_stack = [fields[name] for name in include_keys]
            keys = np.stack(cols_to_stack, axis=1).astype(np.int32)
            if keys.ndim == 1:
                keys = keys.reshape(1, -1)
            N = keys.shape[0]

            global_ids = np.arange(atom_offset, atom_offset + N, dtype=np.int64)

            # build key strings but now they are short
            ks = keys.astype(str)
            key_strings = ks[:, 0]
            for c in range(1, ks.shape[1]):
                key_strings = np.char.add(np.char.add(key_strings, "_"), ks[:, c])

            uniq_keys, inv = np.unique(key_strings, return_inverse=True)
            buckets = [[] for _ in range(len(uniq_keys))]
            inv_list = inv.tolist()
            for row_idx, bucket_id in enumerate(inv_list):
                buckets[bucket_id].append(int(global_ids[row_idx]))

            for uk, ids in zip(uniq_keys.tolist(), buckets):
                masks_dict[uk].extend(ids)

            atom_offset += N
            mol_id += 1

    return masks_dict, atom_offset, mol_id

def convert_to_dgl(adj_batch, attr_batch, logger=None, start_atom_id=0, start_mol_id=0):
    from collections import defaultdict
    masks_dict, start_atom_id, start_mol_id = collect_global_indices_compact(adj_batch, attr_batch, logger, start_atom_id, start_mol_id)   # ✅ unpack
    # print("masks_dict.keys()")
    # print(masks_dict.keys())
    base_graphs = []
    extended_graphs = []
    attr_matrices_all = []

    for i in range(len(adj_batch)):
        args = get_args()
        # both branches identical; keep one
        adj_matrices  = adj_batch[i].view(-1, 100, 100)
        attr_matrices = attr_batch[i].view(-1, 100, 79)

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
            # if remaining_features.numel() and not torch.all(remaining_features == 0):
            #     # print("⚠️ WARNING: Non-zero values found in remaining features!")
            #     nz = remaining_features[remaining_features != 0]
            # else:
            #     print("OK ===========")
            #     # print(f"num_total_nodes {num_total_nodes}")
            #     # print("Non-zero values:", nz[:20])
            #     # print("Indices:", torch.nonzero(remaining_features)[:20])

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
    import gc, itertools, torch, os, copy
    from collections import Counter, defaultdict
    import numpy as np

    # ----------------------------
    # helpers
    # ----------------------------
    def safe_mean(xs):
        return float(sum(xs) / max(1, len(xs)))

    def to_scalar(x):
        """Logging 用に loss を安全に float に変換"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return float(x)

    # ----------------------------
    # dataset and dataloader
    # ----------------------------
    datapath = DATAPATH if conf['train_or_infer'] in ("hptune", "infer", "use_nonredun_cb_infer") else DATAPATH_INFER
    dataset = MoleculeGraphDataset(adj_dir=datapath, attr_dir=datapath)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ここでの clip はほぼ意味がないので残すにしても一度だけ
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    device = next(model.parameters()).device

    for epoch in range(1, conf["max_epoch"] + 1):
        print(f"\n====== epoch {epoch} ======")
        # fresh containers per epoch
        loss_list_list_train = [[] for _ in range(11)]
        loss_list_list_test = [[] for _ in range(11)]
        loss_list = []
        test_loss_list = []

        cb_unique_num_list = []
        cb_unique_num_list_test = []

        # ------------------------------------------
        # 1) K-means 用 latent / attr 収集
        # ------------------------------------------
        all_latents = []
        all_attr = []

        all_masks_dict = defaultdict(list)
        masks_count = defaultdict(int)
        first_batch_feat = None
        start_atom_id = 0
        start_mol_id = 0

        # if (epoch - 1) % 3 == 0:
        print("initial kmeans start ....")
        logger.info("=== epoch {epoch} ==　initial kmeans start ....")
        for idx, (adj_batch, attr_batch) in enumerate(dataloader):

            if idx == 5:
                break

            glist_base, glist, masks_dict, attr_matrices, start_atom_id, start_mol_id = convert_to_dgl(
                adj_batch, attr_batch, logger, start_atom_id, start_mol_id
            )  # 10000 molecules per glist
            print("convert_to_dgl is done")
            all_attr.append(attr_matrices)

            # masks を集約
            for atom_type, masks in masks_dict.items():
                all_masks_dict[atom_type].extend(masks)
                masks_count[atom_type] += len(masks)

            chunk_size = conf["chunk_size"]  # in 10,000 molecules
            for i in range(0, len(glist), chunk_size):
                chunk = glist[i:i + chunk_size]
                attr_chunk = attr_matrices[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]   # only 1-hop

                batched_graph = dgl.batch(chunk)
                batched_graph_base = dgl.batch(chunk_base)

                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"].to(device)

                # model, g, feats, epoch, mask_dict, logger, g_base, chunk_i, mode=None
                latents = evaluate(
                    model,
                    batched_graph,
                    batched_feats,
                    epoch,
                    all_masks_dict,
                    logger,
                    batched_graph_base,
                    idx,
                    "init_kmeans_loop",
                    attr_chunk,
                )
                all_latents.append(latents.cpu())  # save on CPU

                if i == 0 and idx == 0:
                    first_batch_feat = batched_feats.clone().cpu()

                # cleanup small stuff
                del batched_graph, batched_graph_base, batched_feats, chunk, chunk_base
                gc.collect()
                torch.cuda.empty_cache()

            print("latent output is done")
            # glist をクリーンアップ
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            for g in glist_base:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base
            gc.collect()

        # all_latents: [ (#atoms_chunk, D), ... ] -> (N, D)
        all_latents_tensor = torch.cat(all_latents, dim=0)  # [N, D]

        # attr を flatten: list[list[tensor (N_i,27)]] -> (N,27)
        flat_attr_list = [t for batch in all_attr for t in batch]
        all_attr_tensor = torch.cat(flat_attr_list, dim=0)  # (N, 27)

        print("[KMEANS] all latents collected")
        # print(f"[KMEANS] latents shape: {all_latents_tensor.shape}, attr shape: {all_attr_tensor.shape}")
        print("[KMEANS] init_kmeans_final start")
        logger.info("[KMEANS] init_kmeans_final start")

        # if epoch == 1:
        # first_batch_feat は CPU に戻してあるので GPU へ
        first_batch_feat_dev = first_batch_feat.to(device) if first_batch_feat is not None else None
        evaluate(
            model,
            all_latents_tensor.to(device),
            first_batch_feat_dev,
            epoch,
            all_masks_dict,
            logger,
            None,
            None,
            "init_kmeans_final",
            all_attr_tensor.to(device),
        )
        print("[KMEANS] initial kmeans done.")
        logger.info("[KMEANS] initial kmeans done.")
        model.vq._codebook.latent_size_sum = 0

        # ---------------------------
        # 2) TRAIN
        # ---------------------------
        if conf["train_or_infer"] in ("hptune", "train"):
            # re-init codebook
            model.vq._codebook.initted.data.copy_(
                torch.tensor([False], device=model.vq._codebook.initted.device)
            )
            model.latent_size_sum = 0
            print("TRAIN ---------------")
            logger.info("TRAIN ---------------")
            # dataloader は再利用可能（新しい iterator が作られる）
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 5:
                    break
                # print(f"[TRAIN] batch idx {idx}")
                glist_base, glist, masks_2, attr_matrices_all, _, _ = convert_to_dgl(
                    adj_batch, attr_batch, logger
                )
                chunk_size = conf["chunk_size"]

                for i in range(0, len(glist), chunk_size):
                    # print(f"[TRAIN]   chunk {i}")
                    chunk = glist[i:i + chunk_size]
                    batched_graph = dgl.batch(chunk).to(device)
                    attr_chunk = attr_matrices_all[i:i + chunk_size]

                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]

                    # ここで train_sage は「学習込み（backward + optimizer.step）」までやる前提
                    loss, loss_list_train = train_sage(
                        model,
                        batched_graph,
                        batched_feats,
                        optimizer,
                        i,
                        masks_2,
                        logger,
                        epoch,
                        chunk_size,
                        attr_chunk,
                    )

                    # ---- loss 安全チェック & ログ用整形 ----
                    if loss is None:
                        # print("[WARN][TRAIN] train_sage returned loss=None, skipping logging for this chunk")
                        del batched_graph, batched_feats, chunk, attr_chunk
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    # if isinstance(loss, torch.Tensor):
                    #     print(
                    #         "[DEBUG][TRAIN] loss:",
                    #         type(loss),
                    #         getattr(loss, "shape", None),
                    #         getattr(loss, "requires_grad", None),
                    #         getattr(loss, "device", None),
                    #     )
                    # else:
                    #     print("[DEBUG][TRAIN] loss is not a Tensor:", type(loss))

                    # record scalar losses
                    clean_losses = [to_scalar(l) for l in loss_list_train]
                    for j, val in enumerate(clean_losses):
                        loss_list_list_train[j].append(val)
                    loss_list.append(to_scalar(loss))

                    # cleanup
                    del batched_graph, batched_feats, chunk, attr_chunk, loss, loss_list_train
                    gc.collect()
                    torch.cuda.empty_cache()

                # cleanup per batch
                for g in glist:
                    g.ndata.clear()
                    g.edata.clear()
                for g in glist_base:
                    g.ndata.clear()
                    g.edata.clear()
                del glist, glist_base, masks_2, attr_matrices_all
                gc.collect()

        # ---------------------------
        # 3) SAVE MODEL
        # ---------------------------
        if conf["train_or_infer"] != "analysis":
            os.makedirs(".", exist_ok=True)
            state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            model.load_state_dict(state)

        # ---------------------------
        # 4) TEST
        # ---------------------------
        print("TEST ---------------")
        logger.info("TEST ---------------")
        ind_counts = Counter()

        if conf['train_or_infer'] == "hptune":
            start_num, end_num = 5, 6
        elif conf['train_or_infer'] == "analysis":
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
            start_num, end_num = 0, 1
        else:  # infer
            start_num, end_num = 6, 10
        for idx, (adj_batch, attr_batch) in enumerate(
                itertools.islice(dataloader, start_num, end_num),
                start=start_num
        ):
            print(f"[TEST] batch idx {idx}")
            glist_base, glist, masks_3, attr_matrices_all_test, _, _ = convert_to_dgl(
                adj_batch, attr_batch, logger
            )
            chunk_size = conf["chunk_size"]

            # ★ ここ：バッチ内のチャンク番号を 0,1,2,... で作る
            for chunk_i_local, i in enumerate(range(0, len(glist), chunk_size)):
                chunk = glist[i:i + chunk_size]
                chunk_base = glist_base[i:i + chunk_size]

                batched_graph = dgl.batch(chunk).to(device)
                batched_graph_base = dgl.batch(chunk_base).to(device)
                attr_chunk_test = attr_matrices_all_test[i:i + chunk_size]

                # feats取得に no_grad は不要（evaluate 側が no_grad なので）
                batched_feats = batched_graph.ndata["feat"]

                # return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
                test_loss, test_emb, loss_list_test = evaluate(
                    model,
                    batched_graph,
                    batched_feats,
                    epoch,
                    masks_3,
                    logger,
                    batched_graph_base,
                    chunk_i_local,  # ★変更点：idx ではなく 0,1,2,... を渡す
                    "test",
                    attr_chunk_test,
                )

                # record scalar losses
                clean_losses = [to_scalar(l) for l in loss_list_test]
                for j, val in enumerate(clean_losses):
                    loss_list_list_test[j].append(val)
                test_loss_list.append(to_scalar(test_loss))

                # cleanup
                del batched_graph, batched_graph_base, batched_feats, chunk, chunk_base
                del test_loss, test_emb, loss_list_test, attr_chunk_test
                gc.collect()
                torch.cuda.empty_cache()

            # cleanup graphs after idx
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            for g in glist_base:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base, masks_3, attr_matrices_all_test
            gc.collect()

        # ---------------------------
        # 5) stats and save
        # ---------------------------
        kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        os.makedirs(kw, exist_ok=True)

        # train logs   # loss_list_list_train = [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        train_commit = safe_mean(loss_list_list_train[0])
        train_latrep = safe_mean(loss_list_list_train[2])
        train_cbrep = safe_mean(loss_list_list_train[1])
        train_total = safe_mean(loss_list)

        print(f"train - commit_loss: {train_commit:.6f}, "
              f"train - lat_repel_loss: {train_latrep:.6f}, "
              f"train - cb_repel_loss: {train_cbrep:.6f}")
        logger.info(f"train - commit_loss: {train_commit:.6f}, "
                    f"train - lat_repel_loss: {train_latrep:.6f}, "
                    f"train - cb_repel_loss: {train_cbrep:.6f}")
        print(f"train - total_loss: {train_total:.6f}")
        logger.info(f"train - total_loss: {train_total:.6f}")

        # test logs   loss_list_list_test = [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        test_commit = safe_mean(loss_list_list_test[0])
        test_latrep = safe_mean(loss_list_list_test[2])
        test_cbrep = safe_mean(loss_list_list_test[1])
        test_total = safe_mean(test_loss_list)

        print(f"test - commit_loss: {test_commit:.6f}, "
              f"test - lat_repel_loss: {test_latrep:.6f}, "
              f"test - cb_repel_loss: {test_cbrep:.6f}")
        logger.info(f"test - commit_loss: {test_commit:.6f}, "
                    f"test - lat_repel_loss: {test_latrep:.6f}, "
                    f"test - cb_repel_loss: {test_cbrep:.6f}")
        print(f"test - total_loss: {test_total:.6f}")
        logger.info(f"test - total_loss: {test_total:.6f}")

        model.vq._codebook.latent_size_sum = 0

        # cleanup big lists
        loss_list_list_train.clear()
        loss_list_list_test.clear()
        loss_list.clear()
        cb_unique_num_list.clear()
        cb_unique_num_list_test.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # 何か score を返したい場合はここで
    return test_total


