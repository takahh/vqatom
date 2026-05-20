from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from args import get_args

from collections import Counter

DATAPATH = "../data/vqamino_train"
DATAPATH_INFER = "../data/vqamino_infer"

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
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if mode == "init_kmeans_loop":
            # returns latents
            latents = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
            return latents

        if mode == "init_kmeans_final":
            # just run once; side effects (init/dump) happen inside
            model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
            return 0

        if mode == "infer":
            out = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)

            # model returns: (loss=None, ids=(kid,cid,gid,id2safe), extra=None)
            if not (isinstance(out, (tuple, list)) and len(out) >= 2):
                raise TypeError(f"[evaluate] infer expects model to return tuple>=2, got {type(out)}")

            ids = out[1]

            # ids contract:
            # (key_id_full, cluster_id_full, global_id_full, id2safe)
            if isinstance(ids, (tuple, list)) and len(ids) == 4:
                key_id_full, cluster_id_full, global_id_full, id2safe = ids
            elif isinstance(ids, (tuple, list)) and len(ids) == 3:
                key_id_full, cluster_id_full, id2safe = ids
                global_id_full = key_id_full  # fallback
            else:
                raise TypeError(
                    f"[evaluate] infer expects ids 4-tuple (kid,cid,gid,id2safe) (or legacy 3-tuple), "
                    f"got ids type={type(ids)} len={len(ids) if isinstance(ids, (tuple, list)) else 'n/a'}"
                )

            key_id_full = torch.as_tensor(key_id_full).reshape(-1).long()
            cluster_id_full = torch.as_tensor(cluster_id_full).reshape(-1).long()
            global_id_full = torch.as_tensor(global_id_full).reshape(-1).long()
            if not isinstance(id2safe, dict):
                id2safe = {}

            return None, (key_id_full, cluster_id_full, global_id_full, id2safe), None


class ProteinGraphDataset(Dataset):
    """Dataset for VQ-Amino sequence/residue graphs.

    Expected files:
      adj_*.npy  : [B, L, L] or [L, L] adjacency/contact/chain-neighbor matrix
      attr_*.npy : [B, L, F] or [L, F] residue features

    Column 0 of attr is assumed to be an amino-acid id by default
    (0=PAD/unknown, 1..20 canonical residues, or your own mapping).
    """
    def __init__(self, adj_dir, attr_dir):
        self.adj_files = sorted(glob.glob(f"{adj_dir}/adj_*.npy"))
        self.attr_files = sorted(glob.glob(f"{attr_dir}/attr_*.npy"))
        assert len(self.adj_files) == len(self.attr_files), "Mismatch in adjacency and attribute files"

    def __len__(self):
        return len(self.adj_files)

    def __getitem__(self, idx):
        adj_matrix = torch.tensor(np.load(self.adj_files[idx]), dtype=torch.float32)
        attr_matrix = torch.tensor(np.load(self.attr_files[idx]), dtype=torch.float32)
        return adj_matrix, attr_matrix

# Backward-compatible alias for old training code paths.
ProteinGraphDataset = ProteinGraphDataset


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

# -----------------------------
# VQ-Amino grouping utilities
# -----------------------------
# For VQ-Amino, the grouping key should be intrinsic and cheap.
# Default: group residues by amino-acid id only (attr column 0).
# Contextual/environmental information should remain in node features and be
# absorbed by the GNN before quantization.

def _get_vqamino_group_cols():
    """Read optional grouping columns from args.vqamino_group_cols.

    Example: --vqamino_group_cols 0       -> AA only
             --vqamino_group_cols 0,1     -> AA + coarse state
    """
    try:
        args = get_args()
        raw = getattr(args, "vqamino_group_cols", "0")
    except Exception:
        raw = "0"
    if raw is None or str(raw).strip() == "":
        return (0,)
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip() != "")

def collect_global_indices_compact(
    adj_batch,
    attr_batch,
    logger,
    start_residue_id=0,
    start_protein_id=0,
    debug=True,
    debug_max_print=1000,
):
    """Build mask_dict for per-amino-acid/residue codebooks.

    VQ-Atom used chemical keys such as Z/charge/hybridization.
    VQ-Amino uses amino-acid/residue keys. By default, the key is attr[:, 0].
    Any additional context should be encoded in attr features and consumed by
    the GNN, not used for grouping unless explicitly requested.
    """
    from collections import defaultdict
    import numpy as np
    import torch

    def _to_cpu_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    group_cols = _get_vqamino_group_cols()
    masks_dict = defaultdict(list)
    residue_offset = int(start_residue_id)
    protein_id = int(start_protein_id)

    for i in range(len(attr_batch)):
        attr = attr_batch[i]
        adj = adj_batch[i]

        # Support [L,F], [B,L,F], or packed shapes.
        if attr.ndim == 2:
            attr_mats = attr.unsqueeze(0)
        elif attr.ndim == 3:
            attr_mats = attr
        else:
            D = int(attr.shape[-1])
            L = int(adj.shape[-1])
            attr_mats = attr.view(-1, L, D)

        attr_np = _to_cpu_np(attr_mats)

        # Node existence: any nonzero feature row. This treats AA id 0 + all-zero
        # features as padding. If 0 is a real residue id in your data, add a mask
        # feature or shift residues to 1..20.
        node_mask = (np.abs(attr_np).sum(axis=2) > 0)

        for m in range(attr_np.shape[0]):
            nm = node_mask[m]
            if not nm.any():
                protein_id += 1
                continue

            cols = []
            for c in group_cols:
                if c >= attr_np.shape[-1]:
                    raise ValueError(f"vqamino_group_cols contains column {c}, but attr dim is {attr_np.shape[-1]}")
                cols.append(attr_np[m, :, c][nm].astype(np.int32))

            keys = np.stack(cols, axis=1).astype(np.int32)
            if keys.ndim == 1:
                keys = keys.reshape(1, -1)
            N = keys.shape[0]

            global_ids = np.arange(residue_offset, residue_offset + N, dtype=np.int64)

            ks = keys.astype(str)
            key_strings = ks[:, 0]
            for c in range(1, ks.shape[1]):
                key_strings = np.char.add(np.char.add(key_strings, "_"), ks[:, c])

            uniq_keys, inv = np.unique(key_strings, return_inverse=True)
            buckets = [[] for _ in range(len(uniq_keys))]
            for row_idx, bucket_id in enumerate(inv.tolist()):
                buckets[bucket_id].append(int(global_ids[row_idx]))

            for uk, ids in zip(uniq_keys.tolist(), buckets):
                masks_dict[uk].extend(ids)

            residue_offset += N
            protein_id += 1

    return masks_dict, residue_offset, protein_id

def convert_to_dgl(
    adj_batch,
    attr_batch,
    logger=None,
    start_residue_id=0,
    start_protein_id=0,
    device=None,
    two_hop_w=0.5,
    three_hop_w=0.3,
    make_base_bidirected=True,
    add_self_loops=True,
):
    """Convert padded residue graphs into DGL graphs for VQ-Amino.

    No hardcoded molecule size or feature size is used. Each item can be
    [L,L]/[L,F] or [B,L,L]/[B,L,F].
    """
    import torch
    import dgl

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    masks_dict, start_residue_id, start_protein_id = collect_global_indices_compact(
        adj_batch, attr_batch, logger, start_residue_id, start_protein_id
    )

    base_graphs = []
    extended_graphs = []
    attr_matrices_all = []

    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        attr = attr_batch[i]

        if adj.ndim == 2:
            adj_matrices = adj.unsqueeze(0)
        else:
            adj_matrices = adj.view(-1, adj.shape[-2], adj.shape[-1])

        if attr.ndim == 2:
            attr_matrices = attr.unsqueeze(0)
        else:
            attr_matrices = attr.view(-1, attr.shape[-2], attr.shape[-1])

        adj_matrices = adj_matrices.to(device, non_blocking=True)
        attr_matrices = attr_matrices.to(device, non_blocking=True)

        for j in range(attr_matrices.shape[0]):
            adj_matrix = adj_matrices[j]
            attr_matrix = attr_matrices[j]

            nonzero_mask = (attr_matrix.abs().sum(dim=1) > 0)
            N = int(nonzero_mask.sum().item())
            if N <= 0:
                continue

            X = attr_matrix[nonzero_mask]
            W1 = adj_matrix[:N, :N].float()
            A1 = (W1 > 0)

            if make_base_bidirected:
                A1 = A1 | A1.T
            if add_self_loops:
                A1.fill_diagonal_(True)

            bsrc, bdst = A1.nonzero(as_tuple=True)
            base_g = dgl.graph((bsrc, bdst), num_nodes=N, device=device)
            base_g.ndata["feat"] = X
            W1_base = W1.clone()
            if add_self_loops:
                W1_base.fill_diagonal_(1.0)
            es, ed = base_g.edges()
            base_g.edata["weight"] = W1_base[es, ed]
            base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), device=device, dtype=torch.long)
            base_graphs.append(base_g)

            A1f = A1.to(torch.float32)
            A2 = (A1f @ A1f) > 0
            A3 = (A2.to(torch.float32) @ A1f) > 0
            one_hop = A1
            two_hop_only = A2 & ~one_hop
            three_hop_only = A3 & ~(one_hop | A2)

            full_W = W1_base.clone()
            full_W += two_hop_only.to(full_W.dtype) * float(two_hop_w)
            full_W += three_hop_only.to(full_W.dtype) * float(three_hop_w)
            if add_self_loops:
                full_W.fill_diagonal_(1.0)

            Afull = full_W > 0
            if make_base_bidirected:
                Afull = Afull | Afull.T
            if add_self_loops:
                Afull.fill_diagonal_(True)

            sf, df = Afull.nonzero(as_tuple=True)
            extended_g = dgl.graph((sf, df), num_nodes=N, device=device)
            extended_g.ndata["feat"] = X
            e_src, e_dst = extended_g.edges()
            extended_g.edata["weight"] = full_W[e_src, e_dst]

            et = torch.zeros(e_src.numel(), device=device, dtype=torch.long)
            et[one_hop[e_src, e_dst]] = 1
            et[two_hop_only[e_src, e_dst]] = 2
            et[three_hop_only[e_src, e_dst]] = 3
            extended_g.edata["edge_type"] = et

            extended_graphs.append(extended_g)
            attr_matrices_all.append(X)

    return base_graphs, extended_graphs, masks_dict, attr_matrices_all, start_residue_id, start_protein_id


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

def run_infer_after_restore(
    conf,
    model,
    logger,
    checkpoint_path: str,
    *,
    epoch: int | None = None,
    only_save: bool = False,
):
    """
    Restore checkpoint -> infer only -> save per-residue outputs.

    Saves
    -----
    - only_save=False: full dump
        key_id, cluster_id, global_id, id2safe
        + provenance (graph_id_in_chunk, local_node_id, batch_id, chunk_start_in_batch)
    - only_save=True: minimal dump
        global_id, id2safe + meta + global_stats

    Requires
    --------
    evaluate(..., mode="infer") returns:
        (_, (kid, cid, global_id, id2safe), _)

    Assumes in outer scope
    ----------------------
    DATAPATH, DATAPATH_INFER
    ProteinGraphDataset, collate_fn, convert_to_dgl, evaluate
    """

    import os
    import re
    import gc
    import time
    import torch
    import dgl
    from torch.utils.data import DataLoader

    def _log(msg: str):
        if logger is not None:
            try:
                logger.info(msg)
                return
            except Exception:
                pass
        print(msg)

    # ----------------------------
    # helper: pre-create dynamic codebooks so strict=True load can work
    # ----------------------------
    def _ensure_codebook_keys_if_possible(model, state_dict):
        if not (hasattr(model, "vq") and hasattr(model.vq, "_codebook")):
            return
        cb = model.vq._codebook
        if not hasattr(cb, "_get_or_create_safe_key"):
            return

        prefixes = ["vq._codebook.", ""]  # model-level vs cb-only sd

        for prefix in prefixes:
            ea_pat = re.compile(rf"^{re.escape(prefix)}embed_avg_(.+)$")
            cs_pat = re.compile(rf"^{re.escape(prefix)}cluster_size_(.+)$")
            emb_pat = re.compile(rf"^{re.escape(prefix)}embed\.(.+)$")

            created_any = False

            # (1) embed_avg_(orig) => (K,D)
            for k, v in state_dict.items():
                m = ea_pat.match(k)
                if not m or (not torch.is_tensor(v)) or v.ndim != 2:
                    continue
                orig = m.group(1)
                K, D = int(v.shape[0]), int(v.shape[1])

                # buffers (best-effort; your impl may register with safe or orig)
                if not hasattr(cb, f"cluster_size_{orig}"):
                    cb.register_buffer(f"cluster_size_{orig}", torch.zeros((K,), dtype=torch.float32))
                if not hasattr(cb, f"embed_avg_{orig}"):
                    cb.register_buffer(f"embed_avg_{orig}", torch.zeros((K, D), dtype=torch.float32))

                # create embed ParameterDict entry (safe)
                cb._get_or_create_safe_key(orig, K_e=K, D=D, device="cpu")
                created_any = True

            # (2) cluster_size_(orig) only => create size buffer
            for k, v in state_dict.items():
                m = cs_pat.match(k)
                if not m or (not torch.is_tensor(v)) or v.ndim != 1:
                    continue
                orig = m.group(1)
                K = int(v.shape[0])
                if not hasattr(cb, f"cluster_size_{orig}"):
                    cb.register_buffer(f"cluster_size_{orig}", torch.zeros((K,), dtype=torch.float32))
                    created_any = True

            # (3) embed.<safe> directly => ensure ParameterDict has entry
            for k, v in state_dict.items():
                m = emb_pat.match(k)
                if not m or (not torch.is_tensor(v)) or v.ndim != 2:
                    continue
                safe = m.group(1)
                if safe not in cb.embed:
                    K, D = int(v.shape[0]), int(v.shape[1])
                    cb.embed[safe] = torch.nn.Parameter(torch.randn((K, D)) * 0.01, requires_grad=True)
                    created_any = True

            if created_any:
                break

    # ----------------------------
    # 1) RESTORE
    # ----------------------------
    device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    _ensure_codebook_keys_if_possible(model, state)
    strict = bool(conf.get("restore_strict", True))
    # drop non-essential tracking buffers (older ckpt compatibility)
    DROP_PREFIXES = (
        "vq._codebook.usage_ema_k_",
        "vq._codebook.split_cd_k_",
        "vq._codebook.ever_used_k_",
        "vq._codebook.last_used_ep_k_",
    )

    if isinstance(state, dict):
        n0 = len(state)
        state = {k: v for k, v in state.items() if not any(k.startswith(p) for p in DROP_PREFIXES)}
        n1 = len(state)
        if n1 != n0:
            print(f"[restore] dropped {n0 - n1} tracking keys from ckpt for compatibility")

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        print("\n[RESTORE ERROR]")
        print(e)  # ここで Missing/Unexpected が全部出るはず
        # さらに state の先頭キーも出す
        if isinstance(state, dict):
            ks = list(state.keys())
            print("\n[state keys head]", ks[:50])
        raise

    strict = bool(conf.get("restore_strict", True))
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()

    # IMPORTANT: reset latent_size_sum (chunk consistency)
    if hasattr(model, "vq") and hasattr(model.vq, "_codebook"):
        cb = model.vq._codebook
        if hasattr(cb, "latent_size_sum"):
            cb.latent_size_sum = 0

    # ----------------------------
    # 2) DATA
    # ----------------------------
    datapath = DATAPATH if conf.get("train_or_infer") in ("hptune",) else DATAPATH_INFER
    dataset = ProteinGraphDataset(adj_dir=datapath, attr_dir=datapath)
    dataloader = DataLoader(
        dataset,
        batch_size=int(conf.get("infer_batch_size", 16)),
        shuffle=False,
        collate_fn=collate_fn,
    )
    chunk_size = int(conf["chunk_size"])

    # ----------------------------
    # 3) OUTPUT BUFFERS (CPU lists)
    # ----------------------------
    all_key_ids = []
    all_cluster_ids = []
    all_global_ids = []

    all_graph_ids = []
    all_local_node_ids = []
    all_batch_ids = []
    all_chunk_starts = []

    last_id2safe = None

    # ----------------------------
    # 4) INFER (chunked)
    # ----------------------------
    with torch.no_grad():
        for batch_idx, (adj_batch, attr_batch) in enumerate(dataloader):
            glist_base, glist, masks_3, attr_matrices_all, _, _ = convert_to_dgl(
                adj_batch, attr_batch, logger, 0, 0
            )

            for start in range(0, len(glist), chunk_size):
                chunk = glist[start:start + chunk_size]

                batched_graph = dgl.batch(chunk).to(device)
                feats = batched_graph.ndata["feat"]
                attr_list = attr_matrices_all[start:start + chunk_size]

                _, ids, _ = evaluate(
                    model=model,
                    g=batched_graph,
                    feats=feats,
                    epoch=0 if epoch is None else int(epoch),
                    mask_dict=masks_3,
                    logger=logger,
                    g_base=None,
                    chunk_i=start,
                    mode="infer",
                    attr_list=attr_list,
                )

                # EXPECT: kid, cid, gid, id2safe
                kid, cid, gid, id2safe = ids
                last_id2safe = id2safe

                kid = kid.reshape(-1).to(dtype=torch.long)
                cid = cid.reshape(-1).to(dtype=torch.long)
                gid = gid.reshape(-1).to(dtype=torch.long)

                n_nodes = int(batched_graph.num_nodes())
                if kid.numel() != n_nodes or cid.numel() != n_nodes or gid.numel() != n_nodes:
                    raise RuntimeError(
                        f"size mismatch: nodes={n_nodes} kid={kid.numel()} cid={cid.numel()} gid={gid.numel()}"
                    )

                if not only_save:
                    # provenance
                    num_nodes_per_g = [int(g.num_nodes()) for g in chunk]
                    num_nodes_per_g_t = torch.tensor(num_nodes_per_g, device=device, dtype=torch.long)

                    graph_ids = torch.repeat_interleave(
                        torch.arange(len(chunk), device=device, dtype=torch.long),
                        num_nodes_per_g_t
                    )
                    local_node_ids = torch.cat(
                        [torch.arange(n, device=device, dtype=torch.long) for n in num_nodes_per_g],
                        dim=0
                    )

                    batch_ids = torch.full((n_nodes,), int(batch_idx), device=device, dtype=torch.long)
                    chunk_starts = torch.full((n_nodes,), int(start), device=device, dtype=torch.long)

                    all_key_ids.append(kid.detach().cpu())
                    all_cluster_ids.append(cid.detach().cpu())

                    all_graph_ids.append(graph_ids.detach().cpu())
                    all_local_node_ids.append(local_node_ids.detach().cpu())
                    all_batch_ids.append(batch_ids.detach().cpu())
                    all_chunk_starts.append(chunk_starts.detach().cpu())

                all_global_ids.append(gid.detach().cpu())

                # cleanup
                del batched_graph, feats, attr_list, ids, kid, cid, gid
                if not only_save:
                    del graph_ids, local_node_ids, batch_ids, chunk_starts, num_nodes_per_g_t
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # cleanup per batch
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            for g in glist_base:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base, masks_3, attr_matrices_all
            gc.collect()

    if len(all_global_ids) == 0:
        return None

    # ----------------------------
    # 5) CONCAT
    # ----------------------------
    global_id_all = torch.cat(all_global_ids, dim=0)

    if not only_save:
        key_id_all = torch.cat(all_key_ids, dim=0)
        cluster_id_all = torch.cat(all_cluster_ids, dim=0)

        graph_id_all = torch.cat(all_graph_ids, dim=0)
        local_node_id_all = torch.cat(all_local_node_ids, dim=0)
        batch_id_all = torch.cat(all_batch_ids, dim=0)
        chunk_start_all = torch.cat(all_chunk_starts, dim=0)

        n = int(global_id_all.numel())
        if not (
            key_id_all.numel() == cluster_id_all.numel() == graph_id_all.numel() ==
            local_node_id_all.numel() == batch_id_all.numel() == chunk_start_all.numel() == n
        ):
            raise RuntimeError("final pack sanity check failed (numel mismatch)")

    # ----------------------------
    # 6) GLOBAL STATS
    # ----------------------------
    gmin = int(global_id_all.min().item())
    gmax = int(global_id_all.max().item())
    gunique = int(global_id_all.unique().numel())
    gtotal = int(global_id_all.numel())

    _log("--------------------------------------------------")
    _log("[GLOBAL TOKEN STATS]")
    _log(f"min: {gmin}")
    _log(f"max: {gmax}")
    _log(f"unique_count: {gunique}")
    _log(f"total_tokens: {gtotal}")

    gv = None
    if hasattr(model, "vq") and hasattr(model.vq, "_codebook"):
        cb = model.vq._codebook
        if hasattr(cb, "_global_vocab_size"):
            try:
                gv = int(getattr(cb, "_global_vocab_size"))
            except Exception:
                gv = None
    if gv is not None:
        _log(f"global_vocab_size: {gv}")
    _log("--------------------------------------------------")

    # ----------------------------
    # 7) SAVE
    # ----------------------------
    base = conf.get("infer_save_path") or "infer_token_ids.pt"
    root, ext = os.path.splitext(base)
    tag = "_min" if only_save else ""
    if epoch is not None:
        save_path = f"{root}{tag}_ep{int(epoch):04d}{ext}"
    else:
        save_path = f"{root}{tag}{ext}"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    meta = {
        "checkpoint_path": checkpoint_path,
        "chunk_size": int(chunk_size),
        "epoch": None if epoch is None else int(epoch),
        "saved_at": time.strftime("%Y%m%d_%H%M%S"),
        "global_stats": {"min": gmin, "max": gmax, "unique": gunique, "total": gtotal, "global_vocab_size": gv},
    }

    if only_save:
        out = {
            "global_id": global_id_all,
            "id2safe": last_id2safe or {},
            **meta,
        }
    else:
        out = {
            "key_id": key_id_all,
            "cluster_id": cluster_id_all,
            "global_id": global_id_all,
            "id2safe": last_id2safe or {},

            "graph_id_in_chunk": graph_id_all,
            "local_node_id": local_node_id_all,
            "batch_id": batch_id_all,
            "chunk_start_in_batch": chunk_start_all,
            **meta,
        }

    torch.save(out, save_path)
    _log(f"saved: {save_path}")

    return out

def run_inductive(conf, model, optimizer, scheduler, logger):
    import gc, itertools, torch, os, copy
    from collections import Counter, defaultdict
    import numpy as np
    def build_global_id_meta_from_model(model):
        cb = model.vq._codebook
        safe_keys = sorted(cb.embed.keys())

        if not hasattr(cb, "_safe2id"):
            cb._safe2id = {}
        if not hasattr(cb, "_id2safe"):
            cb._id2safe = {}

        next_sid = (max(cb._id2safe.keys()) + 1) if cb._id2safe else 0
        for safe in safe_keys:
            if safe not in cb._safe2id:
                sid = next_sid
                next_sid += 1
                cb._safe2id[safe] = sid
                cb._id2safe[sid] = safe

        offsets = {}
        cur = 0
        for safe in safe_keys:
            K = int(cb.embed[safe].shape[0])
            offsets[safe] = cur
            cur += K

        meta = {
            "safe_keys": safe_keys,
            "global_offsets": offsets,
            "global_vocab_size": cur,
            "pad_id": cur,
            "mask_id": cur + 1,
            "vocab_size": cur + 2,
            "id2safe": dict(cb._id2safe),
        }
        return meta

    def freeze_except_vq(model):
        # freeze all
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze only vq
        for p in model.vq.parameters():
            p.requires_grad = True

    def unfreeze_all(model):
        for p in model.parameters():
            p.requires_grad = True

    def unfreeze_encoder(model):
        for p in model.encoder.parameters():
            p.requires_grad = True

    # freeze_except_vq(model)

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
    dataset = ProteinGraphDataset(adj_dir=datapath, attr_dir=datapath)
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
        # init_kmeans_loop : collect latents
        # ------------------------------------------
        all_latents = []
        all_attr = []

        all_masks_dict = defaultdict(list)
        masks_count = defaultdict(int)
        first_batch_feat = None
        start_residue_id = 0
        start_protein_id = 0

        # if (epoch - 1) % 3 == 0:
        print("initial kmeans start ....")
        logger.info(f"=== epoch {epoch} ==　initial kmeans start ....")
        for idx, (adj_batch, attr_batch) in enumerate(dataloader):

            if idx == 12:
                break

            glist_base, glist, masks_dict, attr_matrices, start_residue_id, start_protein_id = convert_to_dgl(
                adj_batch, attr_batch, logger, start_residue_id, start_protein_id
            )  # protein graphs/residue sequences per glist
            all_attr.append(attr_matrices)

            # masks を集約
            for residue_type, masks in masks_dict.items():
                all_masks_dict[residue_type].extend(masks)
                masks_count[residue_type] += len(masks)

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

            # glist をクリーンアップ
            for g in glist:
                g.ndata.clear()
                g.edata.clear()
            for g in glist_base:
                g.ndata.clear()
                g.edata.clear()
            del glist, glist_base
            gc.collect()

        # all_latents: [ (#residues_chunk, D), ... ] -> (N, D)
        all_latents_tensor = torch.cat(all_latents, dim=0)  # [N, D]

        # attr を flatten: list[list[tensor (N_i,27)]] -> (N,27)
        flat_attr_list = [t for batch in all_attr for t in batch]
        all_attr_tensor = torch.cat(flat_attr_list, dim=0)  # (N, 27)

        print("[KMEANS] all latents collected")
        # print(f"[KMEANS] latents shape: {all_latents_tensor.shape}, attr shape: {all_attr_tensor.shape}")
        print("[KMEANS] init_kmeans_final start")
        logger.info("[KMEANS] init_kmeans_final start")

        # ---------------------------------------------
        # init_kmeans_final : run kmeans and calc. SS
        # ---------------------------------------------
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
                if idx == 12:
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
        # if conf["train_or_infer"] != "analysis":
        #     os.makedirs(".", exist_ok=True)
        #     state = copy.deepcopy(model.state_dict())
        #     torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        #     model.load_state_dict(state)
        # ---------------------------
        if conf["train_or_infer"] != "analysis":
            os.makedirs(".", exist_ok=True)

            global_id_meta = build_global_id_meta_from_model(model)

            ckpt = {
                "model": model.state_dict(),
                "epoch": int(epoch),

                # global vocab info (truth)
                "global_id_meta": global_id_meta,
                "base_vocab": int(global_id_meta["global_vocab_size"]),
                "vocab_size": int(global_id_meta["vocab_size"]),
                "pad_id": int(global_id_meta["pad_id"]),
                "mask_id": int(global_id_meta["mask_id"]),

                "config": conf,
            }
            torch.save(ckpt, f"model_epoch_{epoch}.pt")

        # # ---------------------------
        # # 4) TEST
        # # ---------------------------
        # print("TEST ---------------")
        # logger.info("TEST ---------------")
        # ind_counts = Counter()
        #
        # if conf['train_or_infer'] == "hptune":
        #     start_num, end_num = 5, 6
        # elif conf['train_or_infer'] == "analysis":
        #     dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        #     start_num, end_num = 0, 1
        # else:  # infer
        #     start_num, end_num = 6, 10
        # for idx, (adj_batch, attr_batch) in enumerate(
        #         itertools.islice(dataloader, start_num, end_num),
        #         start=start_num
        # ):
        #     print(f"[TEST] batch idx {idx}")
        #     glist_base, glist, masks_3, attr_matrices_all_test, _, _ = convert_to_dgl(
        #         adj_batch, attr_batch, logger
        #     )
        #     chunk_size = conf["chunk_size"]
        #
        #     # ★ ここ：バッチ内のチャンク番号を 0,1,2,... で作る
        #     for chunk_i_local, i in enumerate(range(0, len(glist), chunk_size)):
        #         chunk = glist[i:i + chunk_size]
        #         chunk_base = glist_base[i:i + chunk_size]
        #
        #         batched_graph = dgl.batch(chunk).to(device)
        #         batched_graph_base = dgl.batch(chunk_base).to(device)
        #         attr_chunk_test = attr_matrices_all_test[i:i + chunk_size]
        #
        #         # feats取得に no_grad は不要（evaluate 側が no_grad なので）
        #         batched_feats = batched_graph.ndata["feat"]
        #
        #         # return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        #         test_loss, test_emb, loss_list_test = evaluate(
        #             model,
        #             batched_graph,
        #             batched_feats,
        #             epoch,
        #             masks_3,
        #             logger,
        #             batched_graph_base,
        #             chunk_i_local,  # ★変更点：idx ではなく 0,1,2,... を渡す
        #             "test",
        #             attr_chunk_test,
        #         )
        #
        #         # record scalar losses
        #         clean_losses = [to_scalar(l) for l in loss_list_test]
        #         for j, val in enumerate(clean_losses):
        #             loss_list_list_test[j].append(val)
        #         test_loss_list.append(to_scalar(test_loss))
        #
        #         # cleanup
        #         del batched_graph, batched_graph_base, batched_feats, chunk, chunk_base
        #         del test_loss, test_emb, loss_list_test, attr_chunk_test
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #
        #     # cleanup graphs after idx
        #     for g in glist:
        #         g.ndata.clear()
        #         g.edata.clear()
        #     for g in glist_base:
        #         g.ndata.clear()
        #         g.edata.clear()
        #     del glist, glist_base, masks_3, attr_matrices_all_test
        #     gc.collect()

        # ---------------------------
        # 5) stats and save
        # ---------------------------

        # epoch end: save infer ids
        # def run_infer_only_after_restore(
        #         conf,
        #         model,
        #         logger,
        #         checkpoint_path: str,
        #         *,
        #         epoch: int | None = None,
        #         only_save: bool = False,
        # ):
        kw = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        os.makedirs(kw, exist_ok=True)
        # commit_loss, codebook_loss, repel_loss, cb_repel_loss, ent_loss
        train_commit = safe_mean(loss_list_list_train[0])
        train_latrep = safe_mean(loss_list_list_train[2])
        train_cbrep = safe_mean(loss_list_list_train[3])
        train_ent = safe_mean(loss_list_list_train[4])
        train_total = safe_mean(loss_list)

        print(f"train - commit_loss: {train_commit:.6f}, "
              f"train - lat_repel_loss: {train_latrep:.6f}, "
              f"train - cb_repel_loss: {train_cbrep:.6f}",
              f"train - ent_loss: {train_ent:.6f}")
        logger.info(
            "train - commit_loss: %.9f, lat_repel_loss: %.6f, cb_repel_loss: %.6f, ent_loss: %.6f",
            train_commit, train_latrep, train_cbrep, train_ent
        )
        print(f"train - total_loss: {train_total:.6f}")
        logger.info("train - total_loss: %.6f", train_total)

        # test logs   loss_list_list_test = [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        test_commit = safe_mean(loss_list_list_test[0])
        test_latrep = safe_mean(loss_list_list_test[2])
        test_cbrep = safe_mean(loss_list_list_test[3])
        test_ent = safe_mean(loss_list_list_test[4])
        test_total = safe_mean(test_loss_list)

        # print(f"test - commit_loss: {test_commit:.6f}, "
        #       f"test - lat_repel_loss: {test_latrep:.6f}, "
        #       f"test - cb_repel_loss: {test_cbrep:.6f}",
        #       f"test - ent_loss: {test_ent:.6f}")
        # logger.info(f"test - commit_loss: {test_commit:.6f}, "
        #             f"test - lat_repel_loss: {test_latrep:.6f}, "
        #             f"test - cb_repel_loss: {test_cbrep:.6f}",
        #             f"test - ent_loss: {test_ent:.6f}")
        # print(f"test - total_loss: {test_total:.6f}")
        # logger.info(f"test - total_loss: {test_total:.6f}")

        model.vq._codebook.latent_size_sum = 0

        # cleanup big lists
        loss_list_list_train.clear()
        loss_list_list_test.clear()
        loss_list.clear()
        cb_unique_num_list.clear()
        cb_unique_num_list_test.clear()
        gc.collect()
        torch.cuda.empty_cache()

        scheduler.step()

        # （optional）LR ログ
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[LR] epoch={epoch} lr={lr:.6e}")

    # 何か score を返したい場合はここで
    # return test_total


