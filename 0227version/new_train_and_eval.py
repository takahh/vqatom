from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading
from train_teacher import get_args
from collections import Counter

DATAPATH = "../data/discret_final"
DATAPATH_INFER = "../data/pretrain_final"

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
            # IMPORTANT: model forward should return (key_id_full, cluster_id_full, id2safe)
            out = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)

            if not (isinstance(out, (tuple, list)) and len(out) == 3):
                raise TypeError(
                    f"[evaluate] mode='infer' expects 3-tuple (key_id_full, cluster_id_full, id2safe), got: {type(out)}"
                )

            key_id_full, cluster_id_full, id2safe = out
            # --- HARDEN: force ids to be 1D ---
            if torch.is_tensor(key_id_full):
                key_id_full = key_id_full.reshape(-1)
            else:
                key_id_full = torch.as_tensor(key_id_full).reshape(-1)

            if torch.is_tensor(cluster_id_full):
                cluster_id_full = cluster_id_full.reshape(-1)
            else:
                cluster_id_full = torch.as_tensor(cluster_id_full).reshape(-1)

            # (optional but recommended)
            key_id_full = key_id_full.long()
            cluster_id_full = cluster_id_full.long()

            # minimal sanity checks
            if not isinstance(key_id_full, torch.Tensor) or key_id_full.ndim != 1:
                raise TypeError(f"[evaluate] key_id_full must be 1D Tensor, got {type(key_id_full)} shape={getattr(key_id_full,'shape',None)}")
            if not isinstance(cluster_id_full, torch.Tensor) or cluster_id_full.ndim != 1:
                raise TypeError(f"[evaluate] cluster_id_full must be 1D Tensor, got {type(cluster_id_full)} shape={getattr(cluster_id_full,'shape',None)}")
            if key_id_full.shape[0] != cluster_id_full.shape[0]:
                raise ValueError(f"[evaluate] key_id_full and cluster_id_full length mismatch: {key_id_full.shape} vs {cluster_id_full.shape}")

            if torch.is_tensor(key_id_full):
                key_id_full = key_id_full.reshape(-1).long()
            else:
                key_id_full = torch.as_tensor(key_id_full).reshape(-1).long()

            if torch.is_tensor(cluster_id_full):
                cluster_id_full = cluster_id_full.reshape(-1).long()
            else:
                cluster_id_full = torch.as_tensor(cluster_id_full).reshape(-1).long()

            # Return a consistent "outputs" object for caller:
            # (loss, embed, loss_list) style is not meaningful here,
            # so we return (None, (key_id_full, cluster_id_full, id2safe), None)
            return None, (key_id_full, cluster_id_full, id2safe), None

        # default (train/test/eval)
        # return loss, embed, loss_list
        # --- guarantee 1D ids (scalar -> [1], empty -> [0]) ---
        outputs = model(g, feats, chunk_i, mask_dict, logger, epoch, g_base, mode, attr_list)
        return outputs

class MoleculeGraphDataset(Dataset):
    def __init__(self, adj_dir, attr_dir):
        self.adj_files = sorted(glob.glob(f"{adj_dir}/adj_*.npy"))
        self.attr_files = sorted(glob.glob(f"{attr_dir}/attr_*.npy"))
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

def convert_to_dgl(
    adj_batch,
    attr_batch,
    logger=None,
    start_atom_id=0,
    start_mol_id=0,
    device=None,
    two_hop_w=0.5,
    three_hop_w=0.3,
    make_base_bidirected=True,
    add_self_loops=True,
):
    """
    Faster GPU conversion:
      - No dgl.to_simple / dgl.to_bidirected
      - No torch.unique hashing
      - Dense boolean k-hop via matmul (N<=100 assumed)
      - No dgl.add_self_loop; we include diagonal edges directly (optional)
    """
    import torch
    import dgl

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # masks likely CPU-centric; keep as-is
    masks_dict, start_atom_id, start_mol_id = collect_global_indices_compact(
        adj_batch, attr_batch, logger, start_atom_id, start_mol_id
    )

    base_graphs = []
    extended_graphs = []
    attr_matrices_all = []

    B = len(adj_batch)
    for i in range(B):
        # move whole packed tensor once
        adj_matrices  = adj_batch[i].view(-1, 100, 100).to(device, non_blocking=True)
        attr_matrices = attr_batch[i].view(-1, 100, 79).to(device, non_blocking=True)

        M = attr_matrices.shape[0]
        for j in range(M):
            adj_matrix  = adj_matrices[j]      # [100,100]
            attr_matrix = attr_matrices[j]     # [100,79]

            # depad
            nonzero_mask = (attr_matrix.abs().sum(dim=1) > 0)
            N = int(nonzero_mask.sum().item())
            if N <= 0:
                continue

            X = attr_matrix[nonzero_mask]                    # [N,79]
            W1 = adj_matrix[:N, :N].float()                  # [N,N] weights for 1-hop
            A1 = (W1 > 0)                                    # [N,N] bool adjacency

            if make_base_bidirected:
                A1 = A1 | A1.T

            if add_self_loops:
                # include self edges directly
                A1.fill_diagonal_(True)

            # -------------------------
            # base graph (1-hop only)
            # -------------------------
            bsrc, bdst = A1.nonzero(as_tuple=True)           # unique by construction
            base_g = dgl.graph((bsrc, bdst), num_nodes=N, device=device)
            base_g.ndata["feat"] = X

            # base weights: use W1, but ensure diagonal weight=1 if self-loops on
            if add_self_loops:
                # cheap: just overwrite diag in a temp view
                W1 = W1.clone()
                W1.fill_diagonal_(1.0)

            es, ed = base_g.edges()
            base_g.edata["weight"] = W1[es, ed]
            base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), device=device, dtype=torch.int32)
            base_graphs.append(base_g)

            # -------------------------
            # k-hop (dense boolean matmul)
            # For N<=100 this is typically faster than sparse setup.
            # -------------------------
            # Use float matmul, threshold to bool.
            A1f = A1.to(torch.float32)                        # [N,N]
            A2 = (A1f @ A1f) > 0                              # 2-hop reachability (includes 1-hop paths too)
            A3 = (A2.to(torch.float32) @ A1f) > 0             # 3-hop reachability

            # If you want "exactly-2-hop" and "exactly-3-hop" classification:
            one_hop = A1
            two_hop_only = A2 & ~one_hop
            three_hop_only = A3 & ~(one_hop | A2)

            # -------------------------
            # extended adjacency weights
            # -------------------------
            full_W = W1.clone()
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

            # edge types:
            # 1 = one-hop
            # 2 = two-hop-only
            # 3 = three-hop-only
            # 0 = other (mostly self-loops)
            et = torch.zeros(e_src.numel(), device=device, dtype=torch.int32)
            et[one_hop[e_src, e_dst]] = 1
            et[two_hop_only[e_src, e_dst]] = 2
            et[three_hop_only[e_src, e_dst]] = 3
            extended_g.edata["edge_type"] = et

            extended_graphs.append(extended_g)
            attr_matrices_all.append(X)

    return base_graphs, extended_graphs, masks_dict, attr_matrices_all, start_atom_id, start_mol_id


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

def run_infer_only_after_restore(conf, model, logger, checkpoint_path):
    """
    Restore checkpoint -> infer only -> save per-atom (key_id, cluster_id) + provenance.

    Notes
    -----
    - This function assumes:
        - DATAPATH, MoleculeGraphDataset, collate_fn, convert_to_dgl, evaluate are defined elsewhere.
        - model expects evaluate(..., mode="infer") to return (_, (kid, cid, id2safe), _)
    - If you have dynamic VQ codebooks: strict=True restore may fail unless the model has
      already registered all codebook keys found in the checkpoint.
    """
    import os
    import re
    import gc
    import time
    import torch
    import dgl
    from torch.utils.data import DataLoader

    # ------------------------------------------------------------
    # 0) helpers
    # ------------------------------------------------------------
    def _to_bool(x):
        try:
            if hasattr(x, "numel") and x.numel() == 1:
                return bool(x.item())
            return bool(x)
        except Exception:
            return x

    def _pick_one_center_key(state_dict):
        for k in state_dict.keys():
            if "vq._codebook.embed.k_" in k:
                return k
        return None

    def _extract_cb_keys_from_state(state_dict):
        """
        Return list of codebook key strings from checkpoint state_dict.
        ex) "vq._codebook.embed.k_6__1_3_0_0_0" -> "6__1_3_0_0_0"
        """
        pat = re.compile(r"^vq\._codebook\.embed\.k_(.+)$")
        keys = set()
        for sk in state_dict.keys():
            m = pat.match(sk)
            if m:
                keys.add(m.group(1))
        return sorted(keys)

    def _ensure_codebook_keys_if_possible(model, state_dict):
        """
        Make strict=True restore work for your EuclideanCodebook which uses:
          - buffers: cluster_size_{orig}, embed_avg_{orig}   (orig key 그대로)
          - params : vq._codebook.embed.{safe}              (safe key in ParameterDict)
        """
        import re
        import torch

        if not (hasattr(model, "vq") and hasattr(model.vq, "_codebook")):
            return

        cb = model.vq._codebook

        # EuclideanCodebook 以外なら何もしない
        if not hasattr(cb, "_get_or_create_safe_key"):
            return

        # state_dict が "model 全体" か "codebook 部分" かで prefix が違う
        # ここは両対応：どっちの形式でも拾えるようにする
        prefixes = ["vq._codebook.", ""]  # try both

        for prefix in prefixes:
            ea_pat = re.compile(rf"^{re.escape(prefix)}embed_avg_(.+)$")
            cs_pat = re.compile(rf"^{re.escape(prefix)}cluster_size_(.+)$")
            emb_pat = re.compile(rf"^{re.escape(prefix)}embed\.(.+)$")

            # 1) embed_avg_ から (orig, K, D) を拾って buffers + embed を用意
            for k, v in state_dict.items():
                m = ea_pat.match(k)
                if not m or not torch.is_tensor(v) or v.ndim != 2:
                    continue
                orig = m.group(1)  # 例: "6_-1_3_0_0_0"
                K, D = int(v.shape[0]), int(v.shape[1])

                # buffers: float32固定（あなたの実装に合わせる）
                buf_cs = f"cluster_size_{orig}"
                buf_ea = f"embed_avg_{orig}"

                if not hasattr(cb, buf_cs):
                    cb.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))
                if not hasattr(cb, buf_ea):
                    cb.register_buffer(buf_ea, torch.zeros((K, D), dtype=torch.float32))

                # embed param: safe key (ParameterDict)
                # device はとりあえず cpu でOK。load_state_dict で上書きされる
                cb._get_or_create_safe_key(orig, K_e=K, D=D, device="cpu")

            # 2) cluster_size_ だけあるケース（embed_avg が無いなら D 不明なので buffers だけ）
            for k, v in state_dict.items():
                m = cs_pat.match(k)
                if not m or not torch.is_tensor(v) or v.ndim != 1:
                    continue
                orig = m.group(1)
                K = int(v.shape[0])
                buf_cs = f"cluster_size_{orig}"
                if not hasattr(cb, buf_cs):
                    cb.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))

            # 3) embed.<safe> を直接拾って ParameterDict に無ければ作る（保険）
            #    safe->orig が分からない場合もあるので、形だけ合わせて作る
            for k, v in state_dict.items():
                m = emb_pat.match(k)
                if not m or not torch.is_tensor(v) or v.ndim != 2:
                    continue
                safe = m.group(1)
                if safe not in cb.embed:
                    K, D = int(v.shape[0]), int(v.shape[1])
                    cb.embed[safe] = torch.nn.Parameter(torch.randn((K, D)) * 0.01, requires_grad=True)

            # この prefix で何か作れたなら終了（両方やると二重登録になり得る）
            # “何か作れたか” は embed_avg を一個でも拾えたかで判定
            if any(ea_pat.match(k) for k in state_dict.keys()):
                break

    # ------------------------------------------------------------
    # 1) RESTORE
    # ------------------------------------------------------------
    device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # (optional) pre-create dynamic codebooks so strict=True works
    _ensure_codebook_keys_if_possible(model, state)

    # Strict load by default; fallback to strict=False if requested
    strict = bool(conf.get("restore_strict", True))
    if strict:
        model.load_state_dict(state, strict=True)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("restore(strict=False) missing:", missing[:20])
        print("restore(strict=False) unexpected:", unexpected[:20])

    model.to(device)
    model.eval()

    # optional: print codebook init status + reset latent_size_sum (if exists)
    if hasattr(model, "vq") and hasattr(model.vq, "_codebook"):
        cb = model.vq._codebook
        if hasattr(cb, "initted"):
            print("codebook initted =", _to_bool(cb.initted))
        if hasattr(cb, "latent_size_sum"):
            cb.latent_size_sum = 0

    # probe key: check centers don't change during infer (debug)
    probe_key = _pick_one_center_key(model.state_dict())
    print("probe key:", probe_key)

    # ------------------------------------------------------------
    # 2) dataset and dataloader
    # ----------------------------
    # 1) dataset and dataloader
    # ----------------------------
    datapath = DATAPATH if conf.get("train_or_infer") in ("hptune") else DATAPATH_INFER
    dataset = MoleculeGraphDataset(adj_dir=datapath, attr_dir=datapath)
    dataloader = DataLoader(
        dataset,
        batch_size=int(conf.get("infer_batch_size", 16)),
        shuffle=False,
        collate_fn=collate_fn,
    )

    chunk_size = int(conf["chunk_size"])

    # outputs (CPU tensors list)
    all_key_ids = []
    all_cluster_ids = []
    all_graph_ids = []
    all_local_node_ids = []
    all_batch_ids = []
    all_chunk_starts = []
    last_id2safe = None

    # ------------------------------------------------------------
    # 3) INFER only
    # ------------------------------------------------------------
    with torch.no_grad():
        for batch_idx, (adj_batch, attr_batch) in enumerate(dataloader):

            glist_base, glist, masks_3, attr_matrices_all, _, _ = convert_to_dgl(
                adj_batch, attr_batch, logger, 0, 0
            )

            # iterate in chunks inside this batch's glist
            for chunk_i_local, start in enumerate(range(0, len(glist), chunk_size)):
                chunk = glist[start:start + chunk_size]

                batched_graph = dgl.batch(chunk).to(device)
                batched_feats = batched_graph.ndata["feat"]

                if probe_key is not None:
                    before = model.state_dict()[probe_key].float().norm().item()
                    print("center norm before:", before)

                # infer returns: (_, (kid, cid, id2safe), _)
                _, ids, _ = evaluate(
                    model=model,
                    g=batched_graph,
                    feats=batched_feats,
                    epoch=int(conf.get("epoch_for_logging", 0)),
                    mask_dict=masks_3,
                    logger=logger,
                    g_base=None,
                    chunk_i=chunk_i_local,
                    mode="infer",
                    attr_list=attr_matrices_all[start:start + chunk_size],
                )

                kid, cid, id2safe = ids
                last_id2safe = id2safe  # may grow over time

                # enforce 1D long
                kid = kid.reshape(-1).long()
                cid = cid.reshape(-1).long()

                # debug
                print("kid:", kid.shape, kid.dtype, int(kid.min().item()), int(kid.max().item()))
                print("cid:", cid.shape, cid.dtype, int(cid.min().item()), int(cid.max().item()))
                print("id2safe size:", len(id2safe))
                print("nodes:", batched_graph.num_nodes(),
                      "kid:", kid.numel(),
                      "cid:", cid.numel(),
                      "graphs_in_chunk:", len(chunk))

                if probe_key is not None:
                    after = model.state_dict()[probe_key].float().norm().item()
                    print("center norm after :", after)

                # ----------------------------------------------------
                # 3.5) provenance mapping within this chunk
                # ----------------------------------------------------
                num_nodes_per_g = [g.num_nodes() for g in chunk]
                num_nodes_per_g_t = torch.tensor(num_nodes_per_g, device=device, dtype=torch.long)

                graph_ids = torch.repeat_interleave(
                    torch.arange(len(chunk), device=device, dtype=torch.long),
                    num_nodes_per_g_t
                )

                local_node_ids = torch.cat(
                    [torch.arange(n, device=device, dtype=torch.long) for n in num_nodes_per_g],
                    dim=0
                )

                batch_ids = torch.full((kid.numel(),), int(batch_idx), device=device, dtype=torch.long)
                chunk_starts = torch.full((kid.numel(),), int(start), device=device, dtype=torch.long)

                # sanity
                if graph_ids.numel() != kid.numel():
                    raise RuntimeError(f"graph_ids mismatch: {graph_ids.numel()} vs kid {kid.numel()}")
                if local_node_ids.numel() != kid.numel():
                    raise RuntimeError(f"local_node_ids mismatch: {local_node_ids.numel()} vs kid {kid.numel()}")
                if cid.numel() != kid.numel():
                    raise RuntimeError(f"cid mismatch: {cid.numel()} vs kid {kid.numel()}")

                # store CPU
                all_key_ids.append(kid.detach().cpu())
                all_cluster_ids.append(cid.detach().cpu())
                all_graph_ids.append(graph_ids.detach().cpu())
                all_local_node_ids.append(local_node_ids.detach().cpu())
                all_batch_ids.append(batch_ids.detach().cpu())
                all_chunk_starts.append(chunk_starts.detach().cpu())

                # cleanup per chunk
                del batched_graph, batched_feats, chunk, ids, kid, cid
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

    # ------------------------------------------------------------
    # 4) PACK + ALWAYS SAVE + RETURN
    # ------------------------------------------------------------
    if len(all_key_ids) == 0:
        return None

    key_id_all = torch.cat(all_key_ids, dim=0)                 # [N_atoms_total]
    cluster_id_all = torch.cat(all_cluster_ids, dim=0)         # [N_atoms_total]
    graph_id_in_chunk_all = torch.cat(all_graph_ids, dim=0)    # [N_atoms_total]
    local_node_id_all = torch.cat(all_local_node_ids, dim=0)   # [N_atoms_total]
    batch_id_all = torch.cat(all_batch_ids, dim=0)             # [N_atoms_total]
    chunk_start_in_batch_all = torch.cat(all_chunk_starts, dim=0)

    # final sanity
    n = key_id_all.numel()
    if not (cluster_id_all.numel() == graph_id_in_chunk_all.numel() == local_node_id_all.numel()
            == batch_id_all.numel() == chunk_start_in_batch_all.numel() == n):
        raise RuntimeError("final pack sanity check failed")

    out = {
        # ids
        "key_id": key_id_all,
        "cluster_id": cluster_id_all,
        "id2safe": last_id2safe or {},

        # provenance
        "graph_id_in_chunk": graph_id_in_chunk_all,
        "local_node_id": local_node_id_all,
        "batch_id": batch_id_all,
        "chunk_start_in_batch": chunk_start_in_batch_all,

        # meta
        "checkpoint_path": checkpoint_path,
        "chunk_size": int(chunk_size),
        "saved_at": time.strftime("%Y%m%d_%H%M%S"),
    }

    save_path = conf.get("infer_save_path") or "infer_token_ids.pt"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(out, save_path)
    print("saved:", save_path)

    return out

def run_inductive(conf, model, optimizer, scheduler, logger):
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
        logger.info(f"=== epoch {epoch} ==　initial kmeans start ....")
        for idx, (adj_batch, attr_batch) in enumerate(dataloader):

            if idx == 12:
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

        scheduler.step()

        # （optional）LR ログ
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[LR] epoch={epoch} lr={lr:.6e}")

    # 何か score を返したい場合はここで
    return test_total


