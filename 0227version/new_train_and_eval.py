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
    logger.info(f"loss_list3 {loss_list3}")
    for l in loss_list3:  # loss_list3 = [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss]
        # handle both Tensor and float / numpy
        if isinstance(l, torch.Tensor):
            loss_list_out.append(float(l.detach().cpu()))
        else:
            loss_list_out.append(float(l))
    logger.info(f"loss_list_out {loss_list_out}")
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
    degree_cap=None,
    ring_size_cap=8,
    arom_nbrs_cap=6,
    fused_id_cap=255,
    # If your new features are already packed in attr_batch, give their column indices:
    # 0  Z (atomic number)
    # 1  degree  (not used; we recompute from adj)
    # 2  charge
    # 3  hyb
    # 4  arom
    # 5  ring
    # 6  hnum
    # 7–24  func base flags (18 dims)
    # 25–26 h-bond donor / acceptor flags (2 dims)  ← NOT used in key (keep CBDICT compatible)
    # 27 ring size
    # 28 aromatic neighbor count (aromNbrs)
    # 29 fused ring ID (fusedId)
    # 30–77 bond_env_raw (48 dims) ← ignored here (key is 100% discrete, same as before)
    ring_size_col=27,
    arom_nbrs_col=28,
    fused_id_col=29,
    # 官能基ベースのフラグ (one-hot / multi-hot) が attr に入っている範囲
    # 例: attr[..., 7:25] に 18 個 (0..17) の官能基フラグがある想定
    func_base_start_col=7,
    n_func_base_flags=18,
    # Or pass them separately (same batching/shape as attr_batch[...,0]):
    ring_size_batch=None,         # list[Tensor] with shape (M,100) per batch item, or a Tensor viewable to (-1,100)
    arom_nbrs_batch=None,         # ditto
    fused_ring_id_batch=None,     # ditto
    # Which fields to include in the string key and in what order:
    include_keys=("Z","charge","hyb","arom","ring","deg",
                  "ringSize","aromNbrs","fusedId","pos","func","hnum"),
    # ★ この base キーだけ詳細集計したいときに使う ("6_0_3_1_1" など; deg は含めない)
    target_base_prefix="6_0_3_1_1",
    debug=True,
    debug_max_print=1000,
):
    """
    Returns:
      masks_dict: {
        'Z_q_h_a_r_deg_ringSize_aromNbrs_fusedId_pos_func_hnum' : [global_idx, ...],
        ...
      }
      atom_offset: next start atom id (global)
      mol_id:      number of processed molecules

    Notes:
      - attr columns assumed base (first 30 dims):
          0: Z
          1: degree (ignored; recomputed from adjacency)
          2: charge
          3: hyb
          4: arom
          5: ring
          6: hnum
          7–24: func base flags (18 dims)
          25–26: H-bond donor/acceptor (ignored in key)
          27: ringSize
          28: aromNbrs
          29: fusedId
        (30–77: bond_env_raw 48 dims; ignored here for CBDICT compatibility)
      - Degree computed from adjacency excluding self-loops
      - ringSize/aromNbrs/fusedId come from attr columns [27,28,29] by default
        or from side-channel tensors if *_col is None.
      - func_base_start_col .. func_base_start_col + n_func_base_flags:
          官能基ベースのフラグ (18個) を想定し、argmax で func_id (0..17) にまとめて key に入れる
      - H-bond donor/acceptor flags (cols 25–26) are currently ignored by this function
        so the number of discrete classes / CBDICT keys does not change.
      - ここでは CBDICT に存在する key だけ masks_dict に残す
      - target_base_prefix が指定されていれば、
        その base キー (Z,charge,hyb,arom,ring,deg,hnum) のうち
        「Z,charge,hyb,arom,ring が prefix 一致する原子」について
        ringSize/aromNbrs/fusedId/deg/pos/hnum の分布を debug 出力する
    """
    from collections import defaultdict, Counter
    import numpy as np
    import torch
    from utils import CBDICT

    # membership を高速にするために set 化
    CBDICT_KEYS = set(CBDICT.keys())
    """    atom.GetAtomicNum(),                 # 0: Z 
        atom.GetDegree(),                    # 1: degree (not used here)
        atom.GetFormalCharge(),              # 2: charge
        int(atom.GetHybridization()),        # 3: hyb (enum int)
        int(atom.GetIsAromatic()),           # 4: arom flag
        int(atom.IsInRing()),                # 5: ring flag
        hcount,                              # 6: total Hs (explicit+implicit)
        *func_flags[idx],                    # 7–24: 官能基フラグ (18 dims)
        *hbond_flags[idx],                   # 25–26: H-bond Donor/Acceptor (2 dims)
        ring_size[idx],                      # 27: ringSize
        arom_nbrs[idx],                      # 28: aromNbrs
        fused_id[idx],                       # 29: fusedId     
        30–77: bond_env_raw (48 dims)       """

    # Base columns in attr: [Z, charge, hyb, arom, ring, hnum]
    COL_Z, COL_CHARGE, COL_HYB, COL_AROM, COL_RING, COL_HNUM = 0, 2, 3, 4, 5, 6
    BASE_COLS = [COL_Z, COL_CHARGE, COL_HYB, COL_AROM, COL_RING, COL_HNUM]

    # include_keys 内での各フィールドの位置（key 文字列を分解するときに使う）
    name_to_idx = {name: idx for idx, name in enumerate(include_keys)}
    base_field_names = ("Z", "charge", "hyb", "arom", "ring", "deg", "hnum")
    base_field_indices = [name_to_idx[n] for n in base_field_names]

    ringSize_idx = name_to_idx.get("ringSize", None)
    aromNbrs_idx = name_to_idx.get("aromNbrs", None)
    fusedId_idx  = name_to_idx.get("fusedId", None)
    pos_idx      = name_to_idx.get("pos", None)
    deg_idx      = name_to_idx.get("deg", None)
    func_idx     = name_to_idx.get("func", None)
    hnum_idx     = name_to_idx.get("hnum", None)

    # 特定クラスの分布集計用
    target_stats = None
    if target_base_prefix is not None:
        target_stats = {
            "count": 0,
            "ringSize": Counter(),
            "aromNbrs": Counter(),
            "fusedId": Counter(),
            "deg": Counter(),
            "pos": Counter(),
            "hnum": Counter(),
        }

    def _to_cpu_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", non_blocking=True).numpy()
        return x

    def _extract_side_feature(side_batch, i, M):
        """
        side_batch: None or
                    - list/tuple: side_batch[i] -> (M,100) or (M,100,1) or (M*100,)
                    - np.ndarray / torch.Tensor: side_batch[i] 同様
        戻り値: np.int32 の (M,100) または None
        """
        if side_batch is None:
            return None

        side_i = side_batch[i]
        side_np = _to_cpu_np(side_i)

        # いろんな shape をそれっぽく (M,100) に揃える
        if side_np.ndim == 1:
            # 長さ M*100 を想定
            side_np = side_np.reshape(M, 100)
        elif side_np.ndim == 2:
            if side_np.shape == (M, 100):
                pass
            elif side_np.shape[0] == M and side_np.shape[1] >= 100:
                side_np = side_np[:, :100]
            elif side_np.shape[0] == M * 100 and side_np.shape[1] == 1:
                side_np = side_np.reshape(M, 100)
            else:
                side_np = side_np.reshape(M, 100)
        elif side_np.ndim == 3:
            # 典型例: (M,100,1)
            if side_np.shape[0] == M and side_np.shape[1] == 100:
                side_np = side_np[..., 0]
            else:
                side_np = side_np.reshape(M, 100)
        else:
            side_np = side_np.reshape(M, 100)

        return side_np.astype(np.int32)

    from collections import defaultdict
    masks_dict = defaultdict(list)
    atom_offset = int(start_atom_id)
    mol_id = int(start_mol_id)

    B = len(attr_batch)

    # ---------- main loop over batch items ----------
    for i in range(B):
        # Shapes (想定):
        #   attr_batch[i]: (M*100*D) 相当 → (M,100,D)
        #   adj_batch[i]:  (M*100*100) 相当 → (M,100,100)
        D = attr_batch[i].shape[-1]
        attr_mats = attr_batch[i].view(-1, 100, D)
        adj_mats  = adj_batch[i].view(-1, 100, 100)

        attr_np = _to_cpu_np(attr_mats)  # (M,100,D)
        adj_np  = _to_cpu_np(adj_mats)   # (M,100,100)

        M = attr_np.shape[0]

        # Base attributes (Z, charge, hyb, arom, ring, hnum)
        A_sel = attr_np[..., BASE_COLS].astype(np.int32)         # (M,100,6)

        # Valid (unpadded) node mask
        node_mask = (np.abs(attr_np).sum(axis=2) > 0)            # (M,100) bool

        # Degree from adjacency (exclude self-loops)
        deg_total = (adj_np != 0).sum(axis=2).astype(np.int32)   # (M,100)
        diag_nz   = (np.abs(np.diagonal(adj_np, axis1=1, axis2=2)) != 0).astype(np.int32)
        degrees   = deg_total - diag_nz
        if degree_cap is not None:
            degrees = np.minimum(degrees, int(degree_cap))

        # ---- ringSize / aromNbrs / fusedId ----
        if ring_size_col is not None:
            ring_size_np = attr_np[..., ring_size_col].astype(np.int32)
        else:
            ring_size_np = _extract_side_feature(ring_size_batch, i, M)
            if ring_size_np is None:
                ring_size_np = np.zeros((M, 100), np.int32)

        if arom_nbrs_col is not None:
            arom_nbrs_np = attr_np[..., arom_nbrs_col].astype(np.int32)
        else:
            arom_nbrs_np = _extract_side_feature(arom_nbrs_batch, i, M)
            if arom_nbrs_np is None:
                arom_nbrs_np = np.zeros((M, 100), np.int32)

        if fused_id_col is not None:
            fused_id_np = attr_np[..., fused_id_col].astype(np.int32)
        else:
            fused_id_np = _extract_side_feature(fused_ring_id_batch, i, M)
            if fused_id_np is None:
                fused_id_np = np.zeros((M, 100), np.int32)

        # ---- functional base flags → func_id (0..n_func_base_flags-1) ----
        if func_base_start_col is not None and n_func_base_flags is not None:
            if attr_np.shape[2] < func_base_start_col + n_func_base_flags:
                raise ValueError(
                    f"attr_np last dim {attr_np.shape[2]} is too small for "
                    f"func_base_start_col={func_base_start_col}, n_func_base_flags={n_func_base_flags}"
                )
            func_flags_np = attr_np[..., func_base_start_col:func_base_start_col + n_func_base_flags]
            # フラグは 0/1 想定だが、一応 argmax で 0..(n_func_base_flags-1) にまとめる
            func_flags_np = func_flags_np.astype(np.float32)
            func_id_np = np.argmax(func_flags_np, axis=2).astype(np.int32)  # (M,100)
        else:
            func_id_np = np.zeros((M, 100), np.int32)

        # Caps
        if ring_size_cap is not None:
            ring_size_np = np.minimum(ring_size_np, int(ring_size_cap))
        if arom_nbrs_cap is not None:
            arom_nbrs_np = np.minimum(arom_nbrs_np, int(arom_nbrs_cap))
        if fused_id_cap is not None:
            fused_id_np = np.minimum(fused_id_np, int(fused_id_cap))

        if i == 0 and debug:
            print("=== [collect_global_indices_compact] batch 0 side-feature summary ===")
            print("  ringSize (all nodes): unique", len(np.unique(ring_size_np)))
            for v in np.unique(ring_size_np):
                print(f"    value={v}  count={int((ring_size_np == v).sum())}")
            print("  aromNbrs (all nodes): unique", len(np.unique(arom_nbrs_np)))
            for v in np.unique(arom_nbrs_np):
                print(f"    value={v}  count={int((arom_nbrs_np == v).sum())}")
            print("  fusedId (all nodes): unique", len(np.unique(fused_id_np)))
            for v in np.unique(fused_id_np):
                print(f"    value={v}  count={int((fused_id_np == v).sum())}")
            print("  func_id (all nodes): unique", len(np.unique(func_id_np)))
            for v in np.unique(func_id_np):
                print(f"    value={v}  count={int((func_id_np == v).sum())}")

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
            hnum   = A_sel[m, :, 5][nm]
            deg    = degrees[m][nm]
            # New features
            rs  = ring_size_np[m][nm]
            an  = arom_nbrs_np[m][nm]
            fid = fused_id_np[m][nm]

            # 官能基カテゴリ (0..n_func_base_flags-1)
            func = func_id_np[m][nm]

            # position flag (例: 芳香 sp2, ring=1, deg=2 の C の outer/inner 区別用)
            pos = (((z == 6) & (hyb == 3) & (arom == 1) & (ring == 1) & (deg == 2))).astype(np.int32)

            # Choose which fields to include and stack in that order
            fields = {
                "Z": z, "charge": charge, "hyb": hyb, "arom": arom, "ring": ring, "deg": deg,
                "ringSize": rs, "aromNbrs": an, "fusedId": fid, "pos": pos, "func": func, "hnum": hnum
            }
            cols_to_stack = [fields[name] for name in include_keys]
            keys = np.stack(cols_to_stack, axis=1).astype(np.int32)   # (N, K)
            if keys.ndim == 1:
                keys = keys.reshape(1, -1)
            N = int(keys.shape[0])

            # Global ids for these atoms
            global_ids = np.arange(atom_offset, atom_offset + N, dtype=np.int64)

            # Build string keys fast with np.char (full key)
            ks = keys.astype(str)       # (N, K) of strings
            key_strings = ks[:, 0]
            for c in range(1, ks.shape[1]):
                key_strings = np.char.add(np.char.add(key_strings, "_"), ks[:, c])

            # ★ target_base_prefix 用の集計：base キーの prefix が一致する原子だけ拾う
            if target_stats is not None:
                base_cols = keys[:, base_field_indices]      # (N, len(base_field_indices))
                bs = base_cols.astype(str)
                base_key_strings = bs[:, 0]
                for c in range(1, bs.shape[1]):
                    base_key_strings = np.char.add(np.char.add(base_key_strings, "_"), bs[:, c])

                # prefix マッチ
                mask = np.char.startswith(base_key_strings, target_base_prefix)

                if mask.any():
                    n_hit = int(mask.sum())
                    target_stats["count"] += n_hit

                    if ringSize_idx is not None:
                        for v in keys[mask, ringSize_idx]:
                            target_stats["ringSize"][int(v)] += 1
                    if aromNbrs_idx is not None:
                        for v in keys[mask, aromNbrs_idx]:
                            target_stats["aromNbrs"][int(v)] += 1
                    if fusedId_idx is not None:
                        for v in keys[mask, fusedId_idx]:
                            target_stats["fusedId"][int(v)] += 1
                    if deg_idx is not None:
                        for v in keys[mask, deg_idx]:
                            target_stats["deg"][int(v)] += 1
                    if pos_idx is not None:
                        for v in keys[mask, pos_idx]:
                            target_stats["pos"][int(v)] += 1
                    if hnum_idx is not None:
                        for v in keys[mask, hnum_idx]:
                            target_stats["hnum"][int(v)] += 1

            # Group by unique key and extend once per key
            uniq_keys, inv = np.unique(key_strings, return_inverse=True)
            buckets = [[] for _ in range(len(uniq_keys))]
            inv_list = inv.tolist()
            for row_idx, bucket_id in enumerate(inv_list):
                buckets[bucket_id].append(int(global_ids[row_idx]))

            # ★ CBDICT に存在する key だけ masks_dict に追加
            for uk, ids in zip(uniq_keys.tolist(), buckets):
                # if uk not in CBDICT_KEYS:
                #     continue
                masks_dict[uk].extend(ids)

            # advance offsets per molecule
            atom_offset += N
            mol_id += 1

    # for key, val in masks_dict.items():
    #     logger.info(f"key {key}, val {len(val)}")
    # logger.info(f"[collect_global_indices_compact] total keys in masks_dict: {len(masks_dict)}")

    # target_base_prefix の集計結果を最後にまとめて出力
    if target_stats is not None and debug:
        lines = []
        lines.append(f"=== [DEBUG target_base_prefix={target_base_prefix}] summary ===")
        lines.append(f"  total atoms: {target_stats['count']}")
        for fld in ("ringSize", "aromNbrs", "fusedId", "deg", "pos", "hnum"):
            ctr = target_stats[fld]
            if not ctr:
                continue
            lines.append(f"  {fld}: {len(ctr)} unique values")
            for v, cnt in sorted(ctr.items()):
                lines.append(f"    value={v}  count={cnt}")
        text = "\n".join(lines)
        if logger is not None:
            logger.info(text)
        else:
            print(text)

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
        attr_matrices = attr_batch[i].view(-1, 100, 78)

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
        print("initial kmeans start ....")

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

        for idx, (adj_batch, attr_batch) in enumerate(dataloader):
            print(f"[KMEANS] batch idx {idx}")

            if idx == 5:
                break
            if idx == 1:
                break

            glist_base, glist, masks_dict, attr_matrices, start_atom_id, start_mol_id = convert_to_dgl(
                adj_batch, attr_batch, logger, start_atom_id, start_mol_id
            )  # 10000 molecules per glist

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

        # print(f"[KMEANS] latents shape: {all_latents_tensor.shape}, attr shape: {all_attr_tensor.shape}")
        print("[KMEANS] init_kmeans_final start")

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

            # dataloader は再利用可能（新しい iterator が作られる）
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 5:
                    break

                if idx == 1:
                    break
                print(f"[TRAIN] batch idx {idx}")
                glist_base, glist, masks_2, attr_matrices_all, _, _ = convert_to_dgl(
                    adj_batch, attr_batch, logger
                )
                chunk_size = conf["chunk_size"]

                for i in range(0, len(glist), chunk_size):
                    print(f"[TRAIN]   chunk {i}")
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
                    logger.info(f"clean_losses {clean_losses}")
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

                #  loss, embed, [commit_loss.item(), repel_loss.item(), cb_repel_loss.item()]
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
        logger.info(f"loss_list_list_train[2] before safe_mean: {loss_list_list_train[2]}")
        train_latrep = safe_mean(loss_list_list_train[2])
        logger.info(f"loss_list_list_train[2] after safe_mean: {loss_list_list_train[2]}")
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

        # test logs
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


