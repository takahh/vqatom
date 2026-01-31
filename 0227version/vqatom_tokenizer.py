import torch
from rdkit import Chem

# あなたの既存コードを import
from smiles_to_npy import smiles_to_graph_with_labels
# from
# from your_pipeline import (
#     smiles_to_graph_with_labels,
#     convert_to_dgl_single,
#     load_trained_model,     # model_epoch_xx.pth を読むやつ
# )
import torch
def convert_to_dgl_single(
    adj,
    attr,
    device=None,
    two_hop_w=0.5,
    three_hop_w=0.3,
    make_base_bidirected=True,
    add_self_loops=True,
    depad_from_attr=True,
):
    """
    Convert ONE molecule (adj, attr) -> (base_g, extended_g, X, meta)

    Inputs
    ------
    adj  : Tensor [100,100] or [N,N]
    attr : Tensor [100,79]  or [N,79]
        - depad_from_attr=True の場合、attr のゼロ行で N を推定して depad する

    Returns
    -------
    base_g     : DGLGraph (1-hop edges)
    extended_g : DGLGraph (1 + 2hop + 3hop weighted edges)
    X          : Tensor [N,79] node features
    meta       : dict (N, nonzero_mask, etc.)
    """
    import torch
    import dgl

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # move once
    adj = adj.to(device, non_blocking=True)
    attr = attr.to(device, non_blocking=True)

    # -------------------------
    # depad
    # -------------------------
    if depad_from_attr:
        nonzero_mask = (attr.abs().sum(dim=1) > 0)  # [100]
        N = int(nonzero_mask.sum().item())
        if N <= 0:
            raise ValueError("No valid atoms found (all-zero attr rows).")
        X = attr[nonzero_mask]          # [N,79]
        W1 = adj[:N, :N].float()        # [N,N]
    else:
        # assume already [N,*]
        N = int(attr.shape[0])
        if N <= 0:
            raise ValueError("attr has zero rows.")
        nonzero_mask = None
        X = attr
        W1 = adj.float()

    # bool adjacency for 1-hop
    A1 = (W1 > 0)                       # [N,N] bool

    if make_base_bidirected:
        A1 = A1 | A1.T

    if add_self_loops:
        A1 = A1.clone()
        A1.fill_diagonal_(True)

    # -------------------------
    # base graph (1-hop)
    # -------------------------
    bsrc, bdst = A1.nonzero(as_tuple=True)
    base_g = dgl.graph((bsrc, bdst), num_nodes=N, device=device)
    base_g.ndata["feat"] = X

    # base weights (diagonal=1 if self-loops)
    if add_self_loops:
        W1w = W1.clone()
        W1w.fill_diagonal_(1.0)
    else:
        W1w = W1

    es, ed = base_g.edges()
    base_g.edata["weight"] = W1w[es, ed]
    base_g.edata["edge_type"] = torch.ones(base_g.num_edges(), device=device, dtype=torch.int32)

    # -------------------------
    # k-hop reachability (dense matmul)
    # -------------------------
    A1f = A1.to(torch.float32)
    A2 = (A1f @ A1f) > 0
    A3 = (A2.to(torch.float32) @ A1f) > 0

    one_hop = A1
    two_hop_only = A2 & ~one_hop
    three_hop_only = A3 & ~(one_hop | A2)

    # -------------------------
    # extended weighted adjacency
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
        Afull = Afull.clone()
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

    meta = {
        "N": N,
        "nonzero_mask": nonzero_mask,
    }
    return base_g, extended_g, X, meta

def load_trained_model(checkpoint_path: str, build_model_fn, conf: dict, device: str = "cuda"):
    """
    build_model_fn: アーキテクチャを作って model を返す関数（あなたの既存コードに合わせる）
    conf: restore_strict などを持つ辞書
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1) build
    model = build_model_fn(conf).to(device)
    model.eval()

    # 2) restore (あなたのコード)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    _ensure_codebook_keys_if_possible(model, state)

    strict = bool(conf.get("restore_strict", True))
    if strict:
        model.load_state_dict(state, strict=True)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("restore(strict=False) missing:", missing[:20])
        print("restore(strict=False) unexpected:", unexpected[:20])

    model.eval()
    return model

class VQAtomTokenizer:
    def __init__(self, model_ckpt: str, device="cpu"):
        self.device = device
        self.model = load_trained_model(model_ckpt, device)
        self.model.eval()

        # --- CBDICT 順で offset を固定 ---
        self.offset = {}
        cur = 0
        for key in self.model.vq._codebook.embed.keys():
            K = self.model.vq._codebook.embed[key].shape[0]
            self.offset[key] = cur
            cur += K
        self.vocab_size = cur

    @torch.no_grad()
    def encode(self, smiles: str):
        # 1) graph 構築
        adj, attr, _ = smiles_to_graph_with_labels(smiles, idx=0)
        g, g_base = convert_to_dgl_single(adj, attr)

        feats = g.ndata["feat"].to(self.device)

        # 2) GNN latent
        latent = self.model.encoder(g, feats)

        # 3) VQ infer
        _, _, key_id, cluster_id, id2safe = self.model.vq(
            latent, attr, mode="infer"
        )

        key_id = key_id.cpu().tolist()
        cluster_id = cluster_id.cpu().tolist()

        # 4) 1D token 化
        tokens = []
        for k, c in zip(key_id, cluster_id):
            safe = id2safe[k]
            tok = self.offset[safe] + c
            tokens.append(int(tok))

        return tokens
