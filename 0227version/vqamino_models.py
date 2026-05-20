from args import get_args
import torch.nn as nn
import torch


class BondWeightLayer(nn.Module):
    def __init__(self, bond_types=4, hidden_dim=64):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)  # Learnable bond representation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.edge_mlp = self.edge_mlp.to(device)  # Move edge MLP to correct device

        nn.init.xavier_uniform_(self.bond_embedding.weight)  # Xavier for embeddings
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, edge_types):
        bond_feats = self.bond_embedding(edge_types)  # Convert bond type to learnable vector
        edge_weight = self.edge_mlp(bond_feats).squeeze()  # Compute edge weight
        edge_weight = edge_weight.squeeze()  # Compute edge weight
        return edge_weight

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


import torch.nn.functional as F
from torch_geometric.nn import GINEConv

import torch.nn as nn
from vqamino_vq import VectorQuantize  # VQ-Amino quantizer


class BondWeightLayer(nn.Module):
    def __init__(self, bond_types=4, hidden_dim=64):
        import torch
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)  # Learnable bond representation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output weight in range (0,1)
        )
        self.edge_mlp = self.edge_mlp.to(device)  # Move edge MLP to correct device

    def forward(self, edge_types):
        bond_feats = self.bond_embedding(edge_types)  # Convert bond type to learnable vector
        edge_weight = self.edge_mlp(bond_feats).squeeze()  # Compute edge weight
        return edge_weight

import torch

def _zeros_like(x, device=None, dtype=None):
    return torch.zeros((), device=device or getattr(x, "device", None), dtype=dtype or getattr(x, "dtype", None))

import torch
def _normalize_quantize_output(qo, logger, device=None, dtype=None):
    """
    Normalize quantizer outputs to 7-tuple:
      (loss, embed, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss)

    Assumption (ALWAYS):
      qo == (total_loss, (commit_loss, codebook_loss, repel_loss, cb_repel_loss))

    Rules:
      - Never detach.
      - Never wrap existing tensors into new tensors.
      - Only convert python/np scalars/None to tensors.
    """

    def _infer_device_dtype(*xs):
        nonlocal device, dtype
        if device is None or dtype is None:
            for x in xs:
                if torch.is_tensor(x):
                    if device is None:
                        device = x.device
                    if dtype is None:
                        dtype = x.dtype
                    return
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

    def _scalar_to_tensor(x):
        if torch.is_tensor(x):
            return x
        if x is None:
            _infer_device_dtype()
            return torch.zeros((), device=device, dtype=dtype)
        _infer_device_dtype()
        return torch.as_tensor(x, device=device, dtype=dtype)

    # ---------------------------
    # only supported shape:
    # (loss, (commit, cb, rep, cb_rep))
    # ---------------------------
    if not (isinstance(qo, (tuple, list)) and len(qo) == 2 and isinstance(qo[1], (tuple, list))):
        raise TypeError(f"Expected (loss, (commit, cb, rep, cb_rep)) but got {type(qo)}: {qo}")

    loss, inner = qo
    if len(inner) != 5:
        raise TypeError(f"Expected inner tuple length 5: (commit, cb, rep, cb_rep, ent), got {len(inner)}: {inner}")

    commit, cb, rep, cb_rep, ent = inner[0], inner[1], inner[2], inner[3], inner[4]

    # total_loss, (commit_loss, codebook_loss, repel_loss, cb_repel_loss, ent_loss)
    logger.info(f"end of model forward, loss {loss},　commit {commit}, ent {ent}")
    _infer_device_dtype(loss, commit, cb, rep, cb_rep, ent)

    embed = None  # qoにはembedが無い前提
    sil = None    # qoにはsilが無い前提（0で埋める）

    return (
        _scalar_to_tensor(loss),
        embed,
        _scalar_to_tensor(commit),
        _scalar_to_tensor(cb),
        _scalar_to_tensor(sil),      # silhouette_loss = 0
        _scalar_to_tensor(rep),
        _scalar_to_tensor(cb_rep),
        _scalar_to_tensor(ent),
    )

import torch.nn as nn
import torch
import torch.nn as nn

class AminoAcidEmbedding(nn.Module):
    """Encode residue-level VQ-Amino features.

    Expected input shape: [N, F]. Column 0 is an amino-acid id.
    Remaining columns are optional numeric/context features (position bins,
    local sequence descriptors, precomputed features, etc.). The module lazily
    creates a projection for the observed feature width, so the training script
    no longer depends on the VQ-Atom [N,79] chemical feature layout.
    """
    def __init__(self, aa_vocab_size=32, aa_emb_dim=32, out_dim=64):
        super().__init__()
        self.aa_embed = nn.Embedding(aa_vocab_size, aa_emb_dim, padding_idx=0)
        self.out_dim = int(out_dim)
        self.aa_emb_dim = int(aa_emb_dim)
        self.extra_proj = None
        self.final = nn.Sequential(
            nn.Linear(self.aa_emb_dim + self.out_dim // 2, self.out_dim),
            nn.ReLU(),
            nn.LayerNorm(self.out_dim),
        )

    def _ensure_extra_proj(self, extra_dim: int, device):
        if self.extra_proj is None or self.extra_proj.in_features != int(extra_dim):
            self.extra_proj = nn.Linear(int(extra_dim), self.out_dim // 2).to(device)

    def forward(self, residue_inputs: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = residue_inputs.to(device=device, non_blocking=True)
        if x.ndim != 2:
            raise ValueError(f"residue_inputs must be [N,F], got {tuple(x.shape)}")
        if x.size(1) < 1:
            raise ValueError("residue_inputs must contain at least column 0 = amino-acid id")

        aa = x[:, 0].long().clamp(0, self.aa_embed.num_embeddings - 1)
        aa_emb = self.aa_embed(aa)

        if x.size(1) > 1:
            extra = x[:, 1:].to(dtype=torch.float32)
        else:
            extra = torch.zeros((x.size(0), 1), device=device, dtype=torch.float32)

        self._ensure_extra_proj(extra.size(1), device)
        extra_emb = self.extra_proj(extra)
        return self.final(torch.cat([aa_emb, extra_emb], dim=-1))

class VQAminoThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super().__init__()
        import torch
        if args is None:
            args = get_args()

        self.feat_embed = AminoAcidEmbedding(
            aa_vocab_size=getattr(args, "aa_vocab_size", 32),
            aa_emb_dim=getattr(args, "aa_emb_dim", 32),
            out_dim=64,
        )
        self.linear_0 = nn.Linear(64, args.hidden_dim)  # h0

        edge_emb_dim = getattr(args, "edge_emb_dim", 32)
        # edge_type: 0=self/other, 1=1-hop, 2=2-hop, 3=3-hop
        self.edge_emb = nn.Embedding(4, edge_emb_dim, padding_idx=0)

        def mlp(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f), nn.ReLU(),
                nn.Linear(out_f, out_f), nn.ReLU()
            )

        nn1 = mlp(args.hidden_dim, hidden_feats)
        nn2 = mlp(hidden_feats, hidden_feats)
        nn3 = mlp(hidden_feats, hidden_feats)

        self.gine1 = GINEConv(nn1, edge_dim=edge_emb_dim)
        self.gine2 = GINEConv(nn2, edge_dim=edge_emb_dim)
        self.gine3 = GINEConv(nn3, edge_dim=edge_emb_dim)

        self.ln_in = nn.LayerNorm(args.hidden_dim)
        self.ln1   = nn.LayerNorm(hidden_feats)
        self.ln2   = nn.LayerNorm(hidden_feats)
        self.ln3   = nn.LayerNorm(hidden_feats)

        # Residual scales
        self.res1 = nn.Parameter(torch.tensor(0.5))
        self.res2 = nn.Parameter(torch.tensor(0.5))
        self.res3 = nn.Parameter(torch.tensor(0.5))

        # If args.hidden_dim != hidden_feats, project skip for hop1
        self.skip0 = None
        if args.hidden_dim != hidden_feats:
            self.skip0 = nn.Linear(args.hidden_dim, hidden_feats, bias=False)

        # JK: concat h0 + h1 + h2 + h3
        jk_dim = args.hidden_dim + 3 * hidden_feats
        self.mix = nn.Sequential(
            nn.Linear(jk_dim, 2 * hidden_feats), nn.ReLU(),
            nn.Linear(2 * hidden_feats, hidden_feats), nn.ReLU(),
        )

        # Project to VQ dim
        self.out_proj   = nn.Linear(hidden_feats, args.hidden_dim)
        self.pre_vq_ln  = nn.LayerNorm(args.hidden_dim)

        self.vq = VectorQuantize(
            dim=args.hidden_dim,
            codebook_size=args.codebook_size,
            decay=getattr(args, "ema_decay", 0.8),
            threshold_ema_dead_code=2,
        )

    def reset_kmeans(self):
        self.vq._codebook.reset_kmeans()

    def forward(self, data, features, chunk_i, mask_dict=None, logger=None, epoch=None,
                batched_graph_base=None, mode=None, attr_list=None):
        import torch
        dev = next(self.parameters()).device
        if mode == "init_kmeans_final":
            if hasattr(data, "to"):
                data = data.to(dev)
            if torch.is_tensor(features):
                features = features.to(dev, non_blocking=True)
            self.vq(data, attr_list, mask_dict, logger, chunk_i, epoch, mode)
            return 0

        # Edges (mirror for undirected) -> dev
        s1, d1 = data.edges()
        s1 = s1.to(dev, non_blocking=True); d1 = d1.to(dev, non_blocking=True)
        src = torch.cat([s1, d1], 0); dst = torch.cat([d1, s1], 0)
        edge_index = torch.stack([src, dst], 0)

        # Edge attributes for residue graph hops: 0=self/other, 1=1-hop, 2=2-hop, 3=3-hop
        eb = data.edata.get("edge_type", torch.zeros(data.num_edges(), dtype=torch.long, device=s1.device))
        eb = eb.to(dev, non_blocking=True).long()
        e  = torch.cat([eb, eb], 0).clamp(0, 3)
        edge_attr = self.edge_emb(e)

        # Node features -> h0
        features = features.to(dev, non_blocking=True)
        h0_raw = self.feat_embed(features)              # [N, 64]
        h0 = self.ln_in(self.linear_0(h0_raw))          # [N, args.hidden_dim]

        # Hop 1
        h0_for1 = self.skip0(h0) if self.skip0 is not None else h0
        h1 = self.gine1(h0, edge_index, edge_attr)
        h1 = self.ln1(h1 * self.res1 + h0_for1)

        # Hop 2
        h2 = self.gine2(h1, edge_index, edge_attr)
        h2 = self.ln2(h2 * self.res2 + h1)
        # Hop 3
        h3 = self.gine3(h2, edge_index, edge_attr)
        h3 = self.ln3(h3 * self.res3 + h2)

        # JK concat
        h_cat = torch.cat([h0, h1, h2, h3], dim=-1)
        h_mid = self.mix(h_cat)
        h_out = self.out_proj(h_mid)

        # K-means 用のループ時は VQ を通さずにそのまま返す
        if mode == "init_kmeans_loop":
            return h_out
        import torch.nn.functional as F
        h_vq = self.pre_vq_ln(h_out)

        # L2 normalize latents here
        h_vq = F.normalize(h_vq, p=2, dim=-1, eps=1e-12)

        quantize_output = self.vq(
            h_vq,
            attr_list,
            mask_dict,
            logger,
            chunk_i,
            epoch,
            mode,
        )
        if mode == "infer":
            # vq.infer returns: (kid_full, cid_full, gid_full, id2safe)
            if isinstance(quantize_output, (tuple, list)) and len(quantize_output) == 4:
                kid_full, cid_full, gid_full, id2safe = quantize_output
                # IMPORTANT: return format expected by evaluate/run_infer_after_restore
                # (they later do: kid, cid, gid, id2safe = ids)
                ids = (kid_full, cid_full, gid_full, id2safe)
                # The rest of your forward might expect (loss, ids, id2safe) or similar.
                # If you already return (_, ids, _) in evaluate, keep that structure.
                return None, ids, None

            # Backward compat (older vq): (gid, cid, id2safe)
            if isinstance(quantize_output, (tuple, list)) and len(quantize_output) == 3:
                gid_full, cid_full, id2safe = quantize_output
                kid_full = gid_full
                ids = (kid_full, cid_full, gid_full, id2safe)
                return None, ids, None

            raise TypeError(f"[model.forward] infer expected vq to return 4-tuple, got {type(quantize_output)}")
        # -------------------------
        # INFER: return token IDs
        # -------------------------
        if mode == "infer":
            # vq は infer のとき (key_id_full, cluster_id_full, id2safe) を返す設計にする
            # ここで normalize は使わない（loss群とは別物なので）
            if not (isinstance(quantize_output, (tuple, list)) and len(quantize_output) == 3):
                raise TypeError(
                    f"[model.forward] mode='infer' expects vq() -> (key_id_full, cluster_id_full, id2safe), "
                    f"got {type(quantize_output)} with len={len(quantize_output) if isinstance(quantize_output,(tuple,list)) else 'NA'}"
                )

            key_id_full, cluster_id_full, id2safe = quantize_output

            # shape guarantee
            key_id_full = key_id_full.reshape(-1).long()
            cluster_id_full = cluster_id_full.reshape(-1).long()

            return key_id_full, cluster_id_full, id2safe

        # quantizer の生の出力（2タプルなど）を正規化して loss 群だけ取り出す
        loss, _embed_ignored, commit_loss, cb_loss, sil_loss, repel_loss, cb_repel_loss, ent_loss = _normalize_quantize_output(
            quantize_output, logger,
            device=h_vq.device if torch.is_tensor(h_vq) else None,
            dtype=getattr(h_vq, "dtype", None)
        )

        # embed は今回は h_vq をそのまま使う（まだ本当の量子化はしていないバージョン）
        embed = h_vq

        # モデルが返すのは (total_loss, embed, loss_list)
        # return loss, embed, [commit_loss, repel_loss, cb_repel_loss]
        #         loss, cb, loss_list3 = outputs
        return loss, embed, [commit_loss, cb_repel_loss, repel_loss, cb_loss, sil_loss, ent_loss]


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"],
                lamb_div_ele=conf["lamb_div_ele"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])

    def forward(self, data, feats, epoch, logger):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats, epoch, logger)

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats):
        if "SAGE" in self.model_name:
            # return self.forward(data, feats)

            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)
