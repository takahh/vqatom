from __future__ import annotations

"""Single-file consolidation of the six uploaded VQAtom modules.

This is a structural merge, not an algorithmic fix.
Fixes: missing training codebook lookup, proportional codebook budget, and EMA buffer key consistency.
Full training and checkpoint compatibility have not been tested.
Original source sections are identified below.
"""


# ==============================================================================
# ORIGINAL MODULE: args.py
# ==============================================================================
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=7, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epoch_at_mode_shift", type=int, default=0, help="Epoch at mode shift")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    # ss_max_total_latent_count
    parser.add_argument(
        "--ss_max_total_latent_count", type=int, default=20000, help="max count of latent used in SS calc."
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_false",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )
    # --------------
    # Dataset
    # --------------
    parser.add_argument("--train_size", type=int, default=5939)
    parser.add_argument("--val_size", type=int, default=1484)
    parser.add_argument("--test_size", type=int, default=1484)
    # parser.add_argument("--train_size", type=int, default=59397)
    # parser.add_argument("--val_size", type=int, default=14849)
    # parser.add_argument("--test_size", type=int, default=14849)
    parser.add_argument("--get_umap_data", action="store_true", help="Enable UMAP data processing")
    parser.add_argument("--use_checkpoint", action="store_true", help="Enable loading saved model")
    parser.add_argument("--percent", type=float, default=1)
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        # default=30,
        default=None,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        # default=20,
        default=None,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )
    # --------------
    # VQ
    # --------------
    parser.add_argument("--codebook_size", type=int, default=1500, help="Codebook size of VQGraph")
    parser.add_argument("--spatial_ivf", action="store_true", help="Use spatial IVF for large atom groups")
    parser.add_argument("--ivf_min_samples", type=int, default=100000)
    parser.add_argument("--ivf_nlist", type=int, default=64)
    parser.add_argument("--ivf_fit_samples", type=int, default=200000)
    parser.add_argument("--ivf_fine_iters", type=int, default=5)
    parser.add_argument("--lamb_edge",  type=float, default=0.003)  # default=0.03)
    parser.add_argument("--lamb_node", type=float, default=0.00008)  # default=0.001)
    parser.add_argument("--lamb_div_ele",  type=float, default=0.002)  # default=0.03)
    parser.add_argument("--dynamic_threshold", action="store_true", help="Use dynamic threshold in loss")

    # --------------
    # Model
    # --------------
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--train_or_infer", type=str, default="train", help="Train or just infer")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="4,4",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for sampler"
    )

    parser.add_argument(
        "--chunk_size", type=int, default=200
    )
    parser.add_argument(
        "--chunk_size2", type=int, default=1000
    )
    """Optimization"""
    parser.add_argument("--accumulation_steps", type=int, default=2) # default=0.0001)
    parser.add_argument("--learning_rate", type=float, default=0.0003) # default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--cosine_epochs", type=float, default=200)
    parser.add_argument(
        "--max_epoch", type=int, default=5, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args


# ==============================================================================
# ORIGINAL MODULE: utils.py
# ==============================================================================
import numpy as np
import torch
import logging
import pytz
import random
import os
import yaml
import shutil
from datetime import datetime
from ogb.nodeproppred import Evaluator
from dgl import function as fn

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]
CORE_ELEMENTS = {"5", "6", "7", "8", "14", "15", "16"}
CBDICT = {
    '1_0_1_0_0_0': 34,   # N=991
    '5_-1_4_0_0_0': 1,   # N=15
    '5_-1_4_0_1_0': 2,   # N=43
    '5_0_3_0_0_0': 11,   # N=305
    '5_0_3_0_1_0': 8,   # N=240
    '6_-1_2_0_0_0': 1,   # N=13
    '6_0_2_0_0_0': 511,   # N=15330
    '6_0_2_0_1_0': 3,   # N=88
    '6_0_3_0_0_0': 7687,   # N=230602
    '6_0_3_0_1_0': 3068,   # N=92016
    '6_0_3_1_1_0': 75098,   # N=2252928
    '6_0_4_0_0_0': 26727,   # N=801795
    '6_0_4_0_1_0': 20629,   # N=618852
    '6_1_3_0_0_0': 1,   # N=9
    '6_1_3_1_1_0': 1,   # N=1
    '7_-1_3_0_0_0': 14,   # N=403
    '7_-1_3_0_1_0': 1,   # N=6
    '7_-1_3_1_1_0': 1,   # N=17
    '7_-1_4_0_0_0': 1,   # N=2
    '7_0_2_0_0_0': 310,   # N=9273
    '7_0_3_0_0_0': 5858,   # N=175723
    '7_0_3_0_1_0': 2597,   # N=77907
    '7_0_3_1_1_0': 8356,   # N=250669
    '7_0_4_0_0_0': 1043,   # N=31290
    '7_0_4_0_1_0': 1333,   # N=39981
    '7_1_2_0_0_0': 14,   # N=393
    '7_1_3_0_0_0': 236,   # N=7074
    '7_1_3_0_1_0': 8,   # N=216
    '7_1_3_1_1_0': 80,   # N=2393
    '7_1_4_0_0_0': 92,   # N=2755
    '7_1_4_0_1_0': 51,   # N=1516
    '8_-1_3_0_0_0': 280,   # N=8390
    '8_-1_4_0_0_0': 42,   # N=1237
    '8_0_3_0_0_0': 15342,   # N=460250
    '8_0_3_0_1_0': 676,   # N=20278
    '8_0_3_1_1_0': 705,   # N=21138
    '8_0_4_0_0_0': 2287,   # N=68608
    '8_0_4_0_1_0': 789,   # N=23651
    '8_1_3_0_1_0': 1,   # N=8
    '8_1_3_1_1_0': 1,   # N=8
    '9_0_4_0_0_0': 2589,   # N=77656
    '14_0_4_0_0_0': 8,   # N=221
    '14_0_4_0_1_0': 1,   # N=13
    '15_0_3_0_0_0': 1,   # N=1
    '15_0_3_1_1_0': 1,   # N=6
    '15_0_4_0_0_0': 137,   # N=4110
    '15_0_4_0_1_0': 9,   # N=257
    '15_0_6_0_0_0': 1,   # N=7
    '15_0_6_0_1_0': 1,   # N=1
    '15_1_4_0_0_0': 2,   # N=54
    '15_1_4_0_1_0': 1,   # N=2
    '16_-1_3_0_0_0': 1,   # N=11
    '16_-1_4_0_0_0': 1,   # N=3
    '16_0_3_0_0_0': 105,   # N=3144
    '16_0_3_0_1_0': 1,   # N=6
    '16_0_3_1_1_0': 714,   # N=21413
    '16_0_4_0_0_0': 1157,   # N=34689
    '16_0_4_0_1_0': 225,   # N=6737
    '16_0_6_0_0_0': 1,   # N=3
    '16_0_6_0_1_0': 1,   # N=1
    '16_0_7_0_0_0': 2,   # N=37
    '16_1_3_0_0_0': 1,   # N=1
    '16_1_3_1_1_0': 2,   # N=31
    '16_1_4_0_0_0': 17,   # N=509
    '16_1_4_0_1_0': 5,   # N=141
    '17_0_4_0_0_0': 1415,   # N=42438
    '34_0_3_0_0_0': 1,   # N=7
    '34_0_3_1_1_0': 3,   # N=90
    '34_0_4_0_0_0': 5,   # N=130
    '34_0_4_0_1_0': 1,   # N=18
    '34_1_3_1_1_0': 1,   # N=3
    '34_1_4_0_0_0': 1,   # N=1
    '35_0_4_0_0_0': 296,   # N=8852
    '53_0_4_0_0_0': 48,   # N=1427
    '53_1_4_0_0_0': 1,   # N=1
    '53_1_4_0_1_0': 1,   # N=6
}


def allocate_codebook_budget(base_counts, budget):
    """Distribute a global code budget proportionally across intrinsic-key groups.

    Every known group receives at least one center. The total is exactly budget.
    The original CBDICT weights are preserved only as allocation weights.
    """
    if not isinstance(budget, int) or budget < len(base_counts):
        raise ValueError(f"codebook_size must be >= {len(base_counts)} (number of groups); got {budget}")
    if any(int(v) <= 0 for v in base_counts.values()):
        raise ValueError("CBDICT must contain positive counts")
    keys = list(base_counts)
    remaining = budget - len(keys)
    total = sum(int(v) for v in base_counts.values())
    shares = [remaining * int(base_counts[k]) / total for k in keys]
    allocations = [1 + int(x) for x in shares]
    leftover = budget - sum(allocations)
    order = sorted(range(len(keys)), key=lambda i: (-(shares[i] - int(shares[i])), i))
    for i in order[:leftover]:
        allocations[i] += 1
    result = dict(zip(keys, allocations))
    assert sum(result.values()) == budget
    return result



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_config(config_path, model_name, dataset):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config["global"]
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config["model_name"] = model_name
    return specific_config


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):
        raise ValueError(f"No such file or directory! {path}")


def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()

def get_logger(filename, console_log=False, log_level=logging.INFO):
    logger = logging.getLogger(f"logger_{filename}")  # unique per file
    logger.propagate = False
    logger.setLevel(log_level)

    # Add handlers only once
    if not logger.handlers:
        file_handler = logging.FileHandler(filename, mode="a")
        formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console_log:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger



def idx_split(idx, ratio, seed=0, train_or_infer=None):
    """
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    """
    set_seed(seed)
    n = len(idx)   # idx starts from 40
    cut = int(n * ratio)  # n 8000, cut 1600, ratio 0.2
    # print(f"n {n}, cut {cut}, ratio {ratio}") # n 8000, cut 1600, ratio 0.2
    if train_or_infer == "train":
        idx_idx_shuffle = torch.randperm(n)
        idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    elif train_or_infer == "infer":
        idx_idx_list = list(range(n))
        idx1_idx, idx2_idx = idx_idx_list[:cut], idx_idx_list[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2  # idx1 is test_ind


def graph_split(idx_train, idx_val, idx_test, rate, seed, train_or_infer):
    """
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)

        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed, train_or_infer)

    idx_obs = torch.cat([idx_train, idx_val])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1 : N1 + N2]
    obs_idx_test = idx_test

    # print(f"obs_idx_train {obs_idx_train}")
    # print(f"obs_idx_train {obs_idx_train.shape}")
    # print(f"obs_idx_val {obs_idx_val}")
    # print(f"obs_idx_val {obs_idx_val.shape}")
    # print(f"obs_idx_test {obs_idx_test}")
    # print(f"obs_idx_test {obs_idx_test.shape}")
    # print(f"idx_test_ind {idx_test_ind}")
    # print(f"idx_test_ind {idx_test_ind.shape}")
    idx_test_ind = torch.tensor(list(range(N1 + N2 + N2, N1 + N2 + N2 + N2 + 1)))
    return obs_idx_train, obs_idx_val, obs_idx_test, obs_idx_all, idx_test_ind


def get_evaluator(dataset):
    if dataset in CPF_data + NonHom_data + BGNN_data:

        def evaluator(out, labels):
            pred = out.argmax(1)
            return pred.eq(labels).float().mean().item()

    elif dataset in OGB_data:
        ogb_evaluator = Evaluator(dataset)

        def evaluator(out, labels):
            pred = out.argmax(1, keepdim=True)
            input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
            return ogb_evaluator.eval(input_dict)["acc"]

    else:
        raise ValueError("Unknown dataset")

    return evaluator


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator


def compute_min_cut_loss(g, out):
    out = out.to("cpu")
    g = g.to("cpu")
    S = out.exp()
    A = g.adj().to_dense()
    D = g.in_degrees().float().diag()
    print(S.device, A.device, D.device)
    min_cut = (
        torch.matmul(torch.matmul(S.transpose(1, 0), A), S).trace()
        / torch.matmul(torch.matmul(S.transpose(1, 0), D), S).trace()
    )
    return min_cut.item()


def feature_prop(feats, g, k):
    """
    Augment node feature by propagating the node features within k-hop neighborhood.
    The propagation is done in the SGC fashion, i.e. hop by hop and symmetrically normalized by node degrees.
    """
    assert feats.shape[0] == g.num_nodes()

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)

    # compute (D^-1/2 A D^-1/2)^k X
    for _ in range(k):
        feats = feats * norm
        g.ndata["h"] = feats
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        feats = g.ndata.pop("h")
        feats = feats * norm

    return feats



# ==============================================================================
# ORIGINAL MODULE: vq.py
# ==============================================================================
import torch.distributed as distributed
from einops import rearrange, repeat, pack, unpack
from torch import nn, einsum



def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x

from einops import rearrange, repeat

def l2norm(t, dim=-1, eps=1e-8):
    return F.normalize(t, dim=dim, eps=eps)

import torch

def noop(x):
    return x

def batched_bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    # x: [H, N] int64 on CUDA
    H, N = x.shape
    out = torch.zeros(H, minlength, device=x.device, dtype=torch.int64)
    one = torch.ones_like(x, dtype=torch.int64)
    out.scatter_add_(1, x, one)            # all CUDA if x is CUDA
    return out

import torch
from typing import Optional, Sequence, Dict, Tuple


def kmeans(
    samples: torch.Tensor,          # [H, N, D]
    num_clusters: int,
    use_cosine_sim: bool = False,
    all_reduce_fn = lambda x: None, # in-place sum for DDP (noop by default)
    eps: float = 1e-12,
    max_iters: int = 20,
    tol: float = 0.0,
    n_block: int = 131072,          # tile size over N (points)
    k_block: int = 4096,            # tile size over K (centers)
    element_names=None,             # optional labels for heads (e.g., ["C","N","O",...])
):
    """
    Lloyd K-Means w/ K-Means++ init, streaming/blocked (no [H,N,K] allocation).

    Returns:
        means:          [H, K, D]
        bins:           [H, K]        # counts per cluster
        used_per_head:  [H]           # number of non-empty clusters per head
        used_per_label: dict[str,int] # only if element_names provided, else None
    """
    import math
    import torch

    # ----------------------------
    # device / shape
    # ----------------------------
    samples = samples.to("cuda", non_blocking=True)
    H, N, D = samples.shape
    device, dtype = samples.device, samples.dtype

    if max_iters <= 0:
        base = 20
        extra = int(10 * max(0.0, math.log10(max(N, 1)) - 2.0))
        max_iters = min(100, base + extra)

    K = int(min(num_clusters, N))
    if K <= 0:
        raise ValueError("No samples to cluster.")

    if use_cosine_sim:
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)

    # ----------------------------
    # deterministic helpers
    # ----------------------------
    def _determinism_enabled() -> bool:
        # True if torch.use_deterministic_algorithms(True) has been set
        return torch.are_deterministic_algorithms_enabled()

    def _with_nondet_allowed(fn, *args, **kwargs):
        """
        Run fn with deterministic algorithms temporarily disabled (only if they are enabled),
        to avoid hard errors on ops like CUDA cumsum.
        """
        prev = _determinism_enabled()
        if not prev:
            return fn(*args, **kwargs)
        try:
            torch.use_deterministic_algorithms(False)
            return fn(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(True)

    # ----------------------------
    # KMeans++ init (blockwise, deterministic-safe)
    # ----------------------------
    @torch.no_grad()
    def kmeanspp_init_blockwise(
        X: torch.Tensor,                 # [H,N,D] on GPU
        K: int,
        cosine: bool = False,
        eps: float = 1e-12,
        sample_block_elems: int = 262_144,
        dtype_prob: torch.dtype = torch.float32,
        deterministic: str = "auto",     # "cpu_cumsum" | "gpu_cumsum" | "gpu_cumsum_nondet_ok" | "auto"
    ) -> torch.Tensor:
        """
        Memory-safe KMeans++ initializer with deterministic-safe sampling.

        Modes:
          - "cpu_cumsum": fully deterministic, does sampling cumsum/searchsorted on CPU.
          - "gpu_cumsum": uses GPU cumsum (will ERROR if deterministic_algorithms=True).
          - "gpu_cumsum_nondet_ok": uses GPU cumsum but temporarily disables determinism just for cumsum.
          - "auto": if determinism enabled -> cpu_cumsum else -> gpu_cumsum
        """
        assert deterministic in ("cpu_cumsum", "gpu_cumsum", "gpu_cumsum_nondet_ok", "auto")
        if deterministic == "auto":
            deterministic = "cpu_cumsum" if _determinism_enabled() else "gpu_cumsum"

        H, N, D = X.shape
        dev = X.device
        C = torch.empty((H, K, D), device=dev, dtype=X.dtype)

        if cosine:
            Xwork = torch.nn.functional.normalize(X, p=2, dim=-1)
            x2 = None
        else:
            Xwork = X
            x2 = (X ** 2).sum(-1).to(dtype_prob)  # [H,N] fp32

        # first seed per head
        idx0 = torch.randint(0, N, (H, 1), device=dev)
        C[:, 0, :] = X.gather(1, idx0.unsqueeze(-1).expand(H, 1, D)).squeeze(1)

        def dist_to_center(x_all: torch.Tensor, c_one: torch.Tensor) -> torch.Tensor:
            # x_all: [H,N,D], c_one: [H,D] -> [H,N] (dtype_prob)
            if cosine:
                d = (1.0 - (x_all * c_one.unsqueeze(1)).sum(-1)).clamp_min_(0)
            else:
                c2 = (c_one ** 2).sum(-1)                 # [H]
                xc = (x_all * c_one.unsqueeze(1)).sum(-1) # [H,N]
                d = (x2 + c2.unsqueeze(1) - 2.0 * xc).clamp_min_(0)
            return d.to(dtype_prob)

        # closest squared distance to chosen centers so far
        closest = dist_to_center(Xwork, C[:, 0, :])  # [H,N] fp32

        def _sample_block_gpu_cumsum(prob: torch.Tensor) -> torch.Tensor:
            """
            prob: [H,N] nonnegative, fp32
            Return idx: [H] long on GPU
            """
            totals = prob.sum(dim=1)                         # [H]
            zero_mask = totals <= 0
            u = torch.rand(H, device=dev, dtype=prob.dtype) * torch.clamp(totals, min=eps)

            idx_out = torch.empty(H, device=dev, dtype=torch.long)
            cum = torch.zeros(H, device=dev, dtype=prob.dtype)
            found = torch.zeros(H, device=dev, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = prob[:, start:end]                  # [H,B]
                block_sum = block.sum(dim=1)                # [H]

                target = (~found) & (cum + block_sum >= u)
                if target.any():
                    h_idx = target.nonzero(as_tuple=False).squeeze(1)
                    block_h = block[h_idx]                  # [Hsel,B]
                    need = (u[h_idx] - cum[h_idx]).unsqueeze(1)

                    # ---- THIS is the deterministic trouble-maker on CUDA ----
                    block_cum = block_h.cumsum(dim=1)
                    pos = torch.searchsorted(block_cum, need, right=False).squeeze(1)
                    pos = pos.clamp_max(block_h.size(1) - 1)

                    idx_out[h_idx] = start + pos
                    found[h_idx] = True

                nf = ~found
                if nf.any():
                    cum[nf] = cum[nf] + block_sum[nf]

                start = end
                if found.all():
                    break

            if (~found).any():
                idx_out[~found] = N - 1
            if zero_mask.any():
                idx_out[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device=dev)
            return idx_out

        def _sample_block_cpu_cumsum(prob: torch.Tensor) -> torch.Tensor:
            """
            Deterministic sampling on CPU:
            prob: [H,N] on GPU fp32 -> moves one head-block at a time as needed.
            Returns idx: [H] on GPU long
            """
            # Move whole prob to CPU can be expensive; but N is big and sampling happens K times.
            # We'll do it blockwise on CPU to keep memory manageable.
            totals = prob.sum(dim=1).detach().cpu()  # [H]
            zero_mask = (totals <= 0)
            # u on CPU for deterministic path
            u = (torch.rand(H, dtype=torch.float64) * torch.clamp(totals.to(torch.float64), min=float(eps))).to(torch.float64)

            idx_out_cpu = torch.empty(H, dtype=torch.long)
            cum = torch.zeros(H, dtype=torch.float64)
            found = torch.zeros(H, dtype=torch.bool)

            start = 0
            while start < N:
                end = min(start + sample_block_elems, N)
                block = prob[:, start:end].detach().cpu().to(torch.float64)  # [H,B]
                block_sum = block.sum(dim=1)                                 # [H]

                target = (~found) & (cum + block_sum >= u)
                if target.any():
                    h_idx = target.nonzero(as_tuple=False).squeeze(1)
                    block_h = block[h_idx]                                   # [Hsel,B]
                    need = (u[h_idx] - cum[h_idx]).unsqueeze(1)              # [Hsel,1]
                    block_cum = block_h.cumsum(dim=1)                        # CPU cumsum deterministic
                    pos = torch.searchsorted(block_cum, need, right=False).squeeze(1)
                    pos = torch.clamp(pos, max=block_h.size(1) - 1)
                    idx_out_cpu[h_idx] = start + pos
                    found[h_idx] = True

                nf = ~found
                if nf.any():
                    cum[nf] = cum[nf] + block_sum[nf]

                start = end
                if found.all():
                    break

            if (~found).any():
                idx_out_cpu[~found] = N - 1
            if zero_mask.any():
                # random but deterministic algorithms demand doesn't constrain RNG reproducibility unless you seed;
                # still fine. If you want fully repeatable, seed torch before calling kmeans.
                idx_out_cpu[zero_mask] = torch.randint(0, N, (int(zero_mask.sum()),), device="cpu")

            return idx_out_cpu.to(device=dev)

        # choose sampler
        def sample_idx(prob: torch.Tensor) -> torch.Tensor:
            if deterministic == "cpu_cumsum":
                return _sample_block_cpu_cumsum(prob)
            elif deterministic == "gpu_cumsum":
                return _sample_block_gpu_cumsum(prob)  # may error if deterministic_algorithms=True
            elif deterministic == "gpu_cumsum_nondet_ok":
                return _with_nondet_allowed(_sample_block_gpu_cumsum, prob)
            else:
                raise RuntimeError("unreachable")

        # pick K-1 more centers
        for k in range(1, K):
            # stabilized probs from closest distances
            prob = closest
            prob = (prob - prob.amin(dim=1, keepdim=True)).clamp_min_(0) + eps
            prob = prob * prob  # standard kmeans++ uses d^2 weighting

            idxk = sample_idx(prob)  # [H] long on GPU
            C[:, k, :] = X[torch.arange(H, device=dev), idxk, :]

            dk = dist_to_center(Xwork, C[:, k, :])  # [H,N]
            closest = torch.minimum(closest, dk)

        if cosine:
            C = torch.nn.functional.normalize(C, p=2, dim=-1)
        return C

    # ---- IMPORTANT CHANGE ----
    # If determinism is enabled, default to CPU sampling (no crash).
    # If you prefer "allow nondet only for cumsum", set deterministic="gpu_cumsum_nondet_ok".
    means = kmeanspp_init_blockwise(
        samples,
        K,
        cosine=use_cosine_sim,
        eps=eps,
        deterministic="gpu_cumsum_nondet_ok",
    )

    # ----------------------------
    # Lloyd steps (streaming/blocked)
    # ----------------------------
    def assign_pass(X, C):
        """
        Returns buckets [H,N] (long)
        Streaming over K tiles and N tiles to find argmin per point.
        """
        H, N, D = X.shape
        _, K, _ = C.shape
        buckets = torch.empty(H, N, device=device, dtype=torch.long)

        for h in range(H):
            if use_cosine_sim:
                best_val = torch.full((N,), -float("inf"), device=device, dtype=X.dtype)
            else:
                best_val = torch.full((N,), float("inf"), device=device, dtype=X.dtype)
            best_idx = torch.zeros((N,), device=device, dtype=torch.long)

            if not use_cosine_sim:
                x2_full = (X[h] ** 2).sum(-1)  # [N]

            for k0 in range(0, K, k_block):
                k1 = min(k0 + k_block, K)
                Ck = C[h, k0:k1]  # [kb,D]

                if use_cosine_sim:
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        sims = X[h, n0:n1] @ Ck.T            # [nb,kb]
                        vals, idxs = sims.max(dim=1)         # [nb]
                        update = vals > best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, idxs + k0, best_idx[n0:n1])
                else:
                    c2 = (Ck ** 2).sum(-1)                   # [kb]
                    for n0 in range(0, N, n_block):
                        n1 = min(n0 + n_block, N)
                        xc = X[h, n0:n1] @ Ck.T              # [nb,kb]
                        d2 = (x2_full[n0:n1].unsqueeze(1) + c2.unsqueeze(0) - 2 * xc).clamp_min_(0)
                        vals, idxs = d2.min(dim=1)           # [nb]
                        update = vals < best_val[n0:n1]
                        best_val[n0:n1] = torch.where(update, vals, best_val[n0:n1])
                        best_idx[n0:n1] = torch.where(update, idxs + k0, best_idx[n0:n1])

            buckets[h] = best_idx

        return buckets

    def update_pass(X, buckets, K, old_means):
        """
        Accumulates sums and counts in chunks of N.
        Returns:
          new_means [H,K,D], bins [H,K]
        """
        H, N, D = X.shape
        new_means = torch.zeros(H, K, D, device=device, dtype=X.dtype)
        bins = torch.zeros(H, K, device=device, dtype=torch.long)

        ones = None
        for h in range(H):
            for n0 in range(0, N, n_block):
                n1 = min(n0 + n_block, N)
                b = buckets[h, n0:n1]                       # [nb]
                x = X[h, n0:n1]                             # [nb,D]
                if ones is None or ones.numel() != b.numel():
                    ones = torch.ones_like(b, dtype=torch.long, device=device)
                bins[h].index_add_(0, b, ones)
                new_means[h].index_add_(0, b, x)

        all_reduce_fn(bins)
        all_reduce_fn(new_means)

        zero = (bins == 0)
        denom = bins.clamp_min(1).unsqueeze(-1)             # [H,K,1]
        new_means = new_means / denom

        # keep old center for empty bins
        new_means = torch.where(zero.unsqueeze(-1), old_means, new_means)

        if use_cosine_sim:
            new_means = torch.nn.functional.normalize(new_means, p=2, dim=-1)

        return new_means, bins

    prev_means = None
    for it in range(max_iters):
        buckets = assign_pass(samples, means)                 # [H,N]
        new_means, bins = update_pass(samples, buckets, K, means)
        if tol > 0.0:
            prev_means = means
        means = new_means
        if tol > 0.0:
            shift = (means - prev_means).pow(2).sum(-1).sqrt().mean()
            if float(shift) <= tol:
                break

    # final counts matched to final means
    buckets = assign_pass(samples, means)
    _, bins = update_pass(samples, buckets, K, means)

    used_per_head = (bins > 0).sum(dim=1)

    used_per_label = None
    if element_names is not None:
        if len(element_names) != H:
            raise ValueError(f"element_names must have length H={H}, got {len(element_names)}")
        used_per_label = {element_names[h]: int(used_per_head[h].item()) for h in range(H)}

    return means, bins, used_per_head, used_per_label



@torch.no_grad()
def spatial_ivf_kmeans(samples, num_clusters, *, nlist=64, fit_samples=200000,
                       coarse_batch=8192, fine_iters=5, block=8192,
                       use_cosine_sim=False, seed=0):
    """Spatial IVF -> local fine KMeans -> full-data Lloyd refinement.

    Input [N,D]; returns [1,K,D] centers and [1,K] counts, compatible with kmeans().
    The IVF regions are fixed; no global nearest-center search across regions.
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    import torch.nn.functional as F

    x = samples.detach().float()
    device = x.device
    N, D = x.shape
    K = min(int(num_clusters), N)
    if K <= 0:
        raise ValueError("Empty samples")
    if use_cosine_sim:
        x = F.normalize(x, dim=-1)
    nlist = min(int(nlist), K, N)
    fit_n = min(N, max(int(fit_samples), K, nlist))
    rng = torch.Generator(device='cpu').manual_seed(int(seed))
    # Sampling indices on CPU avoids a full GPU randperm of millions of items.
    if fit_n < N:
        idx = torch.randint(N, (fit_n,), generator=rng).to(device)
        fit = x.index_select(0, idx)
    else:
        fit = x
    fit_np = fit.cpu().numpy()
    coarse_km = MiniBatchKMeans(n_clusters=nlist, batch_size=max(coarse_batch, nlist),
                                n_init=1, random_state=seed, reassignment_ratio=0.01)
    coarse_km.fit(fit_np)
    coarse = torch.tensor(coarse_km.cluster_centers_, device=device, dtype=torch.float32)
    if use_cosine_sim:
        coarse = F.normalize(coarse, dim=-1)

    def assign(z, centers):
        out = torch.empty(z.shape[0], device=device, dtype=torch.long)
        c2 = (centers * centers).sum(dim=-1)
        for a in range(0, len(z), block):
            v = z[a:a+block]
            score = v @ centers.T
            if not use_cosine_sim:
                score = 2 * score - c2.unsqueeze(0)
            out[a:a+block] = score.argmax(dim=-1)
        return out

    # Count regions on all N points, not merely on the fitting sample.
    coarse_ids = assign(x, coarse)
    region_counts = torch.bincount(coarse_ids, minlength=nlist).cpu()
    # Only occupied regions can receive centers; reserve one per occupied region.
    occupied = torch.where(region_counts > 0)[0]
    weights = region_counts[occupied].double()
    budget = K - len(occupied)
    raw = weights / weights.sum() * budget
    alloc = torch.floor(raw).long() + 1
    rem = K - int(alloc.sum())
    if rem:
        alloc[torch.argsort(raw - torch.floor(raw), descending=True)[:rem]] += 1
    # A region cannot contain more unique centers than points.
    capacity = region_counts[occupied]
    excess = (alloc - capacity).clamp_min(0)
    alloc = torch.minimum(alloc, capacity)
    missing = int(excess.sum())
    while missing:
        free = capacity - alloc
        candidates = torch.where(free > 0)[0]
        if not len(candidates):
            raise RuntimeError('Not enough distinct points for codebook')
        take = min(missing, len(candidates))
        alloc[candidates[:take]] += 1
        missing -= take

    fit_ids = assign(fit, coarse).cpu()
    centers_parts = []
    region_to_slice = {}
    offset = 0
    for r, kr in zip(occupied.tolist(), alloc.tolist()):
        pool = fit[fit_ids.to(device) == r]
        if len(pool) < kr:
            # Add real samples from the full region when fit sample is too small.
            full_region = x[coarse_ids == r]
            pool = full_region
        if kr == 1:
            local = pool.mean(0, keepdim=True)
        else:
            km = MiniBatchKMeans(n_clusters=kr, batch_size=max(1024, kr),
                                 n_init=1, random_state=seed, reassignment_ratio=0.0)
            km.fit(pool.cpu().numpy())
            local = torch.tensor(km.cluster_centers_, device=device, dtype=torch.float32)
        if use_cosine_sim:
            local = F.normalize(local, dim=-1)
        centers_parts.append(local)
        region_to_slice[r] = (offset, offset + kr)
        offset += kr
    centers = torch.cat(centers_parts, dim=0)
    assert centers.shape == (K, D)
    counts = torch.zeros(K, device=device, dtype=torch.long)
    for iteration in range(int(fine_iters)):
        sums = torch.zeros_like(centers)
        counts.zero_()
        for r in occupied.tolist():
            start, end = region_to_slice[r]
            # Process only points assigned to this coarse spatial region.
            ix = torch.where(coarse_ids == r)[0]
            c = centers[start:end]
            c2 = (c*c).sum(-1)
            for a in range(0, len(ix), block):
                indices = ix[a:a+block]
                v = x[indices]
                score = v @ c.T
                if not use_cosine_sim:
                    score = 2*score - c2.unsqueeze(0)
                labels = score.argmax(-1) + start
                sums.index_add_(0, labels, v)
                counts.index_add_(0, labels, torch.ones_like(labels))
        used = counts > 0
        centers[used] = sums[used] / counts[used, None]
        if use_cosine_sim:
            centers = F.normalize(centers, dim=-1)
    # counts correspond to last assignment pass (as in ordinary Lloyd updates).
    return centers.unsqueeze(0), counts.unsqueeze(0)


def compact_clusters(means: torch.Tensor, bins: torch.Tensor, pad_to_max: bool = True):
    """
    Filter out empty clusters head-by-head.

    Args:
        means: [H, K, D]
        bins:  [H, K]
        pad_to_max: if True, returns padded tensors [H, K_used_max, ...];
                    if False, returns Python lists per head (ragged).

    Returns:
        If pad_to_max:
            used_means:  [H, K_used_max, D]
            used_bins:   [H, K_used_max]
            used_mask:   [H, K] boolean (where original clusters were used)
        Else:
            used_means_list: list of H tensors with shapes [K_h, D]
            used_bins_list:  list of H tensors with shapes [K_h]
            used_mask:       [H, K] boolean
    """
    H, K, D = means.shape
    used_mask = bins > 0

    if not pad_to_max:
        means_list, bins_list = [], []
        for h in range(H):
            m = means[h, used_mask[h]]
            b = bins[h, used_mask[h]]
            means_list.append(m)
            bins_list.append(b)
        return means_list, bins_list, used_mask

    max_used = int(used_mask.sum(dim=1).max().item()) if H > 0 else 0
    used_means = means.new_zeros((H, max_used, D))
    used_bins = bins.new_zeros((H, max_used), dtype=bins.dtype)
    for h in range(H):
        idx = used_mask[h].nonzero(as_tuple=False).squeeze(-1)
        k_h = idx.numel()
        if k_h > 0:
            used_means[h, :k_h] = means[h, idx]
            used_bins[h, :k_h] = bins[h, idx]
    return used_means, used_bins, used_mask


def batched_embedding(indices, embed):
    embed = embed.squeeze(0)              # (K, D)
    indices = indices.view(-1).long()     # Ensure shape (B,) and dtype long
    quantized = F.embedding(indices, embed)  # (B, D)
    return quantized.unsqueeze(0)         # (1, B, D)


import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, latent_dim, atom_feat_dim, margin=0.5, temperature=0.1, init_sigmoid_base=0.5):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(margin))
        self.temperature = temperature
        self.sigmoid_base = nn.Parameter(torch.tensor(init_sigmoid_base))
        self.layer_norm_z = nn.LayerNorm(latent_dim)
        self.layer_norm_atom = nn.LayerNorm(latent_dim)
        args = get_args()
        if args.dynamic_threshold:
            self.use_dynamic_threshold = True
        else:
            self.use_dynamic_threshold = False

    def sample_cap(self, t, max_n=200_000, with_replacement=False):
        """
        t: 1D/任意形状 Tensor（flattenして扱う場合は呼び出し側で調整）
        max_n: 取り出す最大サンプル数
        with_replacement: True なら重複あり（超省メモリ）
        """
        import torch
        n = t.numel()
        if n <= max_n:
            return t

        if with_replacement:
            # 重複あり：最小メモリ（長さ max_n のみ確保）
            idx = torch.randint(0, n, (max_n,), device=t.device)
            return t[idx]

        # 重複なし：GPU OOM回避のため CPU で randperm
        idx_cpu = torch.randperm(n, device='cpu')[:max_n]
        idx = idx_cpu.to(t.device, non_blocking=True)
        return t[idx]

    def forward(self, z, chunk, logger, codebook, key):
        import torch
        import torch.nn.functional as F

        device = z.device
        z = z.squeeze()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B, D = z.shape
        if B <= 1:
            return 0, 0, 0, 0, 0

        # ------------------------------------------------------------
        # 0) NORMALIZATION SWITCH (make geometry consistent everywhere)
        # ------------------------------------------------------------
        use_unit_norm = bool(getattr(self, "repel_use_unit_norm", True))
        # If you prefer to follow your VQ setting:
        # use_unit_norm = bool(getattr(self, "use_cosine_sim", False))

        if use_unit_norm:
            # keep gradients; eps avoids NaNs when norm ~ 0
            z_used = F.normalize(z, p=2, dim=-1, eps=1e-8)
            cb_used = F.normalize(codebook, p=2, dim=-1, eps=1e-8) if codebook is not None else None
        else:
            z_used = z
            cb_used = codebook

        N = z_used.shape[0]

        # ---- 0. Subsample for pdist to avoid O(N^2) explosion ----
        max_pdist_points = int(getattr(self, "max_pdist_points", 4096))

        if N > max_pdist_points:
            perm = torch.randperm(N, device=device)
            idx = perm[:max_pdist_points]
            z_for_pdist = z_used.index_select(0, idx)
        else:
            z_for_pdist = z_used

        # pairwise distances in the SAME geometry as everything else
        pdist_z = torch.pdist(z_for_pdist, p=2)  # 1D

        # （巨大時）サンプルを間引き
        sample = pdist_z
        if sample.numel() > 1_000_000:
            sample = self.sample_cap(sample, max_n=200_000)

        with torch.no_grad():
            dynamic_threshold = torch.quantile(sample, 0.10)
            lower_thresh = torch.quantile(sample, 0.25)
            upper_thresh = torch.quantile(sample, 0.75)
            center = (lower_thresh + upper_thresh) / 2

        # ---- ログ（負荷・転送を抑えて）----
        if chunk % 32 == 0:
            with torch.no_grad():
                s = sample
                if s.numel() > 100_000:
                    ridx = torch.randperm(s.numel(), device=s.device)[:100_000]
                    s = s.index_select(0, ridx)
                s_cpu = s.detach().to("cpu", dtype=torch.float32).flatten()
                hist = torch.histc(s_cpu, bins=10, min=0.0, max=1.0)
                vals = hist.tolist()
                # logger.info(f"[VQ_REPEL_HIST] key={key} {vals}")

        # ------------------------------------------------------------
        # 1) latent repel (blockwise) in the SAME geometry
        # ------------------------------------------------------------
        def latent_repel_mid_chunked(
            z_in,
            low,
            high,
            center,
            sigma=3.0,
            sharp=20.0,
            row_block=0,
            col_block=0,
            detach_weight=True,
            use_checkpoint=True,
            stream_backward=False,
            eps=1e-8,
        ):
            import torch
            import torch.utils.checkpoint as cp

            if (not z_in.requires_grad) or (not z_in.is_leaf and z_in.grad_fn is None):
                # encoder freeze / no_grad / detach のときは、この損失は意味がないので 0 にする
                return z_in.new_zeros(())
            assert not (use_checkpoint and stream_backward), \
                "checkpoint と streaming backward は同時に使えません。"

            B, D = z_in.shape
            device, dtype = z_in.device, z_in.dtype

            if self.training:
                assert z_in.requires_grad, "INPUT ERROR: z.requires_grad=False"

            band = float(abs(high - low))
            sigma_val = max(1e-4, 0.20 * band)
            sharp_val = float(sharp) if sharp is not None else 10.0

            low_t = torch.as_tensor(low, device=device, dtype=dtype)
            high_t = torch.as_tensor(high, device=device, dtype=dtype)
            center_t = torch.as_tensor(center, device=device, dtype=dtype)
            sigma_t = torch.as_tensor(sigma_val, device=device, dtype=dtype)

            if row_block <= 0 or row_block > B:
                row_block = B
            if col_block <= 0 or col_block > B:
                col_block = B

            def count_blocks(B, rb, cb):
                cnt = 0
                i = 0
                while i < B:
                    j = i
                    while j < B:
                        cnt += 1
                        j += cb
                    i += rb
                return max(cnt, 1)

            n_blocks_total = count_blocks(B, row_block, col_block)

            def block_loss(zi, zj, low_t, high_t, center_t, sigma_t):
                if self.training:
                    assert zi.requires_grad and zj.requires_grad

                if zi.numel() == 0 or zj.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

                # IMPORTANT: do distances in float32 for stability
                zi32 = zi.float()
                zj32 = zj.float()

                low32 = low_t.float()
                high32 = high_t.float()
                center32 = center_t.float()
                sigma32 = torch.clamp(sigma_t.float(), min=1e-6)

                d = torch.cdist(zi32, zj32, p=2)

                mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
                d = d[mask]
                if d.numel() == 0:
                    return zi.sum() * 0 + zj.sum() * 0

                x1 = (sharp_val * (d - low32)).clamp(-40, 40)
                x2 = (sharp_val * (high32 - d)).clamp(-40, 40)
                w = torch.sigmoid(x1) * torch.sigmoid(x2)
                if detach_weight:
                    w = w.detach()

                exp_arg = -((d - center32) ** 2) / (2 * sigma32 * sigma32)
                exp_arg = exp_arg.clamp(-60, 0)
                bell = torch.exp(exp_arg)

                out = (w * bell).sum() / w.sum().clamp_min(eps)
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                if self.training:
                    assert out.requires_grad
                return out

            def block_loss_ckpt(zi, zj, low_t, high_t, center_t, sigma_t):
                return block_loss(zi, zj, low_t, high_t, center_t, sigma_t)

            total = z_in.new_zeros(())
            i = 0
            while i < B:
                bi = min(row_block, B - i)
                zi = z_in[i:i + bi]

                j = i
                while j < B:
                    bj = min(col_block, B - j)
                    zj = z_in[j:j + bj]

                    if use_checkpoint:
                        lb = cp.checkpoint(
                            block_loss_ckpt,
                            zi, zj, low_t, high_t, center_t, sigma_t,
                            use_reentrant=False,
                        )
                    else:
                        lb = block_loss(zi, zj, low_t, high_t, center_t, sigma_t)

                    total = total + lb
                    j += col_block
                i += row_block

            out = total / n_blocks_total
            if self.training:
                assert out.requires_grad
            return out

        latent_repel_loss_mid = latent_repel_mid_chunked(
            z_used,
            low=lower_thresh, high=upper_thresh, center=center,
            sigma=1.0, sharp=20.0,
            row_block=1024, col_block=1024,
            detach_weight=True,
            use_checkpoint=True,
            stream_backward=False,
        )

        # ------------------------------------------------------------
        # 2) codebook repel (recommend normalize if use_unit_norm)
        # ------------------------------------------------------------
        def repel_codebooks_chunked(cb, sigma=1.0, block=4096):
            """
            cb: [K, D]
            returns avg exp(-d^2/(2*sigma^2)) over all pairs
            """
            K = cb.size(0)
            sum_val = cb.new_zeros(())
            cnt = 0

            for i in range(0, K, block):
                ci = cb[i:i + block]

                if ci.size(0) >= 2:
                    dij = torch.pdist(ci.float(), p=2)
                    if dij.numel() > 0:
                        li = torch.exp(-dij.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + li.sum()
                        cnt += li.numel()
                    del dij

                for j in range(i + block, K, block):
                    cj = cb[j:j + block]
                    if ci.numel() > 0 and cj.numel() > 0:
                        d = torch.cdist(ci.float(), cj.float(), p=2)
                        lj = torch.exp(-d.pow(2) / (2 * sigma ** 2))
                        sum_val = sum_val + lj.sum()
                        cnt += lj.numel()
                        del d, lj
                    del cj
                del ci

            if cnt == 0:
                return sum_val
            return sum_val / cnt

        cb_loss = repel_codebooks_chunked(cb_used) if cb_used is not None else z_used.new_zeros(())

        # ------------------------------------------------------------
        # 3) auxiliary terms (same geometry)
        # ------------------------------------------------------------
        repel_from_2 = torch.exp(- (pdist_z - 2.0) ** 2 / (2 * 3.0 ** 2)).mean()

        def repel_from_zero_1d(d, margin):
            return torch.relu(margin - d).mean()

        latent_repel_loss = repel_from_zero_1d(pdist_z, lower_thresh) + cb_loss

        repel_weight = 1.0
        final_loss = repel_weight * latent_repel_loss_mid
        neg_loss = z.new_tensor(1.0)

        return final_loss, neg_loss, repel_from_2, cb_loss, latent_repel_loss


import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange

# 必要なら
# from utils import CORE_ELEMENTS
# from your_kmeans_module import kmeans
# from your_sampling_module import batched_sample_vectors, sample_vectors_distributed
# from your_distributed_module import distributed, noop
# from your_config_module import get_args
# from your_cb_dict import CBDICT

import re
import torch
import torch.nn as nn
from einops import rearrange

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=True,
        kmeans_iters=100,
        sync_kmeans=True,
        decay=0.1,
        eps=1e-5,
        threshold_ema_dead_code=2,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0,
    ):
        super().__init__()

        self.decay = float(decay)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        args = get_args()
        self.epoch_at_mode_shift = args.epoch_at_mode_shift
        self.train_or_infer = args.train_or_infer
        self.use_checkpoint = args.use_checkpoint

        # Interpret --codebook_size as the TOTAL initial number of centers.
        # CBDICT supplies only relative weights, not absolute center counts.
        self.cb_dict = allocate_codebook_budget(CBDICT, int(codebook_size))
        self.spatial_ivf = bool(args.spatial_ivf)
        self.ivf_min_samples = int(args.ivf_min_samples)
        self.ivf_nlist = int(args.ivf_nlist)
        self.ivf_fit_samples = int(args.ivf_fit_samples)
        self.ivf_fine_iters = int(args.ivf_fine_iters)
        print(f"[CODEBOOK-BUDGET] requested={codebook_size} "
              f"allocated={sum(self.cb_dict.values())} groups={len(self.cb_dict)}")

        assert not (
            use_ddp and num_codebooks > 1 and kmeans_init
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        )
        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and sync_kmeans else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # kmeans 済みかどうか
        self.register_buffer("initted", torch.tensor([not kmeans_init], dtype=torch.bool))

        # --- cluster_size / embed_avg (float32 固定) ---
        for elem in self.cb_dict.keys():
            K_e = int(self.cb_dict[elem])
            elem_str = str(elem)
            self.register_buffer(
                f"cluster_size_{elem_str}",
                torch.zeros(K_e, dtype=torch.float32),
            )
            self.register_buffer(
                f"embed_avg_{elem_str}",
                torch.zeros(K_e, dim, dtype=torch.float32),
            )

        self.learnable_codebook = learnable_codebook

        # element ごとのコードブック本体（ParameterDict のキーは「安全キー」に変換）
        self.embed = nn.ParameterDict()
        self.key_to_safe = {}  # original_key(str) -> safe_key(str)
        self.safe_to_key = {}  # safe_key(str) -> original_key(str)

        for key in self.cb_dict.keys():
            orig = str(key)
            K_e = int(self.cb_dict[key])
            safe = self._get_or_create_safe_key(orig, K_e, dim, device="cpu")
        self.ss_max_total_latent_count = args.ss_max_total_latent_count
        self.latent_size_sum = 0
        self.embed_ind_dict = {}
        self.quantize_dict = {}

    # ------------------------------------------------------------------
    # key まわりのユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_key(k: str) -> str:
        """ParameterDict 用の安全なキーに変換（属性名として有効な文字列）。"""
        s = re.sub(r"[^0-9A-Za-z_]", "_", k)
        if not s or not (s[0].isalpha() or s[0] == "_"):
            s = "k_" + s
        return s

    def ensure_from_state_dict(self, state_dict, prefix: str = ""):
        """
        state_dict に含まれる codebook 関連の buffer/parameter を事前に登録して、
        load_state_dict(strict=True) が通るようにする。

        prefix 例:
          - model 側で呼ぶなら: prefix="vq._codebook."
          - EuclideanCodebook 自身に渡すなら: prefix=""（推奨）
        """
        import torch
        import re
        from torch import nn

        # 1) cluster_size_*, embed_avg_* から orig key を拾って buffers を作る
        cs_pat = re.compile(rf"^{re.escape(prefix)}cluster_size_(.+)$")
        ea_pat = re.compile(rf"^{re.escape(prefix)}embed_avg_(.+)$")

        # まず embed_avg の D を拾いやすいので embed_avg を優先
        for k, v in state_dict.items():
            m = ea_pat.match(k)
            if not m:
                continue
            orig = m.group(1)  # 例: "6_-1_3_0_0_0"
            if not torch.is_tensor(v):
                continue
            K, D = int(v.shape[0]), int(v.shape[1])

            # buffers (orig 名)
            buf_cs = f"cluster_size_{orig}"
            buf_ea = f"embed_avg_{orig}"

            if not hasattr(self, buf_cs):
                self.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))
            if not hasattr(self, buf_ea):
                self.register_buffer(buf_ea, torch.zeros((K, D), dtype=torch.float32))

            # embed parameter (safe 名)
            safe = self._get_or_create_safe_key(orig, K_e=K, D=D, device="cpu")
            # ここで self.embed[safe] が無ければ作られる

        # embed_avg が無いケース向けに cluster_size だけでも拾う
        for k, v in state_dict.items():
            m = cs_pat.match(k)
            if not m:
                continue
            orig = m.group(1)
            if not torch.is_tensor(v):
                continue
            K = int(v.shape[0])

            buf_cs = f"cluster_size_{orig}"
            if not hasattr(self, buf_cs):
                self.register_buffer(buf_cs, torch.zeros((K,), dtype=torch.float32))

            # embed_avg が state に無い場合、D が不明なので embed はここでは作らない
            # （embed 側 key が state にあれば後段で作れる）

        # 2) embed.<safe> から safe key を拾って足りない Parameter を作る
        #    （orig が逆引きできない場合もあるので、とりあえず safe=orig 扱いで作る）
        emb_pat = re.compile(rf"^{re.escape(prefix)}embed\.(.+)$")
        for k, v in state_dict.items():
            m = emb_pat.match(k)
            if not m:
                continue
            safe = m.group(1)  # 例: "k_6__1_3_0_0_0"
            if not torch.is_tensor(v):
                continue

            # v の shape は [K, D] を想定（あなたの実装）
            if v.ndim != 2:
                continue
            K, D = int(v.shape[0]), int(v.shape[1])

            # safe をすでに持っていなければ Parameter を作る
            if safe not in self.embed:
                init = torch.randn((K, D), device="cpu") * 0.01
                self.embed[safe] = nn.Parameter(init, requires_grad=True)

            # mapping も最低限埋める（orig が分からなければ safe を orig 扱いに）
            if safe not in self.safe_to_key:
                self.safe_to_key[safe] = safe
            if self.safe_to_key[safe] not in self.key_to_safe:
                self.key_to_safe[self.safe_to_key[safe]] = safe

    def _get_or_create_safe_key(self, skey: str, K_e=None, D=None, device=None) -> str:
        """
        original key (skey) から safe_key を取得。
        必要なら Parameter を作成（K_e, D, device が与えられている場合）。
        """
        skey = str(skey)
        if skey in self.key_to_safe:
            safe = self.key_to_safe[skey]
        else:
            safe = self._safe_key(skey)
            self.key_to_safe[skey] = safe
            self.safe_to_key[safe] = skey

        if safe not in self.embed and K_e is not None and D is not None and device is not None:
            init = torch.randn(K_e, D, device=device) * 0.01
            self.embed[safe] = nn.Parameter(init, requires_grad=True)

        return safe

    # ------------------------------------------------------------------
    # ユーティリティ / 初期化系
    # ------------------------------------------------------------------
    def reset_kmeans(self):
        self.initted.data.fill_(False)

    def copy_codebook_(self, embed: torch.Tensor, init: torch.Tensor, fill="data", data=None):
        """
        embed: [K, D] or [D, K] current codebook
        init : [K_used, D] or [D, K_used] initializer
        fill : "data" | "repeat" | "randn"
        data : [N, D] latents to sample from if fill=="data"
        """
        def to_KD(t, D_expected):
            if t.shape[-1] == D_expected:
                return t  # [K_used, D]
            if t.shape[0] == D_expected:
                return t.t().contiguous()
            raise ValueError(f"init shape {tuple(t.shape)} not compatible with D={D_expected}")

        if embed.shape[-1] < embed.shape[0]:  # [K, D] とみなす
            K, D = embed.shape
            initKD = to_KD(init, D)
            K_used = initKD.shape[0]
            n = min(K, K_used)
            embed[:n].copy_(initKD[:n])

            if K > n:
                if fill == "data" and data is not None and data.numel() > 0:
                    idx = torch.randint(0, data.size(0), (K - n,), device=embed.device)
                    embed[n:K].copy_(data[idx])
                elif fill == "repeat" and n > 0:
                    reps = (K - n + n - 1) // n
                    embed[n:K].copy_(initKD[:n].repeat((reps, 1))[: K - n])
                elif fill == "randn":
                    embed[n:K].normal_(0, 1e-3)
                else:
                    embed[n:K].copy_(embed[:1])
        else:  # [D, K]
            D, K = embed.shape
            initKD = to_KD(init, D)
            initDK = initKD.t().contiguous()  # [D, K_used]
            n = min(K, initDK.shape[1])
            embed[:, :n].copy_(initDK[:, :n])
            if K > n:
                if fill == "randn":
                    embed[:, n:K].normal_(0, 1e-3)
                else:
                    embed[:, n:K].copy_(embed[:, :1])


    @staticmethod
    def silhouette_score_torch(
        X,
        labels,
        row_block: int = 8192,
        device: str | torch.device | None = None,
    ) -> float:
        """
        Compute Silhouette score in (mini)blocks, on GPU if available.

        Parameters
        ----------
        X : Tensor or nn.Module or nn.ParameterDict
            - Tensor: shape [N, D]
            - nn.Module: must have `.weight` or `.embed` tensor
            - nn.ParameterDict: values are [K, D] parameters → concatenated to [N, D]
        labels : Tensor
            Cluster labels of shape [N] (int).
        row_block : int
            Block size along N for pairwise distance computation.
        device : str or torch.device or None
            Target device. If None: "cuda" if available else "cpu".
        """

        # ---- 0. 正しいテンソル X を取り出す ------------------------------------
        # ParameterDict (or類似) の場合：中身を縦に concat
        if isinstance(X, nn.ParameterDict):
            # 各 value が [K, D] を想定
            X = torch.cat([p.view(p.shape[0], -1) for p in X.values()], dim=0)  # [N, D]

        # nn.Module の場合：.embed または .weight を拾う
        elif isinstance(X, nn.Module):
            weight = getattr(X, "weight", None)
            embed = getattr(X, "embed", None)
            if embed is not None:
                X = embed
            elif weight is not None:
                X = weight
            else:
                raise TypeError("X is a module without .embed/.weight tensor.")

        # ここまで来たら X は Tensor のはず
        if not torch.is_tensor(X):
            raise TypeError(
                f"X must be a Tensor or Module/ParameterDict of Tensors, but got {type(X)}"
            )

        # ---- 1. デバイス決定 & 型整形 -----------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # X: float32, labels: long に統一
        X = X.detach().to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.detach().to(device=device, dtype=torch.long, non_blocking=True)

        # 余計な次元を削る (e.g. [N, D, 1] → [N, D])
        X = X.squeeze()
        labels = labels.squeeze()

        # N チェック
        if X.ndim == 1:
            X = X.unsqueeze(1)  # [N] → [N, 1]

        N = X.shape[0]
        if N <= 1:
            return 0.0

        # ---- 2. ラベルを 0..K-1 にリマップ（空クラスタがあっても無視される） ----
        uniq, inv = labels.unique(sorted=True, return_inverse=True)  # inv: [N] in 0..K-1
        K = uniq.numel()
        if K <= 1:
            return 0.0

        counts = torch.bincount(inv, minlength=K).to(device)  # >0 by construction
        counts_f = counts.float()

        # ---- 3. ブロック毎に Silhouette を計算 --------------------------------
        sil_sum = 0.0
        processed = 0

        for start in range(0, N, row_block):
            end = min(start + row_block, N)
            B = end - start

            # このブロックのサンプル
            Xb = X[start:end]  # [B, D]

            # [B, N] の距離行列
            d_block = torch.cdist(Xb, X, p=2)  # Euclid

            # クラスタごとの距離総和を計算
            inv_index = inv.view(1, N).expand(B, N)  # [B, N]
            sums_per_cluster = torch.zeros(B, K, device=device, dtype=d_block.dtype)
            sums_per_cluster.scatter_add_(1, inv_index, d_block)  # [B, K]

            # クラスタごとの平均距離
            means_per_cluster = sums_per_cluster / counts_f.view(1, K)  # [B, K]

            # ----- a(i): 同じクラスタ内の平均距離（自分自身を除外） -------------
            k_block = inv[start:end]  # [B]
            a_counts = counts[k_block] - 1  # 自分自身を除いた個数

            # とりあえず "含自分" の平均距離
            a_mean = means_per_cluster.gather(1, k_block[:, None]).squeeze(1)  # [B]

            # クラスタサイズが 1 の場合は 0、>1 の場合だけ補正係数を載せる
            scale = torch.zeros_like(a_counts, dtype=a_mean.dtype)
            mask_gt1 = a_counts > 0
            if mask_gt1.any():
                # n / (n - 1) で「自分自身 (距離0) を除いた平均」に補正
                n = counts[k_block][mask_gt1].float()
                scale[mask_gt1] = n / (n - 1.0)
            a_i = a_mean * scale  # [B]; クラスタサイズ 1 のものは 0 のまま

            # ----- b(i): 他クラスタの平均距離の最小値 ---------------------------
            # 自分のクラスタの列は無限大にして除外
            means_per_cluster.scatter_(1, k_block[:, None], float("inf"))
            b_i, _ = means_per_cluster.min(dim=1)  # [B]

            # ----- Silhouette: (b - a) / max(a, b) -------------------------------
            denom = torch.maximum(a_i, b_i)
            sil_block = (b_i - a_i) / denom
            sil_block = torch.nan_to_num(
                sil_block, nan=0.0, posinf=0.0, neginf=0.0
            )

            sil_sum += sil_block.sum().item()
            processed += B

            # メモリ解放
            del Xb, d_block, sums_per_cluster, means_per_cluster, a_mean, a_i, b_i, sil_block

        return float(sil_sum / max(processed, 1))

    import math
    import torch

    @torch.jit.ignore
    def compute_cluster_metrics(self, y: torch.Tensor, K_req: int = None, topk: int = 5):
        """
        y: int64 labels, shape [N] (already compressed to 0..nuniq-1 recommended)
        Returns:
          N, K_req, K_eff, max_frac, topk_frac, entropy, perplexity,
          singleton_ratio (clusters), singleton_point_ratio (points)
        """
        # y must be 1D long
        if y is None or y.numel() == 0:
            return {
                "N": 0,
                "K_req": int(K_req) if K_req is not None else None,
                "K_eff": 0,
                "max_frac": 0.0,
                "topk_frac": 0.0,
                "entropy": 0.0,
                "perplexity": 1.0,
                "singleton_ratio": 0.0,
                "singleton_point_ratio": 0.0,
            }

        y = y.long().view(-1)
        N = int(y.numel())

        # compress labels to 0..nuniq-1 (robust even if already compressed)
        _, y = torch.unique(y, return_inverse=True)

        bc = torch.bincount(y)  # shape [nuniq]
        nz_mask = (bc > 0)
        bc_nz = bc[nz_mask]
        K_eff = int(bc_nz.numel())

        # fractions
        bc_float = bc_nz.float()
        denom = float(max(1, N))
        p = (bc_float / denom).clamp_min(0.0)  # sum to 1 over active clusters

        # max_frac
        max_frac = float(p.max().item()) if K_eff > 0 else 0.0

        # topk_frac
        k = int(min(max(1, topk), K_eff)) if K_eff > 0 else 1
        if K_eff > 0:
            topk_vals = torch.topk(p, k=k, largest=True).values
            topk_frac = float(topk_vals.sum().item())
        else:
            topk_frac = 0.0

        # entropy (natural log) and perplexity
        # H = -sum p log p
        # perplexity = exp(H)
        import math
        if K_eff > 0:
            H = float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())
            ppl = float(math.exp(H))
        else:
            H, ppl = 0.0, 1.0

        # singleton ratios
        singletons = (bc_nz == 1).sum()
        singleton_ratio = float((singletons.float() / max(1.0, float(K_eff))).item())  # clusters
        singleton_point_ratio = float((bc_nz[bc_nz == 1].sum().float() / denom).item())  # points

        out = {
            "N": N,
            "K_req": int(K_req) if K_req is not None else None,
            "K_eff": K_eff,
            "max_frac": max_frac,
            "topk_frac": topk_frac,
            "entropy": H,
            "perplexity": ppl,
            "singleton_ratio": singleton_ratio,
            "singleton_point_ratio": singleton_point_ratio,
        }
        return out

    @torch.jit.ignore
    def init_embed_(self, data, logger, epoch, mask_dict=None, use_cosine_sim: bool = False):
        """
        Initialize per-element codebooks (absolute K from self.cb_dict).

        Option A alignment:
          - store centers as unit-norm at init
          - init EMA buffers consistent with unit-norm centers
          - cluster metrics computed in the same unit-norm space (cosine assignment)
        """
        import os
        import torch
        import torch.nn as nn

        assert mask_dict is not None, "mask_dict is required"
        device = data.device
        D = int(data.shape[-1])

        # -------------------------
        # helpers
        # -------------------------
        def _get_from_dict(d, k):
            if k in d:
                return d[k]
            return d.get(str(k), None)

        def _get_absK(d, k):
            v = _get_from_dict(d, k)
            return None if v is None else int(v)

        def _safe_l2norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
            den = t.norm(dim=-1, keepdim=True).clamp_min(eps)
            t = t / den
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        def _stats(x_nd: torch.Tensor):
            """Robust data stats for padding. Returns (mu, sigma) or None."""
            if x_nd is None or x_nd.numel() == 0:
                return None
            mu = x_nd.mean(dim=0)
            with torch.no_grad():
                devv = (x_nd - mu).pow(2).sum(dim=1).sqrt()
                sigma = torch.quantile(devv, 0.5)
            return (mu, sigma)

        def _pad_to_K(means_1kd, counts_1k, K_req: int, data_stats=None):
            """
            means_1kd: [1, K_run, D]
            counts_1k: [1, K_run]
            return:
              means_kd: [K_req, D]
              counts_k: [K_req]
            """
            H, K_run, Dd = means_1kd.shape
            assert H == 1 and Dd == D

            means_kd = means_1kd[0]  # (K_run, D)
            counts_k = counts_1k[0]  # (K_run,)

            if K_req <= K_run:
                return means_kd[:K_req].contiguous(), counts_k[:K_req].contiguous()

            K_pad = K_req - K_run
            out_means = torch.empty((K_req, D), device=means_kd.device, dtype=means_kd.dtype)
            out_counts = torch.zeros((K_req,), device=counts_k.device, dtype=counts_k.dtype)

            out_means[:K_run].copy_(means_kd)
            out_counts[:K_run].copy_(counts_k)

            if data_stats is not None:
                mu, sigma = data_stats
                noise = torch.randn((K_pad, D), device=means_kd.device, dtype=means_kd.dtype) * (0.01 * sigma + 1e-6)
                out_means[K_run:] = mu + noise
            else:
                out_means[K_run:].zero_()

            return out_means, out_counts

        def _ensure_param(safe: str, shape):
            """Ensure self.embed[safe] exists with exact shape (recreate if mismatch)."""
            if (safe not in self.embed) or (tuple(self.embed[safe].shape) != tuple(shape)):
                self.embed[safe] = nn.Parameter(
                    torch.zeros(shape, device=device, dtype=torch.float32),
                    requires_grad=True,
                )

        def _safe_var_mean(x: torch.Tensor):
            """Mean variance over dim=0, never NaN for N==1."""
            if x is None or x.numel() == 0:
                return float("nan")
            if x.shape[0] < 2:
                return 0.0
            return x.float().var(dim=0, correction=0).mean().item()

        def _safe_std_1d(x: torch.Tensor):
            """Std over a 1D tensor, never NaN for len==1."""
            if x is None or x.numel() == 0:
                return float("nan")
            if x.numel() < 2:
                return 0.0
            return x.float().std(correction=0).item()

        # -------------------------
        # main loop
        # -------------------------
        for raw_key in sorted(mask_dict.keys(), key=lambda x: str(x)):
            skey = str(raw_key)

            idx = _get_from_dict(mask_dict, raw_key)
            if idx is None:
                idx = _get_from_dict(mask_dict, skey)
            if idx is None:
                continue

            masked = data[0][idx]  # (N_i, D)
            N_i = int(masked.shape[0])
            if N_i == 0:
                continue

            # optional dump for UMAP/debug
            if skey == "6_0_3_1_1_0":
                os.makedirs("dumps", exist_ok=True)
                path = f"dumps/masked_latents_epoch{epoch}_{skey}.pt"
                torch.save(
                    {
                        "epoch": int(epoch) if epoch is not None else None,
                        "key": skey,
                        "masked_latents": masked.detach().to("cpu", dtype=torch.float16),
                        "N": int(masked.shape[0]),
                        "D": int(masked.shape[1]),
                    },
                    path,
                )

            # K settings
            K_req = _get_absK(self.cb_dict, skey)
            if K_req is None or K_req <= 0:
                K_req = 1
            K_run = min(K_req, N_i)

            # safe key and buffer names
            safe = self._get_or_create_safe_key(skey, K_req, D, device=device)
            buf_name_cs = f"cluster_size_{skey}"
            buf_name_ea = f"embed_avg_{skey}"

            # -------------------------
            # 1) Initialize ONLY at epoch 1
            #    epoch>1: if missing -> hard error
            # -------------------------
            missing_param = (safe not in self.embed)
            missing_bufs = (not hasattr(self, buf_name_cs)) or (not hasattr(self, buf_name_ea))

            if epoch == 1:
                need_init = True
            else:
                if missing_param or missing_bufs:
                    raise RuntimeError(
                        f"[init_embed_] missing at epoch={epoch} key={skey} "
                        f"missing_param={missing_param} missing_bufs={missing_bufs} safe={safe}"
                    )
                need_init = False

            if need_init:
                samples = masked.to(device).float()
                samples = torch.nn.functional.normalize(samples, dim=-1)  # <-- force L2=1
                norms = samples.norm(dim=-1)
                print("post_norm mean/std:", norms.mean().item(), norms.std().item())

                # and pass samples.unsqueeze(0) into kmeans
                print(
                    f"[KMEANS-CALL] epoch={epoch} skey={skey} use_cosine_sim={use_cosine_sim} "
                    f"samples_norm_mean={norms.mean().item():.4f} "
                    f"samples_norm_std={_safe_std_1d(norms.reshape(-1)):.4f} "
                    f"samples_var={samples.float().var(dim=-1, correction=0).mean().item():.4e}"
                )

                if self.spatial_ivf and N_i >= self.ivf_min_samples and K_run > 1:
                    logger.info(f"[IVF] key={skey} N={N_i} K={K_run} "
                                f"nlist={self.ivf_nlist} fit={self.ivf_fit_samples}")
                    means_1kd, counts_1k = spatial_ivf_kmeans(
                        samples, num_clusters=K_run, nlist=self.ivf_nlist,
                        fit_samples=self.ivf_fit_samples,
                        fine_iters=self.ivf_fine_iters,
                        use_cosine_sim=use_cosine_sim,
                        seed=int(getattr(self, 'ivf_seed', 0)),
                    )
                else:
                    means_1kd, counts_1k, *_ = kmeans(
                        samples.unsqueeze(0),
                        num_clusters=K_run,
                        use_cosine_sim=use_cosine_sim,
                    )

                ds = _stats(masked)
                means_kd, counts_k = _pad_to_K(means_1kd, counts_1k, K_req, data_stats=ds)

                # -------------------------
                # Option A: normalize centers at init (CRITICAL)
                # -------------------------
                means_kd = _safe_l2norm(means_kd.detach().to(device=device, dtype=torch.float32))

                # ensure parameter exists with correct shape, then copy
                _ensure_param(safe, (K_req, D))
                self.embed[safe].data.copy_(means_kd.to(device=device, dtype=self.embed[safe].dtype))

                # init EMA buffers (float32): ea consistent with normalized means
                cs = counts_k.to(device=device, dtype=torch.float32)  # (K_req,)
                ea = means_kd.to(device=device, dtype=torch.float32) * cs.view(-1, 1)  # (K_req, D)

                # replace buffers safely
                if hasattr(self, buf_name_cs):
                    delattr(self, buf_name_cs)
                if hasattr(self, buf_name_ea):
                    delattr(self, buf_name_ea)
                self.register_buffer(buf_name_cs, cs)
                self.register_buffer(buf_name_ea, ea)

                nz = int((cs > 0).sum().item())
                logger.info(f"[init_embed_] Z={skey} N={N_i} K_req={K_req} K_run={K_run} K_used={nz}/{K_req}")

            # -------------------------
            # 2) Cluster metrics every epoch (Option A-consistent)
            # -------------------------
            with torch.no_grad():
                centers_clst = self.embed[safe].detach()  # (K_req, D)

                # active count is just for logging/diagnostics
                cs_now = getattr(self, buf_name_cs, None)
                active = (cs_now > 0) if cs_now is not None else None
                K_active = int(active.sum().item()) if active is not None else -1

                max_n = int(getattr(self, "clst_max_n", 20000))

                # subsample X first (avoid full argmin on huge N)
                if masked.shape[0] > max_n:
                    perm = torch.randperm(masked.shape[0], device=masked.device)[:max_n]
                    x_eval = masked[perm]
                else:
                    x_eval = masked

                # Option A: normalize BOTH before argmax_sim
                x_eval_n = _safe_l2norm(x_eval.float())
                c_n = _safe_l2norm(centers_clst.float())

                # diagnostics
                def diag(tag: str, X: torch.Tensor, centers: torch.Tensor):
                    cn = centers.float().norm(dim=-1)
                    print(
                        f"[DIAG]{tag} "
                        f"X shape={tuple(X.shape)} centers shape={tuple(centers.shape)} "
                        f"X finite={torch.isfinite(X).all().item()} "
                        f"C finite={torch.isfinite(centers).all().item()} "
                        f"X var={_safe_var_mean(X):.3e} "
                        f"C var={_safe_var_mean(centers):.3e} "
                        f"C mean-norm={cn.mean().item():.3e} "
                        f"C std-norm={_safe_std_1d(cn):.3e}"
                    )

                diag(" clst", x_eval_n, c_n)

                # assign labels using ALL centers (K_req) in unit-norm space
                y_eval = self.argmax_sim_blockwise(x_eval_n, c_n, k_block=1024)

                # compute metrics from labels only
                metrics = self.compute_cluster_metrics(y_eval, K_req=K_req, topk=5)

                logger.info(
                    f"[CLST] key={skey} epoch={epoch} "
                    f"N={metrics['N']} K_req={metrics['K_req']} K_eff={metrics['K_eff']} "
                    f"max_frac={metrics['max_frac']:.4f} top5_frac={metrics['topk_frac']:.4f} "
                    f"H={metrics['entropy']:.4f} ppl={metrics['perplexity']:.2f} "
                    f"singleton_ratio={metrics['singleton_ratio']:.4f} "
                    f"singleton_point_ratio={metrics['singleton_point_ratio']:.4f} "
                    f"(K_active={K_active})"
                )

    def _normalize_mask_dict(self, mask_dict, device=None):
        import numpy as np

        if mask_dict is None:
            return None
        norm = {}
        for k, v in mask_dict.items():
            k_int = int(k) if isinstance(k, str) and k.isdigit() else k
            k_str = str(k_int)

            if isinstance(v, (list, tuple, np.ndarray)):
                v = torch.as_tensor(v, dtype=torch.long, device=device)
            elif isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                v = torch.nonzero(v.flatten(), as_tuple=False).flatten().long().to(device)

            norm[k_int] = v
            norm[k_str] = v
        return norm

    import torch

    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def argmax_sim_blockwise(self, x: torch.Tensor, code: torch.Tensor, k_block: int = 1024, eps: float = 1e-12):
        """
        x:    [N, D]
        code: [K, D]
        return best_idx: [N]  (argmax dot similarity)

        A方針なら、呼び出し側で x/code を unit-norm に揃えるのが基本。
        ただし安全のためここでも正規化オプションを入れたいなら後述。
        """
        x_f = x.float()
        c_f = code.float()

        # 早期診断（必要ならコメントアウト）
        if not torch.isfinite(x_f).all():
            bad = (~torch.isfinite(x_f)).sum().item()
            print(f"[argmax] non-finite x: {bad}")
        if not torch.isfinite(c_f).all():
            bad = (~torch.isfinite(c_f)).sum().item()
            print(f"[argmax] non-finite code: {bad}")

        # 念のため：ゼロノルムがあると dot が全部0になったりするのでチェック
        # （Aなら本来あり得ないはず）
        x_norm = x_f.norm(dim=1)
        c_norm = c_f.norm(dim=1)
        if (x_norm <= 0).any():
            print(f"[argmax] WARNING: {(x_norm <= 0).sum().item()} zero-norm x rows.")
        if (c_norm <= 0).any():
            print(f"[argmax] WARNING: {(c_norm <= 0).sum().item()} zero-norm code rows.")

        best_val = torch.full((x_f.size(0),), -float("inf"), device=x.device, dtype=torch.float32)
        best_idx = torch.zeros((x_f.size(0),), device=x.device, dtype=torch.long)

        K = c_f.size(0)
        for k0 in range(0, K, k_block):
            k1 = min(k0 + k_block, K)
            ck = c_f[k0:k1]  # [kb, D]

            sims = x_f @ ck.t()  # [N, kb]

            # ★ NaN/Inf を潰す：NaNは -inf、inf はそのままでもいいが、保守的に posinfも大きくしてOK
            sims = torch.nan_to_num(sims, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))

            vals, idxs = sims.max(dim=1)

            update = vals > best_val
            best_val = torch.where(update, vals, best_val)
            best_idx = torch.where(update, idxs + k0, best_idx)

        if torch.isneginf(best_val).any():
            n_bad = torch.isneginf(best_val).sum().item()
            print(f"[argmax] WARNING: {n_bad} points never updated (all sims non-finite).")

        return best_idx

    # @torch.no_grad()
    # def split_the_winner_ema(
    #         self,
    #         embed,  # (K, D)  centers tensor (writeable view)
    #         ema_sum,  # (K, D)  EMA sum of assigned latents (same shape as embed)
    #         ema_count,  # (K,)    EMA count
    #         usage_ema,  # (K,)    EMA usage (for split decision)
    #         batch_counts,  # (K,)    batch histogram (float)
    #         batch_counts2,
    #         split_thr=0.15,
    #         prune_src_thr=0.005,  # prefer donor codes with p < this
    #         noise_scale=0.02,
    #         cooldown=None,  # (K,) int/long buffer
    #         cooldown_steps=2000,
    #         eps=1e-8,
    # ):

    import torch

    @torch.no_grad()
    def split_the_winner_ema(
            self,
            *,
            embed,  # (K,D)  <-- IMPORTANT: MEANS view, writable
            ema_sum,  # (K,D)  <-- EMA SUM buffer (embed_avg_*), writable
            ema_count,  # (K,)   <-- EMA COUNT buffer (cluster_size_*), writable
            usage_ema,  # (K,)   <-- usage EMA (already updated outside), writable
            batch_counts,  # (K,)   <-- current minibatch histogram (float)
            batch_counts2=None,  # unused
            # PCA-guided split inputs (per-key, per-minibatch)
            X_batch=None,  # (N_i, D) latents for THIS key in THIS minibatch
            assign=None,  # (N_i,) hard assignment in [0..K-1] for THIS key
            split_thr=0.15,
            prune_src_thr=0.005,
            noise_scale=0.02,
            cooldown=None,  # (K,) long (or int), optional
            cooldown_steps=2000,
            eps=1e-8,
            min_points=32,  # PCA needs enough points
            pca_power=0.5,  # step ~ pca_power * std along top PC
            fallback_to_noise=True,
    ):
        """
        Split-the-winner (PCA-guided) that **persists** under EMA recomputation.

        Contract:
          - `embed` is interpreted as MEANS (centers): embed[k] = ema_sum[k] / (ema_count[k] + eps)
          - This function updates BOTH:
              (A) embed[winner], embed[donor]  (means)
              (B) ema_sum / ema_count         so next `means = ema_sum/ema_count` preserves the split
              (C) usage_ema                   split 50/50 so donor doesn't inherit full popularity
              (D) cooldown (optional)

        Winner selection is based on usage_ema distribution p = usage_ema / sum(usage_ema).
        Donor is chosen among low-usage codes if possible.
        PCA direction uses points currently assigned to the winner in THIS minibatch.
        """

        import torch

        # -------------------------
        # 0) basic checks
        # -------------------------
        if embed is None or usage_ema is None:
            return False

        K, D = embed.shape
        device = embed.device

        # -------------------------
        # 1) cooldown tick (optional)
        # -------------------------
        if cooldown is not None:
            # in-place decrement
            cooldown.sub_(1).clamp_(min=0)

        # -------------------------
        # 2) compute p from usage_ema
        # -------------------------
        total = usage_ema.sum()
        if float(total.item()) <= 0.0 or (not torch.isfinite(total)):
            return False

        p = usage_ema / (total + float(eps))

        # -------------------------
        # 3) eligibility mask
        # -------------------------
        if cooldown is not None:
            eligible = (cooldown <= 0)
        else:
            eligible = torch.ones(K, dtype=torch.bool, device=device)

        if not bool(eligible.any()):
            return False

        # -------------------------
        # 4) pick winner among eligible
        # -------------------------
        neg_inf = -float("inf")
        p_eligible = torch.where(eligible, p, torch.full_like(p, neg_inf))
        winner = int(torch.argmax(p_eligible).item())

        winner_p = float(p[winner].item())
        if (not torch.isfinite(p_eligible[winner])) or (winner_p <= float(split_thr)):
            return False

        # -------------------------
        # 5) pick donor (prefer low-usage)
        # -------------------------
        low = (p < float(prune_src_thr)) & eligible
        low[winner] = False

        if bool(low.any()):
            p_low = torch.where(low, p, torch.full_like(p, float("inf")))
            donor = int(torch.argmin(p_low).item())
        else:
            elig2 = eligible.clone()
            elig2[winner] = False
            if not bool(elig2.any()):
                return False
            p_elig2 = torch.where(elig2, p, torch.full_like(p, float("inf")))
            donor = int(torch.argmin(p_elig2).item())

        if donor == winner:
            return False

        # -------------------------
        # 6) need per-key batch points to do a meaningful split
        # -------------------------
        if (X_batch is None) or (assign is None):
            return False  # don't split without local evidence

        if not (X_batch.ndim == 2 and assign.ndim == 1 and X_batch.shape[0] == assign.shape[0]):
            return False

        # ensure device & dtype for stability
        Xb = X_batch.to(device=device, dtype=torch.float32)
        yb = assign.to(device=device)

        # -------------------------
        # 7) isolate winner's assigned points (this is what matters)
        # -------------------------
        aw = (yb == int(winner))
        n_w = int(aw.sum().item())

        # KEY FIX: gate by winner-support only (NOT by K)
        if n_w < int(min_points):
            return False

        Xw = Xb[aw]  # (n_w, D)

        # -------------------------
        # 8) determine split direction (PCA if possible)
        # -------------------------
        did_pca = False
        cw = embed[winner].clone()  # current winner center (mean)

        # center points
        mu = Xw.mean(dim=0, keepdim=True)
        Y = Xw - mu

        # guard: if variance is almost zero, PCA is meaningless
        if torch.isfinite(Y).all():
            var_mean = float(Y.var(dim=0, unbiased=False).mean().item())
        else:
            var_mean = 0.0

        if var_mean > 1e-12:
            try:
                # top principal direction via SVD
                # Y: (n_w, D), Vh: (D, D) or (r, D)
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                v = Vh[0]  # (D,)
                if torch.isfinite(v).all():
                    proj = (Y @ v)  # (n_w,)
                    std = proj.std(unbiased=False) + float(eps)
                    step = float(pca_power) * float(std.item())

                    if torch.isfinite(std).all() and step > 0.0:
                        # move winner and donor in opposite directions
                        embed[winner].copy_(cw - v * step)
                        embed[donor].copy_(cw + v * step)
                        did_pca = True
            except Exception:
                did_pca = False

        # -------------------------
        # 9) fallback: small noise if PCA failed
        # -------------------------
        if (not did_pca) and fallback_to_noise:
            # scale noise by center RMS so it is neither tiny nor huge
            rms = torch.sqrt((cw * cw).mean() + float(eps))
            noise = torch.randn(D, device=device, dtype=torch.float32) * (float(noise_scale) * float(rms.item()))
            embed[winner].copy_(cw)  # keep winner as-is
            embed[donor].copy_(cw + noise)  # seed donor near winner

        if (not did_pca) and (not fallback_to_noise):
            return False

        # -------------------------
        # 10) persist split into EMA buffers
        # -------------------------
        if (ema_sum is not None) and (ema_count is not None):
            # IMPORTANT: do in-place on the original buffers (avoid rebinds)
            if ema_sum.dtype != torch.float32:
                # if your buffers are float32 already, remove this branch
                ema_sum_fp32 = ema_sum.float()
                # but we should not silently detach from the real buffer
                # so instead: write via temporary then copy back
            if ema_count.dtype != torch.float32:
                ema_count_fp32 = ema_count.float()

            # count guard
            cw_cnt = float(ema_count[winner].item())
            if cw_cnt <= 0.0 or (not torch.isfinite(ema_count[winner])):
                seed = 1.0
                ema_count[winner] = seed
                ema_sum[winner].copy_(embed[winner] * seed)
                cw_cnt = seed

            # 50/50 split counts (preserve total)
            cnt_w_new = ema_count[winner] * 0.5
            cnt_d_new = ema_count[winner] - cnt_w_new

            ema_count[winner] = cnt_w_new
            ema_count[donor] = cnt_d_new

            # reconstruct sums from the NEW means so consistency is exact:
            # ema_sum[k] ~= embed[k] * ema_count[k]
            ema_sum[winner].copy_(embed[winner] * ema_count[winner].unsqueeze(0))
            ema_sum[donor].copy_(embed[donor] * ema_count[donor].unsqueeze(0))

        # -------------------------
        # 11) split usage_ema too (so donor survives)
        # -------------------------
        usage_w = usage_ema[winner].clone()
        usage_ema[winner].mul_(0.5)
        usage_ema[donor] = usage_w * 0.5

        # -------------------------
        # 12) cooldown both (optional)
        # -------------------------
        if cooldown is not None:
            cooldown[winner] = int(cooldown_steps)
            cooldown[donor] = int(cooldown_steps)

        return True

    def _resolve_existing_safe_key(self, skey: str) -> str | None:
        """
        Infer時に「既存キーしか使わない」ための safe key 解決。

        返すのは self.embed に実在する safe key だけ。
        ここでは絶対に Parameter を新規作成しない（= _get_or_create_safe_key を呼ばない）。
        """
        skey = str(skey)

        # 1) すでに mapping があるならそれを最優先（これが最も安全）
        safe = None
        if hasattr(self, "key_to_safe") and isinstance(self.key_to_safe, dict):
            safe = self.key_to_safe.get(skey, None)
            if safe is not None and safe in self.embed:
                return safe

        # 2) “規則変換” した safe 候補が実在すれば採用（作成はしない）
        cand = self._safe_key(skey)
        if cand in self.embed:
            return cand

        # 3) 最後の保険：safe_to_key があれば逆引き（skeyに一致するものを探す）
        if hasattr(self, "safe_to_key") and isinstance(self.safe_to_key, dict):
            for s, orig in self.safe_to_key.items():
                if str(orig) == skey and s in self.embed:
                    return s

        # 4) それでも無理なら諦める（inferでは「unknown key」としてスキップ）
        return None

    # ------------------------------------------------------------------
    # Forward: ここで EMA update を dtype 安全 & safe-key 化
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Forward: dtype-safe, safe-key buffers, no crash on empty masks,
    #          entropy as METRIC (not differentiable in hard-argmin EMA),
    #          optional per-key logging.
    # ------------------------------------------------------------------
    import torch
    from einops import rearrange
    @torch.amp.autocast("cuda", enabled=False)
    def forward(
            self,
            x,
            feature=None,
            mask_dict=None,
            logger=None,
            chunk_i=None,
            epoch=None,
            mode=None,
            is_last_batch: bool = False,
    ):
        """
        Per-element vector quantization codebook forward (Option A: unified L2-normalized space).

        Option A invariant:
          - centers (self.embed[safe]) are stored L2-normalized (unit norm)
          - assignment uses Xn (normalized latents) vs C (already normalized centers)
          - EMA uses Xn, then writes back normalized means to self.embed[safe]

        IMPORTANT OUTPUT CONTRACT (for commitment_loss consistency):
          - self.embed_ind_dict[skey] MUST be {"ind": LongTensor(Ni), "safe": str}
            where "safe" is the exact safe-key used to index self.embed[safe] in this forward.
          - Holds for mode="infer" and mode="train"/"eval".

        Returns:
          - mode="infer": (key_id_full[int32], cluster_id_full[int32], id2safe_dict)
          - else       : (quantize_st[1,B,D], embed_ind_dict, self.embed, ent_metric_total[scalar])
        """
        import os, time
        import torch
        from einops import rearrange

        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------
        def _log(msg: str):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        def _safe_l2norm(t: torch.Tensor, eps_: float = 1e-12) -> torch.Tensor:
            den = t.norm(dim=-1, keepdim=True).clamp_min(eps_)
            t = t / den
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        def _target_max_p(N: int) -> float:
            if N >= 1_000_000:
                return float(getattr(self, "split_thr_big", 0.05))
            if N >= 200_000:
                return float(getattr(self, "split_thr_mid", 0.08))
            if N >= 50_000:
                return float(getattr(self, "split_thr_small", 0.12))
            return float(getattr(self, "split_thr_default", 0.20))

        def _hist_sig(idx: torch.Tensor, K: int):
            bc = torch.bincount(idx.long(), minlength=K).float()
            tot = bc.sum().clamp_min(1.0)
            used = int((bc > 0).sum().item())
            maxc = float(bc.max().item()) if bc.numel() else 0.0
            maxp = float((bc / tot).max().item()) if bc.numel() else 0.0
            return used, maxc, maxp

        # Buffers: robust getter that preserves registration
        def _get_buf(name: str, shape, dtype, device):
            shape = tuple(shape)
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.dtype == dtype and tuple(t.shape) == shape and t.device == device:
                return t
            new_t = torch.zeros(*shape, device=device, dtype=dtype)
            # If already registered buffer, replace it in-place in _buffers to keep state_dict behavior.
            if hasattr(self, "_buffers") and (name in self._buffers):
                self._buffers[name] = new_t
            else:
                self.register_buffer(name, new_t)
            return new_t

        # ------------------------------------------------------------------
        # config (override via self.*)
        # ------------------------------------------------------------------
        k_block = int(getattr(self, "assign_k_block", 1024))

        do_split = bool(getattr(self, "do_split_the_winner", True))
        split_once_per_epoch = bool(getattr(self, "split_once_per_epoch", False))
        split_only_last_batch = bool(getattr(self, "split_only_last_batch", False))

        usage_mom = float(getattr(self, "usage_ema_mom", 0.97))
        decay = float(getattr(self, "decay", 0.9))
        eps = float(getattr(self, "eps", 1e-8))
        prune_src_thr = float(getattr(self, "prune_src_thr", 0.005))

        maxp_mix_alpha = float(getattr(self, "maxp_mix_alpha", 0.5))
        maxp_growth_delta = float(getattr(self, "maxp_growth_delta", 0.02))
        min_points_for_split = int(getattr(self, "split_min_points", 32))
        min_N_for_split = int(getattr(self, "split_min_N", 200))

        debug_center_norm = bool(getattr(self, "debug_center_norm", False))

        # revive config
        revive_dead = bool(getattr(self, "revive_dead", True))
        revive_k = int(getattr(self, "revive_k", 512))
        revive_thr = float(getattr(self, "revive_thr", 0.10))
        revive_noise = float(getattr(self, "revive_noise", 0.01))
        revive_use_batch = bool(getattr(self, "revive_use_batch", False))
        revive_every_steps = int(getattr(self, "revive_every_steps", 5))

        # prune-by-age (epoch-last only)
        prune_after_epochs = int(getattr(self, "prune_after_epochs", 3))
        prune_seed_cs = float(getattr(self, "prune_seed_cs", 0.0))
        prune_reseed_from_batch = bool(getattr(self, "prune_reseed_from_batch", True))

        # ------------------------------------------------------------------
        # 0) global offset management (per chunk)
        # ------------------------------------------------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0
        if chunk_i is not None and int(chunk_i) == 0:
            self.latent_size_sum = 0

        # ------------------------------------------------------------------
        # 1) reshape -> flatten (1,B,D)
        # ------------------------------------------------------------------
        x = x.float()
        if x.ndim < 4:
            x = rearrange(x, "... -> 1 ...")
        flatten = x.view(x.shape[0], -1, x.shape[-1])  # (1,B,D)
        B, D = int(flatten.shape[1]), int(flatten.shape[2])

        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        # per-forward outputs (must be reset each call)
        self.quantize_dict = {}
        self.embed_ind_dict = {}

        mask_dict = self._normalize_mask_dict(mask_dict, device=flatten.device)

        # ------------------------------------------------------------------
        # 2) safe-key <-> int id mapping
        # ------------------------------------------------------------------
        if not hasattr(self, "_safe2id"):
            self._safe2id = {}
        if not hasattr(self, "_id2safe"):
            self._id2safe = {}
        if not hasattr(self, "cb_dict"):
            self.cb_dict = {}

        def _safe_id(safe: str) -> int:
            sid = self._safe2id.get(safe)
            if sid is None:
                sid = len(self._safe2id)
                self._safe2id[safe] = sid
                self._id2safe[sid] = safe
            return sid

        def _get_safe_and_code_existing_only(skey: str):
            """
            infer 専用：
              - safe key を解決するが、既存しか使わない
              - 見つからなければ (None, None, None) を返す
            """
            safe = self._resolve_existing_safe_key(skey)
            if safe is None:
                return None, None, None

            code_param = self.embed[safe]  # ParameterDict の既存キー参照のみ
            code = code_param.squeeze(0) if code_param.ndim == 3 else code_param
            return safe, code_param, code

        def _get_safe_and_code(skey: str):
            """Training/init lookup: require a codebook already initialized for this key.

            Never create a random codebook silently: init_embed_ must initialize it.
            """
            safe, code_param, code = _get_safe_and_code_existing_only(skey)
            if safe is None:
                raise KeyError(f"[codebook] missing initialized key {skey!r}; "
                               "check init_embed_ and CBDICT group coverage")
            return safe, code_param, code

        # ------------------------------------------------------------------
        # 3) infer: IDs only (no EMA, no split, no revive, no prune)
        #    returns: (key_id_full, cluster_id_full, global_id_full, id2safe)
        # ------------------------------------------------------------------
        if mode == "infer":
            if mask_dict is None:
                raise ValueError("mode='infer' requires mask_dict (global indices).")

            # --------------------------------------------------------------
            # Build / cache global offsets (safe -> offset) ONCE.
            # infer doesn't grow/split, so this stays valid for the whole run.
            # --------------------------------------------------------------
            if not hasattr(self, "_global_offsets") or not hasattr(self, "_global_vocab_size"):
                # IMPORTANT: fix ordering deterministically
                safe_keys = sorted(self.embed.keys())
                offsets = {}
                cur = 0
                for s in safe_keys:
                    K = int(self.embed[s].shape[0] if self.embed[s].ndim == 2 else self.embed[s].shape[1])
                    offsets[s] = cur
                    cur += K
                self._global_offsets = offsets
                self._global_vocab_size = int(cur)

            offsets = self._global_offsets  # dict: safe -> int offset

            key_id_full = torch.full((B,), -1, device=flatten.device, dtype=torch.int32)
            cluster_id_full = torch.full((B,), -1, device=flatten.device, dtype=torch.int32)
            global_id_full = torch.full((B,), -1, device=flatten.device, dtype=torch.int64)

            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue

                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                X = flatten[0].index_select(0, idx_local)  # (Ni,D)
                if X.numel() == 0:
                    continue

                skey = str(key)

                safe, _code_param, C = _get_safe_and_code_existing_only(skey)
                if safe is None:
                    # unknown / not present in checkpoint codebook => skip (leave -1)
                    if logger is not None:
                        logger.warning(f"[infer] unknown skey not in existing codebook: {skey} (skipped)")
                    continue

                sid = _safe_id(safe)

                with torch.no_grad():
                    Xn = _safe_l2norm(X)
                    idx_code = self.argmax_sim_blockwise(Xn, C, k_block=k_block).to(dtype=torch.long)

                # key_id / cluster_id
                key_id_full.index_copy_(
                    0, idx_local,
                    torch.full((idx_local.numel(),), sid, device=flatten.device, dtype=torch.int32),
                )
                cluster_id_full.index_copy_(0, idx_local, idx_code.to(dtype=torch.int32))

                # global_id = offset[safe] + cid
                off = offsets.get(safe, None)
                if off is None:
                    # safe は存在するが offsets に無い = offsets 構築時の embed keys とズレてる
                    # → ここではスキップして -1 のままにする（安全側）
                    if logger is not None:
                        logger.warning(f"[infer] safe exists but missing in global_offsets: {safe} (skipped)")
                    continue

                global_id_full.index_copy_(
                    0, idx_local,
                    (idx_code.to(dtype=torch.int64) + off)
                )

                # contract for debugging
                self.embed_ind_dict[skey] = {"ind": idx_code, "safe": safe}

                if off is None:
                    # unknown key for this checkpoint: mark as UNK and continue
                    # (keep -1 in key_id/cluster_id/global_id)
                    # optional: log once
                    if logger is not None:
                        logger.warning(f"[infer] unknown key not in ckpt/global_offsets: {safe} (skipped)")
                    continue

                global_id_full.index_copy_(
                    0, idx_local,
                    (idx_code.to(dtype=torch.int64) + off)
                )

                # keep contract for commitment_loss debugging (optional)
                self.embed_ind_dict[skey] = {"ind": idx_code, "safe": safe}

            self.latent_size_sum = global_end
            return key_id_full, cluster_id_full, global_id_full, dict(self._id2safe)

        # ------------------------------------------------------------------
        # 4) init_kmeans_final: init + dump then return
        # ------------------------------------------------------------------
        if mode == "init_kmeans_final":
            self.init_embed_(flatten, logger, epoch, mask_dict=mask_dict)

            if mask_dict is None:
                self.latent_size_sum = global_end
                return 0

            if not hasattr(self, "_kmeans_dump"):
                self._kmeans_dump = {}
            if not hasattr(self, "_kmeans_sig"):
                self._kmeans_sig = {}

            for key, idx_global in mask_dict.items():
                if idx_global is None or idx_global.numel() == 0:
                    continue

                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue

                idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
                X = flatten[0].index_select(0, idx_local)
                if X.numel() == 0:
                    continue

                skey = str(key)
                safe, _code_param, C = _get_safe_and_code(skey)

                with torch.no_grad():
                    Xn = _safe_l2norm(X)
                    idx_code = self.argmax_sim_blockwise(Xn, C, k_block=k_block).to(dtype=torch.long)
                    bc0 = torch.bincount(idx_code, minlength=C.shape[0]).float()
                    sig0 = torch.stack([
                        bc0.sum(),
                        (bc0 > 0).sum().float(),
                        bc0.max(),
                        (bc0 / bc0.sum().clamp_min(1)).max()
                    ])
                    self._kmeans_sig[skey] = {
                        "safe": safe,
                        "C_norm_mean": C.detach().float().norm(dim=1).mean().item(),
                        "sig": sig0.cpu(),
                    }

                entry = self._kmeans_dump.get(skey)
                if entry is None:
                    entry = {"latents": [], "centers": None, "assign": [], "safe": safe}
                    self._kmeans_dump[skey] = entry

                entry["latents"].append(X.detach().to("cpu", dtype=torch.float16))
                entry["assign"].append(idx_code.detach().to("cpu"))
                entry["safe"] = safe
                if entry["centers"] is None:
                    entry["centers"] = C.detach().to("cpu", dtype=torch.float16)

            out = {}
            for k, v in self._kmeans_dump.items():
                out[k] = {
                    "safe": v.get("safe", None),
                    "latents": torch.cat(v["latents"], dim=0) if len(v["latents"]) else None,
                    "centers": v["centers"],
                    "assign": torch.cat(v["assign"], dim=0) if len(v["assign"]) else None,
                    "quantize": None,
                }

            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("dumps", f"init_kmeans_final_dump_{stamp}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(out, path)
            self._kmeans_dump = {}

            self.latent_size_sum = global_end
            return 0

        # ------------------------------------------------------------------
        # 5) train/eval: quantize + EMA + split + revive + prune
        # ------------------------------------------------------------------
        ent_metric_total = torch.zeros((), device=flatten.device, dtype=torch.float32)
        do_ema = bool(self.training and (mode == "train") and (epoch is not None))

        # epoch-wise collectors
        if not hasattr(self, "_mp_epoch"):
            self._mp_epoch = None
        if not hasattr(self, "_mp_list"):
            self._mp_list = []
        if not hasattr(self, "_mp_list_keys"):
            self._mp_list_keys = []

        if epoch is not None:
            ep_int = int(epoch)
            if self._mp_epoch != ep_int:
                self._mp_epoch = ep_int
                self._mp_list = []
                self._mp_list_keys = []

        # per-epoch split gate
        if do_ema and split_once_per_epoch:
            if not hasattr(self, "_split_epoch"):
                self._split_epoch = None
            if not hasattr(self, "_split_done_epoch"):
                self._split_done_epoch = set()
            if self._split_epoch != int(epoch):
                self._split_epoch = int(epoch)
                self._split_done_epoch.clear()

        # prev max_p / step counter
        if do_ema and not hasattr(self, "_prev_maxp"):
            self._prev_maxp = {}
        if do_ema and not hasattr(self, "_step_counter"):
            self._step_counter = 0
        if do_ema:
            self._step_counter += 1

        def _revive_dead_centers_if_needed(
                *,
                means: torch.Tensor,  # (K,D) normalized
                ea: torch.Tensor,  # (K,D)
                cs: torch.Tensor,  # (K,)
                ue: torch.Tensor,  # (K,)
                Xn_dev: torch.Tensor,  # (N,D) normalized
                idx_dev: torch.Tensor,  # (N,) long
        ) -> bool:
            if not revive_dead:
                return False
            if revive_every_steps > 0 and (int(self._step_counter) % int(revive_every_steps)) != 0:
                return False

            K = int(means.shape[0])
            dead = (cs < float(revive_thr))
            n_dead = int(dead.sum().item())
            if n_dead <= 0:
                return False

            if Xn_dev is None or Xn_dev.numel() == 0:
                revive_from_batch = True
            else:
                revive_from_batch = bool(revive_use_batch)

            # how many to revive
            max_seeds = K if revive_from_batch else int(Xn_dev.shape[0])
            k = min(int(revive_k), n_dead, max_seeds)
            if k <= 0:
                return False

            dead_idx = torch.nonzero(dead, as_tuple=False).flatten()[:k]

            if revive_from_batch and (idx_dev is not None) and (idx_dev.numel() > 0):
                bc = torch.bincount(idx_dev, minlength=K).float()
                winners = torch.argsort(bc, descending=True)
                src = winners[:k]
                seeds = means.index_select(0, src).clone()
            else:
                perm = torch.randperm(int(Xn_dev.shape[0]), device=Xn_dev.device)[:k]
                seeds = Xn_dev.index_select(0, perm).clone()

            seeds = _safe_l2norm(seeds + revive_noise * torch.randn_like(seeds), eps_=1e-12)
            means.index_copy_(0, dead_idx, seeds)

            # give them a small nonzero state so they can compete
            cs.index_fill_(0, dead_idx, float(min_points_for_split))
            ue.index_fill_(0, dead_idx, float(min_points_for_split))
            den = cs.unsqueeze(-1).clamp_min(1e-5)
            ea.copy_(means * den)
            return True

        # main per-key loop
        for key, idx_global in (mask_dict.items() if mask_dict is not None else []):
            if idx_global is None or idx_global.numel() == 0:
                continue

            gmask = (idx_global >= global_start) & (idx_global < global_end)
            if not bool(gmask.any()):
                continue

            idx_local = (idx_global[gmask] - global_start).to(device=flatten.device, dtype=torch.long)
            X = flatten[0].index_select(0, idx_local)  # (Ni,D)
            if X.numel() == 0:
                continue

            skey = str(key)
            safe, code_param, C = _get_safe_and_code(skey)

            if debug_center_norm and (epoch == 1) and (chunk_i == 0):
                with torch.no_grad():
                    cn = C.detach().float().norm(dim=1)
                    _log(f"[DBG][C-NORM] key={skey} mean={cn.mean().item():.4f} std={cn.std(correction=0).item():.4f}")

            # normalize latents ONCE per key
            Xn = _safe_l2norm(X.to(dtype=torch.float32), eps_=1e-12)

            # hard assignment: C assumed normalized
            with torch.no_grad():
                idx_code_long = self.argmax_sim_blockwise(Xn, C, k_block=k_block).to(dtype=torch.long)

            Q = C.index_select(0, idx_code_long)
            self.quantize_dict[skey] = Q
            self.embed_ind_dict[skey] = {"ind": idx_code_long, "safe": safe}

            # -------------------------
            # EMA + split + revive + prune
            # -------------------------
            if do_ema:
                dev = code_param.device
                if code_param.ndim == 3:
                    K_e, D_e = int(code_param.shape[1]), int(code_param.shape[2])
                else:
                    K_e, D_e = int(code_param.shape[0]), int(code_param.shape[1])

                # record per-skey K (final truth for this step)
                self.cb_dict[skey] = int(K_e)

                # buffers (registered)
                cs = _get_buf(f"cluster_size_{skey}", (K_e,), torch.float32, dev)
                ea = _get_buf(f"embed_avg_{skey}", (K_e, D_e), torch.float32, dev)
                ue = _get_buf(f"usage_ema_{safe}", (K_e,), torch.float32, dev)
                cd = _get_buf(f"split_cd_{safe}", (K_e,), torch.long, dev)
                ever = _get_buf(f"ever_used_{safe}", (K_e,), torch.uint8, dev)
                lue = _get_buf(f"last_used_ep_{safe}", (K_e,), torch.int32, dev)

                with torch.no_grad():
                    ep_i = int(epoch)

                    idx_dev = idx_code_long.to(device=dev)

                    # snapshot BEFORE update (MUST clone for correct delta/logging)
                    C_pre = code_param.detach()
                    C_pre = C_pre.squeeze(0) if C_pre.ndim == 3 else C_pre
                    C_pre = C_pre.to(device=dev, dtype=torch.float32).clone()

                    # batch counts (K,)
                    batch_counts = torch.zeros((K_e,), device=dev, dtype=torch.float32)
                    batch_counts.index_add_(0, idx_dev, torch.ones_like(idx_dev, device=dev, dtype=torch.float32))

                    # Xn on dev
                    Xn_dev = Xn.to(device=dev, dtype=torch.float32)

                    # batch sum (K,D)
                    batch_sum = torch.zeros((K_e, D_e), device=dev, dtype=torch.float32)
                    batch_sum.index_add_(0, idx_dev, Xn_dev)

                    # update ever/last_used_epoch
                    used_now = torch.nonzero(batch_counts > 0, as_tuple=False).squeeze(1)
                    if used_now.numel() > 0:
                        ever.index_fill_(0, used_now, 1)
                        lue.index_fill_(0, used_now, ep_i)

                    # Entropy regularizer (optional metric accumulation only)
                    denom = batch_counts.sum()
                    if denom.item() > 0.0:
                        p = batch_counts / (denom + 1e-8)
                        H = -(p * (p + 1e-12).log()).sum()

                        ent_w = getattr(self, "entropy_weight", 1e-3)
                        ent_w = 0.0 if ent_w is None else float(ent_w)

                        contrib = (-H) * ent_w
                        ent_metric_total = ent_metric_total + contrib

                        if chunk_i == 0:
                            effK = int((batch_counts > 0).sum().item())
                            max_p_b = float(p.max().item())
                            _log(
                                f"[ENT] epoch={epoch} key={skey} "
                                f"H={H.item():.4f} effK={effK} max_p={max_p_b:.4f} "
                                f"ent_w={ent_w:.3g} contrib={contrib.item():.6f}"
                            )

                    # usage_ema
                    ue.mul_(usage_mom).add_(batch_counts, alpha=(1.0 - usage_mom))

                    # EMA update
                    one_m = 1.0 - decay
                    cs.mul_(decay).add_(batch_counts, alpha=one_m)
                    ea.mul_(decay).add_(batch_sum, alpha=one_m)

                    # means (K,D) and normalize
                    den = cs.unsqueeze(-1).clamp_min(1e-5)
                    means = _safe_l2norm(ea / den, eps_=1e-12)

                    # keep inactive as previous center
                    active = (cs > 0.0)
                    if active.any():
                        means[~active] = _safe_l2norm(C_pre[~active], eps_=1e-12)
                    else:
                        means.copy_(_safe_l2norm(C_pre, eps_=1e-12))

                    # keep ea consistent
                    ea.copy_(means * den)

                    if (epoch == 1) and (chunk_i == 0):
                        mn = means[active].norm(dim=1).mean().item() if active.any() else float("nan")
                        used_u, _maxc_u, maxp_u_batch = _hist_sig(idx_dev, K_e)
                        _log(
                            f"[EMA-MEANS] ep={epoch} key={skey} cs_sum={float(cs.sum()):.1f} "
                            f"used={used_u} maxp_batch={maxp_u_batch:.4f} means_mean_norm(active)={mn:.3f}"
                        )

                    # ---------------------------------------------------------
                    # Max_p policy: mix(cs_dist, ue_dist)
                    # ---------------------------------------------------------
                    total_cs = cs.sum().clamp_min(1e-8)
                    p_cs = cs / total_cs
                    max_p_cs = float(p_cs.max().item())

                    total_u = ue.sum().clamp_min(1e-8)
                    p_u = ue / total_u
                    max_p_u = float(p_u.max().item())

                    a = float(maxp_mix_alpha)
                    p_mix = (1.0 - a) * p_cs + a * p_u
                    max_p = float(p_mix.max().item())

                    N_key = int(cs.sum().item())
                    thr = _target_max_p(N_key)

                    prev = float(self._prev_maxp.get(skey, 0.0))
                    grew = (max_p > prev + float(maxp_growth_delta))
                    self._prev_maxp[skey] = max_p

                    # split gate
                    can_split_now = do_split and (not split_only_last_batch or bool(is_last_batch))
                    can_split_now = can_split_now and (not split_once_per_epoch or (skey not in self._split_done_epoch))

                    if can_split_now and (chunk_i == 0):
                        _log(
                            f"[SPLIT-POLICY] epoch={epoch} key={skey} N~{float(cs.sum()):.1f} "
                            f"max_p={max_p:.4f} (cs={max_p_cs:.4f}, ue={max_p_u:.4f}, prev={prev:.4f}) "
                            f"thr={thr:.4f} K={K_e} grew={bool(grew)}"
                        )

                    do_it = (
                            can_split_now
                            and ((max_p > thr) or bool(grew))
                            and (N_key >= min_N_for_split)
                            and (K_e > 2)
                    )
                    if do_it:
                        did = self.split_the_winner_ema(
                            embed=means,
                            ema_sum=ea,
                            ema_count=cs,
                            usage_ema=ue,
                            batch_counts=batch_counts,
                            batch_counts2=batch_counts,
                            X_batch=Xn_dev,
                            assign=idx_dev.to(device=dev, dtype=torch.long),
                            split_thr=float(thr),
                            prune_src_thr=float(prune_src_thr),
                            cooldown=cd,
                            cooldown_steps=int(getattr(self, "split_cooldown_steps", 2000)),
                            eps=float(eps),
                            min_points=int(min_points_for_split),
                            pca_power=float(getattr(self, "split_pca_power", 0.5)),
                            noise_scale=float(getattr(self, "split_noise_scale", 0.02)),
                            fallback_to_noise=bool(getattr(self, "split_fallback_to_noise", True)),
                        )

                        if split_once_per_epoch:
                            self._split_done_epoch.add(skey)

                        if did:
                            means.copy_(_safe_l2norm(means, eps_=1e-12))
                            den2 = cs.unsqueeze(-1).clamp_min(1e-5)
                            ea.copy_(means * den2)

                            total_cs2 = cs.sum().clamp_min(1e-8)
                            p_cs2 = cs / total_cs2
                            max_p_cs2 = float(p_cs2.max().item())

                            total_u2 = ue.sum().clamp_min(1e-8)
                            p_u2 = ue / total_u2
                            max_p_u2 = float(p_u2.max().item())

                            p_mix2 = (1.0 - a) * p_cs2 + a * p_u2
                            max_p_after = float(p_mix2.max().item())

                            _log(
                                f"[SPLIT][N-DEP] epoch={epoch} key={skey} did_split=True "
                                f"N~{N_key} thr={thr:.4f} max_p_before={max_p:.4f} "
                                f"max_p_after={max_p_after:.4f} (cs={max_p_cs2:.4f}, ue={max_p_u2:.4f})"
                            )
                            self._prev_maxp[skey] = max_p_after

                    # revive
                    revived = _revive_dead_centers_if_needed(
                        means=means, ea=ea, cs=cs, ue=ue, Xn_dev=Xn_dev, idx_dev=idx_dev
                    )
                    if revived:
                        _log(f"[REVIVE] epoch={epoch} key={skey} revived_some_dead_centers=True")

                    # prune-by-age (epoch last batch only)
                    do_prune_now = bool(is_last_batch) and (prune_after_epochs > 0)
                    if do_prune_now:
                        age = ep_i - lue.to(torch.int32)
                        stale = (ever > 0) & (age >= int(prune_after_epochs))
                        stale_idx = torch.nonzero(stale, as_tuple=False).squeeze(1)
                        n_stale = int(stale_idx.numel())

                        if n_stale > 0:
                            if prune_reseed_from_batch and (Xn_dev is not None) and (Xn_dev.numel() > 0):
                                Nn = int(Xn_dev.shape[0])
                                perm = torch.randperm(Nn, device=dev)
                                pick = perm[:n_stale] if Nn >= n_stale else perm.repeat((n_stale + Nn - 1) // Nn)[
                                    :n_stale]
                                seeds = _safe_l2norm(Xn_dev.index_select(0, pick).clone(), eps_=1e-12)
                            else:
                                seeds = _safe_l2norm(
                                    means.index_select(0, stale_idx).clone() + 0.01 * torch.randn((n_stale, D_e),
                                                                                                  device=dev),
                                    eps_=1e-12,
                                )

                            means.index_copy_(0, stale_idx, seeds)

                            # dormant reset
                            cs.index_fill_(0, stale_idx, float(prune_seed_cs))
                            ue.index_fill_(0, stale_idx, 0.0)
                            ea.index_fill_(0, stale_idx, 0.0)

                            # treat as never-used again
                            ever.index_fill_(0, stale_idx, 0)
                            lue.index_fill_(0, stale_idx, 0)

                            if chunk_i == 0:
                                _log(
                                    f"[PRUNE] epoch={epoch} key={skey} pruned={n_stale} after={prune_after_epochs}ep (dormant reset)")

                    # diagnostics
                    if chunk_i == 0:
                        dead_mask = (cs < 1e-3)
                        near_dead_mask = (cs < 0.1)
                        n_dead = int(dead_mask.sum().item())
                        n_near_dead = int(near_dead_mask.sum().item())
                        frac_dead = n_dead / float(K_e)
                        frac_near_dead = n_near_dead / float(K_e)
                        n_ever = int((ever > 0).sum().item())
                        _log(
                            f"[DEAD] epoch={epoch} key={skey} dead={n_dead}/{K_e} ({frac_dead:.3f}) "
                            f"near_dead={n_near_dead}/{K_e} ({frac_near_dead:.3f}) ever_used={n_ever}/{K_e}"
                        )

                    # record max_p for epoch summary
                    total_csR = cs.sum().clamp_min(1e-8)
                    p_csR = cs / total_csR
                    total_uR = ue.sum().clamp_min(1e-8)
                    p_uR = ue / total_uR
                    p_mixR = (1.0 - a) * p_csR + a * p_uR
                    max_p_R = float(p_mixR.max().item())
                    self._mp_list.append(max_p_R)
                    self._mp_list_keys.append((skey, max_p_R, int(cs.sum().item()), int(K_e)))

                    # write centers back to parameter (already normalized)
                    if code_param.ndim == 3:
                        code_param.data.copy_(means.unsqueeze(0).to(dtype=code_param.dtype))
                    else:
                        code_param.data.copy_(means.to(dtype=code_param.dtype))

                    if (epoch == 1) and (chunk_i == 0):
                        C_post = code_param.detach()
                        C_post = C_post.squeeze(0) if C_post.ndim == 3 else C_post
                        delta = (C_post.float() - C_pre).abs().mean().item()
                        if not (delta == delta):
                            raise RuntimeError(f"[EMA-WRITE] delta NaN key={skey}")
                        _log(f"[EMA-WRITE] ep={epoch} key={skey} mean_abs_delta={delta:.3e}")

            # cleanup per key (keep memory pressure low)
            del X, Xn, C, idx_code_long, Q

        # ------------------------------------------------------------------
        # 6) rebuild quantized tensor in original atom order
        # ------------------------------------------------------------------
        quantize_full = torch.empty((B, D), device=flatten.device, dtype=flatten.dtype)

        if mask_dict is not None:
            for key, idx_global in mask_dict.items():
                skey = str(key)
                qk = self.quantize_dict.get(skey, None)
                if qk is None:
                    continue
                gmask = (idx_global >= global_start) & (idx_global < global_end)
                if not bool(gmask.any()):
                    continue
                idx_local = (idx_global[gmask] - global_start).to(device=quantize_full.device, dtype=torch.long)
                qk = qk.to(device=quantize_full.device, dtype=quantize_full.dtype)
                if idx_local.numel() > 0:
                    quantize_full.index_copy_(0, idx_local, qk)

        # fill unused positions with identity
        all_local = []
        if mask_dict is not None:
            for idx in mask_dict.values():
                if idx is None or idx.numel() == 0:
                    continue
                in_chunk = idx[(idx >= global_start) & (idx < global_end)]
                if in_chunk.numel() > 0:
                    all_local.append(in_chunk - global_start)

        used = (
            torch.unique(torch.cat(all_local, dim=0))
            if len(all_local) > 0
            else torch.tensor([], dtype=torch.long, device=flatten.device)
        )
        unused = torch.ones(B, dtype=torch.bool, device=flatten.device)
        if used.numel() > 0:
            unused[used] = False
        if bool(unused.any()):
            quantize_full[unused] = flatten[0][unused]

        # straight-through
        quantize_st = flatten[0] + (quantize_full - flatten[0]).detach()
        quantize_st = quantize_st.unsqueeze(0)

        if chunk_i is not None:
            self.latent_size_sum = global_end

        # ------------------------------------------------------------------
        # end-of-epoch report (only on last batch)
        # ------------------------------------------------------------------
        if (epoch is not None) and bool(is_last_batch) and (logger is not None):
            if len(self._mp_list) > 0:
                mp = torch.tensor(self._mp_list, dtype=torch.float32)
                qs = torch.quantile(mp, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
                logger.info(
                    f"[MAXP][epoch={int(epoch)}] "
                    f"n_keys={len(self._mp_list)} "
                    f"mean={mp.mean().item():.4f} std={mp.std(unbiased=False).item():.4f} "
                    f"p50={qs[0].item():.4f} p75={qs[1].item():.4f} "
                    f"p90={qs[2].item():.4f} p95={qs[3].item():.4f} p99={qs[4].item():.4f} "
                    f"max={mp.max().item():.4f}"
                )

                topK = int(getattr(self, "maxp_report_topK", 20))
                worst = sorted(self._mp_list_keys, key=lambda t: t[1], reverse=True)[:topK]
                for (k, v, nkey, Ke) in worst:
                    logger.info(f"[MAXP-WORST][epoch={int(epoch)}] key={k} max_p={v:.4f} N~{nkey} K={Ke}")

                if bool(getattr(self, "save_maxp_each_epoch", True)):
                    os.makedirs("dumps", exist_ok=True)
                    out = {"epoch": int(epoch), "maxp_list": self._mp_list, "detail": self._mp_list_keys}
                    torch.save(out, f"dumps/maxp_epoch_{int(epoch):04d}.pt")
        if (epoch is not None) and (logger is not None):
            # そのepochで見えた skey->K の合計（枠）
            cb_sum = int(sum(int(v) for v in getattr(self, "cb_dict", {}).values()))
            logger.info(f"[VOCAB-CHECK][epoch={int(epoch)}] cb_dict_sum={cb_sum} vocab_size_if_fixed={cb_sum + 2}")

        return quantize_st, self.embed_ind_dict, self.embed, ent_metric_total


class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.8,
            eps=1e-5,
            kmeans_init=True,
            kmeans_iters=600,
            sync_kmeans=True,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            margin_weight=1,
            spread_weight=0.2,
            pair_weight=0.01,
            lamb_div_ele=1,
            lamb_div_bonds=1,
            lamb_div_aroma=1,
            lamb_div_ringy=1,
            lamb_div_h_num=1,
            lamb_div_equidist=1,
            lamb_div_elec_state=1,
            lamb_div_charge=1,
            commitment_weight=1000,  # using
            codebook_weight=1,  # using
            lamb_sil=0.00001,           # using
            lamb_cb=0.01,           # using
            lamb_div=0.01,           # using
            lamb_equiv_atom=1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False,
            use_cosine=False,
            target_radius=3.0,  # soft cap for ||z||
            radius_weight=1e-3,  # weight for norm regularizer
            tau_init=0.5,  # starting temperature (distance scale)
            tau_min=0.05,
            tau_max=2.0,  # clamp for temperature
            tau_ema=0.9,  # EMA for temperature
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)  # use coocbook_dim if not None
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        # self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.tau_ema = 0.9
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight

        has_codebook_orthogonal_loss = margin_weight > 0

        self.margin_weight = margin_weight
        self.spread_weight = spread_weight
        self.lamb_div_ele = lamb_div_ele
        self.lamb_div_bonds = lamb_div_bonds
        self.lamb_div_aroma = lamb_div_aroma
        self.lamb_div_ringy = lamb_div_ringy
        self.lamb_div_h_num = lamb_div_h_num
        self.lamb_div_elec_state = lamb_div_elec_state
        self.lamb_div_charge = lamb_div_charge
        self.lamb_equiv_atom = lamb_equiv_atom
        self.lamb_sil = lamb_sil
        self.lamb_div = lamb_div
        self.lamb_cb = lamb_cb
        self.lamb_div_equidist = lamb_div_equidist
        self.pair_weight = pair_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        codebook_max_norm = None  # e.g., 4.0 to clamp code vectors
        codebook_class = EuclideanCodebook # if not use_cosine_sim else CosineSimCodebook
        self._codebook = codebook_class(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp,
        )
        args = get_args()
        self.epoch_at_mode_shift = args.epoch_at_mode_shift
        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.compute_contrastive_loss = ContrastiveLoss(dim, 136)

        self.use_cosine = use_cosine
        self.target_radius = float(target_radius)
        self.radius_weight = float(radius_weight)
        self.register_buffer("_tau", torch.tensor(float(tau_init)))
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.tau_ema = float(tau_ema)
        self.codebook_max_norm = codebook_max_norm

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def _cb_to_tensor(self, cb):
        """
        Normalize various "codebook" containers to a [K, D] Tensor.

        対応する型:
          - torch.Tensor
          - nn.Parameter
          - nn.Module (EuclideanCodebook 含む)
              * .weight を持つ場合 → それを使う
              * .embed を持つ場合 → embed.weight か embed 自体を使う
              * .codebook / .codes / .embedding を持つ場合 → それを使う
              * 上記が無くても、named_parameters / named_buffers から
                最初の 2 次元テンソル ([K, D]) を拾う
          - nn.ParameterDict（[K_i, D] を縦に concat）
          - dict[key -> (上記のいずれか)]（縦に concat）
        """
        import torch
        import torch.nn as nn

        # -------- 1. すでに Tensor/Parameter の場合 --------
        if isinstance(cb, torch.Tensor):
            return cb
        if isinstance(cb, nn.Parameter):
            return cb

        # -------- 2. nn.Module (EuclideanCodebook を含む) --------
        if isinstance(cb, nn.Module):
            # (a) .weight を持つモジュール（Embedding, Linear など）
            if hasattr(cb, "weight") and isinstance(cb.weight, torch.Tensor):
                return cb.weight

            # (b) EuclideanCodebook などで .embed を使っている場合
            if hasattr(cb, "embed"):
                e = cb.embed
                if isinstance(e, nn.Embedding):
                    return e.weight
                if isinstance(e, (torch.Tensor, nn.Parameter)):
                    return e

            # (c) その他ありがちな名前をチェック
            for attr in ("codebook", "codes", "embedding"):
                if hasattr(cb, attr):
                    t = getattr(cb, attr)
                    if isinstance(t, (torch.Tensor, nn.Parameter)):
                        return t

            # (d) 最後のフォールバック:
            #     モジュール内の parameter / buffer から
            #     「次元数が 2 のテンソル」を探して最初のものを使う
            #     (recurse=True なので embed.weight のような下位モジュールも拾える)
            for name, p in cb.named_parameters(recurse=True):
                if isinstance(p, torch.Tensor) and p.ndim == 2:
                    # print(f"[DEBUG] using parameter {name} from {type(cb)} as codebook")  # 必要ならログ
                    return p

            for name, b in cb.named_buffers(recurse=True):
                if isinstance(b, torch.Tensor) and b.ndim == 2:
                    # print(f"[DEBUG] using buffer {name} from {type(cb)} as codebook")
                    return b

            # ここまで来ると、2D テンソルが一つも見つからない本当にイレギュラーなケース
            raise TypeError(
                f"_cb_to_tensor: unsupported nn.Module type {type(cb)} "
                f"(no usable weight/embed/codebook/codes/embedding or 2D param/buffer)"
            )

        # -------- 3. nn.ParameterDict: values を全部縦に concat --------
        if isinstance(cb, nn.ParameterDict):
            tensors = []
            for k in sorted(cb.keys()):
                v = cb[k]
                if isinstance(v, (torch.Tensor, nn.Parameter)):
                    tensors.append(v)
                else:
                    raise TypeError(
                        f"_cb_to_tensor: unsupported leaf type {type(v)} "
                        f"in ParameterDict for key {k!r}"
                    )
            if not tensors:
                raise ValueError("_cb_to_tensor: empty ParameterDict")
            return torch.cat(tensors, dim=0)

        # -------- 4. dict: value ごとに再帰的にテンソル化して concat --------
        if isinstance(cb, dict):
            tensors = []
            for k in sorted(cb.keys()):
                v = cb[k]
                t = self._cb_to_tensor(v)  # 再帰
                tensors.append(t)
            if not tensors:
                raise ValueError("_cb_to_tensor: empty dict")
            return torch.cat(tensors, dim=0)

        # -------- 5. それ以外はサポート外 --------
        raise TypeError(f"_cb_to_tensor: unsupported type {type(cb)}")

    def get_codes_from_indices(self, indices):
        indices = indices.long()
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack([indices], 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes, = unpack(codes, ps, 'b * d')
        return codes

    import torch

    def fast_find_equivalence_groups(self, latents):
        from collections import defaultdict
        hash_map = defaultdict(list)
        for idx, vector in enumerate(latents):
            key = tuple(vector.tolist())  # Convert tensor to a hashable tuple
            hash_map[key].append(idx)
        equivalence_groups = [group for group in hash_map.values() if len(group) > 1]
        return equivalence_groups

    def _unwrap_codebook_entry(self, cb):
        """
        Accept: Tensor, nn.Parameter, dict/ParameterDict possibly holding 'embed'.
        Return: Tensor (or Parameter) with shape [K,D] or [1,K,D].
        """
        if isinstance(cb, (torch.Tensor, nn.Parameter)):
            return cb
        if isinstance(cb, (dict, nn.ParameterDict)):
            if 'embed' in cb:
                return self._unwrap_codebook_entry(cb['embed'])
        raise TypeError(f"Unsupported codebook entry type: {type(cb)}. "
                        "Expected Tensor/Parameter or dict/ParameterDict with 'embed'.")

    def commitment_loss(
            self,
            encoder_outputs,
            mask_dict,
            codebook,
            codebook_mod=None,
            logger=None,
            chunk_start=None,
            beta=0.25,
            temperature=None,
            use_cosine=False,
            embed_ind_dict=None,
    ):
        """
        Per-element commitment + codebook + repel losses (assignment-consistent).

        Returns:
            commit_loss, codebook_loss, repel_loss, cb_repel_loss

        Assumptions / fixes vs your version:
          ✅ embed_ind_dict provides assignments consistent with VQ forward (EMA/entropy/split).
             - Most commonly: embed_ind_dict[skey] is length-B (the whole chunk) and aligned to encoder_outputs
             - Sometimes: embed_ind_dict[skey] is already length-N_i (masked-only). We support both.
          ✅ repel losses are computed ONCE (your code double-counted them).
          ✅ safe-key is resolved WITHOUT creating new keys in this loss function:
             - Prefer (safe, indices) packed in embed_ind_dict[skey] if you provide it
             - Else try to find an existing safe key in `codebook` by matching tensor shape
             - If not found, raise (so you don’t accidentally create new clusters during loss)

        Notes:
          - temperature/use_cosine are kept in signature for compatibility; assignment is taken from embed_ind_dict.
          - codebook is expected to be dict/ParameterDict-like where codebook[safe] gives [K_e, D] or [1,K_e,D].
        """
        import torch
        import torch.nn.functional as F
        import torch.nn as nn

        # ---- 0) shape normalize ----
        encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
        device = encoder_outputs.device
        B, D = encoder_outputs.shape
        dtype = encoder_outputs.dtype

        if chunk_start is None:
            chunk_start = 0
        chunk_end = chunk_start + B

        # ---- 1) empty mask_dict => return graph-connected zeros ----
        if mask_dict is None or len(mask_dict) == 0:
            zero = encoder_outputs.sum() * 0.0
            return zero, zero, zero, zero

        # ---- 2) helpers ----
        def _log(msg: str):
            if logger is not None:
                logger.info(msg)
            else:
                # comment out if noisy
                # print(msg)
                pass

        def _as_long_index(x):
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                x = torch.as_tensor(x, device=device)
            elif not isinstance(x, torch.Tensor):
                try:
                    x = torch.as_tensor(x, device=device)
                except Exception as e:
                    raise TypeError(f"mask_dict entry has unsupported type {type(x)}") from e
            else:
                x = x.to(device)

            if x.numel() == 0:
                return None
            if x.dtype == torch.bool:
                x = torch.nonzero(x, as_tuple=False).view(-1)
                if x.numel() == 0:
                    return None
            return x.long()

        def _unwrap_codebook_tensor(cb_param):
            # cb_param: [K,D] or [1,K,D] or nn.Embedding etc.
            if hasattr(self, "_cb_to_tensor"):
                t = self._cb_to_tensor(cb_param)
            else:
                # minimal fallback
                if isinstance(cb_param, nn.Embedding):
                    t = cb_param.weight
                else:
                    t = cb_param
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t

        def _find_existing_safe_key(skey: str, K_e: int, D: int):
            """
            Try to find an existing safe key in `codebook` whose tensor shape matches (K_e, D).
            This avoids calling _get_or_create_safe_key() inside loss.
            """
            if not isinstance(codebook, (dict, nn.ModuleDict, nn.ParameterDict)):
                return None

            candidates = []
            for kk in codebook.keys():
                try:
                    t = _unwrap_codebook_tensor(codebook[kk])
                    if t.ndim == 2 and t.shape[0] == K_e and t.shape[1] == D:
                        # weak string match: safe keys usually contain skey
                        if skey in str(kk):
                            candidates.append(kk)
                except Exception:
                    continue

            if len(candidates) == 1:
                return candidates[0]

            # fallback: if exact skey exists and matches shape
            if skey in codebook:
                t = _unwrap_codebook_tensor(codebook[skey])
                if t.ndim == 2 and t.shape[0] == K_e and t.shape[1] == D:
                    return skey

            return None

        def _extract_assign_and_safe(skey: str):
            """
            Supports multiple embed_ind_dict formats:
              A) embed_ind_dict[skey] = LongTensor of shape (B,)  (aligned with encoder_outputs)
              B) embed_ind_dict[skey] = LongTensor of shape (N_i,) (already masked-only)
              C) embed_ind_dict[skey] = {"ind": LongTensor, "safe": <safe_key_string>}
              D) embed_ind_dict[skey] = (LongTensor, safe_key_string)
            Returns: (nn_all_or_nn_masked, safe_from_dict_or_None)
            """
            if embed_ind_dict is None:
                raise RuntimeError(
                    "[VQ_COMMIT] embed_ind_dict is required to keep assignments consistent with EMA/entropy/split."
                )
            if skey not in embed_ind_dict:
                raise KeyError(f"[VQ_COMMIT] embed_ind_dict missing skey={skey}")

            obj = embed_ind_dict[skey]

            safe_from_dict = None
            nn_obj = None

            if isinstance(obj, dict):
                # common keys: "ind", "idx", "nn_idx", "safe"
                for key in ("ind", "idx", "nn_idx", "assign"):
                    if key in obj:
                        nn_obj = obj[key]
                        break
                safe_from_dict = obj.get("safe", None)
            elif isinstance(obj, (tuple, list)) and len(obj) == 2:
                nn_obj, safe_from_dict = obj[0], obj[1]
            else:
                nn_obj = obj

            if not isinstance(nn_obj, torch.Tensor):
                nn_obj = torch.as_tensor(nn_obj, device=device)
            nn_obj = nn_obj.to(device=device, dtype=torch.long).reshape(-1)

            return nn_obj, safe_from_dict

        # ---- 3) accumulators ----
        total_latent = 0
        total_cb_count = 0

        commit_num = encoder_outputs.new_zeros(())
        codebk_num = encoder_outputs.new_zeros(())
        repel_num = encoder_outputs.new_zeros(())
        cb_repel_num = encoder_outputs.new_zeros(())

        # ---- 4) main loop over keys ----
        for k, g_idx in mask_dict.items():
            g_idx = _as_long_index(g_idx)
            if g_idx is None:
                continue

            # select indices that fall within this chunk
            in_chunk = (g_idx >= chunk_start) & (g_idx < chunk_end)
            if not torch.any(in_chunk):
                continue

            local_idx = (g_idx[in_chunk] - chunk_start).long()
            if local_idx.numel() == 0:
                continue

            z = encoder_outputs.index_select(0, local_idx)  # [N_i, D]
            N_i = z.shape[0]
            if N_i == 0:
                continue

            skey = str(k)

            # determine desired K_e (requested size)
            K_e_default = int(
                getattr(self, "cb_dict", {}).get(
                    skey,
                    getattr(self, "codebook_size", 0)
                )
            )
            if K_e_default <= 0:
                raise RuntimeError(f"[VQ_COMMIT] invalid K_e_default for skey={skey}: {K_e_default}")

            # get assignment tensor (either length-B or length-N_i) and optional safe key
            nn_all_or_masked, safe_from_dict = _extract_assign_and_safe(skey)

            # resolve safe key WITHOUT creating anything
            safe = None
            if safe_from_dict is not None:
                safe = safe_from_dict
            else:
                safe = _find_existing_safe_key(skey, K_e_default, D)

            if safe is None:
                raise RuntimeError(
                    f"[VQ_COMMIT] Could not resolve existing safe key for skey={skey} "
                    f"(wanted shape K_e={K_e_default}, D={D}).\n"
                    f"Fix: pass safe key alongside embed_ind_dict, e.g.\n"
                    f"  embed_ind_dict[skey] = {{'ind': <(B,)>, 'safe': <safe_key>}}\n"
                    f"or ensure codebook contains a key that includes '{skey}' with tensor shape ({K_e_default},{D})."
                )

            if safe not in codebook:
                raise KeyError(
                    f"[VQ_COMMIT] safe-key '{safe}' not found in codebook (from skey='{skey}')."
                )

            cb_param = codebook[safe]
            cb = _unwrap_codebook_tensor(cb_param).to(device=device, dtype=dtype)  # [K_e, D]
            if cb.ndim != 2 or cb.shape[1] != D:
                raise RuntimeError(
                    f"[VQ_COMMIT] codebook tensor shape mismatch skey={skey} safe={safe}: got {tuple(cb.shape)}, expected (*,{D})"
                )
            K_e = cb.shape[0]
            if K_e != K_e_default:
                # not fatal, but warn; assignment OOB check uses actual K_e
                _log(f"[VQ_COMMIT][WARN] skey={skey} expected K_e={K_e_default}, actual K_e={K_e} (safe={safe})")

            # ---- assignment selection aligned with z ----
            # Case A: nn_all_or_masked is length-B (aligned to encoder_outputs)
            # Case B: nn_all_or_masked is length-N_i (already masked-only)
            if nn_all_or_masked.numel() == B:
                nn_idx = nn_all_or_masked.index_select(0, local_idx)  # [N_i]
            elif nn_all_or_masked.numel() == N_i:
                nn_idx = nn_all_or_masked
            else:
                raise ValueError(
                    f"[VQ_COMMIT] nn_idx length mismatch skey={skey}: "
                    f"got {nn_all_or_masked.numel()} (expected B={B} or N_i={N_i})."
                )

            # ---- OOB guard before indexing ----
            min_i = int(nn_idx.min().item())
            max_i = int(nn_idx.max().item())
            if (min_i < 0) or (max_i >= K_e):
                raise RuntimeError(
                    f"[VQ_COMMIT][OOB] skey={skey} safe={safe} "
                    f"K_e={K_e} nn_idx range=[{min_i},{max_i}] cb.shape={tuple(cb.shape)}"
                )

            # ---- gather and losses ----
            e_star = cb.index_select(0, nn_idx)  # [N_i, D]
            _log(f"[VQ_COMMIT][DBG] skey={skey} "
                 f"z_norm={z.float().norm(dim=-1).mean().item():.4f} "
                 f"e_norm={e_star.float().norm(dim=-1).mean().item():.4f} "
                 f"mse32={F.mse_loss(z.float(), e_star.detach().float()).item():.3e}")

            commit_part = F.mse_loss(z, e_star.detach(), reduction="mean")
            codebk_part = F.mse_loss(e_star, z.detach(), reduction="mean")

            total_latent += int(N_i)
            logger.info(f"commit_part {commit_part}, N_i {N_i}")
            commit_num = commit_num + commit_part * N_i
            codebk_num = codebk_num + codebk_part * N_i

            # ---- repel ONCE (your code double-counted this) ----
            # Keep your existing API. We pass cb_param (original object) to preserve old behavior.
            ret = self.compute_contrastive_loss(z, 0, logger, cb_param, k)
            repel_val = ret[0]
            cb_repel_val = ret[3]  # NOTE: verify your compute_contrastive_loss return layout!

            repel_num = repel_num + repel_val * N_i

            total_cb_count += int(K_e)
            cb_repel_num = cb_repel_num + cb_repel_val * K_e

        # ---- 5) aggregate ----
        if total_latent == 0:
            zero = encoder_outputs.sum() * 0.0
            _log(f"[VQ_EMPTY] total_latent=0 in chunk [{chunk_start},{chunk_end})")
            return zero, zero, zero, zero

        commit_loss = beta * (commit_num / float(total_latent))
        codebook_loss = (codebk_num / float(total_latent))
        repel_loss = (repel_num / float(total_latent))

        if total_cb_count > 0:
            cb_repel_loss = cb_repel_num / float(total_cb_count)
        else:
            cb_repel_loss = cb_repel_num * 0.0

        return commit_loss, codebook_loss, repel_loss, cb_repel_loss

    import torch
    import torch.nn.functional as F
    from einops import rearrange

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
            self,
            x,
            feature=None,
            mask_dict=None,
            logger=None,
            chunk_i=None,
            epoch=None,
            mode=None,
    ):
        """
        Forward pass with per-element quantization + EMA update + commitment loss.

        Contract (IMPORTANT):
          - _codebook() must return embed_ind_dict such that:
                embed_ind_dict[skey] = {"ind": LongTensor(N_i), "safe": str}
            (If your _codebook currently returns only LongTensor, update _codebook first.)
          - commitment_loss will use safe from embed_ind_dict to index the codebook container.
        """
        import torch
        from einops import rearrange
        import math

        # -----------------------------
        # 0) Global latent offset
        # -----------------------------
        if not hasattr(self, "latent_size_sum"):
            self.latent_size_sum = 0

        if mode in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = 0
        elif chunk_i is not None and chunk_i == 0:
            self.latent_size_sum = 0

        # -----------------------------
        # 1) Flatten encoder outputs -> [B_total, D]
        # -----------------------------
        x = x.float()

        if getattr(self, "accept_image_fmap", False) and x.ndim == 4:
            encoder_outputs = rearrange(x, "b c h w -> (b h w) c")
        else:
            encoder_outputs = x.reshape(-1, x.size(-1))

        B = int(encoder_outputs.size(0))
        device = encoder_outputs.device
        dtype = encoder_outputs.dtype

        global_start = int(self.latent_size_sum)
        global_end = global_start + B

        # -----------------------------
        # 2) Special modes: init_kmeans_final / infer
        # -----------------------------
        if mode == "init_kmeans_final":
            _ = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode=mode,
            )
            z = torch.zeros((), device=device, dtype=dtype)
            return z, (z, z, z, z)

        if mode == "infer":
            out = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode=mode,
            )

            # Preferred / canonical contract from _codebook:
            # (key_id_full, cluster_id_full, global_id_full, id2safe)
            if isinstance(out, (tuple, list)) and len(out) == 4:
                key_id_full, cluster_id_full, global_id_full, id2safe = out
                key_id_full = key_id_full.reshape(-1).long()
                cluster_id_full = cluster_id_full.reshape(-1).long()
                global_id_full = global_id_full.reshape(-1).long()

                n = key_id_full.numel()
                if cluster_id_full.numel() != n or global_id_full.numel() != n:
                    raise ValueError(
                        f"[vq.forward] infer length mismatch: "
                        f"kid={n}, cid={cluster_id_full.numel()}, gid={global_id_full.numel()}"
                    )
                if not isinstance(id2safe, dict):
                    id2safe = {}

                return key_id_full, cluster_id_full, global_id_full, id2safe

            # Backward compat (very old checkpoints) — keep as fallback if you want:
            if isinstance(out, (tuple, list)) and len(out) == 3:
                # old: (key_id_full, cluster_id_full, id2safe)  -> use key_id as global_id
                key_id_full, cluster_id_full, id2safe = out
                key_id_full = key_id_full.reshape(-1).long()
                cluster_id_full = cluster_id_full.reshape(-1).long()
                if key_id_full.numel() != cluster_id_full.numel():
                    raise ValueError("[vq.forward] infer length mismatch (tuple3)")
                global_id_full = key_id_full
                if not isinstance(id2safe, dict):
                    id2safe = {}
                return key_id_full, cluster_id_full, global_id_full, id2safe

            raise TypeError(
                f"[vq.forward] mode='infer' expects _codebook() -> "
                f"(key_id_full, cluster_id_full, global_id_full, id2safe), got {type(out)} "
                f"len={len(out) if isinstance(out, (tuple, list)) else 'n/a'}"
            )

        # -----------------------------
        # 3) Debug index sanity (optional)
        # -----------------------------
        if getattr(self, "debug_index", False) and (mask_dict is not None) and (B > 0):
            if mode in ("eval", "test", "init_kmeans_final") or (chunk_i == 0):
                assert global_start == 0, f"[INDEX_MISMATCH] chunk head but global_start={global_start}"

            n_checked, n_oob = 0, 0
            for k, idxs in mask_dict.items():
                if idxs is None:
                    continue
                gi = idxs.to(device=device) if isinstance(idxs, torch.Tensor) else torch.as_tensor(idxs, device=device)
                if gi.numel() == 0:
                    continue
                gi = gi.reshape(-1)
                if gi.numel() > 256:
                    gi = gi[:256]
                in_chunk = (gi >= global_start) & (gi < global_end)
                n_oob += int((~in_chunk).sum().item())
                n_checked += int(gi.numel())
                if n_checked >= 1024:
                    break

            assert n_oob == 0, (
                f"[INDEX_MISMATCH] mask_dict contains out-of-chunk indices. "
                f"global_range=[{global_start},{global_end}), checked={n_checked}, oob={n_oob}."
            )

            # quick latent check (cheap)
            k0 = next(iter(mask_dict.keys()), None)
            if k0 is not None:
                gi0 = mask_dict[k0][0] if isinstance(mask_dict[k0], (list, tuple)) else mask_dict[k0].reshape(-1)[
                    0].item()
                li0 = int(gi0) - global_start
                if 0 <= li0 < B:
                    z_norm = float(encoder_outputs[li0].norm().item())
                    assert (z_norm > 0) and (
                        not math.isnan(z_norm)), f"latent invalid at global_id={gi0}, local_id={li0}"

        # -----------------------------
        # 4) Default losses (0-dim tensors)
        # -----------------------------
        z0 = torch.zeros((), device=device, dtype=dtype)
        commit_loss = z0
        codebook_loss = z0
        repel_loss = z0
        cb_repel_loss = z0
        ent_loss = z0

        # -----------------------------
        # 5) Train path: run _codebook first (assign + EMA + entropy)
        # -----------------------------
        if (mask_dict is not None) and (B > 0):
            # quantize_st, self.embed_ind_dict, self.embed, ent_metric_total
            quantize_st, embed_ind_dict, codebook_container, ent_loss = self._codebook(
                encoder_outputs,
                feature=feature,
                mask_dict=mask_dict,
                logger=logger,
                chunk_i=chunk_i,
                epoch=epoch,
                mode="train",
            )

            # ---- HARD REQUIRE: embed_ind_dict contains {"ind","safe"} per skey
            # This prevents "Could not resolve existing safe key..."
            # and keeps commitment aligned with EMA/split assignment.
            if not isinstance(embed_ind_dict, dict):
                raise TypeError("[vq.forward] embed_ind_dict must be dict")
            # (light check only)
            for _k, _v in list(embed_ind_dict.items())[:3]:
                if not (isinstance(_v, dict) and ("ind" in _v) and ("safe" in _v)):
                    raise RuntimeError(
                        "[vq.forward] embed_ind_dict must store {'ind':..., 'safe':...} per skey. "
                        "Fix _codebook() to return that structure."
                    )

            # ---- commitment loss uses assignments from embed_ind_dict
            commit_loss, codebook_loss, repel_loss, cb_repel_loss = self.commitment_loss(
                encoder_outputs=encoder_outputs,
                mask_dict=mask_dict,
                codebook=codebook_container,  # <-- IMPORTANT: must support codebook[safe]
                codebook_mod=self._codebook,  # ok to keep for helpers/logging
                logger=logger,
                chunk_start=global_start,
                beta=getattr(self, "beta", 0.25),
                temperature=getattr(self, "temperature", None),
                use_cosine=getattr(self, "use_cosine", False),
                embed_ind_dict=embed_ind_dict,  # contains safe + ind
            )

        # -----------------------------
        # 6) Update global offset (train only)
        # -----------------------------
        if mode not in ("eval", "test", "init_kmeans_final"):
            self.latent_size_sum = global_end

        # -----------------------------
        # 7) Return
        # -----------------------------
        # (あなたの今の return に合わせて: total_loss = commit + ent)
        total_loss = self.commitment_weight * commit_loss + ent_loss
        logger.info(f"commit loss {commit_loss}, ent_loss {ent_loss}, total_loss {total_loss}")
        return total_loss, (commit_loss, codebook_loss, repel_loss, cb_repel_loss, ent_loss)


# ==============================================================================
# ORIGINAL MODULE: models.py
# ==============================================================================
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

class AtomEmbedding(nn.Module):
    def __init__(self):
        super(AtomEmbedding, self).__init__()

        # ---- 埋め込み定義（離散 0-30 部分）----
        self.degree_embed   = nn.Embedding(num_embeddings=7,   embedding_dim=4)   # 0..6
        self.ring_embed     = nn.Embedding(num_embeddings=2,   embedding_dim=4)   # ring flag+1 など
        self.charge_embed   = nn.Embedding(num_embeddings=8,   embedding_dim=4)   # 0..7 にクリップ
        self.aromatic_embed = nn.Embedding(num_embeddings=2,   embedding_dim=4)   # 0/1
        self.hybrid_embed   = nn.Embedding(num_embeddings=6,   embedding_dim=4)   # 0..5
        self.hydrogen_embed = nn.Embedding(num_embeddings=5,   embedding_dim=4)   # 0..4

        ELEMENTS = [5, 6, 7, 8, 14, 15, 16]
        self.register_buffer(
            "element_lut",
            self._build_element_lut(ELEMENTS)
        )
        self.element_embed = nn.Embedding(num_embeddings=len(ELEMENTS), embedding_dim=4)

        # 0/1 フラグ系は全部 2 クラス想定
        def flag_emb():
            return nn.Embedding(num_embeddings=2, embedding_dim=2)

        # 官能基フラグ 18 個 (7-24)
        self.func_embed_0  = flag_emb()
        self.func_embed_1  = flag_emb()
        self.func_embed_2  = flag_emb()
        self.func_embed_3  = flag_emb()
        self.func_embed_4  = flag_emb()
        self.func_embed_5  = flag_emb()
        self.func_embed_6  = flag_emb()
        self.func_embed_7  = flag_emb()
        self.func_embed_8  = flag_emb()
        self.func_embed_9  = flag_emb()
        self.func_embed_10 = flag_emb()
        self.func_embed_11 = flag_emb()
        self.func_embed_12 = flag_emb()
        self.func_embed_13 = flag_emb()
        self.func_embed_14 = flag_emb()
        self.func_embed_15 = flag_emb()
        self.func_embed_16 = flag_emb()
        self.func_embed_17 = flag_emb()

        # H-bond Donor / Acceptor (25, 26)
        self.h_don_embed   = flag_emb()
        self.h_acc_embed   = flag_emb()

        # ring size / aromatic neighbors / fused-id (27, 28, 29)
        # ringSize のユニーク値: ['0', '3', '4', '5', '6', '7', '8'] → 7種類
        self.ringsize_embed  = nn.Embedding(num_embeddings=7, embedding_dim=4)
        self.aroma_num_embed = nn.Embedding(num_embeddings=5, embedding_dim=4)  # 0..4
        self.fused_if_embed  = nn.Embedding(num_embeddings=8, embedding_dim=4)  # 0..7

        # NEW: het27 (30) 用の embedding (0..26 → 27クラス)
        self.het27_embed = nn.Embedding(num_embeddings=27, embedding_dim=4)

        # 官能基フラグ 18 個 (各 2 次元) → 36 次元を 4 次元に圧縮
        self.func_reduce = nn.Linear(18 * 2, 4)

        # ringSize の元の値 → index の対応（0,3,4,5,6,7,8 → 0..6）
        uniq = ['0', '3', '4', '5', '6', '7', '8']
        uniq_int = sorted(int(x) for x in uniq)  # [0,3,4,5,6,7,8]
        mapping = {v: i for i, v in enumerate(uniq_int)}
        self.register_buffer(
            "ring_values_tensor",
            torch.tensor(uniq_int, dtype=torch.long)
        )
        self.ring_value_to_index = mapping

        # ---- bond_env_raw (31-78, 48 dims) 用の射影 ----
        # 48 次元 → 16 次元に圧縮して concat
        self.bond_env_proj = nn.Linear(48, 16)

        # ---- 離散部分（x0..x6 + flags4 + x25,x26 + x27,x28,x29,x30）を 52→48 に圧縮 ----
        #   x0..x6: 7*4 = 28
        #   flags4: 4
        #   x25,x26: 2+2 = 4
        #   x27,x28,x29,x30: 4*4 = 16
        #   合計 = 28 + 4 + 4 + 16 = 52
        self.disc_proj = nn.Linear(52, 48)

        # このクラスの最終出力次元:
        #   離散部 (disc_proj 後) : 48
        #   bond_env_proj         : 16
        #   合計 = 64
        self.out_dim = 48 + 16

    @staticmethod
    def _build_element_lut(ELEMENTS):
        max_z = max(ELEMENTS)
        lut = torch.zeros(max_z + 1, dtype=torch.long)
        for i, z in enumerate(ELEMENTS):
            lut[z] = i
        return lut

    def forward(self, atom_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode per-atom features into a dense embedding.

        Parameters
        ----------
        atom_inputs : torch.Tensor
            Shape [N, 79].

            0 : Z
            1 : degree
            2 : charge
            3 : hyb
            4 : arom
            5 : ring
            6 : hcount
            7-24  : func_flags (18)
            25    : H-donor flag
            26    : H-acceptor flag
            27    : ringSize (0,3,4,5,6,7,8,...)
            28    : #aromatic neighbors
            29    : fused ring id (0..7)
            30    : het27 (0..26)
            31-78 : bond_env_raw (48)  float
        Returns
        -------
        out : torch.Tensor
            Shape [N, 64] = disc_emb(48) + bond_env_emb(16)
        """
        device = next(self.parameters()).device
        x = atom_inputs.to(device=device, non_blocking=True)

        if x.ndim != 2 or x.size(1) != 79:
            raise ValueError(f"atom_inputs must be [N,79], got {tuple(x.shape)}")

        # ----------------------------
        # helpers
        # ----------------------------
        def clamp_idx(col: int, n_embed: int, lo: int = 0) -> torch.Tensor:
            """Read integer column and clamp to valid embedding range."""
            return x[:, col].long().clamp(lo, n_embed - 1)

        def flag01(col: int) -> torch.Tensor:
            """Read 0/1 flag column."""
            return x[:, col].long().clamp(0, 1)

        # ----------------------------
        # core discrete features
        # ----------------------------

        # 0: Z -> element_lut -> element_embed
        z_raw = x[:, 0].long()
        max_z = int(self.element_lut.shape[0] - 1)
        z_safe = torch.where((0 <= z_raw) & (z_raw <= max_z), z_raw, torch.zeros_like(z_raw))
        idx0 = self.element_lut[z_safe].clamp(0, self.element_embed.num_embeddings - 1)
        x0 = self.element_embed(idx0)

        # 1: degree
        x1 = self.degree_embed(clamp_idx(1, self.degree_embed.num_embeddings))

        # 2: ring flag (+1) for ring_embed (0..)
        ring_plus1 = (x[:, 5].long() + 1).clamp(0, self.ring_embed.num_embeddings - 1)
        x2 = self.ring_embed(ring_plus1)

        # 3: charge
        x3 = self.charge_embed(clamp_idx(2, self.charge_embed.num_embeddings))

        # 4: aromatic (0/1)
        x4 = self.aromatic_embed(clamp_idx(4, self.aromatic_embed.num_embeddings))

        # 5: hybrid
        x5 = self.hybrid_embed(clamp_idx(3, self.hybrid_embed.num_embeddings))

        # 6: #hydrogens
        x6 = self.hydrogen_embed(clamp_idx(6, self.hydrogen_embed.num_embeddings))

        # ----------------------------
        # functional flags (7..24): 18 flags -> per-flag embed -> reduce
        # ----------------------------
        # assumes each func_embed_i is an Embedding(2, d_flag) (you used 2 dims each before)
        func_embeds = [
            self.func_embed_0,  self.func_embed_1,  self.func_embed_2,  self.func_embed_3,
            self.func_embed_4,  self.func_embed_5,  self.func_embed_6,  self.func_embed_7,
            self.func_embed_8,  self.func_embed_9,  self.func_embed_10, self.func_embed_11,
            self.func_embed_12, self.func_embed_13, self.func_embed_14, self.func_embed_15,
            self.func_embed_16, self.func_embed_17,
        ]
        flags = torch.cat([emb(flag01(7 + i)) for i, emb in enumerate(func_embeds)], dim=-1)  # [N, 18*d_flag]
        flags4 = self.func_reduce(flags)  # [N, 4]

        # 25,26: H-bond donor/acceptor flags
        x25 = self.h_don_embed(flag01(25))
        x26 = self.h_acc_embed(flag01(26))

        # ----------------------------
        # ring size mapping (27)
        # ----------------------------
        raw27 = x[:, 27].long()

        # default: last index = "other/unknown"
        mapped27 = torch.full_like(raw27, fill_value=self.ringsize_embed.num_embeddings - 1)

        # map known values via dict {value:int_index}
        # NOTE: this loop is small (few keys), so it's fine
        for v, idx in self.ring_value_to_index.items():
            mapped27[raw27 == int(v)] = int(idx)

        mapped27 = mapped27.clamp(0, self.ringsize_embed.num_embeddings - 1)
        x27 = self.ringsize_embed(mapped27)

        # 28: #aromatic neighbors
        x28 = self.aroma_num_embed(clamp_idx(28, self.aroma_num_embed.num_embeddings))

        # 29: fused ring id
        x29 = self.fused_if_embed(clamp_idx(29, self.fused_if_embed.num_embeddings))

        # 30: het27
        x30 = self.het27_embed(clamp_idx(30, self.het27_embed.num_embeddings))

        # ----------------------------
        # bond_env_raw (31..78): 48 floats -> proj -> 16
        # ----------------------------
        bond_env = x[:, 31:].to(dtype=torch.float32)
        if bond_env.size(1) != 48:
            raise ValueError(f"bond_env_raw must be 48 dims, got {bond_env.size(1)}")
        bond_env_emb = self.bond_env_proj(bond_env)  # [N, 16]

        # ----------------------------
        # concatenate discrete, project, then concat with bond env
        # ----------------------------
        disc_cat = torch.cat(
            [
                x0, x1, x2, x3, x4, x5, x6,   # base discrete embeds
                flags4,                       # [N,4]
                x25, x26,                     # donor/acceptor
                x27, x28, x29, x30,           # ringSize, aromNbrs, fusedId, het27
            ],
            dim=-1,
        )  # expected [N, 52] in your original design

        disc_emb = self.disc_proj(disc_cat)          # [N, 48]
        out = torch.cat([disc_emb, bond_env_emb], dim=-1)  # [N, 64]
        return out

class EquivariantThreeHopGINE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, args):
        super().__init__()
        import torch
        if args is None:
            args = get_args()

        self.feat_embed = AtomEmbedding()
        # AtomEmbedding → 64 dims (48 from discrete + 16 from bond_env)
        self.linear_0 = nn.Linear(64, args.hidden_dim)  # h0

        edge_emb_dim = getattr(args, "edge_emb_dim", 32)
        self.bond_emb = nn.Embedding(5, edge_emb_dim, padding_idx=0)

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

        # Edge attributes (bond types {1..4}, 0 otherwise)
        eb = data.edata.get("weight", torch.zeros(data.num_edges(), dtype=torch.long, device=s1.device))
        eb = eb.to(dev, non_blocking=True)
        e  = torch.cat([eb, eb], 0)
        e  = torch.where((e >= 1) & (e <= 4), e, torch.zeros_like(e))
        edge_attr = self.bond_emb(e.long())

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


# ==============================================================================
# ORIGINAL MODULE: new_train_and_eval.py
# ==============================================================================
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import dgl.dataloading

from collections import Counter

DATAPATH = "../data/discret_final"
DATAPATH_INFER = "../data/pretrain_no_id"

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
        # after A_sel is created
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
    Restore checkpoint -> infer only -> save per-atom outputs.

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
    MoleculeGraphDataset, collate_fn, convert_to_dgl, evaluate
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
    dataset = MoleculeGraphDataset(adj_dir=datapath, attr_dir=datapath)
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
        # init_kmeans_loop : collect latents
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




# ==============================================================================
# ORIGINAL MODULE: train_teacher.py
# ==============================================================================
import argparse
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.autograd.set_detect_anomaly(True)

import multiprocessing

# Set the number of threads for OpenMP
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Set thread affinity to improve data loading performance
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'


def build_model(conf):
    """
    Reconstruct the exact model architecture used in training
    """
    args = conf["args"]   # we pass your training args through conf

    model = EquivariantThreeHopGINE(
        in_feats=args.hidden_dim,
        hidden_feats=args.hidden_dim,
        out_feats=args.hidden_dim,
        args=args,
    )
    return model

import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEBUG_INFO"] = "0"
os.environ["CUDA_VERBOSE_LOGGING"] = "0"


def run(args):

    """ Set seed, device, and logger """
    set_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        print(f"$$$$$$$$$$$$$$$$$$$  USING CPU ????? $$$$$$$$$$$$$$$$$$$")
        device = "cpu"
    torch.cuda.empty_cache()
    if args.feature_noise != 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        )
    if args.feature_aug_k > 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        args.teacher = f"GA{args.feature_aug_k}{args.teacher}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    args.feat_dim = args.hidden_dim
    args.label_dim = 1

    """ Model config """
    conf = {}
    conf = dict(args.__dict__, **conf)
    conf["device"] = device

    """ Model init """
    # in_feats, hidden_feats, out_feats, args
    model = EquivariantThreeHopGINE(in_feats=args.hidden_dim, hidden_feats=args.hidden_dim, out_feats=args.hidden_dim, args=args)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'], weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    from torch.optim.lr_scheduler import CosineAnnealingLR

    T_max = conf["cosine_epochs"]  # 例: 50 or 100
    eta_min = conf.get("min_lr", 5e-6)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
    )
    if args.train_or_infer == "hptune":
        run_inductive(
            conf,
            model,
            optimizer,
            scheduler,
            logger
        )
    elif args.train_or_infer == "infer":
        run_infer_after_restore(conf, model, logger, "../data/model_epoch_3.pt")

def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = ""
        # score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    # print(score_str)


import requests

def stop_instance(instance_id, api_key):
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/stop/"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print("Instance stopped successfully.")
    else:
        print(f"Failed to stop instance: {response.status_code}, {response.text}")



    # # Replace with your Vast.ai instance ID and API key
    # instance_id = "13941138"
    # api_key = "2afa99317d40e9892f7f8b088a84641b4be2279be70103847bb480fb21997354"
    # stop_instance(instance_id, api_key)

# Standalone entrypoint
if __name__ == "__main__":
    main()
