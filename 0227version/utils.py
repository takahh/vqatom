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
    '16_0_4_0_0_0': 356,
    '6_0_3_0_0_0': 625,
    '6_0_3_1_1_0': 1453,
    '6_0_4_0_0_0': 900,
    '7_0_4_0_0_0': 220,
    '8_0_3_0_0_0': 524,
    '6_0_4_0_1_0': 1138,
    '7_0_3_0_0_0': 637,
    '7_0_3_0_1_0': 791,
    '7_0_3_1_1_0': 883,
    '8_0_4_0_1_0': 317,
    '35_0_4_0_0_0': 73,
    '6_0_2_0_0_0': 146,
    '7_0_2_0_0_0': 32,
    '8_0_4_0_0_0': 259,
    '8_0_3_1_1_0': 194,
    '7_0_4_0_1_0': 389,
    '17_0_4_0_0_0': 143,
    '9_0_4_0_0_0': 142,
    '7_-1_3_0_0_0': 13,
    '7_1_2_0_0_0': 8,
    '6_0_3_0_1_0': 857,
    '8_0_3_0_1_0': 223,
    '53_0_4_0_0_0': 20,
    '1_0_1_0_0_0': 6,
    '7_1_4_0_0_0': 65,
    '16_0_3_1_1_0': 216,
    '15_0_4_0_0_0': 64,
    '16_0_4_0_1_0': 186,
    '7_1_3_0_0_0': 59,
    '8_-1_3_0_0_0': 44,
    '16_1_4_0_0_0': 18,
    '8_-1_4_0_0_0': 28,
    '7_1_4_0_1_0': 53,
    '16_0_3_0_0_0': 62,
    '5_0_3_0_0_0': 13,
    '7_1_3_1_1_0': 53,
    '14_0_4_0_0_0': 7,
    '15_0_4_0_1_0': 8,
    '7_1_3_0_1_0': 8,
    '5_-1_4_0_1_0': 2,
    '15_1_4_0_0_0': 2,
    '34_0_3_1_1_0': 4,
}



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

