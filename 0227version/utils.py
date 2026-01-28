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

