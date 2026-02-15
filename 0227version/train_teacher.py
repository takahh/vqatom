import argparse
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)

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

from models import EquivariantThreeHopGINE

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
    from models import Model
    from new_train_and_eval import run_inductive

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
    from models import EquivariantThreeHopGINE
    from args import get_args
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
    from new_train_and_eval import run_infer_only_after_restore
    if args.train_or_infer == "hptune":
        run_inductive(
            conf,
            model,
            optimizer,
            scheduler,
            logger
        )
    elif args.train_or_infer == "infer":
        run_infer_only_after_restore(conf, model, logger, "../data/model_epoch_4.pth")

def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    from args import get_args
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


if __name__ == "__main__":
    main()

    # # Replace with your Vast.ai instance ID and API key
    # instance_id = "13941138"
    # api_key = "2afa99317d40e9892f7f8b088a84641b4be2279be70103847bb480fb21997354"
    # stop_instance(instance_id, api_key)