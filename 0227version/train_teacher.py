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

import multiprocessing

# Set the number of threads for OpenMP
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Set thread affinity to improve data loading performance
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'


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
        "--samples_latent_in_kmeans", type=int, default=10000, help="samples of latent in init_embed"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
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
    parser.add_argument("--learning_rate", type=float, default=0.0001) # default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
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

import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEBUG_INFO"] = "0"
os.environ["CUDA_VERBOSE_LOGGING"] = "0"


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """
    from models import Model
    from new_train_and_eval import run_inductive

    """ Set seed, device, and logger """
    set_seed(args.seed)

    # print("Is GPU available?", torch.cuda.is_available())
    # print("Current device:", torch.cuda.current_device())
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
    from dataloader import load_data
    #
    # """ Load data """
    # g, labels, idx_train, idx_val, idx_test = load_data(
    #     args.dataset,
    #     args.data_path,
    #     split_idx=args.split_idx,
    #     seed=args.seed,
    #     labelrate_train=args.labeate_train,
    #     labelrate_val=args.labelrate_val,
    #     train_or_infer=args.train_or_infer,
    #     percent=args.percent
    # )
    # # logger.info(f"Total {g.number_of_nodes()} nodes.")
    args.feat_dim = args.hidden_dim
    args.label_dim = 1

    # if 0 < args.feature_noise <= 1:
    #     feats = (
    #         1 - args.feature_noise
    #     ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    # if args.model_config_path is not None:
    #     conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    # logger.info(f"conf: {conf}")
    """ Model init """
    from models import EquivariantThreeHopGINE
    # in_feats, hidden_feats, out_feats, args
    model = EquivariantThreeHopGINE(in_feats=args.hidden_dim, hidden_feats=args.hidden_dim, out_feats=args.hidden_dim, args=args)
    # model = Model(conf)

    # if conf["train_or_infer"] == "infer":
    #     model.load_state_dict(torch.load("./model_epoch_200.pth", weights_only=False))

    if conf["train_or_infer"] != "hptune" or conf["use_checkpoint"] == True:
        thiskey = f"{conf['codebook_size']}_{conf['hidden_dim']}"
        best_epoch_dict = {'10000_16': 8}
        model.load_state_dict(torch.load(f"/vqatom/0227version/model_epoch_{best_epoch_dict[thiskey]}.pth", weights_only=False, map_location=device))
        buffer_dict = torch.load(f"/vqatom/0227version/model_buffers_{best_epoch_dict[thiskey]}.pth", map_location=device, weights_only=False)
        model.__dict__.update(buffer_dict)
        print(f"LOADED best epoch number {best_epoch_dict[thiskey]} model ^^^^^^^^^^^^^")

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'], weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # criterion = torch.nn.NLLLoss()
    # evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []
    latent_train_list = None
    # if args.exp_setting == "tran":
    #     # indices = (idx_train, idx_val, idx_test)
    #
    #     # propagate node feature
    #     if args.feature_aug_k > 0:
    #         feats = feature_prop(feats, g, args.feature_aug_k)
    #
    #     # out, score_val, score_test, h_list, dist, codebook =
    #     out, score_val, score_test, h_list, dist, codebook, loss_list = run_transductive(
    #         conf,
    #         model,
    #         g,
    #         feats,
    #         labels,
    #         indices,
    #         criterion,
    #         evaluator,
    #         optimizer,
    #         logger,
    #         loss_and_score,
    #     )
    #     # print(dist.shape)
    #
    #     score_lst = [score_test]

    if args.exp_setting == "ind":
        # indices = (obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind)  indices[4] is idx_test_ind
        # --------------------------------------------
        # make train/valid/test data into mini-batches
        # --------------------------------------------
        # indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed, args.train_or_infer)

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        # if args.feature_aug_k > 0:
        #     idx_obs = indices[3]
        #     obs_g = g.subgraph(idx_obs)
        #     obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
        #     feats = feature_prop(feats, g, args.feature_aug_k)
        #     feats[idx_obs] = obs_feats

        # ----------------------------------------------------------------------------
        # run the model
        # ----------------------------------------------------------------------------
        # out, score_val, score_test_tran, score_test_ind, h_list, dist, codebook, latents_trans, latents_ind, latent_train_list = run_inductive(
        #     conf,
        #     model,
        #     optimizer,
        #     args.accumulation_steps,
        #     logger
        # )
        run_inductive(
            conf,
            model,
            optimizer,
            args.accumulation_steps,
            logger
        )
        # score_lst = [score_test_tran, score_test_ind]
    #
    # logger.info(
    #     f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    # )
    # logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    # def squeezeit(dlist):
    #     dlist = [torch.squeeze(x) for x in dlist]
    #     return dlist
    # # print(f"{squeezeit(latent_train_list)} squeezeit(latent_train_list)")
    # latent_train_list = torch.cat(squeezeit(latent_train_list), dim=0)
    # # Generate random indices for sampling
    # random_indices = torch.randperm(latent_train_list.size(0))[:30000]
    # # Sample the tensor
    # latent_train_list_sampled = latent_train_list[random_indices]

    # """ Saving teacher outputs """
    # out_np = out.detach().cpu().numpy()
    # out_codebook = codebook.detach().cpu().numpy()
    # out_emb = h_list[0].detach().cpu().numpy()
    # dist_vq = dist.detach().cpu().numpy()
    # # latents_ind = F.normalize(latents_ind, p=2, dim=-1)
    # # latents_trans = F.normalize(latents_trans, p=2, dim=-1)
    # # print(f"latent : tran {latents_trans.shape}")
    # # print(f"latent : ind {latents_ind.shape}")
    # np.savez(output_dir.joinpath("tea_soft_labels"), out_np)
    # np.savez(output_dir.joinpath("codebook_embeddings"), out_codebook)
    # np.savez(output_dir.joinpath("codebook"), codebook.cpu().detach().numpy())
    # np.savez(output_dir.joinpath("out_emb_list"), out_emb)
    # # np.savez(output_dir.joinpath("latents_trans"), latents_trans.cpu())
    # # np.savez(output_dir.joinpath("latents_ind"), latents_ind.cpu())
    # np.savez(output_dir.joinpath("latent_train_list"), latent_train_list_sampled.cpu().detach().numpy())
    # np.savez_compressed(output_dir.joinpath("tea_soft_token_assignments"), dist_vq)

    """ Saving loss curve and model """
    # if args.save_results:
        # Loss curves
    # loss_and_score = np.array(loss_and_score.cpu())
    # np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

    # Model
    # torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    # if args.exp_setting == "tran" and args.compute_min_cut:
    #     min_cut = compute_min_cut_loss(g, out)
    #     with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
    #         f.write(f"{min_cut :.4f}\n")
    #
    # return score_lst


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


if __name__ == "__main__":
    main()

    # # Replace with your Vast.ai instance ID and API key
    # instance_id = "13941138"
    # api_key = "2afa99317d40e9892f7f8b088a84641b4be2279be70103847bb480fb21997354"
    # stop_instance(instance_id, api_key)