import numpy as np
import copy
import torch
import dgl
from utils import set_seed
import dgl.dataloading
from train_teacher import get_args


"""
1. Train and eval
"""

import dgl
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import dgl
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def filter_small_graphs_from_blocks(input_nodes, output_nodes, blocks, step, mode, min_size=6):
    filtered_blocks = []
    filtered_input_nodes = []
    filtered_output_nodes = []

    for blk_idx, block in enumerate(blocks):
        src, dst = block.edges()  # Get edge list directly
        # Convert to sparse adjacency matrix format (FASTER than dense numpy conversion)
        num_nodes = block.num_nodes()
        adj_matrix_sparse = csr_matrix(
            (torch.ones_like(src).cpu().numpy(), (src.cpu().numpy(), dst.cpu().numpy())),
            shape=(num_nodes, num_nodes)
        )

        # Identify connected components (independent small graphs)
        num_components, labels = connected_components(csgraph=adj_matrix_sparse, directed=False)

        # Filter out small graphs and nodes > 9999
        keep_nodes = []
        labels_tensor = torch.tensor(labels, device=input_nodes.device)  # Faster than NumPy conversion
        for i in range(num_components):
            component_nodes = torch.where(labels_tensor == i)[0]
            valid_nodes = [node for node in component_nodes.tolist() if node <= 9999]  # ✅ Remove nodes > 9999
            if len(valid_nodes) >= min_size:
                keep_nodes.extend(valid_nodes)

        args = get_args()
        train_size = args.train_size
        batch_size = args.batch_size

        if keep_nodes:
            filtered_blocks.append(block)  # Keep this block

            # ✅ Map `keep_nodes` back to global indices before indexing
            global_keep_nodes = input_nodes[keep_nodes]  # Convert local to global node indices
            # print(f"input nodes selected {global_keep_nodes[:20]} {global_keep_nodes[-20:]}")
            # print(f"input nodes {input_nodes[:20]} {input_nodes[-20:]}")

            if mode == "train":
                valid_input_nodes = input_nodes[(input_nodes < (step + 1) * batch_size) & torch.isin(input_nodes, global_keep_nodes)]
                valid_output_nodes = output_nodes[(output_nodes < (step + 1) * batch_size) & torch.isin(output_nodes, global_keep_nodes)]
            else:
                valid_input_nodes = input_nodes[(input_nodes < train_size + (step + 1) * batch_size) & torch.isin(input_nodes, global_keep_nodes)]
                valid_output_nodes = output_nodes[(output_nodes < train_size + (step + 1) * batch_size) & torch.isin(output_nodes, global_keep_nodes)]
            # print(f" valid_input_nodes selected {len(valid_input_nodes)}")

            filtered_input_nodes.append(valid_input_nodes)
            filtered_output_nodes.append(valid_output_nodes)

    if not filtered_blocks:  # If no valid blocks, return empty tensors
        return (
            torch.tensor([], dtype=torch.int64, device=input_nodes.device),
            torch.tensor([], dtype=torch.int64, device=output_nodes.device),
            []
        )

    return torch.cat(filtered_input_nodes), torch.cat(filtered_output_nodes), filtered_blocks


def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    _, logits, loss, dist, codebooklogits, loss_list = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss += criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val, loss_list


def transform_node_feats(a):
    transformed = torch.empty_like(a)
    transformed[:, 0] = torch.where(a[:, 0] == 6, 1,
                        torch.where(a[:, 0] == 8, 20, torch.where(a[:, 0] == 7, 10,
                        torch.where(a[:, 0] == 17, 5, torch.where(a[:, 0] == 9, 15,
                        torch.where(a[:, 0] == 35, 8, torch.where(a[:, 0] == 16, 3,
                        torch.where(a[:, 0] == 15, 12, torch.where(a[:, 0] == 1, 18,
                        torch.where(a[:, 0] == 5, 2, torch.where(a[:, 0] == 53, 16,
                        torch.where(a[:, 0] == 14, 4, torch.where(a[:, 0] == 34, 6,
                        torch.where(a[:, 0] == 19, 7, torch.where(a[:, 0] == 11, 9,
                        torch.where(a[:, 0] == 3, 11, torch.where(a[:, 0] == 30, 13,
                        torch.where(a[:, 0] == 33, 14, torch.where(a[:, 0] == 12, 17,
                        torch.where(a[:, 0] == 52, 19, -2))))))))))))))))))))

    transformed[:, 1] = torch.where(a[:, 1] == 1, 1,
    torch.where(a[:, 1] == 2, 20, torch.where(a[:, 1] == 3, 10,
    torch.where(a[:, 1] == 0, 15, torch.where(a[:, 1] == 4, 5,
    torch.where(a[:, 1] == 6, 7,
    torch.where(a[:, 1] == 5, 12, -2)))))))

    transformed[:, 2] = torch.where(a[:, 2] == 0, 1,
    torch.where(a[:, 2] == 1, 20, torch.where(a[:, 2] == -1, 10,
    torch.where(a[:, 2] == 3, 5,
    torch.where(a[:, 2] == 2, 15, -2)))))

    transformed[:, 3] = torch.where(a[:, 3] == 4, 1,
    torch.where(a[:, 3] == 3, 20, torch.where(a[:, 3] == 1, 10,
    torch.where(a[:, 3] == 2, 5, torch.where(a[:, 3] == 7, 15,
    torch.where(a[:, 3] == 6, 18, -2))))))

    transformed[:, 4] = torch.where(a[:, 4] == 0, 1,
    torch.where(a[:, 4] == 1, 20, -2))

    transformed[:, 5] = torch.where(a[:, 5] == 0, 1,
    torch.where(a[:, 5] == 1, 20, -2))

    transformed[:, 6] = torch.where(a[:, 6] == 3, 1,
    torch.where(a[:, 6] == 0, 20, torch.where(a[:, 6] == 1, 10,
    torch.where(a[:, 6] == 2, 15, torch.where(a[:, 6] == 4, 5, -2)))))

    return transformed


def train_sage(model, dataloader, feats, labels, criterion, optimizer, epoch, accumulation_steps=1, lamb=1):
    device = feats.device
    model.train()
    total_loss = 0
    loss_list, latent_list = [], []
    cb_list = []
    loss_list_list = []  # Initialize a list for tracking loss_list3 over steps
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()

    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]  # Convert blocks to device
        # print("----TRAIN ------")
        # print(f"Original Input Nodes: min {input_nodes.min()} max {input_nodes.max()} shape {input_nodes.shape}")
        # print(
        #     f"Original Output Nodes: min {output_nodes.min()} max {output_nodes.max()} shape {output_nodes.shape}")
        for i, block in enumerate(blocks):
            src, dst = block.edges()
            # print(f"Block {i}: min src {src.min()} max src {src.max()} shape {src.shape}")
            # print(f"Block {i}: min dst {dst.min()} max dst {dst.max()} shape {dst.shape}")

        input_nodes, output_nodes, blocks = filter_small_graphs_from_blocks(input_nodes, output_nodes, blocks, step, "train",min_size=6)
        # print(f"[train] input_nodes: {input_nodes[:20]}, {input_nodes[-20:]}")
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_feats = transform_node_feats(batch_feats)
        with torch.cuda.amp.autocast():
            _, logits, loss, _, cb, loss_list3, latent_train, quantized, latents = model(blocks, batch_feats, epoch)
            loss = loss * lamb / accumulation_steps
        if not torch.isfinite(loss):
            continue
        # Initialize loss_list_list with empty sublists if it's the first step
        if step == 0:
            loss_list_list = [[] for _ in range(len(loss_list3))]
        # Append each element from loss_list3 to the corresponding sublist
        for i, loss_value in enumerate(loss_list3):
            loss_list_list[i].append(loss_value.item())
        scaler.scale(loss).backward()
        # Accumulation steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        latent_list.append(latent_train.detach().cpu())
        cb_list.append(cb.detach().cpu())
        loss_list.append(loss.detach().cpu())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, loss_list_list, latent_list, latents



def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        _, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


"""
Train student MLP model with Token-Based GNN-MLP Distillation.
"""


def train_mini_batch_token(model, feats, codebook_embeddings, tea_soft_token_assignments_all, batch_size, criterion,
                           optimizer, lamb=1, temperature=4):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        h_list, _ = model(None, feats[idx_batch[i]])
        tea_soft_token_assignments = tea_soft_token_assignments_all[idx_batch[i]]

        # Compute student soft token assignments by calculating the L2 distance between student features and teacher codebook embeddings.
        stu_soft_token_assignments = - torch.cdist(h_list[-1], codebook_embeddings, p=2)
        tea_soft_token_assignments = tea_soft_token_assignments / temperature
        stu_soft_token_assignments = stu_soft_token_assignments / temperature
        tea_soft_token_assignments = tea_soft_token_assignments.softmax(dim=-1)
        tea_soft_token_assignments = torch.squeeze(tea_soft_token_assignments)
        stu_soft_token_assignments = stu_soft_token_assignments.log_softmax(dim=-1)
        stu_soft_token_assignments = torch.squeeze(stu_soft_token_assignments)

        # Compare student and teacher soft token assignments by KL divergence.
        loss = criterion(stu_soft_token_assignments, tea_soft_token_assignments) * temperature * temperature
        loss *= lamb
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    # this line explicitly set self.training True
    model.eval()
    with torch.no_grad():
        h_list, logits, _ , dist, codebook, loss_list, latent_vectors, sample_list = model.inference(data, feats)
        out, loss, score = None, None, None

        # out = logits.log_softmax(dim=1)
        # if idx_eval is None:
        #     loss = criterion(out, labels)
        #     score = evaluator(out, labels)
        # else:
        #     loss = criterion(out[idx_eval], labels[idx_eval])
        #     score = evaluator(out[idx_eval], labels[idx_eval])
        #  out, loss_test_ind, acc_ind, h_list, dist, codebook, loss_list1, latent_ind
    return out, loss, score, h_list, dist, codebook, loss_list, latent_vectors, sample_list


def evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            _, logits = model.inference(None, feats[batch_size * i: batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score


"""
2. Run teacher
"""


def run_transductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighbors.
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    loss_list = [0, 0, 0]
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss, loss_list = train_sage(model, data, feats, labels, criterion, optimizer)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss, loss_list = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test, acc = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                out, loss_train, score_train,  h_list, dist, codebook, loss_list = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_train
                )
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                acc = evaluator(out[idx_test], labels[idx_test])
            logger.info(loss_list)            # loss_list : [feature_rec_loss, edge_rec_loss, commit_loss]
            logger.info(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {acc:.4f} | feature_loss: {loss_list[0].item(): 4f}| edge_loss: {loss_list[1].item(): 4f}| commit_loss: {loss_list[2].item(): 4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    acc,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_val
        )
    else:
        out, _, score_val, h_list, dist, codebook, loss_list = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )

    acc = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, acc: {acc :.4f}"
    )
    return out, score_val, acc, h_list, dist, codebook, loss_list


def run_inductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        accumulation_steps=1
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """
    # indices = [obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind]
    # idx_obs = idx_train + idx_val + idx_test_tran
    # idx_test_ind :  idx_test_ind + idx_test_tran = idx_test
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    # # obs_idx_test start with 20 or 40 ?
    # print_and_shape(obs_idx_train, "obs_idx_train")
    # print_and_shape(obs_idx_val, "obs_idx_val")
    # print_and_shape(obs_idx_test, "obs_idx_test")
    # print_and_shape(idx_obs, "idx_obs")
    # print_and_shape(idx_test_ind, "idx_test_ind")

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)
    test_label = labels[obs_idx_test]
    test_feats = feats[obs_idx_test]
    test_g = g.subgraph(obs_idx_test)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        # sampler = dgl.dataloading.MultiLayerNeighborSampler(
        #     [eval(fanout) for fanout in conf["fan_out"].split(",")]
        # )

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        # obs_dataloader = dgl.dataloading.NodeDataLoader(
        # -------------------------
        # all obs data PARTIAL sampling (for training)
        # -------------------------
        obs_dataloader = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        # obs_dataloader_eval = dgl.dataloading.NodeDataLoader(
        # -------------------------
        # all obs data FULL sampling (for inference/test)
        # -------------------------
        print("Does the original graph have edge (0, 10000)?", obs_g.has_edges_between(0, 10000))

        obs_dataloader_eval = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_val,
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        for step, (input_nodes, output_nodes, blocks) in enumerate(obs_dataloader_eval):
            # print(f"Batch {step}: -----------------------------------------")
            input_nodes = torch.unique(input_nodes, dim=0)
            output_nodes = torch.unique(output_nodes, dim=0)
            # print(f"  Input nodes: {len(input_nodes)}")
            # print(f"  Output nodes: {len(output_nodes)}")

        # -------------------------
        # all data, FULL sampling
        # -------------------------
        dataloader_eval = dgl.dataloading.DataLoader(
            test_g,
            torch.arange(test_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval

    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        # data_eval = g

    state = None
    best_epoch, best_score_val, count = 0, 100, 0
    cb_at_best, train_latents_at_best = None, None
    latent_ind, latent_trans, latent_train = None, None, None
    cb_just_trained = None
    init_cb_list = None
    out, acc_ind, h_list, dist = None, None, None, None
    acc_tran, acc_ind = 0, 0
    score_val = 0
    for epoch in range(1, conf["max_epoch"] + 1):
        # print(f"epoch {epoch} -- Train")

        # --------------------------------
        # train
        # --------------------------------
        if "SAGE" in model.model_name:
            # --------------------------------
            # run only in train mode
            # --------------------------------
            if conf["train_or_infer"] == "train":
                # print("TRAIN STARAT -------------!")
                # partial sampling, only obs data
                # this loss is label loss
                loss, loss_list_list, latent_train, latents = train_sage(
                    model, obs_data, obs_feats, obs_labels, criterion, optimizer, epoch, accumulation_steps
                )
                model.encoder.reset_kmeans()
                cb_new = model.encoder.vq._codebook.init_embed_(latents)
                # save codebook and vectors every epoch
                # cb_just_trained = np.concatenate([a.cpu().detach().numpy() for a in cb_just_trained[-1]])
                np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())
                latents = torch.squeeze(latents)
                # random_indices = np.random.choice(latent_train.shape[0], 20000, replace=False)
                np.savez(f"./latents_{epoch}", latents.cpu().detach().numpy())
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        # if epoch % conf["eval_interval"] == 0:
        if "MLP" in model.model_name:
            _, loss_train, score_train = evaluate_mini_batch(
                model, feats_train, labels_train, criterion, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion, batch_size, evaluator
            )
            _, loss_test_tran, acc_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, acc_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion,
                batch_size,
                evaluator,
            )
        else:
            # --------------------------------
            # run only in train mode
            # --------------------------------
            if conf["train_or_infer"] == "train":
                # --------------------------------------
                # test/inference, no sampling, full obs graph
                # --------------------------------------
                # the "loss_train" is test loss in training
                # out, loss, score, h_list, dist, codebook, loss_list, latent_vectors, embed_ind_list

                # -----------------------------
                # 1 st evaluate
                # -----------------------------
                # out, loss, score, h_list, dist, codebook, loss_list, latent_vectors, embed_ind_list, input_nodes
                # print("EVAL 1 STARAT -------------!")
                obs_out, loss_train, score_train, h_list, dist, codebook, loss_list1, latent_trans, sample_list1 = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # loss_val = criterion(
                #     obs_out[obs_idx_val], obs_labels[obs_idx_val]
                # ).item()
                # score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                # loss_test_tran = criterion(
                #     obs_out[obs_idx_test], obs_labels[obs_idx_test]
                # ).item()
                # acc_tran = evaluator(
                #     obs_out[obs_idx_test], obs_labels[obs_idx_test]
                # )

            # -------------------------------------------------
            # 3. Evaluate the inductive part (idx_test_ind),
            # which is unlabeled, unseen in training
            # -------------------------------------------------
            # ---------------------------------------------------------------
            # save 'INPUT' embed indices for comparison to actual molecules
            # ---------------------------------------------------------------
            idx_test_ind_tosave = idx_test_ind[:8000]
            # --------------------------------
            # run only in train mode
            # --------------------------------
            # if conf["train_or_infer"] == "infer":
            #     np.savez(f"./idx_test_ind_tosave_first8000_{epoch}", idx_test_ind_tosave)
            # -----------------------------------------------
            # Evaluate the inductive part with the full graph
            # -----------------------------------------------
            # -----------------------------
            # 2 nd evaluate
            # -----------------------------
            # print(f"EVAL 2 -------------------")
            # # out, loss, score, h_list, dist, codebook, loss_list, latent_vectors, embed_ind_list
            # out, loss_test_ind, acc_ind, h_list, dist, codebook, loss_list1, latent_ind, sample_list2 = evaluate(
            #     model,
            #     data_eval,   #
            #     test_feats,       #
            #     test_label,      #
            #     criterion,
            #     evaluator,
            #     idx_test_ind
            # )

            # -----------------------------------------------------
            # save embed indices for comparison to actual molecules
            # -----------------------------------------------------
            # sample_list = [sample_ind, sample_feat, sample_adj]
            np.savez(f"./sample_emb_ind_{epoch}", sample_list1[0].cpu())
            np.savez(f"./sample_node_feat_{epoch}", sample_list1[1].cpu())
            np.savez(f"./sample_adj_{epoch}", sample_list1[2].cpu()[:1000, :1000])
            # [tensor([2., 1., 1.,  ..., 1., 1., 1.], device='cuda:0'), tensor([2., 1., 1.,  ..., 1., 1., 1.], device='cuda:0'), tensor([2., 1., 1.,  ..., 1., 1., 1.], device='cuda:0'), tensor([2., 1., 1.,  ..., 1., 1., 1.], device='cuda:0'), tensor([2., 1., 2.,  ..., 1., 1., 1.], device='cuda:0'), tensor([2., 1., 2.,  ..., 1., 1., 1.], device='cuda:0')]
            np.savez(f"./sample_bond_order_{epoch}", sample_list1[3].cpu())
            np.savez(f"./sample_bond_to_edge_0_{epoch}", sample_list1[4][0].cpu())
            np.savez(f"./sample_bond_to_edge_1_{epoch}", sample_list1[4][1].cpu())
            # print("sample_list1[4] len")
            # print(len(sample_list1[4]))
            # print("len(sample_list1[4][0])")
            # print(len(sample_list1[4][0]))
            # print("converted_list")
            # print(converted_list[0])
            # print(converted_list)

        if conf["train_or_infer"] == "train":

            # loss_total = float(loss_list1[0] + loss_list1[1] + loss_list1[2])
            logger.info(f"------------epoch {epoch:3d} -----------------------")
            # logger.info(f"train_known_g, epoch {epoch:3d}, feature_loss: {loss_list[0].item(): 4f}| edge_loss: {loss_list[1].item(): 4f}|"
            logger.info(f" div_element_loss: {sum(loss_list_list[0])/len(loss_list_list[0]): 4f},|  ")
            # logger.info(f" div_element_loss: {loss_list[0].item(): 4f},| commit_loss: {loss_list[1].item(): 4f}, margin loss {loss_list[4].item(): 4f}, "
            #             f"spread loss {loss_list[3].item(): 4f}, pair loss {loss_list[4].item(): 4f}, "
            #             f"bond_num_div_loss: {loss_list[5].item(): 4f}, aroma_div_loss: {loss_list[6].item(): 4f}, "
            #             f"ringy_div_loss: {loss_list[7].item(): 4f}, h_num_div_loss: {loss_list[8].item(): 4f}, sil_loss: {loss_list[9].item(): 4f}, ")
            # logger.info(f"test_known_g, epoch {epoch:3d}, div_element_loss: {loss_list0[0].item(): 4f},| commit_loss: {loss_list0[1].item(): 4f}")
            # logger.info(f"test_unknown_g, epoch {epoch:3d}, div_element_loss: {loss_list1[0].item(): 4f},| commit_loss: {loss_list1[1].item(): 4f}")
            # logger.info(f"test_known_g, epoch {epoch:3d}, feature_loss: {loss_list0[0].item(): 4f}| edge_loss: {loss_list0[1].item(): 4f}| div_element_loss: {loss_list0[2].item(): 4f},| commit_loss: {loss_list0[3].item(): 4f}")
            # logger.info(f"test_unknown_g, epoch {epoch:3d}, feature_loss: {loss_list1[0].item(): 4f}| edge_loss: {loss_list1[1].item(): 4f}| div_element_loss: {loss_list1[2].item(): 4f},| commit_loss: {loss_list1[3].item(): 4f}")

            print(f"------------epoch {epoch:3d} -----------------------")  # raw_feat_loss, raw_edge_rec_loss, raw_commit_loss, margin_loss, spread_loss, pair_los
            # print(f" div_element_loss: {sum(loss_list)/len(loss_list): 4f},|  ")
            print(f"train_known_g, div_element_loss: {sum(loss_list_list[0])/len(loss_list_list[0]): 7f}, "
                  f"bond_num_div_loss: {sum(loss_list_list[1])/len(loss_list_list[1]): 7f}, "
                  f" aroma_div_loss: {sum(loss_list_list[2])/len(loss_list_list[2]): 7f}, "
                  f"ringy_div_loss: {sum(loss_list_list[3])/len(loss_list_list[3]): 7f}, "
                  f"h_num_div_loss: {sum(loss_list_list[4])/len(loss_list_list[4]): 7f}, "
                  f"elec_state_div_loss: {sum(loss_list_list[5])/len(loss_list_list[5]): 7f}, "
                  f"charge_div_loss: {sum(loss_list_list[6])/len(loss_list_list[6]): 7f}, "
                  f"sil_loss: {sum(loss_list_list[9])/len(loss_list_list[9]): 7f}")
                  # f"ringy_div_loss: {loss_list[7]: 4f}, h_num_div_loss: {loss_list[8]: 4f}, sil_loss: {loss_list[9].item(): 4f}")
            # print(f"test_known_g, div_element_loss: {loss_list0[0].item(): 4f}, | commit_loss: {loss_list0[1].item(): 4f}")
            # print(f"test_unknown_g, "
            #       f"div_element_loss: {loss_list1[0].item(): 4f},| commit_loss: {loss_list1[1].item(): 4f},"
            #       f"bond_num_div_loss: {loss_list1[5].item(): 4f}, aroma_div_loss: {loss_list1[6].item(): 4f}, "
            #       f"ringy_div_loss: {loss_list1[7].item(): 4f}, h_num_div_loss: {loss_list1[8].item(): 4f}, sil_loss: {loss_list1[9].item(): 4f}")

            # print(f"test_unknown_g, div_element_loss: {loss_list1[0]:7f}, ")
            # print(f"loss_list1 {loss_list1}")
            # print(f"loss_list1[5] {loss_list1[5]}")
            # print(f"loss_list1[4] {loss_list1[4]}")
            print(f"train_known_g, div_element_loss: {sum(loss_list1[0])/len(loss_list1[0]): 7f}, "
                  f"bond_num_div_loss: {sum(loss_list1[1])/len(loss_list1[1]): 7f}, "
                  f" aroma_div_loss: {sum(loss_list1[2])/len(loss_list1[2]): 7f}, "
                  f"ringy_div_loss: {sum(loss_list1[3])/len(loss_list1[3]): 7f}, "
                  f"h_num_div_loss: {sum(loss_list1[4])/len(loss_list1[4]): 7f}, "
                  f"elec_state_div_loss: {sum(loss_list1[5])/len(loss_list1[5]): 7f}, "
                  f"charge_div_loss: {sum(loss_list1[6])/len(loss_list1[6]): 7f}, "
                  f"sil_loss: {sum(loss_list1[9])/len(loss_list1[9]): 7f}")
            # print(
            #     f"------------epoch {epoch:3d} -----------------------")  # raw_feat_loss, raw_edge_rec_loss, raw_commit_loss, margin_loss, spread_loss, pair_los
            # print(f"train_known_g, feature_loss: {loss_list[0].item(): 4f}| edge_loss: {loss_list[1].item(): 4f}| "
            #       f"div_element_loss: {loss_list[2].item(): 4f},| commit_loss: {loss_list[3].item(): 4f},"
            #       f"bond_num_div_loss: {loss_list[7].item(): 4f}, aroma_div_loss: {loss_list[8].item(): 4f}, "
            #       f"ringy_div_loss: {loss_list[9].item(): 4f}, h_num_div_loss: {loss_list[10].item(): 4f}, sil_loss: {loss_list[11].item(): 4f}")
            # print(
            #     f"test_known_g, feature_loss: {loss_list0[0].item(): 4f}| edge_loss: {loss_list0[1].item(): 4f}| div_element_loss: {loss_list0[2].item(): 4f}, | commit_loss: {loss_list0[3].item(): 4f}")
            # print(f"test_unknown_g, feature_loss: {loss_list1[0].item(): 4f}| edge_loss: {loss_list1[1].item(): 4f}| "
            #       f"div_element_loss: {loss_list1[2].item(): 4f},| commit_loss: {loss_list1[3].item(): 4f},"
            #       f"bond_num_div_loss: {loss_list1[7].item(): 4f}, aroma_div_loss: {loss_list1[8].item(): 4f}, "
            #       f"ringy_div_loss: {loss_list1[9].item(): 4f}, h_num_div_loss: {loss_list1[10].item(): 4f}, sil_loss: {loss_list1[11].item(): 4f}")

            # loss_and_score += [
            #     [
            #         epoch,
            #         loss_train,
            #         loss_val,
            #         loss_test_tran,
            #         loss_test_ind,
            #         score_train,
            #         score_val,
            #         acc_tran,
            #         acc_ind,
            #     ]
            # ]
            # print(f"loss_total {loss_total:4f}, best_score_val {best_score_val: 5f}")
            # --------------------------------
            # check if edge loss is decreasing
            # --------------------------------
            # if loss_list[1].item() < best_score_val:
                # best_epoch = epoch
                # best_score_val = loss_list[1].item()
            state = copy.deepcopy(model.state_dict())
                # cb_at_best = cb_just_trained
                # train_latents_at_best = latent_train
                # print(f"best epoch is {best_epoch} !!!!!!!!!")
                # count = 0
            # else:
            #     count += 1

            # if count == conf["patience"] or epoch == conf["max_epoch"]:
            #     break

            # --------------------------------
            # save model every epoch
            # --------------------------------
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

            model.load_state_dict(state)
            # if "MLP" in model.model_name:
            #     obs_out, _, score_val = evaluate_mini_batch(
            #         model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
            #     )
            #     out, _, acc_ind = evaluate_mini_batch(
            #         model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
            #     )
            #
            # else:
            #     logger.info(
            #         f"started the final tran test"
            #     )
            #     # -----------------------------
            #     # 3 rd evaluate
            #     # -----------------------------
            #     obs_out, _, score_val, h_list, dist, codebook, loss_list, latent_trans = evaluate(
            #         model,
            #         obs_data_eval,
            #         obs_feats,
            #         obs_labels,
            #         criterion,
            #         evaluator,
            #         obs_idx_val,
            #     )
            #     logger.info(
            #         f"started the final ind test"
            #     )
            #     # -----------------------------
            #     # 4 th evaluate
            #     # -----------------------------
            #     out, _, acc_ind, h_list, dist, codebook, loss_list, latent_ind = evaluate(
            #         model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
            #     )

            # acc_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
            # out[idx_obs] = obs_out
            # logger.info(
            #     f"Best valid model at epoch: {best_epoch :3d}, acc_tran: {acc_tran :.4f}, acc_ind: {acc_ind :.4f}"
            # )
    #      out, score_val, score_test_tran, score_test_ind, h_list, dist, codebook, latents_trans, latents_ind, latent_train_list

    return out, score_val, acc_tran, acc_ind, h_list, dist, cb_at_best, latent_trans, latent_ind, train_latents_at_best


"""
3. Distill
"""


def distill_run_transductive(
        conf,
        model,
        feats,
        labels,
        out_t_all,
        out_codebook_embeddings,
        out_tea_soft_token_assignments,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb_soft_labels = conf["lamb_soft_labels"]
    lamb_soft_tokens = conf["lamb_soft_tokens"]
    temperature = conf["temperature"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    out_codebook_embeddings = out_codebook_embeddings.to(device)
    out_tea_soft_token_assignments = out_tea_soft_token_assignments.to(device)

    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    count, best_acc = 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        # soft token assignments distillation
        loss_token = train_mini_batch_token(
            model, feats_t, out_codebook_embeddings, out_tea_soft_token_assignments, batch_size, criterion_t, optimizer,
            lamb_soft_tokens, temperature
        )
        # soft label distillation
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb_soft_labels
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb_soft_labels
        )
        loss = loss_token + loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_test, acc = evaluate_mini_batch(
                model, feats_test, labels_test, criterion_l, batch_size, evaluator
            )
            if epoch % 5 == 0:
                print(
                    f"Ep {epoch:3d} | loss: {loss:.4f} | acc: {acc:.4f}")
                logger.info(
                    f"Ep {epoch:3d} | loss: {loss:.4f} | acc: {acc:.4f}"
                )
            loss_and_score += [
                [epoch, loss_l, loss_test, score_l, acc]
            ]
            if acc >= best_acc:
                best_acc = acc
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, _ = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    acc = evaluator(out[idx_test], labels_test)
    logger.info(f"Best Accuracy: {acc}")
    return out, acc


def distill_run_inductive(
        conf,
        model,
        feats,
        labels,
        out_t_all,
        out_codebook_embeddings,
        out_tea_soft_token_assignments,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb_soft_labels = conf["lamb_soft_labels"]
    lamb_soft_tokens = conf["lamb_soft_tokens"]
    temperature = conf["temperature"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    out_codebook_embeddings = out_codebook_embeddings.to(device)
    out_tea_soft_token_assignments = out_tea_soft_token_assignments.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_acc, count = 0, 0
    print(conf["max_epoch"])
    for epoch in range(1, conf["max_epoch"] + 1):
        # soft token assignments distillation
        loss_token = train_mini_batch_token(
            model, feats_t, out_codebook_embeddings, out_tea_soft_token_assignments, batch_size, criterion_t, optimizer,
            lamb_soft_tokens, temperature
        )
        # soft label distillation
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb_soft_labels
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb_soft_labels
        )
        loss = loss_token + loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, acc = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test_tran, acc_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, acc_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )
            if epoch % 5 == 0:
                print(
                    f"Ep {epoch:3d} | loss: {loss:.4f} | acc_tran: {acc_tran:.4f} | acc_ind: {acc_ind:.4f}")
                logger.info(
                    f"Ep {epoch:3d} | loss: {loss:.4f} | acc_tran: {acc_tran:.4f} | acc_ind: {acc_ind:.4f}")
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    acc,
                    acc_tran,
                    acc_ind,
                ]
            ]

            if acc >= best_acc:
                best_acc = acc
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            print(f"break cz count {count}, epoch {epoch}")
            break

    model.load_state_dict(state)
    obs_out, _, acc = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, acc_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    acc_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best acc_tran: {acc_tran :.4f}, acc_ind: {acc_ind :.4f}"
    )

    return out, acc_tran, acc_ind
