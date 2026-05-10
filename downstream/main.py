import sys
import numpy as np
import copy
import torch
import os
import pickle
import logging
import json
from pathlib import  Path
from torch.utils.data import DataLoader
from protacloader import PROTACSet, collater
from model import GraphConv, ProtacModel, SageConv, GATTConv, EGNNConv
from train_and_test import train, valids
from config.config import get_args
import time

import sys
from prepare_data import GraphData
from dataset import VOCAB
from nn_utils import load_model, setup_seed

TRAIN_NAME = "test"

args = get_args()
setup_seed(args.seed)
root = str(getattr(args, "dataset_root", "data/PROTAC"))


def _select_device():
    requested = str(getattr(args, "device", "auto")).lower()
    if requested == "mps":
        return torch.device("mps")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_aligned_indices(name_list, labels, train_ratio, seed):
    labeled_indices = [i for i, y in enumerate(labels) if int(y) != -1]
    unlabeled_indices = [i for i, y in enumerate(labels) if int(y) == -1]

    rng = np.random.default_rng(seed)
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    if labeled_indices.size > 0:
        rng.shuffle(labeled_indices)
    train_size = int(len(labeled_indices) * train_ratio)
    train_indices = labeled_indices[:train_size].tolist() + unlabeled_indices
    test_indices = labeled_indices[train_size:].tolist()
    return train_indices, test_indices

def main():
    protac_graphs = GraphData('protac', root=root,
                             select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    ligase_pocket = GraphData("ligase_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)
    target_pocket = GraphData("target_pocket", root,
                              select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3,
                               conv_name=args.conv_name)

    with open(os.path.join(root, '{}.json'.format(args.dataset_type)), 'r') as f:
        name_dic = json.load(f)

    name_list = list(name_dic.keys())


    label = torch.load(os.path.join(target_pocket.processed_dir, "label.pt"), weights_only=False)
    feature = torch.load(os.path.join(target_pocket.processed_dir, "feature.pt"), weights_only=False)

    # Keep classification indexing aligned with processed graph/feature tensors.
    max_feature_rows = int(feature.shape[0])
    filtered_name_list = []
    for key in name_list:
        try:
            idx = int(key)
        except ValueError:
            continue
        if 0 <= idx < max_feature_rows:
            filtered_name_list.append(key)
    name_list = filtered_name_list

    n_common = min(len(name_list), len(label), len(feature), len(protac_graphs), len(ligase_pocket), len(target_pocket))
    if n_common == 0:
        raise RuntimeError("No aligned classification samples available after index filtering")
    name_list = name_list[:n_common]
    label = label[:n_common]
    feature = feature[:n_common]
    if not args.feature:
        feature = np.random.rand(feature.shape[0], feature.shape[1])

    protac_set = PROTACSet(
        name_list,
        protac_graphs,
        ligase_pocket,
        target_pocket,
        feature,
        label,
    )
    data_size = len(protac_set)
    train_size = int(data_size * args.train_rate)
    test_size = data_size - train_size
    pos_num, neg_num = 0, 0
    for key, value in name_dic.items():
        if value['label'] == 0:
            neg_num += 1
        elif value['label'] == 1:
            pos_num += 1
    logging.info(f"all data: {data_size}")
    logging.info(f"train data: {train_size}")
    logging.info(f"test data: {test_size}")
    logging.info(f"positive label number: {pos_num}")
    logging.info(f"negative label number: {neg_num}")
    train_indicies, test_indicies = _split_aligned_indices(name_list, label, args.train_rate, args.seed)

    train_dataset = torch.utils.data.Subset(protac_set, train_indicies)
    test_dataset = torch.utils.data.Subset(protac_set, test_indicies)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collater,drop_last=False, shuffle=False)

    if args.conv_name == "GCN":

        ligase_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.e3_dim, hidden_size=args.hidden_size)
        target_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.tar_dim, hidden_size=args.hidden_size)
        protac_model = GraphConv(num_embeddings=118, graph_dim=args.protac_dim, hidden_size=args.hidden_size)
    elif args.conv_name == "GAT":
        ligase_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        target_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        protac_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)

    elif args.conv_name == "EGNN":
        ligase_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1,graph_nf=args.e3_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        protac_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.protac_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        target_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.tar_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
    else:
        raise ValueError("conv_type Error")
    model = ProtacModel(
        protac_model,
        ligase_pocket_model,
        target_pocket_model,
        args.hidden_size,
        protac_dim=args.protac_dim,
        tar_dim=args.tar_dim,
        e3_dim=args.e3_dim,
    )
    device = _select_device()
    writer = None

    if args.mode == 'Train':
        model = train(
            model,
            train_loader=trainloader,
            valid_loader=testloader,
            device=device,
            writer=writer,
            LOSS_NAME=TRAIN_NAME,
            args=args
        )
    loss_l, acc_l, auc_l, f1_l, pre_l, rec_l = [], [], [], [], [], []
    load_model(model, args, loaded_epoch=args.epoch)
    for i in range(10):
        loss, acc, auc, f1, pre, rec, latent_val = valids(model.to(device),
                                test_loader=testloader,
                                device=device)
        saved_dict = {'latent_emb': latent_val}
        np.save('./latent/latent_ours_{}.npy'.format(args.hidden_size), saved_dict)

        print('Test Loss: {} | Accuracy: {} | AUC: {} | F1: {} | PRE: {} | REC: {}'.format(loss, acc, auc, f1, pre, rec))
        loss_l.append(loss)
        acc_l.append(acc)
        auc_l.append(auc)
        f1_l.append(f1)
        pre_l.append(pre)
        rec_l.append(rec)
    test_end = time.time()
    print("------------------- Final Test -------------------")
    print('Base model: ', args.conv_name)
    print('Train rate: ', args.train_rate)
    print('Dataset_type: ', args.dataset_type)
    print('Loss in infer: ', np.mean(loss_l))
    print('Accuracy in infer: ', np.mean(acc_l))
    print('AUC in infer: ', np.mean(auc_l))
    print('F1 in infer: ', np.mean(f1_l))
    print('PRE in infer: ', np.mean(pre_l))
    print('REC in infer: ', np.mean(rec_l))

    save_dir = Path(getattr(args, "save_dir", "runs_classification"))
    if not save_dir.is_absolute():
        save_dir = Path.cwd() / save_dir
    run_name = str(getattr(args, "run_name", "classification_run"))
    run_dir = (save_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = {
        "loss": float(np.mean(loss_l)),
        "accuracy": float(np.mean(acc_l)),
        "auc": float(np.mean(auc_l)),
        "f1": float(np.mean(f1_l)),
        "precision": float(np.mean(pre_l)),
        "recall": float(np.mean(rec_l)),
        "dataset_root": str(Path(root).resolve()),
        "dataset_type": args.dataset_type,
        "run_name": run_name,
        "feature": bool(args.feature),
        "protac_dim": int(args.protac_dim),
    }
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics: {run_dir / 'final_metrics.json'}")

if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()
