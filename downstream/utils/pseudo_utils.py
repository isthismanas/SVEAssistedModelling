import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import random
import json
import pickle as pkl


class EntropyLoss(nn.Module):
    def __init__(self, reduction=True):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)
        return b


def cosine_similarity(x1, x2, reduction=True):
    cos_sim = nn.CosineSimilarity(dim=-1)
    if reduction:
        sim = cos_sim(x1, x2).mean()
    else:
        sim = cos_sim(x1, x2)
    return sim


class CE_uniform(nn.Module):
    def __init__(self, n_id_classes, reduction=True):
        super(CE_uniform, self).__init__()
        self.reduction = reduction
        self.n_id_classes = n_id_classes

    def forward(self, x):
        b = (1/self.n_id_classes) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)
        return b


def get_consistent_loss_new(x1, x2, f1=None, f2=None):
    x1 = x1.mean(axis=0)
    if f1 is not None:
        x1 = f1(x1)
    if f2 is not None:
        x2 = f2(x2)
    loss = cosine_similarity(x1, x2)
    return -1.0 * loss


def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(att.mean(axis=1)).detach()
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0
    return loss


def split_dataset(json_path, train_ratio):
    with open(json_path, 'r') as f:
        name_dict = json.load(f)

    num_label = 0
    for val in name_dict.values():
        if val['label'] != -1:
            num_label += 1
    test_size = int(num_label * (1 - train_ratio))
    train_indices, test_indices = [], []
    labels_l = []
    test_cnt = 0
    random.seed(111)
    val = list(name_dict.values())
    indx = list(np.arange(len(val)))
    random.shuffle(indx)

    for idx in indx:
        value = val[idx]
        if value['label'] != -1 and test_cnt < test_size:
            test_indices.append(int(idx))
            test_cnt += 1
        else:
            train_indices.append(int(idx))

    keys = np.array(list(name_dict.keys()))
    dic = {'indices': test_indices, 'labels': labels_l, 'id': keys[test_indices]}
    os.makedirs('./latent', exist_ok=True)
    with open('./latent/test_indices.pkl', 'wb') as f:
        pkl.dump(dic, f)

    return train_indices, test_indices
