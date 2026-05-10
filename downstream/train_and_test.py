import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from utils.pseudo_utils import EntropyLoss, get_consistent_loss_new, cosine_similarity, local_ent_loss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from nn_utils import save_model

def valids(model, test_loader, device):
    Path('./eval').mkdir(exist_ok=True)
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        latent_list = []
        loss = []
        iteration = 0
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs, latent = model(
                data_sample['protac_graphs'].to(device),
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['feature'].to(device)
            )
            labeled_mask = torch.where(y==-1, False, True)
            outputs, y = outputs[labeled_mask], y[labeled_mask]
            loss_val = criterion(outputs, y)
            loss.append(loss_val.item())
            latent_list.append(latent)
            y_score = y_score + torch.nn.functional.softmax(outputs,1)[:,1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs,1)[1].cpu().tolist()
            y_true = y_true + y.cpu().tolist()
            iteration += 1

        model.train()
        saved_dict = {'score': y_score, 'pred': y_pred, 'label': y_true}
        np.save('./eval/PROTAC_demo_saved.npy', saved_dict)
        latent = torch.cat(latent_list, 0)


    return (sum(loss)/iteration, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score),
            f1_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred), recall_score(y_true, y_pred), latent)


def train(model, args, train_loader=None, valid_loader=None, device=None, writer=None, LOSS_NAME=None):
    Path('./eval').mkdir(exist_ok=True)
    Path('./latent').mkdir(exist_ok=True)
    Path('./log').mkdir(exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_l)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0

    print("------------------- Supervised Train -------------------")
    for epoch in range(args.epoch):
        model.train()
        epoch_losses = []
        epoch_true = []
        epoch_pred = []
        epoch_score = []

        for data_sample in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs, latent = model(
                data_sample['protac_graphs'].to(device),
                data_sample['ligase_pocket'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['feature'].to(device)
            )

            y = data_sample['label'].to(device)
            labeled_mask = torch.where(y == -1, False, True)
            if not labeled_mask.any():
                continue

            logits = outputs[labeled_mask]
            y_labeled = y[labeled_mask]
            loss = criterion(logits, y_labeled)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_score.extend(torch.nn.functional.softmax(logits, 1)[:, 1].detach().cpu().tolist())
            epoch_pred.extend(torch.max(logits, 1)[1].detach().cpu().tolist())
            epoch_true.extend(y_labeled.detach().cpu().tolist())

        if epoch % 10 == 0 or epoch == args.epoch - 1:
            if epoch_true:
                train_acc = accuracy_score(epoch_true, epoch_pred)
                train_auc = roc_auc_score(epoch_true, epoch_score) if len(set(epoch_true)) > 1 else float('nan')
                train_f1 = f1_score(epoch_true, epoch_pred, average='macro')
                train_pre = precision_score(epoch_true, epoch_pred, zero_division=0)
                train_rec = recall_score(epoch_true, epoch_pred, zero_division=0)
            else:
                train_acc = train_auc = train_f1 = train_pre = train_rec = float('nan')

            print('Epoch: {}/{}  '.format(epoch + 1, args.epoch), end='')
            print(
                'Train Loss: {} | Train acc: {}| Train auc: {}|Valid f1: {}| Train pre: {}| Train rec: {}'.format(
                    float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    train_acc,
                    train_auc,
                    train_f1,
                    train_pre,
                    train_rec,
                )
            )
            model.eval()
            val_loss, acc, auc, f1, pre, rec, latent_val = valids(model, valid_loader, device)
            if acc > best_acc:
                best_acc = acc
                np.save('./latent/latent_{}.npy'.format(args.hidden_size), {'latent_emb': latent_val})
                save_model(epoch + 1, model, args)
            print('Valid loss: {} | Valid acc: {}| Valid auc: {}|Valid f1: {}| Valid pre: {}| Valid rec: {}'.format(val_loss, acc, auc, f1, pre, rec))
            print('\n')

    save_model(args.epoch, model, args)
    _, acc, auc, f1, pre, rec, _ = valids(model, valid_loader, device)
    print('Final validation')
    print('Valid acc: {}| Valid auc: {}|Valid f1: {}| Valid pre: {}| Valid rec: {}'.format(acc, auc, f1, pre, rec))


    return model
