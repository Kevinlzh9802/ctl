import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter
from inclearn.tools.utils import to_onehot
from inclearn.deeprtc.utils import deep_rtc_nloss
from inclearn.datasets.data import tgt_to_tgt0, tgt_to_tgt0_no_tax


def finetune_last_layer(logger, network, loader, n_class, device, nepoch=30, lr=0.1, scheduling=None, lr_decay=0.1,
                        weight_decay=5e-4, loss_type="ce", temperature=5.0, test_loader=None, save_path='',
                        index_map=None):
    if scheduling is None:
        scheduling = [15, 35]
    network.eval()
    n_module = network.module
    # if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    # else:
    #    network.module.convnet.eval()
    optim = SGD(n_module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")
    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        all_preds = None
        all_is_correct = np.array([])
        for inputs, targets in loader:
            if device.type == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
                network = network.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            if n_module.taxonomy == 'rtc':
                outputs = network(inputs)
                nout = outputs['nout']
                loss = deep_rtc_nloss(nout, targets, n_module.leaf_id, n_module.node_labels, n_module.device)

                max_z = torch.max(outputs["output"], dim=1)[0]
                preds = torch.eq(outputs["output"], max_z.view(-1, 1))
                leaf_id_indexes = tgt_to_tgt0(targets, n_module.leaf_id, n_module.device)
                iscorrect = torch.gather(preds, 1, leaf_id_indexes.view(-1, 1)).flatten().float()

                if all_preds is None:
                    all_preds = np.empty([0, preds.shape[1]])
                all_preds = np.concatenate((all_preds, preds.cpu()))
                all_is_correct = np.concatenate((all_is_correct, iscorrect.cpu()))
                # print(loss)
                loss.backward()
                optim.step()
                total_loss += loss
                total_correct += iscorrect.sum()
                total_count += inputs.size(0)
            else:
                outputs = network(inputs)['output']
                targets_0 = tgt_to_tgt0_no_tax(targets, index_map, device)
                _, preds = outputs.max(1)
                optim.zero_grad()
                loss = criterion(outputs / temperature, targets_0.long())
                loss.backward()
                optim.step()
                total_loss += loss * inputs.size(0)
                total_correct += (preds == targets_0).sum()
                total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
        if i == nepoch - 1:
            np.savetxt(save_path + f'_epoch_{i}_preds.txt', np.array(all_preds), fmt='%2.2f')
            np.savetxt(save_path + f'_epoch_{i}_iscorrect.txt', np.array(all_is_correct), fmt='%2.2f')
    return network


def extract_features(model, loader, device):
    targets, features = [], []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in loader:
            if device == 'cuda':
                _inputs = _inputs.cuda()
            else:
                _inputs = _inputs
            _targets = _targets.numpy()
            _features = model(_inputs)['feature'].detach().cpu().numpy()
            features.append(_features)
            targets.append(_targets)
    if len(targets) == 1:
        return np.array(features), np.array(targets)
    else:
        return np.concatenate(features), np.concatenate(targets)


def calc_class_mean(network, loader, class_idx, metric):
    EPSILON = 1e-8
    features, targets = extract_features(network, loader)
    # norm_feats = features/(np.linalg.norm(features, axis=1)[:,np.newaxis]+EPSILON)
    # examplar_mean = norm_feats.mean(axis=0)
    examplar_mean = features.mean(axis=0)
    if metric == "cosine" or metric == "weight":
        examplar_mean /= (np.linalg.norm(examplar_mean) + EPSILON)
    return examplar_mean


def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means

