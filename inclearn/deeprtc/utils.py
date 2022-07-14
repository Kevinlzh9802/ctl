import torch
import numpy as np


def deep_rtc_nloss(nout, targets, leaf_id, node_labels, device):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    nloss = []
    for idx in range(targets.size(0)):
        index = targets.cpu().numpy()[idx]
        if index in leaf_id.keys():
            index = leaf_id[index]
        for n_id, n_l in node_labels[index]:
            if device.type == 'cuda':
                res = criterion(nout[n_id][idx, :].view(1, -1).cuda(), torch.tensor([n_l]).cuda())
            else:
                res = criterion(nout[n_id][idx, :].view(1, -1), torch.tensor([n_l]))
            nloss.append(res)
    nloss = torch.mean(torch.stack(nloss))
    return nloss


def deep_rtc_sts_loss(output, targets, sfmx_base, leaf_id, device):
    leaf_id_indexes = targets_from_0(targets, leaf_id, device)
    gt_z = torch.gather(output, 1, leaf_id_indexes.view(-1, 1))
    stsloss = torch.mean(-gt_z + torch.log(torch.clamp(sfmx_base.view(-1, 1), 1e-17, 1e17)))
    return stsloss


def targets_from_0(targets, leaf_id, device):
    leaf_id_index_list = []
    for target_i in list(np.array(targets.cpu())):
        if target_i in leaf_id.keys():
            leaf_id_index_list.append(leaf_id[target_i])
    if device.type == 'cuda':
        leaf_id_indexes = torch.tensor(leaf_id_index_list).cuda()
    else:
        leaf_id_indexes = torch.tensor(leaf_id_index_list)
    return leaf_id_indexes
