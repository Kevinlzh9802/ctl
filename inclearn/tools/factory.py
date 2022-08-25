import time

import torch
from torch import nn
from torch import optim

from inclearn import models
from inclearn.convnet import resnet, cifar_resnet, modified_resnet_cifar, preact_resnet
from inclearn.datasets import data, dataset


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet32":
        return cifar_resnet.resnet32()
    elif convnet_type == "modified_resnet32":
        return modified_resnet_cifar.resnet32(**kwargs)
    elif convnet_type == "preact_resnet18":
        return preact_resnet.PreActResNet18(**kwargs)
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(cfg, logger, inc_dataset):
    if cfg["model"] == "incmodel":
        return models.IncModel(cfg, logger, inc_dataset)
    if cfg["model"] == "weight_align":
        return models.Weight_Align(cfg, logger, inc_dataset)
    if cfg["model"] == "bic":
        return models.BiC(cfg, logger, inc_dataset)
    else:
        raise NotImplementedError(cfg["model"])


def get_data(cfg):
    return data.IncrementalDataset(
        trial_i=cfg['trial'],
        dataset_name=cfg["dataset"],
        is_distributed=cfg["is_distributed"],
        random_order=cfg["random_classes"],
        shuffle=True,
        batch_size=cfg["batch_size"],
        sample_rate=cfg["sample_rate"],
        workers=0,
        device=cfg["device"],
        validation_split=cfg["validation"],
        resampling=cfg["resampling"],
        increment=cfg["increment"],
        data_folder=cfg["data_folder"],
        mode_train=cfg["exp"]["mode_train"],
        taxonomy=cfg["taxonomy"]
    )


def set_device(cfg):
    device_type = cfg["device"]
    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))
    cfg["device"] = device


def set_acc_detail_path(cfg, mode):
    cfg["acc_detail_path"] += (cfg["exp"]["name"] + '/' + mode)


class MyCustomLoader:
    def __init__(self, rank=0, print_rank=0):
        self.rank = rank
        self.print_enable = (self.rank == print_rank)

    def info(self, content=''):
        # assert isinstance(content, str)
        if self.print_enable:
            print(str(time.time()) + ' | ' + content)
