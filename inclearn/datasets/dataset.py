import os.path as osp
import numpy as np
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
import torch


def get_datasets(dataset_names):
    return [get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def get_dataset(dataset_name):
    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif "imagenet100" in dataset_name:
        return iImageNet100
    elif dataset_name == "imagenet":
        return iImageNet
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None

    
def split_data(data_c):
    # currently for trees with depth of 2
    DataLoaderList = []
    tg = data_c.targets.copy()
    for class_i in range(len(np.unique(tg))):
        DataLoaderList.append(torch.utils.data.DataLoader((data_c.data[tg==tg[class_i]], data_c.targets[tg==tg[class_i]])))
    return DataLoaderList

