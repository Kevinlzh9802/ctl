import os.path as osp
import numpy as np

import albumentations as A
from inclearn.deeprtc.libs import Tree
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict


def get_datasets(dataset_names):
    return [get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def get_dataset(dataset_name):
    # here returns an object of the specific class (no instance yet!)
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


class iCIFAR10(DataHandler):
    base_dataset_cls = datasets.cifar.CIFAR10
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_folder, train, device, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 10

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [4, 0, 2, 5, 8, 3, 1, 6, 9, 7]


class iCIFAR100(iCIFAR10):
    base_dataset_cls = datasets.cifar.CIFAR100
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    # data_name_hier_dict = OrderedDict({
    #     'vehicles_1': {'motorcycle': {}, 'bus': {}, 'train': {}, 'bicycle': {}, 'pickup_truck': {}},
    #     'trees': {'palm_tree': {}, 'willow_tree': {}, 'maple_tree': {}, 'oak_tree': {}, 'pine_tree': {}},
    #     'large_man-made_outdoor_things': {'bridge': {}, 'road': {}, 'skyscraper': {}, 'house': {}, 'castle': {}},
    #     'food_containers': {'can': {}, 'cup': {}, 'plate': {}, 'bowl': {}, 'bottle': {}},
    #     'small_mammals': {'hamster': {}, 'mouse': {}, 'shrew': {}, 'rabbit': {}, 'squirrel': {}},
    #     'large_omnivores_and_herbivores': {'cattle': {}, 'camel': {}, 'chimpanzee': {}, 'kangaroo': {}, 'elephant': {}},
    #     'flowers': {'rose': {}, 'tulip': {}, 'poppy': {}, 'orchid': {}, 'sunflower': {}},
    #     'large_natural_outdoor_scenes': {'forest': {}, 'plain': {}, 'cloud': {}, 'mountain': {}, 'sea': {}},
    #     'reptiles': {'turtle': {}, 'crocodile': {}, 'dinosaur': {}, 'lizard': {}, 'snake': {}},
    #     'household_furniture': {'wardrobe': {}, 'bed': {}, 'couch': {}, 'chair': {}, 'table': {}},
    #     'fruit_and_vegetables': {'apple': {}, 'pear': {}, 'mushroom': {}, 'sweet_pepper': {}, 'orange': {}},
    #     'large_carnivores': {'bear': {}, 'leopard': {}, 'tiger': {}, 'wolf': {}, 'lion': {}},
    #     'vehicles_2': {'streetcar': {}, 'tractor': {}, 'tank': {}, 'lawn_mower': {}, 'rocket': {}},
    #     'people': {'man': {}, 'boy': {}, 'girl': {}, 'baby': {}, 'woman': {}},
    #     'insects': {'butterfly': {}, 'bee': {}, 'beetle': {}, 'caterpillar': {}, 'cockroach': {}},
    #     'household_electrical_devices': {'lamp': {}, 'television': {}, 'telephone': {}, 'keyboard': {}, 'clock': {}},
    #     'non-insect_invertebrates': {'crab': {}, 'snail': {}, 'lobster': {}, 'worm': {}, 'spider': {}},
    #     'aquatic_mammals': {'dolphin': {}, 'whale': {}, 'otter': {}, 'seal': {}, 'beaver': {}},
    #     'fish': {'aquarium_fish': {}, 'flatfish': {}, 'ray': {}, 'trout': {}, 'shark': {}},
    #     'medium_mammals': {'raccoon': {}, 'fox': {}, 'porcupine': {}, 'skunk': {}, 'possum': {}}
    # })

    data_name_hier_dict = OrderedDict({
        'medium_mammals': {'raccoon': {}, 'fox': {}, 'porcupine': {}, 'skunk': {}, 'possum': {}},
        'fish': {'aquarium_fish': {}, 'flatfish': {}, 'ray': {}, 'trout': {}, 'shark': {}},
        'aquatic_mammals': {'dolphin': {}, 'whale': {}, 'otter': {}, 'seal': {}, 'beaver': {}},
        'non-insect_invertebrates': {'crab': {}, 'snail': {}, 'lobster': {}, 'worm': {}, 'spider': {}},
        'household_electrical_devices': {'lamp': {}, 'television': {}, 'telephone': {}, 'keyboard': {}, 'clock': {}},
        'insects': {'butterfly': {}, 'bee': {}, 'beetle': {}, 'caterpillar': {}, 'cockroach': {}},
        'people': {'man': {}, 'boy': {}, 'girl': {}, 'baby': {}, 'woman': {}},
        'vehicles_2': {'streetcar': {}, 'tractor': {}, 'tank': {}, 'lawn_mower': {}, 'rocket': {}},
        'large_carnivores': {'bear': {}, 'leopard': {}, 'tiger': {}, 'wolf': {}, 'lion': {}},
        'fruit_and_vegetables': {'apple': {}, 'pear': {}, 'mushroom': {}, 'sweet_pepper': {}, 'orange': {}},
        'household_furniture': {'wardrobe': {}, 'bed': {}, 'couch': {}, 'chair': {}, 'table': {}},
        'reptiles': {'turtle': {}, 'crocodile': {}, 'dinosaur': {}, 'lizard': {}, 'snake': {}},
        'large_natural_outdoor_scenes': {'forest': {}, 'plain': {}, 'cloud': {}, 'mountain': {}, 'sea': {}},
        'flowers': {'rose': {}, 'tulip': {}, 'poppy': {}, 'orchid': {}, 'sunflower': {}},
        'large_omnivores_and_herbivores': {'cattle': {}, 'camel': {}, 'chimpanzee': {}, 'kangaroo': {}, 'elephant': {}},
        'small_mammals': {'hamster': {}, 'mouse': {}, 'shrew': {}, 'rabbit': {}, 'squirrel': {}},
        'food_containers': {'can': {}, 'cup': {}, 'plate': {}, 'bowl': {}, 'bottle': {}},
        'large_man-made_outdoor_things': {'bridge': {}, 'road': {}, 'skyscraper': {}, 'house': {}, 'castle': {}},
        'trees': {'palm_tree': {}, 'willow_tree': {}, 'maple_tree': {}, 'oak_tree': {}, 'pine_tree': {}},
        'vehicles_1': {'motorcycle': {}, 'bus': {}, 'train': {}, 'bicycle': {}, 'pickup_truck': {}}
    })

    data_label_index_dict = {
        'medium_mammals': -20, 'fish': -19, 'aquatic_mammals': -18, 'non-insect_invertebrates': -17,
        'household_electrical_devices': -16, 'insects': -15, 'people': -14, 'vehicles_2': -13, 'large_carnivores': -12,
        'fruit_and_vegetables': -11, 'household_furniture': -10, 'reptiles': -9, 'large_natural_outdoor_scenes': -8,
        'flowers': -7, 'large_omnivores_and_herbivores': -6, 'small_mammals': -5, 'food_containers': -4,
        'large_man-made_outdoor_things': -3, 'trees': -2, 'vehicles_1': -1, 'apple': 0, 'aquarium_fish': 1, 'baby': 2,
        'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11,
        'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
        'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26,
        'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33,
        'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40,
        'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47,
        'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54,
        'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61,
        'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69,
        'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77,
        'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84,
        'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91,
        'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

    taxonomy_tree = Tree('cifar100', data_name_hier_dict, data_label_index_dict)
    used_nodes, leaf_id, node_labels = taxonomy_tree.prepro()

    def __init__(self, data_folder, train, device, is_fine_label=False):
        super().__init__(data_folder, train, is_fine_label)
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 100
        self.transform_type = 'torchvision'

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        idx_to_label = {cls.data_label_index_dict[x]: x for x in cls.data_label_index_dict}
        if trial_i == 0:
            return [
                62, 54, 84, 20, 94, 22, 40, 29, 78, 27, 26, 79, 17, 76, 68, 88, 3, 19, 31, 21, 33, 60, 24, 14, 6, 10,
                16, 82, 70, 92, 25, 5, 28, 9, 61, 36, 50, 90, 8, 48, 47, 56, 11, 98, 35, 93, 44, 64, 75, 66, 15, 38, 97,
                42, 43, 12, 37, 55, 72, 95, 18, 7, 23, 71, 49, 53, 57, 86, 39, 87, 34, 63, 81, 89, 69, 46, 2, 1, 73, 32,
                67, 91, 0, 51, 83, 13, 58, 80, 74, 65, 4, 30, 45, 77, 99, 85, 41, 96, 59, 52
            ]
        elif trial_i == 1:
            return cls.lt_convert([
                [68, 56, 78, 8, 23],
                [84, 90, 65, 74, 76],
                [40, 89, 3, 92, 55],
                [9, 26, 80, 43, 38],
                [58, 70, 77, 1, 85],
                [19, 17, 50, 28, 53],
                [13, 81, 45, 82, 6],
                [59, 83, 16, 15, 44],
                [91, 41, 72, 60, 79],
                [52, 20, 10, 31, 54],
                [37, 95, 14, 71, 96],
                [98, 97, 2, 64, 66],
                [42, 22, 35, 86, 24],
                [34, 87, 21, 99, 0],
                [88, 27, 18, 94, 11],
                [12, 47, 25, 30, 46],
                [62, 69, 36, 61, 7],
                [63, 75, 5, 32, 4],
                [51, 48, 73, 93, 39],
                [67, 29, 49, 57, 33]], idx_to_label)
        elif trial_i == 2:  # PODNet
            return [
                87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
                88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6,
                46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
                40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
            ]
        elif trial_i == 3:  # PODNet
            return [
                58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70,
                90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88,
                95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97,
                82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61
            ]
        elif trial_i == 4:  # PODNet
            return [
                71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36,
                90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64,
                18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97,
                75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11
            ]
        elif trial_i == 5:
            label_list = [
                ['medium_mammals', 'fish', 'aquatic_mammals', 'non-insect_invertebrates',
                 'household_electrical_devices', 'insects', 'people', 'vehicles_2', 'large_carnivores',
                 'fruit_and_vegetables', 'household_furniture', 'reptiles', 'large_natural_outdoor_scenes',
                 'flowers', 'large_omnivores_and_herbivores', 'small_mammals', 'food_containers',
                 'large_man-made_outdoor_things', 'trees', 'vehicles_1'],
                #           ['vehicles_1', 'trees', 'large_man-made_outdoor_things', 'food_containers', 'small_mammals',
                #            'large_omnivores_and_herbivores', 'flowers', 'large_natural_outdoor_scenes', 'reptiles',
                #            'household_furniture', 'fruit_and_vegetables', 'large_carnivores', 'vehicles_2', 'people',
                #            'insects', 'household_electrical_devices', 'non-insect_invertebrates', 'aquatic_mammals',
                #            'fish', 'medium_mammals'],
                # ['household_furniture', 'large_man-made_outdoor_things', 'medium_mammals',
                #  'food_containers', 'reptiles', 'vehicles_2', 'large_natural_outdoor_scenes',
                #  'large_omnivores_and_herbivores', 'flowers', 'small_mammals', 'trees',
                #  'vehicles_1', 'large_carnivores', 'people', 'aquatic_mammals', 'fish',
                #  'insects', 'fruit_and_vegetables', 'non-insect_invertebrates',
                #  'household_electrical_devices'],
                ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
                ['palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree'],
                ['bridge', 'road', 'skyscraper', 'house', 'castle'],
                ['can', 'cup', 'plate', 'bowl', 'bottle'],
                ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],  #
                ['cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant'],
                ['rose', 'tulip', 'poppy', 'orchid', 'sunflower'],
                ['forest', 'plain', 'cloud', 'mountain', 'sea'],
                ['turtle', 'crocodile', 'dinosaur', 'lizard', 'snake'],
                ['wardrobe', 'bed', 'couch', 'chair', 'table'],
                ['apple', 'pear', 'mushroom', 'sweet_pepper', 'orange'],
                ['bear', 'leopard', 'tiger', 'wolf', 'lion'],
                ['streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket'],
                ['man', 'boy', 'girl', 'baby', 'woman'],  #
                ['butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach'],
                ['lamp', 'television', 'telephone', 'keyboard', 'clock'],
                ['crab', 'snail', 'lobster', 'worm', 'spider'],  #
                ['dolphin', 'whale', 'otter', 'seal', 'beaver'],  #
                ['aquarium_fish', 'flatfish', 'ray', 'trout', 'shark'],  #
                ['raccoon', 'fox', 'porcupine', 'skunk', 'possum']]
            return label_list

        elif trial_i == 6:
            label_list = [['medium_mammals', 'fish', 'aquatic_mammals', 'non-insect_invertebrates',
                           'household_electrical_devices', 'insects', 'people', 'vehicles_2', 'large_carnivores',
                           'fruit_and_vegetables', 'household_furniture', 'reptiles', 'large_natural_outdoor_scenes',
                           'flowers', 'large_omnivores_and_herbivores', 'small_mammals', 'food_containers',
                           'large_man-made_outdoor_things', 'trees', 'vehicles_1'],
                          ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],
                          ['man', 'boy', 'girl', 'baby', 'woman'],
                          ['crab', 'snail', 'lobster', 'worm', 'spider'],
                          ['dolphin', 'whale', 'otter', 'seal', 'beaver'],  #
                          ['aquarium_fish', 'flatfish', 'ray', 'trout', 'shark'],
                          ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
                          ['palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree'],
                          ['bridge', 'road', 'skyscraper', 'house', 'castle'],
                          ['can', 'cup', 'plate', 'bowl', 'bottle'],
                          ['cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant'],
                          ['rose', 'tulip', 'poppy', 'orchid', 'sunflower'],
                          ['forest', 'plain', 'cloud', 'mountain', 'sea'],
                          ['turtle', 'crocodile', 'dinosaur', 'lizard', 'snake'],  #
                          ['wardrobe', 'bed', 'couch', 'chair', 'table'],
                          ['apple', 'pear', 'mushroom', 'sweet_pepper', 'orange'],
                          ['bear', 'leopard', 'tiger', 'wolf', 'lion'],
                          ['streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket'],
                          ['butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach'],  #
                          ['lamp', 'television', 'telephone', 'keyboard', 'clock'],
                          ['raccoon', 'fox', 'porcupine', 'skunk', 'possum']]
            return label_list

        elif trial_i == -1:
            label_list = [
                # ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel']
                # ['poppy', 'orchid', 'table', 'chair', 'wardrobe']
                ['people'],
                ['man', 'boy', 'girl', 'baby', 'woman']
                # 'crab', 'snail', 'lobster', 'worm', 'spider',
                # 'dolphin', 'whale', 'otter', 'seal', 'beaver',  #
                # 'aquarium_fish', 'flatfish', 'ray', 'trout', 'shark',
                # 'motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck',
                # 'palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree',
                # 'bridge', 'road', 'skyscraper', 'house', 'castle',
                # 'can', 'cup', 'plate', 'bowl', 'bottle',
                # 'cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant',
                # 'rose', 'tulip', 'poppy', 'orchid', 'sunflower',
                # 'forest', 'plain', 'cloud', 'mountain', 'sea',
                # 'turtle', 'crocodile', 'dinosaur', 'lizard', 'snake',  #
                # 'wardrobe', 'bed', 'couch', 'chair', 'table',
                # 'apple', 'pear', 'mushroom', 'sweet_pepper', 'orange',
                # 'bear', 'leopard', 'tiger', 'wolf', 'lion',
                # 'streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket',
                # 'butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach',  #
                # 'lamp', 'television', 'telephone', 'keyboard', 'clock',
                # 'raccoon', 'fox', 'porcupine', 'skunk', 'possum'
            ]
            return label_list

    @classmethod
    def label_to_target(cls, class_order_list):
        class_id_list = []
        for task_label_list in class_order_list:
            task_id_list = []
            for task_label in task_label_list:
                task_id_list.append(cls.data_label_index_dict[task_label])
            class_id_list.append(task_id_list)
        return class_id_list

    @staticmethod
    def lt_convert(ori_list, mapping):
        new_list = []
        for ori_sub_list in ori_list:
            new_sub_list = []
            for task_label in ori_sub_list:
                new_sub_list.append(mapping[task_label])
            new_list.append(new_sub_list)
        return new_list


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iImageNet(DataHandler):
    base_dataset_cls = datasets.ImageFolder
    transform_type = 'albumentations'
    if transform_type == 'albumentations':
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=63 / 255),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, data_folder, train, is_fine_label=False):
        if train is True:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "train"))
        else:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "val"))
        self.data, self.targets = zip(*self.base_dataset.samples)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.n_cls = 1000

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [
            54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419, 274, 108,
            928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289,
            123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722, 748, 14, 77, 437, 394, 859,
            279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400, 471, 632, 275, 730, 105, 523, 224, 186,
            478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954,
            465, 533, 585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659,
            360, 136, 578, 163, 427, 70, 226, 925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39,
            326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957,
            691, 155, 820, 584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409,
            156, 455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248,
            333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933, 306, 378,
            76, 227, 426, 403, 322, 321, 808, 393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917,
            611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592, 573, 128, 243, 520, 887, 892,
            696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 678, 745, 845, 208, 188,
            674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619, 217, 631, 934, 932, 568, 353, 863, 827, 425, 420,
            99, 823, 113, 974, 438, 874, 343, 118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407,
            56, 927, 655, 809, 839, 640, 297, 34, 497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535,
            139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 767, 666, 22, 525, 902, 233, 250, 825,
            79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586,
            729, 253, 486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452,
            245, 487, 706, 2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 826,
            668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795, 132, 145, 368, 147, 327,
            713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246, 449, 492, 162, 97, 59, 357, 198,
            519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793, 151,
            847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987,
            801, 629, 491, 605, 112, 429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725,
            480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213,
            196, 743, 817, 433, 328, 970, 969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317,
            926, 269, 161, 209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564,
            185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170,
            679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861, 692, 686,
            277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599,
            187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 267, 867, 772, 604, 618,
            346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 869, 955, 17, 506, 963,
            786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717,
            116, 488, 796, 983, 646, 499, 53, 1, 603, 45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19,
            965, 143, 555, 687, 235, 790, 125, 173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998,
            991, 469, 967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 684, 862,
            574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308,
            881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178, 489,
            37, 197, 789, 530, 111, 876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953,
            270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749,
            916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 194, 152, 981, 938, 854,
            483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 559, 756, 25, 211, 158,
            723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 331, 901, 416, 873, 754, 900, 435, 762,
            124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 697, 417
        ]


# trial 2
def imagenet1000_label_dict_index_trial2():
    return {'physical_entity': -1, 'abstraction': -2, 'matter': -3, 'causal_agent': -4, 'object': -5, 'bubble': 971,
            'communication': -6, 'food': -7, 'food_1': -8, 'substance': -9, 'person': -10, 'whole': -11,
            'geological_formation': -12, 'traffic_light': 920, 'street_sign': 919, 'nutriment': -13, 'foodstuff': -14,
            'hay': 958, 'menu': 922, 'beverage': -15, 'bread': -16, 'produce': -17, 'toilet_tissue': 999,
            'ballplayer': 981, 'scuba_diver': 983, 'groom': 982, 'animal': -18, 'natural_object': -19, 'organism': -20,
            'artifact': -21, 'valley': 979, 'geyser': 974, 'shore': -22, 'cliff': 972, 'natural_elevation': -23,
            'course': -24, 'dish': -25, 'condiment': -26, 'dough': 961, 'espresso': 967, 'alcohol': -27,
            'loaf_of_bread': -28, 'pretzel': 932, 'bagel': 931, 'vegetable': -29, 'vertebrate': -30,
            'invertebrate': -31, 'fruit': -32, 'sandbar': 977, 'flower': -33, 'fungus': -34, 'fabric': -35,
            'pillow': 721, 'sheet': -36, 'consumer_goods': -37, 'rain_barrel': 756, 'plaything': -38, 'structure': -39,
            'instrumentality': -40, 'stage': 819, 'decoration': -41, 'commodity': -42, 'covering': -43, 'lakeside': 975,
            'seashore': 978, 'mountain': -44, 'ridge': -45, 'promontory': 976, 'plate': 923, 'dessert': -46,
            'pizza': 963, 'consomme': 925, 'burrito': 965, 'meat_loaf': 962, 'hot_pot': 926, 'sandwich': -47,
            'potpie': 964, 'sauce': -48, 'guacamole': 924, 'red_wine': 966, 'punch': -49, 'french_loaf': 930,
            'solanaceous_vegetable': -50, 'mashed_potato': 935, 'mushroom': 947, 'artichoke': 944,
            'cruciferous_vegetable': -51, 'cucumber': 943, 'cardoon': 946, 'squash': -52, 'mammal': -53, 'bird': -54,
            'fish': -55, 'reptile': -56, 'amphibian_1': -57, 'worm': -58, 'echinoderm': -59, 'arthropod': -60,
            'coelenterate': -61, 'mollusk': -62, 'ear': 998, 'seed': -63, 'hip': 989, 'acorn': 988, 'edible_fruit': -64,
            'yellow_ladys_slipper': 986, 'daisy': 985, 'agaric': 992, 'stinkhorn': 994, 'hen-of-the-woods': 996,
            'coral_fungus': 991, 'earthstar': 995, 'bolete': 997, 'gyromitra': 993, 'piece_of_cloth': -66,
            'velvet': 885, 'wool': 911, 'scoreboard': 781, 'clothing': -67, 'home_appliance': -68, 'teddy': 850,
            'coil': 506, 'bridge': -69, 'mountain_tent': 672, 'fountain': 562, 'patio': 706, 'housing': -70,
            'dock': 536, 'establishment': -71, 'barrier': -72, 'beacon': 437, 'triumphal_arch': 873, 'altar': 406,
            'castle': 483, 'lumbermill': 634, 'memorial': -73, 'building': -74, 'column': -75,
            'supporting_structure': -76, 'furnishing': -77, 'comic_book': 917, 'system': -78, 'implement': -79,
            'toiletry': -80, 'chain': 488, 'device': -81, 'conveyance': -82, 'equipment': -83, 'container': -84,
            'necklace': 679, 'bath_towel': 434, 'protective_covering': -85, 'mask': 643, 'shoji': 789, 'footwear': -86,
            'cloak': 501, 'top': -87, 'book_jacket': 921, 'floor_cover': -88, 'cloth_covering': -89, 'alp': 970,
            'volcano': 980, 'coral_reef': 973, 'frozen_dessert': -90, 'trifle': 927, 'hotdog': 934, 'cheeseburger': 933,
            'chocolate_sauce': 960, 'carbonara': 959, 'eggnog': 969, 'cup': 968, 'bell_pepper': 945, 'cauliflower': 938,
            'head_cabbage': 936, 'broccoli': 937, 'summer_squash': -91, 'winter_squash': -92, 'monotreme': -93,
            'placental': -94, 'marsupial': -95, 'tusker': 101, 'oscine': -96, 'piciform_bird': -97, 'ostrich': 9,
            'cock': 7, 'hummingbird': 94, 'game_bird': -98, 'coraciiform_bird': -99, 'aquatic_bird': -100,
            'parrot': -101, 'coucal': 91, 'hen': 8, 'bird_of_prey': -102, 'teleost_fish': -103, 'elasmobranch': -104,
            'food_fish': -105, 'diapsid': -106, 'turtle': -107, 'salamander': -108, 'frog': -109, 'flatworm': 110,
            'nematode': 111, 'starfish': 327, 'sea_urchin': 328, 'sea_cucumber': 329, 'arachnid': -110, 'trilobite': 69,
            'crustacean': -111, 'insect': -112, 'centipede': 79, 'jellyfish': 107, 'anthozoan': -113,
            'chambered_nautilus': 117, 'gastropod': -114, 'chiton': 116, 'corn': 987, 'buckeye': 990, 'rapeseed': 984,
            'custard_apple': 956, 'fig': 952, 'strawberry': 949, 'pineapple': 953, 'citrus': -115, 'pomegranate': 957,
            'jackfruit': 955, 'granny_smith': 948, 'banana': 954, 'bib': 443, 'towel': -116, 'handkerchief': 591,
            'dishrag': 533, 'seat_belt': 785, 'mitten': 658, 'protective_garment': -117, 'headdress': -118, 'gown': 910,
            'military_uniform': 652, 'hosiery': -119, 'womans_clothing': -120, 'pajama': 697, 'attire': -121,
            'garment': -122, 'white_goods': -123, 'sewing_machine': 786, 'vacuum': 882, 'kitchen_appliance': -124,
            'iron': 606, 'viaduct': 888, 'steel_arch_bridge': 821, 'suspension_bridge': 839, 'mobile_home': 660,
            'dwelling': -125, 'mercantile_establishment': -126, 'prison': 743, 'movable_barrier': -127, 'dam': 525,
            'grille': 581, 'breakwater': 460, 'fence': -128, 'bannister': 421, 'megalith': 649, 'brass': 458,
            'barn': 425, 'theater': -129, 'place_of_worship': -130, 'shed': -131, 'restaurant': 762, 'library': 624,
            'greenhouse': 580, 'residence': -132, 'planetarium': 727, 'totem_pole': 863, 'obelisk': 682,
            'pedestal': 708, 'framework': -133, 'furniture': -134, 'maze': 646, 'communication_system': -135,
            'pen': -136, 'pole': 733, 'space_bar': 810, 'cooking_utensil': -137, 'stick': -138, 'racket': 752,
            'tool': -139, 'cleaning_implement': -140, 'paddle': 693, 'rubber_eraser': 767, 'pencil_sharpener': 710,
            'sunscreen': 838, 'makeup': -141, 'hair_spray': 585, 'perfume': 711, 'lotion': 631,
            'source_of_illumination': -142, 'instrument': -143, 'lighter': 626, 'filter': -144, 'whistle': 902,
            'heater': -145, 'paintbrush': 696, 'hard_disc': 592, 'remote_control': 761, 'pick': 714, 'restraint': -146,
            'electric_fan': 545, 'neck_brace': 678, 'electro-acoustic_transducer': -147, 'ski': 795, 'sunglass': 836,
            'trap': -148, 'mechanism': -149, 'crane': 517, 'breathing_device': -150, 'wing': 908,
            'electronic_device': -151, 'reflector': -152, 'support': -153, 'hand_blower': 589, 'keyboard': -154,
            'musical_instrument': -155, 'machine': -156, 'stretcher': 830, 'vehicle': -157, 'public_transport': -158,
            'sports_equipment': -159, 'electronic_equipment': -160, 'parachute': 701, 'camera': -161,
            'game_equipment': -162, 'photocopier': 713, 'gear': -163, 'packet': 692, 'pot_1': 738, 'vessel': -165,
            'cassette': 481, 'box': -166, 'shaker': -167, 'tray': 868, 'soap_dispenser': 804, 'milk_can': 653,
            'wooden_spoon': 910, 'envelope': 549, 'ashcan': 412, 'wheeled_vehicle': -168, 'glass': -169, 'dish_1': -170,
            'wallet': 893, 'bag': -171, 'piggy_bank': 719, 'basket': -172, 'measuring_cup': 647, 'armor_plate': -173,
            'roof': -174, 'sheath': -175, 'cap_1': -176, 'blind': -177, 'screen': 509, 'lampshade': 619,
            'shelter': -178, 'mask_1': -179, 'binder': 446, 'armor': -180, 'cowboy_boot': 514, 'clog': 502,
            'shoe': -181, 'manhole_cover': 640, 'cap': -182, 'doormat': 539, 'prayer_rug': 741, 'quilt': 750,
            'band_aid': 419, 'ice_lolly': 929, 'ice_cream': 928, 'spaghetti_squash': 940, 'zucchini': 939,
            'butternut_squash': 942, 'acorn_squash': 941, 'platypus': 103, 'echidna': 102, 'edentate': -183,
            'aquatic_mammal': -184, 'ungulate': -185, 'rodent': -186, 'carnivore': -187, 'leporid': -188,
            'elephant': -189, 'primate': -190, 'koala': 105, 'wallaby': 104, 'wombat': 106, 'corvine_bird': -191,
            'water_ouzel': 20, 'finch': -192, 'chickadee': 19, 'thrush': -193, 'toucan': 96, 'jacamar': 95,
            'phasianid': -194, 'grouse': -195, 'hornbill': 93, 'bee_eater': 92, 'wading_bird': -196, 'black_swan': 100,
            'european_gallinule': 136, 'seabird': -197, 'anseriform_bird': -198, 'sulphur-crested_cockatoo': 89,
            'lorikeet': 90, 'african_grey': 87, 'macaw': 88, 'bald_eagle': 22, 'vulture': 23, 'kite': 21,
            'great_grey_owl': 24, 'spiny-finned_fish': -199, 'soft-finned_fish': -200, 'ganoid': -201, 'ray': -202,
            'shark': -203, 'barracouta': 389, 'coho': 391, 'snake': -204, 'lizard': -205, 'triceratops': 51,
            'crocodilian_reptile': -206, 'sea_turtle': -207, 'box_turtle': 37, 'terrapin': 36, 'mud_turtle': 35,
            'european_fire_salamander': 25, 'newt': -208, 'ambystomid': -209, 'tailed_frog': 32, 'bullfrog': 30,
            'tree_frog': 31, 'tick': 78, 'spider': -210, 'scorpion': 71, 'harvestman': 70, 'decapod_crustacean': -211,
            'isopod': 126, 'homopterous_insect': -212, 'butterfly': -213, 'beetle': -214, 'orthopterous_insect': -215,
            'walking_stick': 313, 'fly': 308, 'hymenopterous_insect': -216, 'dictyopterous_insect': -217,
            'odonate': -218, 'lacewing': 318, 'sea_anemone': 108, 'brain_coral': 109, 'slug': 114, 'sea_slug': 115,
            'conch': 112, 'snail': 113, 'orange': 950, 'lemon': 951, 'paper_towel': 700, 'knee_pad': 615, 'apron': 411,
            'helmet': -219, 'hat': -220, 'cap_2': -221, 'vestment': 887, 'academic_gown': 400,
            'christmas_stocking': 496, 'sock': 806, 'maillot_1': 639, 'gown_1': 578, 'wig': 903, 'scarf': -224,
            'diaper': 529, 'skirt': -225, 'suit': 834, 'swimsuit': -226, 'jean': 608, 'brassiere': 459,
            'overgarment': -227, 'robe': -228, 'sweater': -229, 'necktie': -230, 'jersey': 610, 'refrigerator': 760,
            'washer': 897, 'dishwasher': 534, 'espresso_maker': 550, 'waffle_iron': 891, 'microwave': 651, 'oven': -231,
            'toaster': 859, 'cliff_dwelling': 500, 'yurt': 915, 'shop': -232, 'grocery_store': 582, 'sliding_door': 799,
            'turnstile': 877, 'chainlink_fence': 489, 'picket_fence': 716, 'stone_wall': 825, 'worm_fence': 912,
            'cinema': 498, 'home_theater': 598, 'stupa': 832, 'mosque': 668, 'church': 497, 'apiary': 410,
            'boathouse': 449, 'palace': 698, 'monastery': 663, 'honeycomb': 599, 'plate_rack': 729, 'four-poster': 564,
            'baby_bed': -233, 'table': -234, 'cabinet': -235, 'file': 553, 'entertainment_center': 548,
            'dining_table': 532, 'bookcase': 453, 'table_lamp': 846, 'wardrobe': 894, 'seat': -236, 'chiffonier': 493,
            'radio': 754, 'television': 851, 'fountain_pen': 563, 'quill': 749, 'ballpoint': 418, 'pan': -237,
            'spatula': 813, 'crock_pot': 521, 'pot': 837, 'drumstick': 542, 'spindle': 816, 'staff': -238,
            'matchstick': 644, 'hand_tool': -239, 'edge_tool': -240, 'plow': 730, 'power_drill': 740, 'lawn_mower': 621,
            'swab': 840, 'broom': 462, 'face_powder': 551, 'lipstick': 629, 'torch': 862, 'lamp': -241,
            'optical_instrument': -242, 'weapon': -243, 'measuring_instrument': -244, 'magnifier': -245,
            'guillotine': 583, 'medical_instrument': -246, 'magnetic_compass': 635, 'oil_filter': 686, 'strainer': 828,
            'space_heater': 811, 'stove': 827, 'disk_brake': 535, 'fastener': -247, 'muzzle': 676, 'loudspeaker': 632,
            'microphone': 650, 'mousetrap': 674, 'spider_web': 815, 'radiator': 753, 'puck': 746, 'control': -248,
            'mechanical_device': -249, 'snorkel': 801, 'oxygen_mask': 691, 'mouse': 673, 'screen_1': 782,
            'solar_dish': 807, 'car_mirror': 475, 'pier': 718, 'tripod': 872, 'maypole': 645,
            'typewriter_keyboard': 878, 'stringed_instrument': -251, 'percussion_instrument': -252,
            'keyboard_instrument': -253, 'wind_instrument': -254, 'power_tool': -255, 'cash_machine': 480,
            'abacus': 398, 'farm_machine': -256, 'computer': -257, 'slot_machine': -258, 'military_vehicle': -259,
            'sled': -260, 'craft': -261, 'missile': 657, 'bus': -262, 'bullet_train': 466, 'gymnastic_apparatus': -263,
            'golf_equipment': -264, 'weight': -265, 'cassette_player': 482, 'modem': 662, 'telephone': -266,
            'tape_player': 848, 'ipod': 605, 'cd_player': 485, 'monitor': 664, 'oscilloscope': 688, 'peripheral': -267,
            'polaroid_camera': 732, 'reflex_camera': 759, 'puzzle': -268, 'ball': -269, 'pool_table': 736,
            'carpenters_kit': 477, 'drilling_platform': 540, 'mortar': 666, 'ladle': 618, 'tub': 876, 'pitcher': 725,
            'jar': -271, 'coffee_mug': 504, 'bucket': 463, 'barrel': 427, 'bathtub': 435, 'reservoir': -272,
            'washbasin': 896, 'bottle': -273, 'safe': 771, 'pencil_box': 709, 'mailbox': 637, 'crate': 519,
            'chest': 492, 'carton': 478, 'saltshaker': 773, 'cocktail_shaker': 503, 'unicycle': 880, 'car_1': -274,
            'motor_scooter': 670, 'cart': -275, 'handcart': -276, 'bicycle': -277, 'tricycle': 870,
            'self-propelled_vehicle': -278, 'beer_glass': 441, 'goblet': 572, 'bowl': -279, 'petri_dish': 712,
            'backpack': 414, 'sleeping_bag': 797, 'mailbag': 636, 'purse': 748, 'plastic_bag': 728,
            'shopping_basket': 790, 'hamper': 588, 'breastplate': 461, 'pickelhaube': 715, 'vault': 884, 'dome': 538,
            'tile_roof': 858, 'thatch': 853, 'holster': 597, 'scabbard': 777, 'lens_cap': 622, 'thimble': 855,
            'curtain': -280, 'window_shade': 905, 'fire_screen': 556, 'window_screen': 904, 'mosquito_net': 669,
            'birdhouse': 448, 'umbrella': 879, 'bell_cote': 442, 'gasmask': 570, 'ski_mask': 796, 'shield': 787,
            'body_armor': -281, 'running_shoe': 770, 'loafer': 630, 'sandal': 774, 'bottlecap': 455, 'nipple': 680,
            'armadillo': 363, 'three-toed_sloth': 364, 'dugong': 149, 'sea_lion': 150, 'whale': -282,
            'even-toed_ungulate': -283, 'equine': -284, 'guinea_pig': 338, 'marmot': 336, 'porcupine': 334,
            'hamster': 333, 'beaver': 337, 'fox_squirrel': 335, 'feline': -285, 'canine': -286, 'viverrine': -287,
            'bear': -288, 'musteline_mammal': -289, 'procyonid': -290, 'rabbit': -291, 'hare': 331,
            'indian_elephant': 385, 'african_elephant': 386, 'anthropoid_ape': -292, 'monkey': -293, 'lemur': -294,
            'magpie': 18, 'jay': 17, 'goldfinch': 11, 'junco': 13, 'brambling': 10, 'indigo_bunting': 14,
            'house_finch': 12, 'bulbul': 16, 'robin': 15, 'peacock': 84, 'partridge': 86, 'quail': 85,
            'prairie_chicken': 83, 'ruffed_grouse': 82, 'ptarmigan': 81, 'black_grouse': 80, 'bustard': 138,
            'shorebird': -295, 'heron': -296, 'limpkin': 135, 'spoonbill': 129, 'american_coot': 137, 'stork': -297,
            'crane_1': 134, 'flamingo': 130, 'king_penguin': 145, 'albatross': 146, 'pelican': 144, 'duck': -299,
            'goose': 99, 'lionfish': 396, 'percoid_fish': -300, 'puffer': 397, 'eel': 390, 'cyprinid': -301, 'gar': 395,
            'sturgeon': 394, 'electric_ray': 5, 'stingray': 6, 'great_white_shark': 2, 'hammerhead': 4,
            'tiger_shark': 3, 'viper': -302, 'colubrid_snake': -303, 'boa': -304, 'sea_snake': 65, 'elapid': -305,
            'agamid': -306, 'banded_gecko': 38, 'green_lizard': 46, 'komodo_dragon': 48, 'gila_monster': 45,
            'whiptail': 41, 'iguanid': -307, 'alligator_lizard': 44, 'african_chameleon': 47, 'african_crocodile': 49,
            'american_alligator': 50, 'leatherback_turtle': 34, 'loggerhead': 33, 'eft': 27, 'common_newt': 26,
            'axolotl': 29, 'spotted_salamander': 28, 'garden_spider': 74, 'barn_spider': 73, 'black_widow': 75,
            'tarantula': 76, 'wolf_spider': 77, 'black_and_gold_garden_spider': 72, 'hermit_crab': 125, 'crab': -308,
            'lobster': -309, 'crayfish': 124, 'cicada': 316, 'leafhopper': 317, 'admiral': 321, 'lycaenid': 326,
            'sulphur_butterfly': 325, 'ringlet': 322, 'cabbage_butterfly': 324, 'monarch': 323, 'leaf_beetle': 304,
            'tiger_beetle': 300, 'long-horned_beetle': 303, 'scarabaeid_beetle': -310, 'ground_beetle': 302,
            'weevil': 307, 'ladybug': 301, 'cricket': 312, 'grasshopper': 311, 'ant': 310, 'bee': 309, 'cockroach': 314,
            'mantis': 315, 'dragonfly': 319, 'damselfly': 320, 'crash_helmet': 518, 'football_helmet': 560,
            'bearskin': 439, 'bonnet': 452, 'cowboy_hat': 515, 'sombrero': 808, 'bathing_cap': 433, 'shower_cap': 793,
            'mortarboard': 667, 'stole': 824, 'feather_boa': 552, 'hoopskirt': 601, 'sarong': 775, 'overskirt': 689,
            'miniskirt': 655, 'maillot': 638, 'bikini': 445, 'swimming_trunks': 842, 'poncho': 735, 'coat': -311,
            'kimono': 614, 'abaya': 399, 'cardigan': 474, 'sweatshirt': 841, 'bow_tie': 457, 'bolo_tie': 451,
            'windsor_tie': 906, 'rotisserie': 766, 'dutch_oven': 544, 'confectionery': 509, 'shoe_shop': 788,
            'toyshop': 865, 'bookshop': 454, 'tobacco_shop': 860, 'bakery': 415, 'barbershop': 424, 'butcher_shop': 467,
            'bassinet': 431, 'crib': 520, 'cradle': 516, 'desk': 526, 'medicine_chest': 648, 'china_cabinet': 495,
            'chair': -312, 'toilet_seat': 861, 'park_bench': 703, 'studio_couch': 831, 'wok': 909, 'frying_pan': 567,
            'caldron': 469, 'teapot': 849, 'coffeepot': 505, 'crutch': 523, 'flagpole': 557, 'shovel': 792,
            'opener': -313, 'plunger': 731, 'hammer': 587, 'screwdriver': 784, 'hatchet': 596, 'knife': -314,
            'plane': 726, 'jack-o-lantern': 607, 'spotlight': 818, 'candle': 470, 'sunglasses': 837, 'projector': 745,
            'binoculars': 447, 'gun': -316, 'projectile': 744, 'bow': 456, 'rule': 769, 'scale': 778, 'odometer': 685,
            'barometer': 426, 'timepiece': -317, 'radio_telescope': 755, 'loupe': 633, 'stethoscope': 823,
            'syringe': 845, 'buckle': 464, 'nail': 677, 'safety_pin': 772, 'knot': 616, 'screw': 783, 'lock': -318,
            'hair_slide': 584, 'switch': 844, 'joystick': 613, 'hook': 600, 'gas_pump': 571, 'carousel': 476,
            'reel': 758, 'wheel': -319, 'swing': 843, 'harp': 594, 'banjo': 420, 'bowed_stringed_instrument': -320,
            'guitar': -321, 'steel_drum': 822, 'maraca': 641, 'marimba': 642, 'gong': 577, 'chime': 494, 'drum': 541,
            'piano': -322, 'organ': 687, 'panpipe': 699, 'brass_1': -323, 'woodwind': -324, 'ocarina': 684,
            'free-reed_instrument': -325, 'chain_saw': 491, 'harvester': 595, 'thresher': 856,
            'personal_computer': -326, 'slide_rule': 798, 'web_site': 916, 'vending_machine': 886, 'slot': 800,
            'warship': -327, 'half_track': 586, 'bobsled': 450, 'dogsled': 537, 'vessel_1': -328, 'aircraft': -329,
            'space_shuttle': 812, 'minibus': 654, 'trolleybus': 874, 'school_bus': 779, 'horizontal_bar': 602,
            'parallel_bars': 702, 'balance_beam': 416, 'golfcart': 575, 'dumbbell': 543, 'barbell': 422,
            'dial_telephone': 528, 'pay-phone': 707, 'cellular_telephone': 487, 'data_input_device': -330,
            'printer': 742, 'jigsaw_puzzle': 611, 'crossword_puzzle': 918, 'tennis_ball': 852, 'volleyball': 890,
            'golf_ball': 574, 'punching_bag': 747, 'ping-pong_ball': 722, 'soccer_ball': 805, 'rugby_ball': 768,
            'baseball': 429, 'basketball': 430, 'croquet_ball': 522, 'beaker': 438, 'vase': 883, 'water_tower': 900,
            'jug': -331, 'water_bottle': 898, 'pill_bottle': 720, 'beer_bottle': 440, 'wine_bottle': 907,
            'pop_bottle': 737, 'freight_car': 565, 'passenger_car': 705, 'jinrikisha': 612, 'oxcart': 690,
            'horse_cart': 603, 'shopping_cart': 791, 'barrow': 428, 'bicycle-built-for-two': 444, 'mountain_bike': 671,
            'streetcar': 829, 'forklift': 561, 'tank': 847, 'tractor': 866, 'recreational_vehicle': 757,
            'tracked_vehicle': -332, 'locomotive': -333, 'motor_vehicle': -334, 'mixing_bowl': 659, 'soup_bowl': 809,
            'theater_curtain': 854, 'shower_curtain': 794, 'bulletproof_vest': 465, 'chain_mail': 490, 'cuirass': 524,
            'grey_whale': 147, 'killer_whale': 148, 'bovid': -335, 'arabian_camel': 354, 'swine': -336, 'llama': 355,
            'hippopotamus': 344, 'sorrel': 339, 'zebra': 340, 'cat': -337, 'big_cat': -338, 'hyena': 276,
            'wild_dog': -339, 'dog': -340, 'fox': -341, 'wolf': -342, 'meerkat': 299, 'mongoose': 298,
            'american_black_bear': 295, 'brown_bear': 294, 'ice_bear': 296, 'sloth_bear': 297, 'mink': 357,
            'badger': 362, 'otter': 360, 'weasel': 356, 'skunk': 361, 'polecat': 358, 'black-footed_ferret': 359,
            'giant_panda': 388, 'lesser_panda': 387, 'wood_rabbit': 330, 'angora': 332, 'lesser_ape': -343,
            'great_ape': -344, 'new_world_monkey': -345, 'old_world_monkey': -346, 'indri': 384, 'madagascar_cat': 383,
            'ruddy_turnstone': 139, 'sandpiper': -347, 'dowitcher': 142, 'oystercatcher': 143, 'little_blue_heron': 131,
            'bittern': 133, 'american_egret': 132, 'black_stork': 128, 'white_stork': 127, 'red-breasted_merganser': 98,
            'drake': 97, 'rock_beauty': 392, 'anemone_fish': 393, 'goldfish': 1, 'tench': 0, 'horned_viper': 66,
            'rattlesnake': -348, 'thunder_snake': 52, 'ringneck_snake': 53, 'garter_snake': 57, 'water_snake': 58,
            'vine_snake': 59, 'king_snake': 56, 'green_snake': 55, 'hognose_snake': 54, 'night_snake': 60,
            'boa_constrictor': 61, 'rock_python': 62, 'indian_cobra': 63, 'green_mamba': 64, 'frilled_lizard': 43,
            'agama': 42, 'common_iguana': 39, 'american_chameleon': 40, 'king_crab': 121, 'rock_crab': 119,
            'fiddler_crab': 120, 'dungeness_crab': 118, 'american_lobster': 122, 'spiny_lobster': 123,
            'rhinoceros_beetle': 306, 'dung_beetle': 305, 'trench_coat': 869, 'fur_coat': 568, 'lab_coat': 617,
            'folding_chair': 559, 'barber_chair': 423, 'throne': 857, 'rocking_chair': 765, 'can_opener': 473,
            'corkscrew': 512, 'cleaver': 499, 'letter_opener': 623, 'cannon': 471, 'firearm': -349, 'timer': -350,
            'sundial': 835, 'digital_watch': 531, 'hourglass': 604, 'clock': -351, 'combination_lock': 507,
            'padlock': 695, 'car_wheel': 479, 'paddlewheel': 694, 'potters_wheel': 739, 'pinwheel': 723, 'violin': 889,
            'cello': 486, 'electric_guitar': 546, 'acoustic_guitar': 402, 'upright': 881, 'grand_piano': 579,
            'french_horn': 566, 'cornet': 513, 'trombone': 875, 'beating-reed_instrument': -353, 'flute': 558,
            'accordion': 401, 'harmonica': 593, 'desktop_computer': 527, 'portable_computer': -354,
            'aircraft_carrier': 403, 'submarine': 833, 'boat': -355, 'ship': -356, 'sailing_vessel': -357,
            'heavier-than-air_craft': -358, 'lighter-than-air_craft': -359, 'computer_keyboard': 508,
            'whiskey_jug': 901, 'water_jug': 899, 'snowmobile': 802, 'steam_locomotive': 820,
            'electric_locomotive': 547, 'amphibian': 408, 'car': -360, 'go-kart': 573, 'truck': -361, 'snowplow': 803,
            'moped': 665, 'ox': 345, 'ibex': 350, 'bison': 347, 'ram': 348, 'antelope': -362, 'bighorn': 349,
            'water_buffalo': 346, 'hog': 341, 'warthog': 343, 'wild_boar': 342, 'wildcat': -363, 'domestic_cat': -364,
            'tiger': 292, 'cheetah': 293, 'lion': 291, 'snow_leopard': 289, 'leopard': 288, 'jaguar': 290, 'dhole': 274,
            'dingo': 273, 'african_hunting_dog': 275, 'mexican_hairless': 268, 'basenji': 253, 'spitz': -365,
            'pug': 254, 'newfoundland': 256, 'great_pyrenees': 257, 'brabancon_griffon': 262, 'poodle': -366,
            'corgi': -367, 'dalmatian': 251, 'working_dog': -368, 'leonberg': 255, 'toy_dog': -369, 'hunting_dog': -370,
            'arctic_fox': 279, 'grey_fox': 280, 'red_fox': 277, 'kit_fox': 278, 'timber_wolf': 269, 'red_wolf': 271,
            'coyote': 272, 'white_wolf': 270, 'gibbon': 368, 'siamang': 369, 'gorilla': 366, 'orangutan': 365,
            'chimpanzee': 367, 'howler_monkey': 379, 'marmoset': 377, 'spider_monkey': 381, 'titi': 380,
            'capuchin': 378, 'squirrel_monkey': 382, 'proboscis_monkey': 376, 'macaque': 373, 'colobus': 375,
            'langur': 374, 'baboon': 372, 'guenon': 370, 'patas': 371, 'red-backed_sandpiper': 140, 'redshank': 141,
            'sidewinder': 68, 'diamondback': 67, 'assault_rifle': 413, 'rifle': 764, 'revolver': 763,
            'parking_meter': 704, 'stopwatch': 826, 'digital_clock': 530, 'wall_clock': 892, 'analog_clock': 409,
            'sax': 776, 'double-reed_instrument': -371, 'laptop': 620, 'notebook': 681, 'hand-held_computer': 590,
            'gondola': 576, 'fireboat': 554, 'lifeboat': 625, 'small_boat': -372, 'speedboat': 814,
            'container_ship': 510, 'liner': 628, 'wreck': 913, 'pirate': 724, 'sailboat': -373, 'schooner': 780,
            'airliner': 404, 'warplane': 895, 'balloon': 417, 'airship': 405, 'beach_wagon': 436, 'cab': 468,
            'jeep': 609, 'minivan': 656, 'ambulance': 407, 'limousine': 627, 'sports_car': 817, 'model_t': 661,
            'racer': 751, 'convertible': 511, 'fire_engine': 555, 'van': -374, 'garbage_truck': 569,
            'trailer_truck': 867, 'pickup': 717, 'tow_truck': 864, 'gazelle': 353, 'hartebeest': 351, 'impala': 352,
            'cougar': 286, 'lynx': 287, 'siamese_cat': 284, 'persian_cat': 283, 'tabby': 281, 'egyptian_cat': 285,
            'tiger_cat': 282, 'chow': 260, 'keeshond': 261, 'pomeranian': 259, 'samoyed': 258, 'toy_poodle': 265,
            'standard_poodle': 267, 'miniature_poodle': 266, 'pembroke': 263, 'cardigan_1': 264, 'bull_mastiff': 243,
            'sennenhunde': -376, 'watchdog': -377, 'shepherd_dog': -378, 'boxer': 242, 'tibetan_mastiff': 244,
            'saint_bernard': 247, 'french_bulldog': 245, 'toy_spaniel': -379, 'eskimo_dog': 248, 'great_dane': 246,
            'hound': -380, 'toy_terrier': 158, 'pekinese': 154, 'japanese_spaniel': 152, 'maltese_dog': 153,
            'shih-tzu': 155, 'chihuahua': 151, 'rhodesian_ridgeback': 159, 'sporting_dog': -381, 'terrier': -382,
            'pinscher': -383, 'bassoon': 432, 'oboe': 683, 'yawl': 914, 'canoe': 472, 'trimaran': 871, 'catamaran': 484,
            'moving_van': 675, 'police_van': 734, 'malamute': 249, 'siberian_husky': 250, 'appenzeller': 240,
            'greater_swiss_mountain_dog': 238, 'entlebucher': 241, 'bernese_mountain_dog': 239,
            'belgian_sheepdog': -384, 'schipperke': 223, 'kuvasz': 222, 'shetland_sheepdog': 230, 'kelpie': 227,
            'bouvier_des_flandres': 233, 'komondor': 228, 'old_english_sheepdog': 229, 'border_collie': 232,
            'collie': 231, 'briard': 226, 'german_shepherd': 235, 'rottweiler': 234, 'greyhound': -385, 'papillon': 157,
            'blenheim_spaniel': 156, 'bluetick': 164, 'scottish_deerhound': 177, 'ibizan_hound': 173, 'wolfhound': -386,
            'bloodhound': 163, 'beagle': 162, 'otterhound': 175, 'weimaraner': 178, 'redbone': 168, 'foxhound': -387,
            'afghan_hound': 160, 'saluki': 176, 'basset': 161, 'spaniel': -388, 'norwegian_elkhound': 174,
            'black-and-tan_coonhound': 165, 'setter': -389, 'pointer': -390, 'retriever': -391, 'schnauzer': -392,
            'kerry_blue_terrier': 183, 'scotch_terrier': 199, 'tibetan_terrier': 200,
            'west_highland_white_terrier': 203, 'australian_terrier': 193, 'dandie_dinmont': 194, 'lhasa': 204,
            'bedlington_terrier': 181, 'airedale': 191, 'norwich_terrier': 186, 'silky_terrier': 201,
            'norfolk_terrier': 185, 'yorkshire_terrier': 187, 'wirehair': -393, 'bullterrier': -394, 'cairn': 192,
            'boston_bull': 195, 'soft-coated_wheaten_terrier': 202, 'springer_spaniel': -396, 'irish_terrier': 184,
            'border_terrier': 182, 'wire-haired_fox_terrier': 188, 'affenpinscher': 252, 'doberman': 236,
            'miniature_pinscher': 237, 'groenendael': 224, 'malinois': 225, 'whippet': 172, 'italian_greyhound': 171,
            'irish_wolfhound': 170, 'borzoi': 169, 'english_foxhound': 167, 'walker_hound': 166, 'clumber': 216,
            'sussex_spaniel': 220, 'cocker_spaniel': 219, 'brittany_spaniel': 215, 'irish_water_spaniel': 221,
            'english_setter': 212, 'gordon_setter': 214, 'irish_setter': 213, 'german_short-haired_pointer': 210,
            'vizsla': 211, 'golden_retriever': 207, 'flat-coated_retriever': 205, 'labrador_retriever': 208,
            'chesapeake_bay_retriever': 209, 'curly-coated_retriever': 206, 'standard_schnauzer': 198,
            'miniature_schnauzer': 196, 'giant_schnauzer': 197, 'lakeland_terrier': 189, 'sealyham_terrier': 190,
            'american_staffordshire_terrier': 180, 'staffordshire_bullterrier': 179, 'english_springer': 217,
            'welsh_springer_spaniel': 218, 'sled_dog': -375}


# trial 3
def imagenet1000_label_dict_index_trial3():
    return {'physical_entity': -1, 'abstraction': -2, 'matter': -3, 'causal_agent': -4, 'object': -5, 'bubble': 971,
            'communication': -6, 'food': -7, 'food_1': -8, 'substance': -9, 'person': -10, 'whole': -11,
            'geological_formation': -12, 'traffic_light': 920, 'street_sign': 919, 'nutriment': -13, 'foodstuff': -14,
            'hay': 958, 'menu': 922, 'beverage': -15, 'bread': -16, 'produce': -17, 'toilet_tissue': 999,
            'ballplayer': 981, 'scuba_diver': 983, 'groom': 982, 'animal': -18, 'natural_object': -19, 'organism': -20,
            'artifact': -21, 'valley': 979, 'geyser': 974, 'shore': -22, 'cliff': 972, 'natural_elevation': -23,
            'course': -24, 'dish': -25, 'condiment': -26, 'dough': 961, 'espresso': 967, 'alcohol': -27,
            'loaf_of_bread': -28, 'pretzel': 932, 'bagel': 931, 'vegetable': -29, 'vertebrate': -30,
            'invertebrate': -31, 'fruit': -32, 'sandbar': 977, 'flower': -33, 'fungus': -34, 'fabric': -35,
            'pillow': 721, 'sheet': -36, 'consumer_goods': -37, 'rain_barrel': 756, 'plaything': -38, 'structure': -39,
            'instrumentality': -40, 'stage': 819, 'decoration': -41, 'commodity': -42, 'covering': -43, 'lakeside': 975,
            'seashore': 978, 'mountain': -44, 'ridge': -45, 'promontory': 976, 'plate': 923, 'dessert': -46,
            'pizza': 963, 'consomme': 925, 'burrito': 965, 'meat_loaf': 962, 'hot_pot': 926, 'sandwich': -47,
            'potpie': 964, 'sauce': -48, 'guacamole': 924, 'red_wine': 966, 'punch': -49, 'french_loaf': 930,
            'solanaceous_vegetable': -50, 'mashed_potato': 935, 'mushroom': 947, 'artichoke': 944,
            'cruciferous_vegetable': -51, 'cucumber': 943, 'cardoon': 946, 'squash': -52, 'mammal': -53, 'bird': -54,
            'fish': -55, 'reptile': -56, 'amphibian_1': -57, 'worm': -58, 'echinoderm': -59, 'arthropod': -60,
            'coelenterate': -61, 'mollusk': -62, 'ear': 998, 'seed': -63, 'hip': 989, 'acorn': 988, 'edible_fruit': -64,
            'yellow_ladys_slipper': 986, 'daisy': 985, 'agaric': 992, 'stinkhorn': 994, 'hen-of-the-woods': 996,
            'coral_fungus': 991, 'earthstar': 995, 'bolete': 997, 'gyromitra': 993, 'piece_of_cloth': -66,
            'velvet': 885, 'wool': 911, 'scoreboard': 781, 'clothing': -67, 'home_appliance': -68, 'teddy': 850,
            'coil': 506, 'bridge': -69, 'mountain_tent': 672, 'fountain': 562, 'patio': 706, 'housing': -70,
            'dock': 536, 'establishment': -71, 'barrier': -72, 'beacon': 437, 'triumphal_arch': 873, 'altar': 406,
            'castle': 483, 'lumbermill': 634, 'memorial': -73, 'building': -74, 'column': -75,
            'supporting_structure': -76, 'furnishing': -77, 'comic_book': 917, 'system': -78, 'implement': -79,
            'toiletry': -80, 'chain': 488, 'device': -81, 'conveyance': -82, 'equipment': -83, 'container': -84,
            'necklace': 679, 'bath_towel': 434, 'protective_covering': -85, 'mask': 643, 'shoji': 789, 'footwear': -86,
            'cloak': 501, 'top': -87, 'book_jacket': 921, 'floor_cover': -88, 'cloth_covering': -89, 'alp': 970,
            'volcano': 980, 'coral_reef': 973, 'frozen_dessert': -90, 'trifle': 927, 'hotdog': 934, 'cheeseburger': 933,
            'chocolate_sauce': 960, 'carbonara': 959, 'eggnog': 969, 'cup': 968, 'bell_pepper': 945, 'cauliflower': 938,
            'head_cabbage': 936, 'broccoli': 937, 'summer_squash': -91, 'winter_squash': -92, 'monotreme': -93,
            'placental': -94, 'marsupial': -95, 'tusker': 101, 'oscine': -96, 'piciform_bird': -97, 'ostrich': 9,
            'cock': 7, 'hummingbird': 94, 'game_bird': -98, 'coraciiform_bird': -99, 'aquatic_bird': -100,
            'parrot': -101, 'coucal': 91, 'hen': 8, 'bird_of_prey': -102, 'teleost_fish': -103, 'elasmobranch': -104,
            'food_fish': -105, 'diapsid': -106, 'turtle': -107, 'salamander': -108, 'frog': -109, 'flatworm': 110,
            'nematode': 111, 'starfish': 327, 'sea_urchin': 328, 'sea_cucumber': 329, 'arachnid': -110, 'trilobite': 69,
            'crustacean': -111, 'insect': -112, 'centipede': 79, 'jellyfish': 107, 'anthozoan': -113,
            'chambered_nautilus': 117, 'gastropod': -114, 'chiton': 116, 'corn': 987, 'buckeye': 990, 'rapeseed': 984,
            'custard_apple': 956, 'fig': 952, 'strawberry': 949, 'pineapple': 953, 'citrus': -115, 'pomegranate': 957,
            'jackfruit': 955, 'granny_smith': 948, 'banana': 954, 'bib': 443, 'towel': -116, 'handkerchief': 591,
            'dishrag': 533, 'seat_belt': 785, 'mitten': 658, 'protective_garment': -117, 'headdress': -118, 'gown': 910,
            'military_uniform': 652, 'hosiery': -119, 'womans_clothing': -120, 'pajama': 697, 'attire': -121,
            'garment': -122, 'white_goods': -123, 'sewing_machine': 786, 'vacuum': 882, 'kitchen_appliance': -124,
            'iron': 606, 'viaduct': 888, 'steel_arch_bridge': 821, 'suspension_bridge': 839, 'mobile_home': 660,
            'dwelling': -125, 'mercantile_establishment': -126, 'prison': 743, 'movable_barrier': -127, 'dam': 525,
            'grille': 581, 'breakwater': 460, 'fence': -128, 'bannister': 421, 'megalith': 649, 'brass': 458,
            'barn': 425, 'theater': -129, 'place_of_worship': -130, 'shed': -131, 'restaurant': 762, 'library': 624,
            'greenhouse': 580, 'residence': -132, 'planetarium': 727, 'totem_pole': 863, 'obelisk': 682,
            'pedestal': 708, 'framework': -133, 'furniture': -134, 'maze': 646, 'communication_system': -135,
            'pen': -136, 'pole': 733, 'space_bar': 810, 'cooking_utensil': -137, 'stick': -138, 'racket': 752,
            'tool': -139, 'cleaning_implement': -140, 'paddle': 693, 'rubber_eraser': 767, 'pencil_sharpener': 710,
            'sunscreen': 838, 'makeup': -141, 'hair_spray': 585, 'perfume': 711, 'lotion': 631,
            'source_of_illumination': -142, 'instrument': -143, 'lighter': 626, 'filter': -144, 'whistle': 902,
            'heater': -145, 'paintbrush': 696, 'hard_disc': 592, 'remote_control': 761, 'pick': 714, 'restraint': -146,
            'electric_fan': 545, 'neck_brace': 678, 'electro-acoustic_transducer': -147, 'ski': 795, 'sunglass': 836,
            'trap': -148, 'mechanism': -149, 'crane': 517, 'breathing_device': -150, 'wing': 908,
            'electronic_device': -151, 'reflector': -152, 'support': -153, 'hand_blower': 589, 'keyboard': -154,
            'musical_instrument': -155, 'machine': -156, 'stretcher': 830, 'vehicle': -157, 'public_transport': -158,
            'sports_equipment': -159, 'electronic_equipment': -160, 'parachute': 701, 'camera': -161,
            'game_equipment': -162, 'photocopier': 713, 'gear': -163, 'packet': 692, 'pot_1': 738, 'vessel': -165,
            'cassette': 481, 'box': -166, 'shaker': -167, 'tray': 868, 'soap_dispenser': 804, 'milk_can': 653,
            'wooden_spoon': 910, 'envelope': 549, 'ashcan': 412, 'wheeled_vehicle': -168, 'glass': -169, 'dish_1': -170,
            'wallet': 893, 'bag': -171, 'piggy_bank': 719, 'basket': -172, 'measuring_cup': 647, 'armor_plate': -173,
            'roof': -174, 'sheath': -175, 'cap_1': -176, 'blind': -177, 'screen': 509, 'lampshade': 619,
            'shelter': -178, 'mask_1': -179, 'binder': 446, 'armor': -180, 'cowboy_boot': 514, 'clog': 502,
            'shoe': -181, 'manhole_cover': 640, 'cap': -182, 'doormat': 539, 'prayer_rug': 741, 'quilt': 750,
            'band_aid': 419, 'ice_lolly': 929, 'ice_cream': 928, 'spaghetti_squash': 940, 'zucchini': 939,
            'butternut_squash': 942, 'acorn_squash': 941, 'platypus': 103, 'echidna': 102, 'edentate': -183,
            'aquatic_mammal': -184, 'ungulate': -185, 'rodent': -186, 'carnivore': -187, 'leporid': -188,
            'elephant': -189, 'primate': -190, 'koala': 105, 'wallaby': 104, 'wombat': 106, 'corvine_bird': -191,
            'water_ouzel': 20, 'finch': -192, 'chickadee': 19, 'thrush': -193, 'toucan': 96, 'jacamar': 95,
            'phasianid': -194, 'grouse': -195, 'hornbill': 93, 'bee_eater': 92, 'wading_bird': -196, 'black_swan': 100,
            'european_gallinule': 136, 'seabird': -197, 'anseriform_bird': -198, 'sulphur-crested_cockatoo': 89,
            'lorikeet': 90, 'african_grey': 87, 'macaw': 88, 'bald_eagle': 22, 'vulture': 23, 'kite': 21,
            'great_grey_owl': 24, 'spiny-finned_fish': -199, 'soft-finned_fish': -200, 'ganoid': -201, 'ray': -202,
            'shark': -203, 'barracouta': 389, 'coho': 391, 'snake': -204, 'lizard': -205, 'triceratops': 51,
            'crocodilian_reptile': -206, 'sea_turtle': -207, 'box_turtle': 37, 'terrapin': 36, 'mud_turtle': 35,
            'european_fire_salamander': 25, 'newt': -208, 'ambystomid': -209, 'tailed_frog': 32, 'bullfrog': 30,
            'tree_frog': 31, 'tick': 78, 'spider': -210, 'scorpion': 71, 'harvestman': 70, 'decapod_crustacean': -211,
            'isopod': 126, 'homopterous_insect': -212, 'butterfly': -213, 'beetle': -214, 'orthopterous_insect': -215,
            'walking_stick': 313, 'fly': 308, 'hymenopterous_insect': -216, 'dictyopterous_insect': -217,
            'odonate': -218, 'lacewing': 318, 'sea_anemone': 108, 'brain_coral': 109, 'slug': 114, 'sea_slug': 115,
            'conch': 112, 'snail': 113, 'orange': 950, 'lemon': 951, 'paper_towel': 700, 'knee_pad': 615, 'apron': 411,
            'helmet': -219, 'hat': -220, 'cap_2': -221, 'vestment': 887, 'academic_gown': 400,
            'christmas_stocking': 496, 'sock': 806, 'maillot_1': 639, 'gown_1': 578, 'wig': 903, 'scarf': -224,
            'diaper': 529, 'skirt': -225, 'suit': 834, 'swimsuit': -226, 'jean': 608, 'brassiere': 459,
            'overgarment': -227, 'robe': -228, 'sweater': -229, 'necktie': -230, 'jersey': 610, 'refrigerator': 760,
            'washer': 897, 'dishwasher': 534, 'espresso_maker': 550, 'waffle_iron': 891, 'microwave': 651, 'oven': -231,
            'toaster': 859, 'cliff_dwelling': 500, 'yurt': 915, 'shop': -232, 'grocery_store': 582, 'sliding_door': 799,
            'turnstile': 877, 'chainlink_fence': 489, 'picket_fence': 716, 'stone_wall': 825, 'worm_fence': 912,
            'cinema': 498, 'home_theater': 598, 'stupa': 832, 'mosque': 668, 'church': 497, 'apiary': 410,
            'boathouse': 449, 'palace': 698, 'monastery': 663, 'honeycomb': 599, 'plate_rack': 729, 'four-poster': 564,
            'baby_bed': -233, 'table': -234, 'cabinet': -235, 'file': 553, 'entertainment_center': 548,
            'dining_table': 532, 'bookcase': 453, 'table_lamp': 846, 'wardrobe': 894, 'seat': -236, 'chiffonier': 493,
            'radio': 754, 'television': 851, 'fountain_pen': 563, 'quill': 749, 'ballpoint': 418, 'pan': -237,
            'spatula': 813, 'crock_pot': 521, 'pot': 837, 'drumstick': 542, 'spindle': 816, 'staff': -238,
            'matchstick': 644, 'hand_tool': -239, 'edge_tool': -240, 'plow': 730, 'power_drill': 740, 'lawn_mower': 621,
            'swab': 840, 'broom': 462, 'face_powder': 551, 'lipstick': 629, 'torch': 862, 'lamp': -241,
            'optical_instrument': -242, 'weapon': -243, 'measuring_instrument': -244, 'magnifier': -245,
            'guillotine': 583, 'medical_instrument': -246, 'magnetic_compass': 635, 'oil_filter': 686, 'strainer': 828,
            'space_heater': 811, 'stove': 827, 'disk_brake': 535, 'fastener': -247, 'muzzle': 676, 'loudspeaker': 632,
            'microphone': 650, 'mousetrap': 674, 'spider_web': 815, 'radiator': 753, 'puck': 746, 'control': -248,
            'mechanical_device': -249, 'snorkel': 801, 'oxygen_mask': 691, 'mouse': 673, 'screen_1': 782,
            'solar_dish': 807, 'car_mirror': 475, 'pier': 718, 'tripod': 872, 'maypole': 645,
            'typewriter_keyboard': 878, 'stringed_instrument': -251, 'percussion_instrument': -252,
            'keyboard_instrument': -253, 'wind_instrument': -254, 'power_tool': -255, 'cash_machine': 480,
            'abacus': 398, 'farm_machine': -256, 'computer': -257, 'slot_machine': -258, 'military_vehicle': -259,
            'sled': -260, 'craft': -261, 'missile': 657, 'bus': -262, 'bullet_train': 466, 'gymnastic_apparatus': -263,
            'golf_equipment': -264, 'weight': -265, 'cassette_player': 482, 'modem': 662, 'telephone': -266,
            'tape_player': 848, 'ipod': 605, 'cd_player': 485, 'monitor': 664, 'oscilloscope': 688, 'peripheral': -267,
            'polaroid_camera': 732, 'reflex_camera': 759, 'puzzle': -268, 'ball': -269, 'pool_table': 736,
            'carpenters_kit': 477, 'drilling_platform': 540, 'mortar': 666, 'ladle': 618, 'tub': 876, 'pitcher': 725,
            'jar': -271, 'coffee_mug': 504, 'bucket': 463, 'barrel': 427, 'bathtub': 435, 'reservoir': -272,
            'washbasin': 896, 'bottle': -273, 'safe': 771, 'pencil_box': 709, 'mailbox': 637, 'crate': 519,
            'chest': 492, 'carton': 478, 'saltshaker': 773, 'cocktail_shaker': 503, 'unicycle': 880, 'car_1': -274,
            'motor_scooter': 670, 'cart': -275, 'handcart': -276, 'bicycle': -277, 'tricycle': 870,
            'self-propelled_vehicle': -278, 'beer_glass': 441, 'goblet': 572, 'bowl': -279, 'petri_dish': 712,
            'backpack': 414, 'sleeping_bag': 797, 'mailbag': 636, 'purse': 748, 'plastic_bag': 728,
            'shopping_basket': 790, 'hamper': 588, 'breastplate': 461, 'pickelhaube': 715, 'vault': 884, 'dome': 538,
            'tile_roof': 858, 'thatch': 853, 'holster': 597, 'scabbard': 777, 'lens_cap': 622, 'thimble': 855,
            'curtain': -280, 'window_shade': 905, 'fire_screen': 556, 'window_screen': 904, 'mosquito_net': 669,
            'birdhouse': 448, 'umbrella': 879, 'bell_cote': 442, 'gasmask': 570, 'ski_mask': 796, 'shield': 787,
            'body_armor': -281, 'running_shoe': 770, 'loafer': 630, 'sandal': 774, 'bottlecap': 455, 'nipple': 680,
            'armadillo': 363, 'three-toed_sloth': 364, 'dugong': 149, 'sea_lion': 150, 'whale': -282,
            'even-toed_ungulate': -283, 'equine': -284, 'guinea_pig': 338, 'marmot': 336, 'porcupine': 334,
            'hamster': 333, 'beaver': 337, 'fox_squirrel': 335, 'feline': -285, 'canine': -286, 'viverrine': -287,
            'bear': -288, 'musteline_mammal': -289, 'procyonid': -290, 'rabbit': -291, 'hare': 331,
            'indian_elephant': 385, 'african_elephant': 386, 'anthropoid_ape': -292, 'monkey': -293, 'lemur': -294,
            'magpie': 18, 'jay': 17, 'goldfinch': 11, 'junco': 13, 'brambling': 10, 'indigo_bunting': 14,
            'house_finch': 12, 'bulbul': 16, 'robin': 15, 'peacock': 84, 'partridge': 86, 'quail': 85,
            'prairie_chicken': 83, 'ruffed_grouse': 82, 'ptarmigan': 81, 'black_grouse': 80, 'bustard': 138,
            'shorebird': -295, 'heron': -296, 'limpkin': 135, 'spoonbill': 129, 'american_coot': 137, 'stork': -297,
            'crane_1': 134, 'flamingo': 130, 'king_penguin': 145, 'albatross': 146, 'pelican': 144, 'duck': -299,
            'goose': 99, 'lionfish': 396, 'percoid_fish': -300, 'puffer': 397, 'eel': 390, 'cyprinid': -301, 'gar': 395,
            'sturgeon': 394, 'electric_ray': 5, 'stingray': 6, 'great_white_shark': 2, 'hammerhead': 4,
            'tiger_shark': 3, 'viper': -302, 'colubrid_snake': -303, 'boa': -304, 'sea_snake': 65, 'elapid': -305,
            'agamid': -306, 'banded_gecko': 38, 'green_lizard': 46, 'komodo_dragon': 48, 'gila_monster': 45,
            'whiptail': 41, 'iguanid': -307, 'alligator_lizard': 44, 'african_chameleon': 47, 'african_crocodile': 49,
            'american_alligator': 50, 'leatherback_turtle': 34, 'loggerhead': 33, 'eft': 27, 'common_newt': 26,
            'axolotl': 29, 'spotted_salamander': 28, 'garden_spider': 74, 'barn_spider': 73, 'black_widow': 75,
            'tarantula': 76, 'wolf_spider': 77, 'black_and_gold_garden_spider': 72, 'hermit_crab': 125, 'crab': -308,
            'lobster': -309, 'crayfish': 124, 'cicada': 316, 'leafhopper': 317, 'admiral': 321, 'lycaenid': 326,
            'sulphur_butterfly': 325, 'ringlet': 322, 'cabbage_butterfly': 324, 'monarch': 323, 'leaf_beetle': 304,
            'tiger_beetle': 300, 'long-horned_beetle': 303, 'scarabaeid_beetle': -310, 'ground_beetle': 302,
            'weevil': 307, 'ladybug': 301, 'cricket': 312, 'grasshopper': 311, 'ant': 310, 'bee': 309, 'cockroach': 314,
            'mantis': 315, 'dragonfly': 319, 'damselfly': 320, 'crash_helmet': 518, 'football_helmet': 560,
            'bearskin': 439, 'bonnet': 452, 'cowboy_hat': 515, 'sombrero': 808, 'bathing_cap': 433, 'shower_cap': 793,
            'mortarboard': 667, 'stole': 824, 'feather_boa': 552, 'hoopskirt': 601, 'sarong': 775, 'overskirt': 689,
            'miniskirt': 655, 'maillot': 638, 'bikini': 445, 'swimming_trunks': 842, 'poncho': 735, 'coat': -311,
            'kimono': 614, 'abaya': 399, 'cardigan': 474, 'sweatshirt': 841, 'bow_tie': 457, 'bolo_tie': 451,
            'windsor_tie': 906, 'rotisserie': 766, 'dutch_oven': 544, 'confectionery': 509, 'shoe_shop': 788,
            'toyshop': 865, 'bookshop': 454, 'tobacco_shop': 860, 'bakery': 415, 'barbershop': 424, 'butcher_shop': 467,
            'bassinet': 431, 'crib': 520, 'cradle': 516, 'desk': 526, 'medicine_chest': 648, 'china_cabinet': 495,
            'chair': -312, 'toilet_seat': 861, 'park_bench': 703, 'studio_couch': 831, 'wok': 909, 'frying_pan': 567,
            'caldron': 469, 'teapot': 849, 'coffeepot': 505, 'crutch': 523, 'flagpole': 557, 'shovel': 792,
            'opener': -313, 'plunger': 731, 'hammer': 587, 'screwdriver': 784, 'hatchet': 596, 'knife': -314,
            'plane': 726, 'jack-o-lantern': 607, 'spotlight': 818, 'candle': 470, 'sunglasses': 837, 'projector': 745,
            'binoculars': 447, 'gun': -316, 'projectile': 744, 'bow': 456, 'rule': 769, 'scale': 778, 'odometer': 685,
            'barometer': 426, 'timepiece': -317, 'radio_telescope': 755, 'loupe': 633, 'stethoscope': 823,
            'syringe': 845, 'buckle': 464, 'nail': 677, 'safety_pin': 772, 'knot': 616, 'screw': 783, 'lock': -318,
            'hair_slide': 584, 'switch': 844, 'joystick': 613, 'hook': 600, 'gas_pump': 571, 'carousel': 476,
            'reel': 758, 'wheel': -319, 'swing': 843, 'harp': 594, 'banjo': 420, 'bowed_stringed_instrument': -320,
            'guitar': -321, 'steel_drum': 822, 'maraca': 641, 'marimba': 642, 'gong': 577, 'chime': 494, 'drum': 541,
            'piano': -322, 'organ': 687, 'panpipe': 699, 'brass_1': -323, 'woodwind': -324, 'ocarina': 684,
            'free-reed_instrument': -325, 'chain_saw': 491, 'harvester': 595, 'thresher': 856,
            'personal_computer': -326, 'slide_rule': 798, 'web_site': 916, 'vending_machine': 886, 'slot': 800,
            'warship': -327, 'half_track': 586, 'bobsled': 450, 'dogsled': 537, 'vessel_1': -328, 'aircraft': -329,
            'space_shuttle': 812, 'minibus': 654, 'trolleybus': 874, 'school_bus': 779, 'horizontal_bar': 602,
            'parallel_bars': 702, 'balance_beam': 416, 'golfcart': 575, 'dumbbell': 543, 'barbell': 422,
            'dial_telephone': 528, 'pay-phone': 707, 'cellular_telephone': 487, 'data_input_device': -330,
            'printer': 742, 'jigsaw_puzzle': 611, 'crossword_puzzle': 918, 'tennis_ball': 852, 'volleyball': 890,
            'golf_ball': 574, 'punching_bag': 747, 'ping-pong_ball': 722, 'soccer_ball': 805, 'rugby_ball': 768,
            'baseball': 429, 'basketball': 430, 'croquet_ball': 522, 'beaker': 438, 'vase': 883, 'water_tower': 900,
            'jug': -331, 'water_bottle': 898, 'pill_bottle': 720, 'beer_bottle': 440, 'wine_bottle': 907,
            'pop_bottle': 737, 'freight_car': 565, 'passenger_car': 705, 'jinrikisha': 612, 'oxcart': 690,
            'horse_cart': 603, 'shopping_cart': 791, 'barrow': 428, 'bicycle-built-for-two': 444, 'mountain_bike': 671,
            'streetcar': 829, 'forklift': 561, 'tank': 847, 'tractor': 866, 'recreational_vehicle': 757,
            'tracked_vehicle': -332, 'locomotive': -333, 'motor_vehicle': -334, 'mixing_bowl': 659, 'soup_bowl': 809,
            'theater_curtain': 854, 'shower_curtain': 794, 'bulletproof_vest': 465, 'chain_mail': 490, 'cuirass': 524,
            'grey_whale': 147, 'killer_whale': 148, 'bovid': -335, 'arabian_camel': 354, 'swine': -336, 'llama': 355,
            'hippopotamus': 344, 'sorrel': 339, 'zebra': 340, 'cat': -337, 'big_cat': -338, 'hyena': 276,
            'wild_dog': -339, 'dog': -340, 'fox': -341, 'wolf': -342, 'meerkat': 299, 'mongoose': 298,
            'american_black_bear': 295, 'brown_bear': 294, 'ice_bear': 296, 'sloth_bear': 297, 'mink': 357,
            'badger': 362, 'otter': 360, 'weasel': 356, 'skunk': 361, 'polecat': 358, 'black-footed_ferret': 359,
            'giant_panda': 388, 'lesser_panda': 387, 'wood_rabbit': 330, 'angora': 332, 'lesser_ape': -343,
            'great_ape': -344, 'new_world_monkey': -345, 'old_world_monkey': -346, 'indri': 384, 'madagascar_cat': 383,
            'ruddy_turnstone': 139, 'sandpiper': -347, 'dowitcher': 142, 'oystercatcher': 143, 'little_blue_heron': 131,
            'bittern': 133, 'american_egret': 132, 'black_stork': 128, 'white_stork': 127, 'red-breasted_merganser': 98,
            'drake': 97, 'rock_beauty': 392, 'anemone_fish': 393, 'goldfish': 1, 'tench': 0, 'horned_viper': 66,
            'rattlesnake': -348, 'thunder_snake': 52, 'ringneck_snake': 53, 'garter_snake': 57, 'water_snake': 58,
            'vine_snake': 59, 'king_snake': 56, 'green_snake': 55, 'hognose_snake': 54, 'night_snake': 60,
            'boa_constrictor': 61, 'rock_python': 62, 'indian_cobra': 63, 'green_mamba': 64, 'frilled_lizard': 43,
            'agama': 42, 'common_iguana': 39, 'american_chameleon': 40, 'king_crab': 121, 'rock_crab': 119,
            'fiddler_crab': 120, 'dungeness_crab': 118, 'american_lobster': 122, 'spiny_lobster': 123,
            'rhinoceros_beetle': 306, 'dung_beetle': 305, 'trench_coat': 869, 'fur_coat': 568, 'lab_coat': 617,
            'folding_chair': 559, 'barber_chair': 423, 'throne': 857, 'rocking_chair': 765, 'can_opener': 473,
            'corkscrew': 512, 'cleaver': 499, 'letter_opener': 623, 'cannon': 471, 'firearm': -349, 'timer': -350,
            'sundial': 835, 'digital_watch': 531, 'hourglass': 604, 'clock': -351, 'combination_lock': 507,
            'padlock': 695, 'car_wheel': 479, 'paddlewheel': 694, 'potters_wheel': 739, 'pinwheel': 723, 'violin': 889,
            'cello': 486, 'electric_guitar': 546, 'acoustic_guitar': 402, 'upright': 881, 'grand_piano': 579,
            'french_horn': 566, 'cornet': 513, 'trombone': 875, 'beating-reed_instrument': -353, 'flute': 558,
            'accordion': 401, 'harmonica': 593, 'desktop_computer': 527, 'portable_computer': -354,
            'aircraft_carrier': 403, 'submarine': 833, 'boat': -355, 'ship': -356, 'sailing_vessel': -357,
            'heavier-than-air_craft': -358, 'lighter-than-air_craft': -359, 'computer_keyboard': 508,
            'whiskey_jug': 901, 'water_jug': 899, 'snowmobile': 802, 'steam_locomotive': 820,
            'electric_locomotive': 547, 'amphibian': 408, 'car': -360, 'go-kart': 573, 'truck': -361, 'snowplow': 803,
            'moped': 665, 'ox': 345, 'ibex': 350, 'bison': 347, 'ram': 348, 'antelope': -362, 'bighorn': 349,
            'water_buffalo': 346, 'hog': 341, 'warthog': 343, 'wild_boar': 342, 'wildcat': -363, 'domestic_cat': -364,
            'tiger': 292, 'cheetah': 293, 'lion': 291, 'snow_leopard': 289, 'leopard': 288, 'jaguar': 290, 'dhole': 274,
            'dingo': 273, 'african_hunting_dog': 275, 'mexican_hairless': 268, 'basenji': 253, 'spitz': -365,
            'pug': 254, 'newfoundland': 256, 'great_pyrenees': 257, 'brabancon_griffon': 262, 'poodle': -366,
            'corgi': -367, 'dalmatian': 251, 'working_dog': -368, 'leonberg': 255, 'toy_dog': -369, 'hunting_dog': -370,
            'arctic_fox': 279, 'grey_fox': 280, 'red_fox': 277, 'kit_fox': 278, 'timber_wolf': 269, 'red_wolf': 271,
            'coyote': 272, 'white_wolf': 270, 'gibbon': 368, 'siamang': 369, 'gorilla': 366, 'orangutan': 365,
            'chimpanzee': 367, 'howler_monkey': 379, 'marmoset': 377, 'spider_monkey': 381, 'titi': 380,
            'capuchin': 378, 'squirrel_monkey': 382, 'proboscis_monkey': 376, 'macaque': 373, 'colobus': 375,
            'langur': 374, 'baboon': 372, 'guenon': 370, 'patas': 371, 'red-backed_sandpiper': 140, 'redshank': 141,
            'sidewinder': 68, 'diamondback': 67, 'assault_rifle': 413, 'rifle': 764, 'revolver': 763,
            'parking_meter': 704, 'stopwatch': 826, 'digital_clock': 530, 'wall_clock': 892, 'analog_clock': 409,
            'sax': 776, 'double-reed_instrument': -371, 'laptop': 620, 'notebook': 681, 'hand-held_computer': 590,
            'gondola': 576, 'fireboat': 554, 'lifeboat': 625, 'small_boat': -372, 'speedboat': 814,
            'container_ship': 510, 'liner': 628, 'wreck': 913, 'pirate': 724, 'sailboat': -373, 'schooner': 780,
            'airliner': 404, 'warplane': 895, 'balloon': 417, 'airship': 405, 'beach_wagon': 436, 'cab': 468,
            'jeep': 609, 'minivan': 656, 'ambulance': 407, 'limousine': 627, 'sports_car': 817, 'model_t': 661,
            'racer': 751, 'convertible': 511, 'fire_engine': 555, 'van': -374, 'garbage_truck': 569,
            'trailer_truck': 867, 'pickup': 717, 'tow_truck': 864, 'gazelle': 353, 'hartebeest': 351, 'impala': 352,
            'cougar': 286, 'lynx': 287, 'siamese_cat': 284, 'persian_cat': 283, 'tabby': 281, 'egyptian_cat': 285,
            'tiger_cat': 282, 'chow': 260, 'keeshond': 261, 'pomeranian': 259, 'samoyed': 258, 'toy_poodle': 265,
            'standard_poodle': 267, 'miniature_poodle': 266, 'pembroke': 263, 'cardigan_1': 264, 'bull_mastiff': 243,
            'sennenhunde': -376, 'watchdog': -377, 'shepherd_dog': -378, 'boxer': 242, 'tibetan_mastiff': 244,
            'saint_bernard': 247, 'french_bulldog': 245, 'toy_spaniel': -379, 'eskimo_dog': 248, 'great_dane': 246,
            'hound': -380, 'toy_terrier': 158, 'pekinese': 154, 'japanese_spaniel': 152, 'maltese_dog': 153,
            'shih-tzu': 155, 'chihuahua': 151, 'rhodesian_ridgeback': 159, 'sporting_dog': -381, 'terrier': -382,
            'pinscher': -383, 'bassoon': 432, 'oboe': 683, 'yawl': 914, 'canoe': 472, 'trimaran': 871, 'catamaran': 484,
            'moving_van': 675, 'police_van': 734, 'malamute': 249, 'siberian_husky': 250, 'appenzeller': 240,
            'greater_swiss_mountain_dog': 238, 'entlebucher': 241, 'bernese_mountain_dog': 239,
            'belgian_sheepdog': -384, 'schipperke': 223, 'kuvasz': 222, 'shetland_sheepdog': 230, 'kelpie': 227,
            'bouvier_des_flandres': 233, 'komondor': 228, 'old_english_sheepdog': 229, 'border_collie': 232,
            'collie': 231, 'briard': 226, 'german_shepherd': 235, 'rottweiler': 234, 'greyhound': -385, 'papillon': 157,
            'blenheim_spaniel': 156, 'bluetick': 164, 'scottish_deerhound': 177, 'ibizan_hound': 173, 'wolfhound': -386,
            'bloodhound': 163, 'beagle': 162, 'otterhound': 175, 'weimaraner': 178, 'redbone': 168, 'foxhound': -387,
            'afghan_hound': 160, 'saluki': 176, 'basset': 161, 'spaniel': -388, 'norwegian_elkhound': 174,
            'black-and-tan_coonhound': 165, 'setter': -389, 'pointer': -390, 'retriever': -391, 'schnauzer': -392,
            'kerry_blue_terrier': 183, 'scotch_terrier': 199, 'tibetan_terrier': 200,
            'west_highland_white_terrier': 203, 'australian_terrier': 193, 'dandie_dinmont': 194, 'lhasa': 204,
            'bedlington_terrier': 181, 'airedale': 191, 'norwich_terrier': 186, 'silky_terrier': 201,
            'norfolk_terrier': 185, 'yorkshire_terrier': 187, 'wirehair': -393, 'bullterrier': -394, 'cairn': 192,
            'boston_bull': 195, 'soft-coated_wheaten_terrier': 202, 'springer_spaniel': -396, 'irish_terrier': 184,
            'border_terrier': 182, 'wire-haired_fox_terrier': 188, 'affenpinscher': 252, 'doberman': 236,
            'miniature_pinscher': 237, 'groenendael': 224, 'malinois': 225, 'whippet': 172, 'italian_greyhound': 171,
            'irish_wolfhound': 170, 'borzoi': 169, 'english_foxhound': 167, 'walker_hound': 166, 'clumber': 216,
            'sussex_spaniel': 220, 'cocker_spaniel': 219, 'brittany_spaniel': 215, 'irish_water_spaniel': 221,
            'english_setter': 212, 'gordon_setter': 214, 'irish_setter': 213, 'german_short-haired_pointer': 210,
            'vizsla': 211, 'golden_retriever': 207, 'flat-coated_retriever': 205, 'labrador_retriever': 208,
            'chesapeake_bay_retriever': 209, 'curly-coated_retriever': 206, 'standard_schnauzer': 198,
            'miniature_schnauzer': 196, 'giant_schnauzer': 197, 'lakeland_terrier': 189, 'sealyham_terrier': 190,
            'american_staffordshire_terrier': 180, 'staffordshire_bullterrier': 179, 'english_springer': 217,
            'welsh_springer_spaniel': 218, 'sled_dog': -375, 'other_oscine': -450, 'other_aquatic_bird': -451,
            'other_wheeled_vehicle': -452}


# trial setting 1
# imagenet1000_label_dict_index = imagenet1000_label_dict_index_trial2
imagenet1000_label_dict_index = imagenet1000_label_dict_index_trial3


class iImageNet100(DataHandler):
    base_dataset_cls = datasets.ImageFolder
    # base_dataset_cls = datasets.ImageNet
    transform_type = 'albumentations'
    if transform_type == 'albumentations':
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            # A.ColorJitter(brightness=63 / 255),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # trial 2
    index_list_trial2 = [
        212, 250, 193, 217, 147, 387, 285, 350, 283, 286, 353, 334, 150, 249, 362, 246, 166, 218, 172, 177, 148,
        357, 386, 178, 202, 194, 271, 229, 290, 175, 163, 191, 276, 299, 197, 380, 364, 339, 359, 251, 165, 157,
        361, 179, 268, 233, 356, 266, 264, 225, 349, 335, 375, 282, 204, 352, 272, 187, 256, 294, 277, 174, 234,
        351, 176, 280, 223, 154, 262, 203, 190, 370, 298, 384, 292, 170, 342, 241, 340, 348, 245, 365, 253, 288,
        239, 153, 185, 158, 211, 192, 382, 224, 216, 284, 367, 228, 160, 152, 376, 338
    ]

    # trial 3
    index_list_trial3 = [398, 146, 279, 414, 428, 438, 337, 100, 10, 464, 16, 138, 471, 378, 479, 480, 491, 293, 492,
                         19, 513,
                         519, 527, 274, 136, 561, 565, 368, 11, 99, 583, 338, 584, 333, 594, 351, 595, 344, 12, 379,
                         276, 352,
                         14, 17, 612, 13, 145, 618, 135, 291, 131, 18, 636, 637, 336, 268, 666, 670, 676, 683, 345, 694,
                         695,
                         86, 371, 84, 709, 283, 725, 728, 334, 739, 81, 746, 748, 85, 755, 757, 139, 82, 769, 771, 772,
                         776,
                         797, 129, 829, 837, 844, 847, 292, 282, 269, 380, 866, 875, 876, 880, 20, 340]

    # trial setting 2
    # index_list = index_list_trial2
    index_list = index_list_trial3

    # trial 2
    data_name_hier_dict_100_trial2 = {
        'carnivore': {
            'canine': {
                'hunting_dog': {
                    'sporting_dog': {'english_setter': {}, 'english_springer': {}, 'welsh_springer_spaniel': {},
                                     'clumber': {}, 'vizsla': {}},
                    'terrier': {'australian_terrier': {}, 'soft-coated_wheaten_terrier': {}, 'dandie_dinmont': {},
                                'airedale': {}, 'giant_schnauzer': {}, 'staffordshire_bullterrier': {},
                                'lhasa': {}, 'yorkshire_terrier': {}, 'west_highland_white_terrier': {},
                                'sealyham_terrier': {}, 'norfolk_terrier': {}, 'cairn': {}},
                    'hound': {'walker_hound': {}, 'whippet': {}, 'scottish_deerhound': {}, 'weimaraner': {},
                              'otterhound': {}, 'bloodhound': {}, 'black-and-tan_coonhound': {},
                              'norwegian_elkhound': {}, 'saluki': {}, 'irish_wolfhound': {}, 'afghan_hound': {}}
                },
                'working_dog': {
                    'shepherd_dog': {'old_english_sheepdog': {}, 'bouvier_des_flandres': {}, 'malinois': {},
                                     'groenendael': {}, 'rottweiler': {}, 'komondor': {}},
                    'other_working_dog': {'siberian_husky': {}, 'malamute': {}, 'great_dane': {}, 'schipperke': {},
                                          'entlebucher': {}, 'bernese_mountain_dog': {}, 'french_bulldog': {}}
                },
                'other_dogs': {'dalmatian': {}, 'papillon': {}, 'pekinese': {}, 'maltese_dog': {},
                               'toy_terrier': {}, 'japanese_spaniel': {}, 'mexican_hairless': {},
                               'miniature_poodle': {}, 'cardigan_1': {}, 'newfoundland': {},
                               'brabancon_griffon': {}, 'basenji': {}},

                'other_canines': {'red_wolf': {}, 'coyote': {}, 'hyena': {}, 'red_fox': {}, 'grey_fox': {}}
            },
            'feline': {'egyptian_cat': {}, 'persian_cat': {}, 'tiger_cat': {}, 'siamese_cat': {}, 'cougar': {},
                       'jaguar': {}, 'tiger': {}, 'leopard': {}},
            'other_carnivores': {'badger': {}, 'mink': {}, 'black-footed_ferret': {}, 'skunk': {}, 'weasel': {},
                                 'meerkat': {}, 'mongoose': {}, 'brown_bear': {}, 'lesser_panda': {}}
        },
        'primate': {'titi': {}, 'squirrel_monkey': {}, 'colobus': {}, 'guenon': {}, 'proboscis_monkey': {},
                    'indri': {}, 'orangutan': {}, 'chimpanzee': {}},
        'ungulate': {'ibex': {}, 'gazelle': {}, 'impala': {}, 'hartebeest': {}, 'bighorn': {}, 'ram': {},
                     'wild_boar': {}, 'sorrel': {}, 'zebra': {}},
        'other_placentals': {'grey_whale': {}, 'killer_whale': {}, 'sea_lion': {}, 'fox_squirrel': {},
                             'guinea_pig': {}, 'porcupine': {}, 'three-toed_sloth': {}, 'african_elephant': {}},
    }

    # trial 3
    data_name_hier_dict_100_trial3 = {
        'mammal': {'ungulate': {'hippopotamus': {}, 'ox': {}, 'hartebeest': {}, 'impala': {}, 'zebra': {}},
                   'rodent': {'guinea_pig': {}, 'marmot': {}, 'porcupine': {}, 'hamster': {}, 'beaver': {}},
                   'primate': {'titi': {}, 'capuchin': {}, 'howler_monkey': {}, 'patas': {}, 'gibbon': {}},
                   'feline': {'tiger_cat': {}, 'tiger': {}, 'persian_cat': {}, 'cheetah': {}, 'lion': {}},
                   'canine': {'hyena': {}, 'dhole': {}, 'mexican_hairless': {}, 'arctic_fox': {}, 'timber_wolf': {}}},
        'bird': {'game_bird': {'ruffed_grouse': {}, 'peacock': {}, 'ptarmigan': {}, 'partridge': {}, 'quail': {}},
                 'finch': {'goldfinch': {}, 'junco': {}, 'brambling': {}, 'indigo_bunting': {}, 'house_finch': {}},
                 'wading_bird': {'bustard': {}, 'ruddy_turnstone': {}, 'little_blue_heron': {}, 'limpkin': {},
                                 'spoonbill': {}},
                 'other_oscine': {'bulbul': {}, 'jay': {}, 'magpie': {}, 'chickadee': {}, 'water_ouzel': {}},
                 'other_aquatic_bird': {'goose': {}, 'black_swan': {}, 'european_gallinule': {}, 'king_penguin': {},
                                        'albatross': {}}},
        'device': {'instrument': {'sunglasses': {}, 'cannon': {}, 'rule': {}, 'radio_telescope': {}, 'guillotine': {}},
                   'restraint': {'buckle': {}, 'padlock': {}, 'hair_slide': {}, 'safety_pin': {}, 'muzzle': {}},
                   'mechanism': {'paddlewheel': {}, 'potters_wheel': {}, 'puck': {}, 'car_wheel': {}, 'switch': {}},
                   'musical_instrument': {'harp': {}, 'sax': {}, 'trombone': {}, 'oboe': {}, 'cornet': {}},
                   'machine': {'chain_saw': {}, 'cash_machine': {}, 'abacus': {}, 'harvester': {},
                               'desktop_computer': {}}},
        'container': {'vessel': {'mortar': {}, 'ladle': {}, 'tub': {}, 'pitcher': {}, 'beaker': {}},
                      'box': {'safe': {}, 'pencil_box': {}, 'mailbox': {}, 'crate': {}, 'chest': {}},
                      'bag': {'backpack': {}, 'sleeping_bag': {}, 'mailbag': {}, 'purse': {}, 'plastic_bag': {}},
                      'self-propelled_vehicle': {'streetcar': {}, 'forklift': {}, 'tank': {}, 'tractor': {},
                                                 'recreational_vehicle': {}},
                      'other_wheeled_vehicle': {'barrow': {}, 'freight_car': {}, 'jinrikisha': {}, 'motor_scooter': {},
                                                'unicycle': {}}}}

    # trial setting 3
    # data_name_hier_dict_100 = data_name_hier_dict_100_trial2
    data_name_hier_dict_100 = data_name_hier_dict_100_trial3

    data_label_index_dict = imagenet1000_label_dict_index()

    # trial setting 4

    # trial 2
    # data_label_index_dict['other_placentals'] = -400
    # data_label_index_dict['other_carnivores'] = -401
    # data_label_index_dict['other_canines'] = -402
    # data_label_index_dict['other_dogs'] = -403
    # data_label_index_dict['other_working_dog'] = -404

    # trial 3
    data_label_index_dict['other_oscine'] = -450
    data_label_index_dict['other_aquatic_bird'] = -451
    data_label_index_dict['other_wheeled_vehicle'] = -452

    taxonomy_tree = Tree('imagenet1000', data_name_hier_dict_100, data_label_index_dict)

    used_nodes, leaf_id, node_labels = taxonomy_tree.prepro()

    def __init__(self, data_folder, train, device, is_fine_label=False):
        if device.type == 'cuda':
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "training"))
        else:
            if train is True:
                self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "train"))
            else:
                self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "val"))
        self.data, self.targets = zip(*self.base_dataset.samples)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.n_cls = 100

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        if trial_i == 1:
            return [
                68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17,
                50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14,
                71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30,
                46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
            ]
        elif trial_i == 2:
            return [
                ['carnivore', 'primate', 'ungulate', 'other_placentals'],  # placental
                ['canine', 'feline', 'other_carnivores'],  # carnivore
                ['titi', 'squirrel_monkey', 'colobus', 'guenon', 'proboscis_monkey', 'indri', 'orangutan',
                 'chimpanzee'],  # primate
                # ungulate
                ['ibex', 'gazelle', 'impala', 'hartebeest', 'bighorn', 'ram', 'wild_boar', 'sorrel', 'zebra'],
                ['grey_whale', 'killer_whale', 'sea_lion', 'fox_squirrel', 'guinea_pig', 'porcupine',
                 'three-toed_sloth', 'african_elephant'],  # other_placentals
                ['hunting_dog', 'working_dog', 'other_dogs', 'other_canines'],  # canine

                # feline
                ['egyptian_cat', 'persian_cat', 'tiger_cat', 'siamese_cat', 'cougar', 'jaguar', 'tiger', 'leopard'],
                ['badger', 'mink', 'black-footed_ferret', 'skunk', 'weasel', 'meerkat', 'mongoose', 'brown_bear',
                 'lesser_panda'],  # other carnivores
                ['sporting_dog', 'terrier', 'hound'],  # hunting_dog
                ['shepherd_dog', 'other_working_dog'],  # working_dog
                # other dogs
                ['dalmatian', 'papillon', 'pekinese', 'maltese_dog', 'toy_terrier', 'japanese_spaniel',
                 'mexican_hairless', 'miniature_poodle', 'cardigan_1', 'newfoundland', 'brabancon_griffon', 'basenji'],
                ['red_wolf', 'coyote', 'hyena', 'red_fox', 'grey_fox'],  # other canines

                ['english_setter', 'english_springer', 'welsh_springer_spaniel', 'clumber', 'vizsla'],
                ['australian_terrier', 'soft-coated_wheaten_terrier', 'dandie_dinmont', 'airedale', 'giant_schnauzer',
                 'staffordshire_bullterrier', 'lhasa', 'yorkshire_terrier', 'west_highland_white_terrier',
                 'sealyham_terrier', 'norfolk_terrier', 'cairn'],
                ['walker_hound', 'whippet', 'scottish_deerhound', 'weimaraner', 'otterhound', 'bloodhound',
                 'black-and-tan_coonhound', 'norwegian_elkhound', 'saluki', 'irish_wolfhound', 'afghan_hound'],
                ['old_english_sheepdog', 'bouvier_des_flandres', 'malinois', 'groenendael', 'rottweiler', 'komondor'],
                ['siberian_husky', 'malamute', 'great_dane', 'schipperke', 'entlebucher', 'bernese_mountain_dog',
                 'french_bulldog']
            ]

        elif trial_i == 3:
            # trial setting 5
            # BFS
            # return [
            #     ['mammal', 'bird', 'device', 'container'],  # init
            #     ['ungulate', 'rodent', 'primate', 'feline', 'canine'],  # mammal
            #     ['game_bird', 'finch', 'wading_bird', 'other_oscine', 'other_aquatic_bird'],  # bird
            #     ['instrument', 'restraint', 'mechanism', 'musical_instrument', 'machine'],  # device
            #     ['vessel', 'box', 'bag', 'self-propelled_vehicle', 'other_wheeled_vehicle'],  # container
            #     ['hippopotamus', 'ox', 'hartebeest', 'impala', 'zebra'],  # ungulate
            #     ['guinea_pig', 'marmot', 'porcupine', 'hamster', 'beaver'],  # rodent
            #     ['titi', 'capuchin', 'howler_monkey', 'patas', 'gibbon'],  # primate
            #     ['tiger_cat', 'tiger', 'persian_cat', 'cheetah', 'lion'],  # feline
            #     ['hyena', 'dhole', 'mexican_hairless', 'arctic_fox', 'timber_wolf'],  # canine
            #     ['ruffed_grouse', 'peacock', 'ptarmigan', 'partridge', 'quail'],  # game_bird
            #     ['goldfinch', 'junco', 'brambling', 'indigo_bunting', 'house_finch'],  # finch
            #     ['bustard', 'ruddy_turnstone', 'little_blue_heron', 'limpkin', 'spoonbill'],  # wading_bird
            #     ['bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel'],  # other_oscine
            #     ['goose', 'black_swan', 'european_gallinule', 'king_penguin', 'albatross'],  # other_aquatic_bird
            #     ['sunglasses', 'cannon', 'rule', 'radio_telescope', 'guillotine'],  # instrument
            #     ['buckle', 'padlock', 'hair_slide', 'safety_pin', 'muzzle'],  # restraint
            #     ['paddlewheel', 'potters_wheel', 'puck', 'car_wheel', 'switch'],  # mechanism
            #     ['harp', 'sax', 'trombone', 'oboe', 'cornet'],  # musical_instrument
            #     ['chain_saw', 'cash_machine', 'abacus', 'harvester', 'desktop_computer'],  # machine
            #     ['mortar', 'ladle', 'tub', 'pitcher', 'beaker'],  # vessel
            #     ['safe', 'pencil_box', 'mailbox', 'crate', 'chest'],  # box
            #     ['backpack', 'sleeping_bag', 'mailbag', 'purse', 'plastic_bag'],  # bag
            #     ['streetcar', 'forklift', 'tank', 'tractor', 'recreational_vehicle'],  # self-propelled_vehicle
            #     ['barrow', 'freight_car', 'jinrikisha', 'motor_scooter', 'unicycle'],  # other_wheeled_vehicle
            # ]


            # DFS
            # return [
            #     ['mammal', 'bird', 'device', 'container'],  # init
            #
            #     ['ungulate', 'rodent', 'primate', 'feline', 'canine'],  # mammal
            #
            #     ['hippopotamus', 'ox', 'hartebeest', 'impala', 'zebra'],  # ungulate
            #     ['guinea_pig', 'marmot', 'porcupine', 'hamster', 'beaver'],  # rodent
            #     ['titi', 'capuchin', 'howler_monkey', 'patas', 'gibbon'],  # primate
            #     ['tiger_cat', 'tiger', 'persian_cat', 'cheetah', 'lion'],  # feline
            #     ['hyena', 'dhole', 'mexican_hairless', 'arctic_fox', 'timber_wolf'],  # canine
            #
            #
            #     ['game_bird', 'finch', 'wading_bird', 'other_oscine', 'other_aquatic_bird'],  # bird
            #
            #     ['ruffed_grouse', 'peacock', 'ptarmigan', 'partridge', 'quail'],  # game_bird
            #     ['goldfinch', 'junco', 'brambling', 'indigo_bunting', 'house_finch'],  # finch
            #     ['bustard', 'ruddy_turnstone', 'little_blue_heron', 'limpkin', 'spoonbill'],  # wading_bird
            #     ['bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel'],  # other_oscine
            #     ['goose', 'black_swan', 'european_gallinule', 'king_penguin', 'albatross'],  # other_aquatic_bird
            #
            #
            #     ['instrument', 'restraint', 'mechanism', 'musical_instrument', 'machine'],  # device
            #
            #     ['sunglasses', 'cannon', 'rule', 'radio_telescope', 'guillotine'],  # instrument
            #     ['buckle', 'padlock', 'hair_slide', 'safety_pin', 'muzzle'],  # restraint
            #     ['paddlewheel', 'potters_wheel', 'puck', 'car_wheel', 'switch'],  # mechanism
            #     ['harp', 'sax', 'trombone', 'oboe', 'cornet'],  # musical_instrument
            #     ['chain_saw', 'cash_machine', 'abacus', 'harvester', 'desktop_computer'],  # machine
            #
            #
            #     ['vessel', 'box', 'bag', 'self-propelled_vehicle', 'other_wheeled_vehicle'],  # container
            #
            #     ['mortar', 'ladle', 'tub', 'pitcher', 'beaker'],  # vessel
            #     ['safe', 'pencil_box', 'mailbox', 'crate', 'chest'],  # box
            #     ['backpack', 'sleeping_bag', 'mailbag', 'purse', 'plastic_bag'],  # bag
            #     ['streetcar', 'forklift', 'tank', 'tractor', 'recreational_vehicle'],  # self-propelled_vehicle
            #     ['barrow', 'freight_car', 'jinrikisha', 'motor_scooter', 'unicycle'],  # other_wheeled_vehicle
            # ]

            # DER
            return [
                ['hippopotamus', 'ox', 'hartebeest', 'impala', 'zebra'],  # ungulate
                ['guinea_pig', 'marmot', 'porcupine', 'hamster', 'beaver'],  # rodent
                ['titi', 'capuchin', 'howler_monkey', 'patas', 'gibbon'],  # primate
                ['tiger_cat', 'tiger', 'persian_cat', 'cheetah', 'lion'],  # feline
                ['hyena', 'dhole', 'mexican_hairless', 'arctic_fox', 'timber_wolf'],  # canine

                ['ruffed_grouse', 'peacock', 'ptarmigan', 'partridge', 'quail'],  # game_bird
                ['goldfinch', 'junco', 'brambling', 'indigo_bunting', 'house_finch'],  # finch
                ['bustard', 'ruddy_turnstone', 'little_blue_heron', 'limpkin', 'spoonbill'],  # wading_bird
                ['bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel'],  # other_oscine
                ['goose', 'black_swan', 'european_gallinule', 'king_penguin', 'albatross'],  # other_aquatic_bird

                ['sunglasses', 'cannon', 'rule', 'radio_telescope', 'guillotine'],  # instrument
                ['buckle', 'padlock', 'hair_slide', 'safety_pin', 'muzzle'],  # restraint
                ['paddlewheel', 'potters_wheel', 'puck', 'car_wheel', 'switch'],  # mechanism
                ['harp', 'sax', 'trombone', 'oboe', 'cornet'],  # musical_instrument
                ['chain_saw', 'cash_machine', 'abacus', 'harvester', 'desktop_computer'],  # machine

                ['mortar', 'ladle', 'tub', 'pitcher', 'beaker'],  # vessel
                ['safe', 'pencil_box', 'mailbox', 'crate', 'chest'],  # box
                ['backpack', 'sleeping_bag', 'mailbag', 'purse', 'plastic_bag'],  # bag
                ['streetcar', 'forklift', 'tank', 'tractor', 'recreational_vehicle'],  # self-propelled_vehicle
                ['barrow', 'freight_car', 'jinrikisha', 'motor_scooter', 'unicycle'],  # other_wheeled_vehicle
            ]




