import os.path as osp
import numpy as np

import albumentations as A
from inclearn.deeprtc.libs import Tree
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms


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

    def __init__(self, data_folder, train, is_fine_label=False):
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
    data_name_hier_dict = {
        'vehicles_1': {'motorcycle': {}, 'bus': {}, 'train': {}, 'bicycle': {}, 'pickup_truck': {}},
        'trees': {'palm_tree': {}, 'willow_tree': {}, 'maple_tree': {}, 'oak_tree': {}, 'pine_tree': {}},
        'large_man-made_outdoor_things': {'bridge': {}, 'road': {}, 'skyscraper': {}, 'house': {}, 'castle': {}},
        'food_containers': {'can': {}, 'cup': {}, 'plate': {}, 'bowl': {}, 'bottle': {}},
        'small_mammals': {'hamster': {}, 'mouse': {}, 'shrew': {}, 'rabbit': {}, 'squirrel': {}},
        'large_omnivores_and_herbivores': {'cattle': {}, 'camel': {}, 'chimpanzee': {}, 'kangaroo': {}, 'elephant': {}},
        'flowers': {'rose': {}, 'tulip': {}, 'poppy': {}, 'orchid': {}, 'sunflower': {}},
        'large_natural_outdoor_scenes': {'forest': {}, 'plain': {}, 'cloud': {}, 'mountain': {}, 'sea': {}},
        'reptiles': {'turtle': {}, 'crocodile': {}, 'dinosaur': {}, 'lizard': {}, 'snake': {}},
        'household_furniture': {'wardrobe': {}, 'bed': {}, 'couch': {}, 'chair': {}, 'table': {}},
        'fruit_and_vegetables': {'apple': {}, 'pear': {}, 'mushroom': {}, 'sweet_pepper': {}, 'orange': {}},
        'large_carnivores': {'bear': {}, 'leopard': {}, 'tiger': {}, 'wolf': {}, 'lion': {}},
        'vehicles_2': {'streetcar': {}, 'tractor': {}, 'tank': {}, 'lawn_mower': {}, 'rocket': {}},
        'people': {'man': {}, 'boy': {}, 'girl': {}, 'baby': {}, 'woman': {}},
        'insects': {'butterfly': {}, 'bee': {}, 'beetle': {}, 'caterpillar': {}, 'cockroach': {}},
        'household_electrical_devices': {'lamp': {}, 'television': {}, 'telephone': {}, 'keyboard': {}, 'clock': {}},
        'non-insect_invertebrates': {'crab': {}, 'snail': {}, 'lobster': {}, 'worm': {}, 'spider': {}},
        'aquatic_mammals': {'dolphin': {}, 'whale': {}, 'otter': {}, 'seal': {}, 'beaver': {}},
        'fish': {'aquarium_fish': {}, 'flatfish': {}, 'ray': {}, 'trout': {}, 'shark': {}},
        'medium_mammals': {'raccoon': {}, 'fox': {}, 'porcupine': {}, 'skunk': {}, 'possum': {}}}

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

    def __init__(self, data_folder, train, is_fine_label=False):
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
            label_list = [['medium_mammals', 'fish', 'aquatic_mammals', 'non-insect_invertebrates',
                           'household_electrical_devices', 'insects', 'people', 'vehicles_2', 'large_carnivores',
                           'fruit_and_vegetables', 'household_furniture', 'reptiles', 'large_natural_outdoor_scenes',
                           'flowers', 'large_omnivores_and_herbivores', 'small_mammals', 'food_containers',
                           'large_man-made_outdoor_things', 'trees', 'vehicles_1'],
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

    data_name_hier_dict = {'physical_entity': {'matter': {'food': {'nutriment': {'course': {'plate': {}, 'dessert':
        {'frozen_dessert': {'ice_lolly': {}, 'ice_cream': {}}, 'trifle': {}}}, 'dish':
        {'pizza': {}, 'consomme': {}, 'burrito': {}, 'meat_loaf': {}, 'hot_pot': {},
         'sandwich': {'hotdog': {}, 'cheeseburger': {}}, 'potpie': {}}}, 'foodstuff':
        {'condiment': {'sauce': {'chocolate_sauce': {}, 'carbonara': {}}, 'guacamole': {}},
         'dough': {}}, 'hay': {}, 'menu': {}, 'beverage': {'espresso': {}, 'alcohol': {'red_wine': {}, 'punch':
        {'eggnog': {}, 'cup': {}}}}}, 'food_1': {'bread': {'loaf_of_bread': {'french_loaf': {}},
        'pretzel': {}, 'bagel': {}}, 'produce': {'vegetable': {'solanaceous_vegetable': {'bell_pepper': {}},
        'mashed_potato': {}, 'mushroom': {}, 'artichoke': {},
        'cruciferous_vegetable': {'cauliflower': {}, 'head_cabbage': {}, 'broccoli': {}}, 'cucumber': {},
                                'cardoon': {}, 'squash': {'summer_squash': {'spaghetti_squash': {}, 'zucchini': {}},
        'winter_squash': {'butternut_squash': {}, 'acorn_squash': {}}}}}},
        'substance': {'toilet_tissue': {}}}, 'causal_agent': {'person': {'ballplayer': {}, 'scuba_diver': {},
        'groom': {}}}, 'object': {'whole': {'animal': {'vertebrate': {'mammal': {'monotreme': {'platypus': {},
        'echidna': {}}, 'placental': {'edentate': {'armadillo': {}, 'three-toed_sloth': {}},
        'aquatic_mammal': {'dugong': {}, 'sea_lion': {}, 'whale': {'grey_whale': {}, 'killer_whale': {}}},
        'ungulate': {'even-toed_ungulate': {'bovid': {'ox': {}, 'ibex': {}, 'bison': {}, 'ram': {},
        'antelope': {'gazelle': {}, 'hartebeest': {}, 'impala': {}}, 'bighorn': {}, 'water_buffalo': {}},
        'arabian_camel': {}, 'swine': {'hog': {}, 'warthog': {}, 'wild_boar': {}}, 'llama': {}, 'hippopotamus': {}},
        'equine': {'sorrel': {}, 'zebra': {}}}, 'rodent': {'guinea_pig': {}, 'marmot': {}, 'porcupine': {},
        'hamster': {}, 'beaver': {}, 'fox_squirrel': {}}, 'carnivore': {'feline': {'cat': {'wildcat': {'cougar': {},
        'lynx': {}}, 'domestic_cat': {'siamese_cat': {}, 'persian_cat': {}, 'tabby': {}, 'egyptian_cat': {},
        'tiger_cat': {}}}, 'big_cat': {'tiger': {}, 'cheetah': {}, 'lion': {}, 'snow_leopard': {}, 'leopard': {},
                                       'jaguar': {}}},
        'canine': {'hyena': {}, 'wild_dog': {'dhole': {}, 'dingo': {}, 'african_hunting_dog': {}},
        'dog': {'mexican_hairless': {}, 'basenji': {}, 'spitz': {'chow': {}, 'keeshond': {}, 'pomeranian': {},
                                                                 'samoyed': {}},
        'pug': {}, 'newfoundland': {}, 'great_pyrenees': {}, 'brabancon_griffon': {},
                'poodle': {'toy_poodle': {}, 'standard_poodle': {}, 'miniature_poodle': {}},
                'corgi': {'pembroke': {}, 'cardigan_1': {}},
                'dalmatian': {}, 'working_dog': {'bull_mastiff': {},
                        'sled_dog': {'malamute': {}, 'siberian_husky': {}},
                        'sennenhunde': {'appenzeller': {}, 'greater_swiss_mountain_dog': {}, 'entlebucher': {},
                        'bernese_mountain_dog': {}},
                        'watchdog': {'pinscher': {'affenpinscher': {}, 'doberman': {}, 'miniature_pinscher': {}},
        'schipperke': {}, 'kuvasz': {}}, 'boxer': {}, 'tibetan_mastiff': {}, 'saint_bernard': {},
        'french_bulldog': {}, 'shepherd_dog': {'shetland_sheepdog': {}, 'kelpie': {}, 'bouvier_des_flandres': {},
        'komondor': {}, 'old_english_sheepdog': {}, 'border_collie': {}, 'collie': {}, 'briard': {},
        'german_shepherd': {}, 'rottweiler': {}, 'belgian_sheepdog': {'groenendael': {}, 'malinois': {}}},
        'eskimo_dog': {}, 'great_dane': {}}, 'leonberg': {}, 'toy_dog': {'toy_spaniel': {'papillon': {},
        'blenheim_spaniel': {}}, 'toy_terrier': {}, 'pekinese': {}, 'japanese_spaniel': {}, 'maltese_dog': {},
        'shih-tzu': {}, 'chihuahua': {}}, 'hunting_dog': {'rhodesian_ridgeback': {}, 'hound': {'bluetick': {},
        'scottish_deerhound': {}, 'ibizan_hound': {}, 'greyhound': {'whippet': {}, 'italian_greyhound': {}},
        'bloodhound': {}, 'beagle': {}, 'otterhound': {}, 'weimaraner': {}, 'redbone': {},
        'wolfhound': {'irish_wolfhound': {}, 'borzoi': {}}, 'afghan_hound': {}, 'saluki': {}, 'basset': {},
        'foxhound': {'english_foxhound': {}, 'walker_hound': {}}, 'norwegian_elkhound': {},
        'black-and-tan_coonhound': {}}, 'sporting_dog': {'spaniel': {'clumber': {}, 'sussex_spaniel': {},
        'cocker_spaniel': {}, 'brittany_spaniel': {}, 'springer_spaniel': {'english_springer': {},
        'welsh_springer_spaniel': {}}, 'irish_water_spaniel': {}}, 'setter': {'english_setter': {},
        'gordon_setter': {}, 'irish_setter': {}}, 'pointer': {'german_short-haired_pointer': {}, 'vizsla': {}},
        'retriever': {'golden_retriever': {}, 'flat-coated_retriever': {}, 'labrador_retriever': {},
        'chesapeake_bay_retriever': {}, 'curly-coated_retriever': {}}}, 'terrier': {'kerry_blue_terrier': {},
        'scotch_terrier': {}, 'tibetan_terrier': {}, 'west_highland_white_terrier': {}, 'australian_terrier': {},
        'dandie_dinmont': {}, 'lhasa': {}, 'bedlington_terrier': {}, 'airedale': {}, 'norwich_terrier': {},
        'silky_terrier': {}, 'norfolk_terrier': {}, 'yorkshire_terrier': {}, 'schnauzer': {'standard_schnauzer': {},
        'miniature_schnauzer': {}, 'giant_schnauzer': {}}, 'wirehair': {'lakeland_terrier': {},
        'sealyham_terrier': {}}, 'cairn': {}, 'boston_bull': {}, 'soft-coated_wheaten_terrier': {},
        'bullterrier': {'american_staffordshire_terrier': {}, 'staffordshire_bullterrier': {}}, 'irish_terrier': {},
        'border_terrier': {}, 'wire-haired_fox_terrier': {}}}}, 'fox': {'arctic_fox': {}, 'grey_fox': {},
        'red_fox': {}, 'kit_fox': {}}, 'wolf': {'timber_wolf': {}, 'red_wolf': {}, 'coyote': {}, 'white_wolf': {}}},
        'viverrine': {'meerkat': {}, 'mongoose': {}}, 'bear': {'american_black_bear': {}, 'brown_bear': {},
        'ice_bear': {}, 'sloth_bear': {}}, 'musteline_mammal': {'mink': {}, 'badger': {}, 'otter': {}, 'weasel': {},
        'skunk': {}, 'polecat': {}, 'black-footed_ferret': {}}, 'procyonid': {'giant_panda': {}, 'lesser_panda': {}}},
        'leporid': {'rabbit': {'wood_rabbit': {}, 'angora': {}}, 'hare': {}}, 'elephant': {'indian_elephant': {}, 'african_elephant': {}}, 'primate': {'anthropoid_ape': {'lesser_ape': {'gibbon': {}, 'siamang': {}}, 'great_ape': {'gorilla': {}, 'orangutan': {}, 'chimpanzee': {}}}, 'monkey': {'new_world_monkey': {'howler_monkey': {}, 'marmoset': {}, 'spider_monkey': {}, 'titi': {}, 'capuchin': {}, 'squirrel_monkey': {}}, 'old_world_monkey': {'proboscis_monkey': {}, 'macaque': {}, 'colobus': {}, 'langur': {}, 'baboon': {}, 'guenon': {}, 'patas': {}}}, 'lemur': {'indri': {}, 'madagascar_cat': {}}}}, 'marsupial': {'koala': {}, 'wallaby': {}, 'wombat': {}}, 'tusker': {}}, 'bird': {'oscine': {'corvine_bird': {'magpie': {}, 'jay': {}}, 'water_ouzel': {}, 'finch': {'goldfinch': {}, 'junco': {}, 'brambling': {}, 'indigo_bunting': {}, 'house_finch': {}}, 'chickadee': {}, 'thrush': {'bulbul': {}, 'robin': {}}}, 'piciform_bird': {'toucan': {}, 'jacamar': {}}, 'ostrich': {}, 'cock': {}, 'hummingbird': {}, 'game_bird': {'phasianid': {'peacock': {}, 'partridge': {}, 'quail': {}}, 'grouse': {'prairie_chicken': {}, 'ruffed_grouse': {}, 'ptarmigan': {}, 'black_grouse': {}}}, 'coraciiform_bird': {'hornbill': {}, 'bee_eater': {}}, 'aquatic_bird': {'wading_bird': {'bustard': {}, 'shorebird': {'ruddy_turnstone': {}, 'sandpiper': {'red-backed_sandpiper': {}, 'redshank': {}}, 'dowitcher': {}, 'oystercatcher': {}}, 'heron': {'little_blue_heron': {}, 'bittern': {}, 'american_egret': {}}, 'limpkin': {}, 'spoonbill': {}, 'american_coot': {}, 'stork': {'black_stork': {}, 'white_stork': {}}, 'crane_1': {}, 'flamingo': {}}, 'black_swan': {}, 'european_gallinule': {}, 'seabird': {'king_penguin': {}, 'albatross': {}, 'pelican': {}}, 'anseriform_bird': {'duck': {'red-breasted_merganser': {}, 'drake': {}}, 'goose': {}}}, 'parrot': {'sulphur-crested_cockatoo': {}, 'lorikeet': {}, 'african_grey': {}, 'macaw': {}}, 'coucal': {}, 'hen': {}, 'bird_of_prey': {'bald_eagle': {}, 'vulture': {}, 'kite': {}, 'great_grey_owl': {}}}, 'fish': {'teleost_fish': {'spiny-finned_fish': {'lionfish': {}, 'percoid_fish': {'rock_beauty': {}, 'anemone_fish': {}}, 'puffer': {}}, 'soft-finned_fish': {'eel': {}, 'cyprinid': {'goldfish': {}, 'tench': {}}}, 'ganoid': {'gar': {}, 'sturgeon': {}}}, 'elasmobranch': {'ray': {'electric_ray': {}, 'stingray': {}}, 'shark': {'great_white_shark': {}, 'hammerhead': {}, 'tiger_shark': {}}}, 'food_fish': {'barracouta': {}, 'coho': {}}}, 'reptile': {'diapsid': {'snake': {'viper': {'horned_viper': {}, 'rattlesnake': {'sidewinder': {}, 'diamondback': {}}}, 'colubrid_snake': {'thunder_snake': {}, 'ringneck_snake': {}, 'garter_snake': {}, 'water_snake': {}, 'vine_snake': {}, 'king_snake': {}, 'green_snake': {}, 'hognose_snake': {}, 'night_snake': {}}, 'boa': {'boa_constrictor': {}, 'rock_python': {}}, 'sea_snake': {}, 'elapid': {'indian_cobra': {}, 'green_mamba': {}}}, 'lizard': {'agamid': {'frilled_lizard': {}, 'agama': {}}, 'banded_gecko': {}, 'green_lizard': {}, 'komodo_dragon': {}, 'gila_monster': {}, 'whiptail': {}, 'iguanid': {'common_iguana': {}, 'american_chameleon': {}}, 'alligator_lizard': {}, 'african_chameleon': {}}, 'triceratops': {}, 'crocodilian_reptile': {'african_crocodile': {}, 'american_alligator': {}}}, 'turtle': {'sea_turtle': {'leatherback_turtle': {}, 'loggerhead': {}}, 'box_turtle': {}, 'terrapin': {}, 'mud_turtle': {}}}, 'amphibian_1': {'salamander': {'european_fire_salamander': {}, 'newt': {'eft': {}, 'common_newt': {}}, 'ambystomid': {'axolotl': {}, 'spotted_salamander': {}}}, 'frog': {'tailed_frog': {}, 'bullfrog': {}, 'tree_frog': {}}}}, 'invertebrate': {'worm': {'flatworm': {}, 'nematode': {}}, 'echinoderm': {'starfish': {}, 'sea_urchin': {}, 'sea_cucumber': {}}, 'arthropod': {'arachnid': {'tick': {}, 'spider': {'garden_spider': {}, 'barn_spider': {}, 'black_widow': {}, 'tarantula': {}, 'wolf_spider': {}, 'black_and_gold_garden_spider': {}}, 'scorpion': {}, 'harvestman': {}}, 'trilobite': {}, 'crustacean': {'decapod_crustacean': {'hermit_crab': {}, 'crab': {'king_crab': {}, 'rock_crab': {}, 'fiddler_crab': {}, 'dungeness_crab': {}}, 'lobster': {'american_lobster': {}, 'spiny_lobster': {}}, 'crayfish': {}}, 'isopod': {}}, 'insect': {'homopterous_insect': {'cicada': {}, 'leafhopper': {}}, 'butterfly': {'admiral': {}, 'lycaenid': {}, 'sulphur_butterfly': {}, 'ringlet': {}, 'cabbage_butterfly': {}, 'monarch': {}}, 'beetle': {'leaf_beetle': {}, 'tiger_beetle': {}, 'long-horned_beetle': {}, 'scarabaeid_beetle': {'rhinoceros_beetle': {}, 'dung_beetle': {}}, 'ground_beetle': {}, 'weevil': {}, 'ladybug': {}}, 'orthopterous_insect': {'cricket': {}, 'grasshopper': {}}, 'walking_stick': {}, 'fly': {}, 'hymenopterous_insect': {'ant': {}, 'bee': {}}, 'dictyopterous_insect': {'cockroach': {}, 'mantis': {}}, 'odonate': {'dragonfly': {}, 'damselfly': {}}, 'lacewing': {}}, 'centipede': {}}, 'coelenterate': {'jellyfish': {}, 'anthozoan': {'sea_anemone': {}, 'brain_coral': {}}}, 'mollusk': {'chambered_nautilus': {}, 'gastropod': {'slug': {}, 'sea_slug': {}, 'conch': {}, 'snail': {}}, 'chiton': {}}}}, 'natural_object': {'fruit': {'ear': {}, 'seed': {'corn': {}, 'buckeye': {}, 'rapeseed': {}}, 'hip': {}, 'acorn': {}, 'edible_fruit': {'custard_apple': {}, 'fig': {}, 'strawberry': {}, 'pineapple': {}, 'citrus': {'orange': {}, 'lemon': {}}, 'pomegranate': {}, 'jackfruit': {}, 'granny_smith': {}, 'banana': {}}}, 'sandbar': {}}, 'organism': {'flower': {'yellow_ladys_slipper': {}, 'daisy': {}}, 'fungus': {'agaric': {}, 'stinkhorn': {}, 'hen-of-the-woods': {}, 'coral_fungus': {}, 'earthstar': {}, 'bolete': {}, 'gyromitra': {}}}, 'artifact': {'fabric': {'piece_of_cloth': {'bib': {}, 'towel': {'paper_towel': {}}, 'handkerchief': {}, 'dishrag': {}}, 'velvet': {}, 'wool': {}}, 'pillow': {}, 'sheet': {'scoreboard': {}}, 'consumer_goods': {'clothing': {'seat_belt': {}, 'mitten': {}, 'protective_garment': {'knee_pad': {}, 'apron': {}}, 'headdress': {'helmet': {'crash_helmet': {}, 'football_helmet': {}}, 'hat': {'bearskin': {}, 'bonnet': {}, 'cowboy_hat': {}, 'sombrero': {}}, 'cap_2': {'bathing_cap': {}, 'shower_cap': {}, 'mortarboard': {}}}, 'gown': {'vestment': {}, 'academic_gown': {}}, 'military_uniform': {}, 'hosiery': {'christmas_stocking': {}, 'sock': {}, 'maillot_1': {}}, 'womans_clothing': {'gown_1': {}}, 'pajama': {}, 'attire': {'wig': {}}, 'garment': {'scarf': {'stole': {}, 'feather_boa': {}}, 'diaper': {}, 'skirt': {'hoopskirt': {}, 'sarong': {}, 'overskirt': {}, 'miniskirt': {}}, 'suit': {}, 'swimsuit': {'maillot': {}, 'bikini': {}, 'swimming_trunks': {}}, 'jean': {}, 'brassiere': {}, 'overgarment': {'poncho': {}, 'coat': {'trench_coat': {}, 'fur_coat': {}, 'lab_coat': {}}}, 'robe': {'kimono': {}, 'abaya': {}}, 'sweater': {'cardigan': {}, 'sweatshirt': {}}, 'necktie': {'bow_tie': {}, 'bolo_tie': {}, 'windsor_tie': {}}, 'jersey': {}}}, 'home_appliance': {'white_goods': {'refrigerator': {}, 'washer': {}, 'dishwasher': {}}, 'sewing_machine': {}, 'vacuum': {}, 'kitchen_appliance': {'espresso_maker': {}, 'waffle_iron': {}, 'microwave': {}, 'oven': {'rotisserie': {}, 'dutch_oven': {}}, 'toaster': {}}, 'iron': {}}}, 'rain_barrel': {}, 'plaything': {'teddy': {}}, 'structure': {'coil': {}, 'bridge': {'viaduct': {}, 'steel_arch_bridge': {}, 'suspension_bridge': {}}, 'mountain_tent': {}, 'fountain': {}, 'patio': {}, 'housing': {'mobile_home': {}, 'dwelling': {'cliff_dwelling': {}, 'yurt': {}}}, 'dock': {}, 'establishment': {'mercantile_establishment': {'shop': {'confectionery': {}, 'shoe_shop': {}, 'toyshop': {}, 'bookshop': {}, 'tobacco_shop': {}, 'bakery': {}, 'barbershop': {}, 'butcher_shop': {}}, 'grocery_store': {}}, 'prison': {}}, 'barrier': {'movable_barrier': {'sliding_door': {}, 'turnstile': {}}, 'dam': {}, 'grille': {}, 'breakwater': {}, 'fence': {'chainlink_fence': {}, 'picket_fence': {}, 'stone_wall': {}, 'worm_fence': {}}, 'bannister': {}}, 'beacon': {}, 'triumphal_arch': {}, 'altar': {}, 'castle': {}, 'lumbermill': {}, 'memorial': {'megalith': {}, 'brass': {}}, 'building': {'barn': {}, 'theater': {'cinema': {}, 'home_theater': {}}, 'place_of_worship': {'stupa': {}, 'mosque': {}, 'church': {}}, 'shed': {'apiary': {}, 'boathouse': {}}, 'restaurant': {}, 'library': {}, 'greenhouse': {}, 'residence': {'palace': {}, 'monastery': {}}, 'planetarium': {}}, 'column': {'totem_pole': {}, 'obelisk': {}}, 'supporting_structure': {'pedestal': {}, 'framework': {'honeycomb': {}, 'plate_rack': {}}}}, 'instrumentality': {'furnishing': {'furniture': {'four-poster': {}, 'baby_bed': {'bassinet': {}, 'crib': {}, 'cradle': {}}, 'table': {'desk': {}}, 'cabinet': {'medicine_chest': {}, 'china_cabinet': {}}, 'file': {}, 'entertainment_center': {}, 'dining_table': {}, 'bookcase': {}, 'table_lamp': {}, 'wardrobe': {}, 'seat': {'chair': {'folding_chair': {}, 'barber_chair': {}, 'throne': {}, 'rocking_chair': {}}, 'toilet_seat': {}, 'park_bench': {}, 'studio_couch': {}}, 'chiffonier': {}}}, 'comic_book': {}, 'system': {'maze': {}, 'communication_system': {'radio': {}, 'television': {}}}, 'implement': {'pen': {'fountain_pen': {}, 'quill': {}, 'ballpoint': {}}, 'pole': {}, 'space_bar': {}, 'cooking_utensil': {'pan': {'wok': {}, 'frying_pan': {}}, 'spatula': {}, 'crock_pot': {}, 'pot': {'caldron': {}, 'teapot': {}, 'coffeepot': {}}}, 'stick': {'drumstick': {}, 'spindle': {}, 'staff': {'crutch': {}, 'flagpole': {}}, 'matchstick': {}}, 'racket': {}, 'tool': {'hand_tool': {'shovel': {}, 'opener': {'can_opener': {}, 'corkscrew': {}}, 'plunger': {}, 'hammer': {}, 'screwdriver': {}}, 'edge_tool': {'hatchet': {}, 'knife': {'cleaver': {}, 'letter_opener': {}}, 'plane': {}}, 'plow': {}, 'power_drill': {}, 'lawn_mower': {}}, 'cleaning_implement': {'swab': {}, 'broom': {}}, 'paddle': {}, 'rubber_eraser': {}, 'pencil_sharpener': {}}, 'toiletry': {'sunscreen': {}, 'makeup': {'face_powder': {}, 'lipstick': {}}, 'hair_spray': {}, 'perfume': {}, 'lotion': {}}, 'chain': {}, 'device': {'source_of_illumination': {'torch': {}, 'lamp': {'jack-o-lantern': {}, 'spotlight': {}, 'candle': {}}}, 'instrument': {'optical_instrument': {'sunglasses': {}, 'projector': {}, 'binoculars': {}}, 'weapon': {'gun': {'cannon': {}, 'firearm': {'assault_rifle': {}, 'rifle': {}, 'revolver': {}}}, 'projectile': {}, 'bow': {}}, 'measuring_instrument': {'rule': {}, 'scale': {}, 'odometer': {}, 'barometer': {}, 'timepiece': {'timer': {'parking_meter': {}, 'stopwatch': {}}, 'sundial': {}, 'digital_watch': {}, 'hourglass': {}, 'clock': {'digital_clock': {}, 'wall_clock': {}, 'analog_clock': {}}}}, 'magnifier': {'radio_telescope': {}, 'loupe': {}}, 'guillotine': {}, 'medical_instrument': {'stethoscope': {}, 'syringe': {}}, 'magnetic_compass': {}}, 'lighter': {}, 'filter': {'oil_filter': {}, 'strainer': {}}, 'whistle': {}, 'heater': {'space_heater': {}, 'stove': {}}, 'paintbrush': {}, 'hard_disc': {}, 'remote_control': {}, 'pick': {}, 'restraint': {'disk_brake': {}, 'fastener': {'buckle': {}, 'nail': {}, 'safety_pin': {}, 'knot': {}, 'screw': {}, 'lock': {'combination_lock': {}, 'padlock': {}}, 'hair_slide': {}}, 'muzzle': {}}, 'electric_fan': {}, 'neck_brace': {}, 'electro-acoustic_transducer': {'loudspeaker': {}, 'microphone': {}}, 'ski': {}, 'sunglass': {}, 'trap': {'mousetrap': {}, 'spider_web': {}}, 'mechanism': {'radiator': {}, 'puck': {}, 'control': {'switch': {}, 'joystick': {}}, 'mechanical_device': {'hook': {}, 'gas_pump': {}, 'carousel': {}, 'reel': {}, 'wheel': {'car_wheel': {}, 'paddlewheel': {}, 'potters_wheel': {}, 'pinwheel': {}}, 'swing': {}}}, 'crane': {}, 'breathing_device': {'snorkel': {}, 'oxygen_mask': {}}, 'wing': {}, 'electronic_device': {'mouse': {}, 'screen_1': {}}, 'reflector': {'solar_dish': {}, 'car_mirror': {}}, 'support': {'pier': {}, 'tripod': {}, 'maypole': {}}, 'hand_blower': {}, 'keyboard': {'typewriter_keyboard': {}}, 'musical_instrument': {'stringed_instrument': {'harp': {}, 'banjo': {}, 'bowed_stringed_instrument': {'violin': {}, 'cello': {}}, 'guitar': {'electric_guitar': {}, 'acoustic_guitar': {}}}, 'percussion_instrument': {'steel_drum': {}, 'maraca': {}, 'marimba': {}, 'gong': {}, 'chime': {}, 'drum': {}}, 'keyboard_instrument': {'piano': {'upright': {}, 'grand_piano': {}}, 'organ': {}}, 'wind_instrument': {'panpipe': {}, 'brass_1': {'french_horn': {}, 'cornet': {}, 'trombone': {}}, 'woodwind': {'beating-reed_instrument': {'sax': {}, 'double-reed_instrument': {'bassoon': {}, 'oboe': {}}}, 'flute': {}}, 'ocarina': {}, 'free-reed_instrument': {'accordion': {}, 'harmonica': {}}}}, 'machine': {'power_tool': {'chain_saw': {}}, 'cash_machine': {}, 'abacus': {}, 'farm_machine': {'harvester': {}, 'thresher': {}}, 'computer': {'personal_computer': {'desktop_computer': {}, 'portable_computer': {'laptop': {}, 'notebook': {}, 'hand-held_computer': {}}}, 'slide_rule': {}, 'web_site': {}}, 'slot_machine': {'vending_machine': {}, 'slot': {}}}}, 'conveyance': {'stretcher': {}, 'vehicle': {'military_vehicle': {'warship': {'aircraft_carrier': {}, 'submarine': {}}, 'half_track': {}}, 'sled': {'bobsled': {}, 'dogsled': {}}, 'craft': {'vessel_1': {'boat': {'gondola': {}, 'fireboat': {}, 'lifeboat': {}, 'small_boat': {'yawl': {}, 'canoe': {}}, 'speedboat': {}}, 'ship': {'container_ship': {}, 'liner': {}, 'wreck': {}, 'pirate': {}}, 'sailing_vessel': {'sailboat': {'trimaran': {}, 'catamaran': {}}, 'schooner': {}}}, 'aircraft': {'heavier-than-air_craft': {'airliner': {}, 'warplane': {}}, 'lighter-than-air_craft': {'balloon': {}, 'airship': {}}}, 'space_shuttle': {}}, 'missile': {}}, 'public_transport': {'bus': {'minibus': {}, 'trolleybus': {}, 'school_bus': {}}, 'bullet_train': {}}}, 'equipment': {'sports_equipment': {'gymnastic_apparatus': {'horizontal_bar': {}, 'parallel_bars': {}, 'balance_beam': {}}, 'golf_equipment': {'golfcart': {}}, 'weight': {'dumbbell': {}, 'barbell': {}}}, 'electronic_equipment': {'cassette_player': {}, 'modem': {}, 'telephone': {'dial_telephone': {}, 'pay-phone': {}, 'cellular_telephone': {}}, 'tape_player': {}, 'ipod': {}, 'cd_player': {}, 'monitor': {}, 'oscilloscope': {}, 'peripheral': {'data_input_device': {'computer_keyboard': {}}, 'printer': {}}}, 'parachute': {}, 'camera': {'polaroid_camera': {}, 'reflex_camera': {}}, 'game_equipment': {'puzzle': {'jigsaw_puzzle': {}, 'crossword_puzzle': {}}, 'ball': {'tennis_ball': {}, 'volleyball': {}, 'golf_ball': {}, 'punching_bag': {}, 'ping-pong_ball': {}, 'soccer_ball': {}, 'rugby_ball': {}, 'baseball': {}, 'basketball': {}, 'croquet_ball': {}}, 'pool_table': {}}, 'photocopier': {}, 'gear': {'carpenters_kit': {}, 'drilling_platform': {}}}, 'container': {'packet': {}, 'pot_1': {}, 'vessel': {'mortar': {}, 'ladle': {}, 'tub': {}, 'pitcher': {}, 'jar': {'beaker': {}, 'vase': {}}, 'coffee_mug': {}, 'bucket': {}, 'barrel': {}, 'bathtub': {}, 'reservoir': {'water_tower': {}}, 'washbasin': {}, 'bottle': {'jug': {'whiskey_jug': {}, 'water_jug': {}}, 'water_bottle': {}, 'pill_bottle': {}, 'beer_bottle': {}, 'wine_bottle': {}, 'pop_bottle': {}}}, 'cassette': {}, 'box': {'safe': {}, 'pencil_box': {}, 'mailbox': {}, 'crate': {}, 'chest': {}, 'carton': {}}, 'shaker': {'saltshaker': {}, 'cocktail_shaker': {}}, 'tray': {}, 'soap_dispenser': {}, 'milk_can': {}, 'wooden_spoon': {}, 'envelope': {}, 'ashcan': {}, 'wheeled_vehicle': {'unicycle': {}, 'car_1': {'freight_car': {}, 'passenger_car': {}}, 'motor_scooter': {}, 'cart': {'jinrikisha': {}, 'oxcart': {}, 'horse_cart': {}}, 'handcart': {'shopping_cart': {}, 'barrow': {}}, 'bicycle': {'bicycle-built-for-two': {}, 'mountain_bike': {}}, 'tricycle': {}, 'self-propelled_vehicle': {'streetcar': {}, 'forklift': {}, 'tank': {}, 'tractor': {}, 'recreational_vehicle': {}, 'tracked_vehicle': {'snowmobile': {}}, 'locomotive': {'steam_locomotive': {}, 'electric_locomotive': {}}, 'motor_vehicle': {'amphibian': {}, 'car': {'beach_wagon': {}, 'cab': {}, 'jeep': {}, 'minivan': {}, 'ambulance': {}, 'limousine': {}, 'sports_car': {}, 'model_t': {}, 'racer': {}, 'convertible': {}}, 'go-kart': {}, 'truck': {'fire_engine': {}, 'van': {'moving_van': {}, 'police_van': {}}, 'garbage_truck': {}, 'trailer_truck': {}, 'pickup': {}, 'tow_truck': {}}, 'snowplow': {}, 'moped': {}}}}, 'glass': {'beer_glass': {}, 'goblet': {}}, 'dish_1': {'bowl': {'mixing_bowl': {}, 'soup_bowl': {}}, 'petri_dish': {}}, 'wallet': {}, 'bag': {'backpack': {}, 'sleeping_bag': {}, 'mailbag': {}, 'purse': {}, 'plastic_bag': {}}, 'piggy_bank': {}, 'basket': {'shopping_basket': {}, 'hamper': {}}, 'measuring_cup': {}}}, 'stage': {}, 'decoration': {'necklace': {}}, 'commodity': {'bath_towel': {}}, 'covering': {'protective_covering': {'armor_plate': {'breastplate': {}, 'pickelhaube': {}}, 'roof': {'vault': {}, 'dome': {}, 'tile_roof': {}, 'thatch': {}}, 'sheath': {'holster': {}, 'scabbard': {}}, 'cap_1': {'lens_cap': {}, 'thimble': {}}, 'blind': {'curtain': {'theater_curtain': {}, 'shower_curtain': {}}, 'window_shade': {}}, 'screen': {'fire_screen': {}, 'window_screen': {}, 'mosquito_net': {}}, 'lampshade': {}, 'shelter': {'birdhouse': {}, 'umbrella': {}, 'bell_cote': {}}, 'mask_1': {'gasmask': {}, 'ski_mask': {}}, 'binder': {}, 'armor': {'shield': {}, 'body_armor': {'bulletproof_vest': {}, 'chain_mail': {}, 'cuirass': {}}}}, 'mask': {}, 'shoji': {}, 'footwear': {'cowboy_boot': {}, 'clog': {}, 'shoe': {'running_shoe': {}, 'loafer': {}, 'sandal': {}}}, 'cloak': {}, 'top': {'manhole_cover': {}, 'cap': {'bottlecap': {}, 'nipple': {}}}, 'book_jacket': {}, 'floor_cover': {'doormat': {}, 'prayer_rug': {}}, 'cloth_covering': {'quilt': {}, 'band_aid': {}}}}}, 'geological_formation': {'valley': {}, 'geyser': {}, 'shore': {'lakeside': {}, 'seashore': {}}, 'cliff': {}, 'natural_elevation': {'mountain': {'alp': {}, 'volcano': {}}, 'ridge': {'coral_reef': {}}, 'promontory': {}}}}}, 'abstraction': {'bubble': {}, 'communication': {'traffic_light': {}, 'street_sign': {}}}}

    # data_name_hier_dict_100 = {
    #     'carnivore': {
    #         'canine': {
    #             'dog': {
    #                 'hunting_dog': {
    #                     'sporting_dog': {'english_setter': {}, 'english_springer': {}, 'welsh_springer_spaniel': {},
    #                                      'clumber': {}, 'vizsla': {}},
    #                     'terrier': {'australian_terrier': {}, 'soft-coated_wheaten_terrier': {}, 'dandie_dinmont': {},
    #                                 'airedale': {}, 'giant_schnauzer': {}, 'staffordshire_bullterrier': {},
    #                                 'lhasa': {}, 'yorkshire_terrier': {}, 'west_highland_white_terrier': {},
    #                                 'sealyham_terrier': {}, 'norfolk_terrier': {}, 'cairn': {}},
    #                     'hound': {'walker_hound': {}, 'whippet': {}, 'scottish_deerhound': {}, 'weimaraner': {},
    #                               'otterhound': {}, 'bloodhound': {}, 'black-and-tan_coonhound': {},
    #                               'norwegian_elkhound': {}, 'saluki': {}, 'irish_wolfhound': {}, 'afghan_hound': {}}
    #                 },
    #                 'working_dog': {
    #                     'shepherd_dog': {'old_english_sheepdog': {}, 'bouvier_des_flandres': {}, 'malinois': {},
    #                                      'groenendael': {}, 'rottweiler': {}, 'komondor': {}},
    #                     'other_working_dog': {'siberian_husky': {}, 'malamute': {}, 'great_dane': {}, 'schipperke': {},
    #                                           'entlebucher': {}, 'bernese_mountain_dog': {}, 'french_bulldog': {}}
    #                 },
    #                 'other_dogs': {'dalmatian': {}, 'papillon': {}, 'pekinese': {}, 'maltese_dog': {},
    #                                'toy_terrier': {}, 'japanese_spaniel': {}, 'mexican_hairless': {},
    #                                'miniature_poodle': {}, 'cardigan_1': {}, 'newfoundland': {},
    #                                'brabancon_griffon': {}, 'basenji': {}}
    #             },
    #             'other_canines': {'red_wolf': {}, 'coyote': {}, 'hyena': {}, 'red_fox': {}, 'grey_fox': {}}
    #         },
    #         'feline': {'egyptian_cat': {}, 'persian_cat': {}, 'tiger_cat': {}, 'siamese_cat': {}, 'cougar': {},
    #                    'jaguar': {}, 'tiger': {}, 'leopard': {}},
    #         'other_carnivores': {'badger': {}, 'mink': {}, 'black-footed_ferret': {}, 'skunk': {}, 'weasel': {},
    #                              'meerkat': {}, 'mongoose': {}, 'brown_bear': {}, 'lesser_panda': {}}
    #     },
    #     'primate': {'titi': {}, 'squirrel_monkey': {}, 'colobus': {}, 'guenon': {}, 'proboscis_monkey': {},
    #                 'indri': {}, 'orangutan': {}, 'chimpanzee': {}},
    #     'ungulate': {'ibex': {}, 'gazelle': {}, 'impala': {}, 'hartebeest': {}, 'bighorn': {}, 'ram': {},
    #                  'wild_boar': {}, 'sorrel': {}, 'zebra': {}},
    #     'other_placentals': {'grey_whale': {}, 'killer_whale': {}, 'sea_lion': {}, 'fox_squirrel': {},
    #                          'guinea_pig': {}, 'porcupine': {}, 'three-toed_sloth': {}, 'african_elephant': {}},
    # }
    data_name_hier_dict_100 = {
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

    data_label_index_dict = {'physical_entity': -1, 'abstraction': -2, 'matter': -3, 'causal_agent': -4, 'object': -5,
                             'bubble': 991, 'communication': -6, 'food': -7, 'food_1': -8, 'substance': -9,
                             'person': -10, 'whole': -11, 'geological_formation': -12, 'traffic_light': 860,
                             'street_sign': 931, 'nutriment': -13, 'foodstuff': -14, 'hay': 815, 'menu': 802,
                             'beverage': -15, 'bread': -16, 'produce': -17, 'toilet_tissue': 888, 'ballplayer': 953,
                             'scuba_diver': 981, 'groom': 847, 'animal': -18, 'natural_object': -19, 'organism': -20,
                             'artifact': -21, 'valley': 359, 'geyser': 367, 'shore': -22, 'cliff': 358,
                             'natural_elevation': -23, 'course': -24, 'dish': -25, 'condiment': -26, 'dough': 863,
                             'espresso': 946, 'alcohol': -27, 'loaf_of_bread': -28, 'pretzel': 974, 'bagel': 767,
                             'vegetable': -29, 'vertebrate': -30, 'invertebrate': -31, 'fruit': -32, 'sandbar': 363,
                             'flower': -33, 'fungus': -34, 'fabric': -35, 'pillow': 887, 'sheet': -36,
                             'consumer_goods': -37, 'rain_barrel': 926, 'plaything': -38, 'structure': -39,
                             'instrumentality': -40, 'stage': 803, 'decoration': -41, 'commodity': -42,
                             'covering': -43, 'lakeside': 365, 'seashore': 366, 'mountain': -44, 'ridge': -45,
                             'promontory': 362, 'plate': 753, 'dessert': -46, 'pizza': 947, 'consomme': 843,
                             'burrito': 899, 'meat_loaf': 805, 'hot_pot': 770, 'sandwich': -47, 'potpie': 829,
                             'sauce': -48, 'guacamole': 812, 'red_wine': 797, 'punch': -49, 'french_loaf': 872,
                             'solanaceous_vegetable': -50, 'mashed_potato': 733, 'mushroom': 745, 'artichoke': 743,
                             'cruciferous_vegetable': -51, 'cucumber': 742, 'cardoon': 744, 'squash': -52,
                             'mammal': -53, 'bird': -54, 'fish': -55, 'reptile': -56, 'amphibian_1': -57, 'worm': -58,
                             'echinoderm': -59, 'arthropod': -60, 'coelenterate': -61, 'mollusk': -62, 'ear': 328,
                             'seed': -63, 'hip': 327, 'acorn': 326, 'edible_fruit': -64, 'yellow_ladys_slipper': -65,
                             'daisy': 356, 'agaric': 912, 'stinkhorn': 892, 'hen-of-the-woods': 811,
                             'coral_fungus': 885, 'earthstar': 877, 'bolete': 980, 'gyromitra': 955,
                             'piece_of_cloth': -66, 'velvet': 968, 'wool': 814, 'scoreboard': 728, 'clothing': -67,
                             'home_appliance': -68, 'teddy': 785, 'coil': 697, 'bridge': -69, 'mountain_tent': 727,
                             'fountain': 711, 'patio': 678, 'housing': -70, 'dock': 714, 'establishment': -71,
                             'barrier': -72, 'beacon': 732, 'triumphal_arch': 677, 'altar': 676, 'castle': 700,
                             'lumbermill': 696, 'memorial': -73, 'building': -74, 'column': -75,
                             'supporting_structure': -76, 'furnishing': -77, 'comic_book': 929, 'system': -78,
                             'implement': -79, 'toiletry': -80, 'chain': 826, 'device': -81, 'conveyance': -82,
                             'equipment': -83, 'container': -84, 'necklace': 754, 'bath_towel': 908,
                             'protective_covering': -85, 'mask': 780, 'shoji': 831, 'footwear': -86, 'cloak': 796,
                             'top': -87, 'book_jacket': 773, 'floor_cover': -88, 'cloth_covering': -89, 'alp': 360,
                             'volcano': 361, 'coral_reef': 364, 'frozen_dessert': -90, 'trifle': 792, 'hotdog': 884,
                             'cheeseburger': 992, 'chocolate_sauce': 952, 'carbonara': 998, 'eggnog': 822, 'cup': 858,
                             'bell_pepper': 734, 'cauliflower': 737, 'head_cabbage': 735, 'broccoli': 736,
                             'summer_squash': -91, 'winter_squash': -92, 'monotreme': -93, 'placental': -94,
                             'marsupial': -95, 'tusker': 213, 'oscine': -96, 'piciform_bird': -97, 'ostrich': 384,
                             'cock': 382, 'hummingbird': 414, 'game_bird': -98, 'coraciiform_bird': -99,
                             'aquatic_bird': -100, 'parrot': -101, 'coucal': 411, 'hen': 383, 'bird_of_prey': -102,
                             'teleost_fish': -103, 'elasmobranch': -104, 'food_fish': -105, 'diapsid': -106,
                             'turtle': -107, 'salamander': -108, 'frog': -109, 'flatworm': 649, 'nematode': 650,
                             'starfish': 224, 'sea_urchin': 656, 'sea_cucumber': 657, 'arachnid': -110,
                             'trilobite': 600, 'crustacean': -111, 'insect': -112, 'centipede': 610, 'jellyfish': 646,
                             'anthozoan': -113, 'chambered_nautilus': 225, 'gastropod': -114, 'chiton': 655,
                             'corn': 330, 'buckeye': 331, 'rapeseed': 329, 'custard_apple': 324, 'fig': 320,
                             'strawberry': 228, 'pineapple': 321, 'citrus': -115, 'pomegranate': 325, 'jackfruit': 323,
                             'granny_smith': 317, 'banana': 322, 'bib': 940, 'towel': -116, 'handkerchief': 749,
                             'dishrag': 820, 'seat_belt': 588, 'mitten': 870, 'protective_garment': -117,
                             'headdress': -118, 'gown': 910, 'military_uniform': 865, 'hosiery': -119,
                             'womans_clothing': -120, 'pajama': 758, 'attire': -121, 'garment': -122,
                             'white_goods': -123, 'sewing_machine': 558, 'vacuum': 665, 'kitchen_appliance': -124,
                             'iron': 658, 'viaduct': 681, 'steel_arch_bridge': 679, 'suspension_bridge': 680,
                             'mobile_home': 289, 'dwelling': -125, 'mercantile_establishment': -126, 'prison': 701,
                             'movable_barrier': -127, 'dam': 719, 'grille': 724, 'breakwater': 718, 'fence': -128,
                             'bannister': 717, 'megalith': 716, 'brass': 715, 'barn': 682, 'theater': -129,
                             'place_of_worship': -130, 'shed': -131, 'restaurant': 693, 'library': 686,
                             'greenhouse': 683, 'residence': -132, 'planetarium': 692, 'totem_pole': 699,
                             'obelisk': 698, 'pedestal': 731, 'framework': -133, 'furniture': -134, 'maze': 921,
                             'communication_system': -135, 'pen': -136, 'pole': 922, 'space_bar': 857,
                             'cooking_utensil': -137, 'stick': -138, 'racket': 859, 'tool': -139,
                             'cleaning_implement': -140, 'paddle': 825, 'rubber_eraser': 996, 'pencil_sharpener': 849,
                             'sunscreen': 809, 'makeup': -141, 'hair_spray': 894, 'perfume': 882, 'lotion': 893,
                             'source_of_illumination': -142, 'instrument': -143, 'lighter': 545, 'filter': -144,
                             'whistle': 501, 'heater': -145, 'paintbrush': 503, 'hard_disc': 572,
                             'remote_control': 577, 'pick': 574, 'restraint': -146, 'electric_fan': 511,
                             'neck_brace': 594, 'electro-acoustic_transducer': -147, 'ski': 589, 'sunglass': 573,
                             'trap': -148, 'mechanism': -149, 'crane': 544, 'breathing_device': -150, 'wing': 502,
                             'electronic_device': -151, 'reflector': -152, 'support': -153, 'hand_blower': 504,
                             'keyboard': -154, 'musical_instrument': -155, 'machine': -156, 'stretcher': 956,
                             'vehicle': -157, 'public_transport': -158, 'sports_equipment': -159,
                             'electronic_equipment': -160, 'parachute': 941, 'camera': -161, 'game_equipment': -162,
                             'photocopier': 788, 'gear': -163, 'packet': 920, 'pot_1': -164, 'vessel': -165,
                             'cassette': 889, 'box': -166, 'shaker': -167, 'tray': 765, 'soap_dispenser': 959,
                             'milk_can': 874, 'wooden_spoon': 950, 'envelope': 878, 'ashcan': 751,
                             'wheeled_vehicle': -168, 'glass': -169, 'dish_1': -170, 'wallet': 927, 'bag': -171,
                             'piggy_bank': 930, 'basket': -172, 'measuring_cup': 945, 'armor_plate': -173,
                             'roof': -174, 'sheath': -175, 'cap_1': -176, 'blind': -177, 'screen': 509,
                             'lampshade': 813, 'shelter': -178, 'mask_1': -179, 'binder': 834, 'armor': -180,
                             'cowboy_boot': 909, 'clog': 978, 'shoe': -181, 'manhole_cover': 762, 'cap': -182,
                             'doormat': 971, 'prayer_rug': 768, 'quilt': 975, 'band_aid': 966, 'ice_lolly': 967,
                             'ice_cream': 973, 'spaghetti_squash': 739, 'zucchini': 738, 'butternut_squash': 741,
                             'acorn_squash': 740, 'platypus': 216, 'echidna': 214, 'edentate': -183,
                             'aquatic_mammal': -184, 'ungulate': -185, 'rodent': -186, 'carnivore': -187,
                             'leporid': -188, 'elephant': -189, 'primate': -190, 'koala': 212, 'wallaby': 215,
                             'wombat': 217, 'corvine_bird': -191, 'water_ouzel': 395, 'finch': -192, 'chickadee': 394,
                             'thrush': -193, 'toucan': 416, 'jacamar': 415, 'phasianid': -194, 'grouse': -195,
                             'hornbill': 413, 'bee_eater': 412, 'wading_bird': -196, 'black_swan': 420,
                             'european_gallinule': 437, 'seabird': -197, 'anseriform_bird': -198,
                             'sulphur-crested_cockatoo': 409, 'lorikeet': 410, 'african_grey': 407, 'macaw': 408,
                             'bald_eagle': 397, 'vulture': 398, 'kite': 396, 'great_grey_owl': 399,
                             'spiny-finned_fish': -199, 'soft-finned_fish': -200, 'ganoid': -201, 'ray': -202,
                             'shark': -203, 'barracouta': 446, 'coho': 447, 'snake': -204, 'lizard': -205,
                             'triceratops': 473, 'crocodilian_reptile': -206, 'sea_turtle': -207, 'box_turtle': 461,
                             'terrapin': 460, 'mud_turtle': 459, 'european_fire_salamander': 493, 'newt': -208,
                             'ambystomid': -209, 'tailed_frog': 500, 'bullfrog': 498, 'tree_frog': 499, 'tick': 609,
                             'spider': -210, 'scorpion': 602, 'harvestman': 601, 'decapod_crustacean': -211,
                             'isopod': 611, 'homopterous_insect': -212, 'butterfly': -213, 'beetle': -214,
                             'orthopterous_insect': -215, 'walking_stick': 632, 'fly': 628,
                             'hymenopterous_insect': -216, 'dictyopterous_insect': -217, 'odonate': -218,
                             'lacewing': 637, 'sea_anemone': 647, 'brain_coral': 648, 'slug': 653, 'sea_slug': 654,
                             'conch': 651, 'snail': 652, 'orange': 318, 'lemon': 319, 'paper_towel': 876,
                             'knee_pad': 772, 'apron': 844, 'helmet': -219, 'hat': -220, 'cap_2': -221,
                             'vestment': 789, 'academic_gown': 895, 'christmas_stocking': 800, 'sock': 985,
                             'maillot_1': -222, 'gown_1': -223, 'wig': 898, 'scarf': -224, 'diaper': 965,
                             'skirt': -225, 'suit': 793, 'swimsuit': -226, 'jean': 747, 'brassiere': 871,
                             'overgarment': -227, 'robe': -228, 'sweater': -229, 'necktie': -230, 'jersey': 960,
                             'refrigerator': 667, 'washer': 668, 'dishwasher': 666, 'espresso_maker': 659,
                             'waffle_iron': 664, 'microwave': 660, 'oven': -231, 'toaster': 663,
                             'cliff_dwelling': 712, 'yurt': 713, 'shop': -232, 'grocery_store': 702,
                             'sliding_door': 725, 'turnstile': 726, 'chainlink_fence': 720, 'picket_fence': 721, 'stone_wall': 723, 'worm_fence': 722, 'cinema': 694, 'home_theater': 695, 'stupa': 691, 'mosque': 690, 'church': 689, 'apiary': 687, 'boathouse': 688, 'palace': 684, 'monastery': 685, 'honeycomb': 729, 'plate_rack': 730, 'four-poster': 298, 'baby_bed': -233, 'table': -234, 'cabinet': -235, 'file': 304, 'entertainment_center': 315, 'dining_table': 314, 'bookcase': 299, 'table_lamp': 303, 'wardrobe': 316, 'seat': -236, 'chiffonier': 302, 'radio': 862, 'television': 943, 'fountain_pen': 933, 'quill': 861, 'ballpoint': 906, 'pan': -237, 'spatula': 675, 'crock_pot': 669, 'pot': 837, 'drumstick': 798, 'spindle': 774, 'staff': -238, 'matchstick': 983, 'hand_tool': -239, 'edge_tool': -240, 'plow': 380, 'power_drill': 372, 'lawn_mower': 373, 'swab': 827, 'broom': 850, 'face_powder': 807, 'lipstick': 866, 'torch': 593, 'lamp': -241, 'optical_instrument': -242, 'weapon': -243, 'measuring_instrument': -244, 'magnifier': -245, 'guillotine': 516, 'medical_instrument': -246, 'magnetic_compass': 531, 'oil_filter': 512, 'strainer': 513, 'space_heater': 514, 'stove': 515, 'disk_brake': 578, 'fastener': -247, 'muzzle': 587, 'loudspeaker': 507, 'microphone': 508, 'mousetrap': 598, 'spider_web': 599, 'radiator': 570, 'puck': 571, 'control': -248, 'mechanical_device': -249, 'snorkel': 506, 'oxygen_mask': 505, 'mouse': 510, 'screen_1': -250, 'solar_dish': 576, 'car_mirror': 575, 'pier': 595, 'tripod': 596, 'maypole': 597, 'typewriter_keyboard': 543, 'stringed_instrument': -251, 'percussion_instrument': -252, 'keyboard_instrument': -253, 'wind_instrument': -254, 'power_tool': -255, 'cash_machine': 547, 'abacus': 546, 'farm_machine': -256, 'computer': -257, 'slot_machine': -258, 'military_vehicle': -259, 'sled': -260, 'craft': -261, 'missile': 250, 'bus': -262, 'bullet_train': 886, 'gymnastic_apparatus': -263, 'golf_equipment': -264, 'weight': -265, 'cassette_player': 928, 'modem': 763, 'telephone': -266, 'tape_player': 977, 'ipod': 979, 'cd_player': 986, 'monitor': 868, 'oscilloscope': 869, 'peripheral': -267, 'polaroid_camera': 856, 'reflex_camera': 964, 'puzzle': -268, 'ball': -269, 'pool_table': 313, 'carpenters_kit': -270, 'drilling_platform': 833, 'mortar': 823, 'ladle': 891, 'tub': 764, 'pitcher': 982, 'jar': -271, 'coffee_mug': 995, 'bucket': 819, 'barrel': 904, 'bathtub': 883, 'reservoir': -272, 'washbasin': 905, 'bottle': -273, 'safe': 752, 'pencil_box': 841, 'mailbox': 916, 'crate': 897, 'chest': 761, 'carton': 748, 'saltshaker': 951, 'cocktail_shaker': 760, 'unicycle': 291, 'car_1': -274, 'motor_scooter': 259, 'cart': -275, 'handcart': -276, 'bicycle': -277, 'tricycle': 290, 'self-propelled_vehicle': -278, 'beer_glass': 810, 'goblet': 954, 'bowl': -279, 'petri_dish': 782, 'backpack': 846, 'sleeping_bag': 942, 'mailbag': 817, 'purse': 938, 'plastic_bag': 963, 'shopping_basket': 949, 'hamper': 839, 'breastplate': 948, 'pickelhaube': 925, 'vault': 989, 'dome': 896, 'tile_roof': 779, 'thatch': 988, 'holster': 786, 'scabbard': 808, 'lens_cap': 987, 'thimble': 757, 'curtain': -280, 'window_shade': 903, 'fire_screen': 918, 'window_screen': 911, 'mosquito_net': 851, 'birdhouse': 838, 'umbrella': 219, 'bell_cote': 932, 'gasmask': 970, 'ski_mask': 775, 'shield': 799, 'body_armor': -281, 'running_shoe': 759, 'loafer': 972, 'sandal': 750, 'bottlecap': 778, 'nipple': 914, 'armadillo': 177, 'three-toed_sloth': 37, 'dugong': 192, 'sea_lion': 13, 'whale': -282, 'even-toed_ungulate': -283, 'equine': -284, 'guinea_pig': 100, 'marmot': 182, 'porcupine': 12, 'hamster': 156, 'beaver': 194, 'fox_squirrel': 52, 'feline': -285, 'canine': -286, 'viverrine': -287, 'bear': -288, 'musteline_mammal': -289, 'procyonid': -290, 'rabbit': -291, 'hare': 128, 'indian_elephant': 193, 'african_elephant': 23, 'anthropoid_ape': -292, 'monkey': -293, 'lemur': -294, 'magpie': 393, 'jay': 392, 'goldfinch': 386, 'junco': 388, 'brambling': 385, 'indigo_bunting': 389, 'house_finch': 387, 'bulbul': 391, 'robin': 390, 'peacock': 404, 'partridge': 406, 'quail': 405, 'prairie_chicken': 403, 'ruffed_grouse': 402, 'ptarmigan': 401, 'black_grouse': 400, 'bustard': 431, 'shorebird': -295, 'heron': -296, 'limpkin': 429, 'spoonbill': 423, 'american_coot': 430, 'stork': -297, 'crane_1': -298, 'flamingo': 424, 'king_penguin': 439, 'albatross': 440, 'pelican': 438, 'duck': -299, 'goose': 419, 'lionfish': 453, 'percoid_fish': -300, 'puffer': 454, 'eel': 450, 'cyprinid': -301, 'gar': 456, 'sturgeon': 455, 'electric_ray': 444, 'stingray': 445, 'great_white_shark': 441, 'hammerhead': 443, 'tiger_shark': 442, 'viper': -302, 'colubrid_snake': -303, 'boa': -304, 'sea_snake': 489, 'elapid': -305, 'agamid': -306, 'banded_gecko': 462, 'green_lizard': 470, 'komodo_dragon': 472, 'gila_monster': 469, 'whiptail': 465, 'iguanid': -307, 'alligator_lizard': 468, 'african_chameleon': 471, 'african_crocodile': 474, 'american_alligator': 475, 'leatherback_turtle': 458, 'loggerhead': 457, 'eft': 495, 'common_newt': 494, 'axolotl': 497, 'spotted_salamander': 496, 'garden_spider': 605, 'barn_spider': 604, 'black_widow': 606, 'tarantula': 607, 'wolf_spider': 608, 'black_and_gold_garden_spider': 603, 'hermit_crab': 619, 'crab': -308, 'lobster': -309, 'crayfish': 618, 'cicada': 635, 'leafhopper': 636, 'admiral': 640, 'lycaenid': 645, 'sulphur_butterfly': 644, 'ringlet': 641, 'cabbage_butterfly': 643, 'monarch': 642, 'leaf_beetle': 624, 'tiger_beetle': 620, 'long-horned_beetle': 623, 'scarabaeid_beetle': -310, 'ground_beetle': 622, 'weevil': 627, 'ladybug': 621, 'cricket': 631, 'grasshopper': 630, 'ant': 223, 'bee': 629, 'cockroach': 633, 'mantis': 634, 'dragonfly': 638, 'damselfly': 639, 'crash_helmet': 777, 'football_helmet': 783, 'bearskin': 848, 'bonnet': 804, 'cowboy_hat': 880, 'sombrero': 924, 'bathing_cap': 784, 'shower_cap': 867, 'mortarboard': 853, 'stole': 997, 'feather_boa': 795, 'hoopskirt': 801, 'sarong': 937, 'overskirt': 936, 'miniskirt': 879, 'maillot': 976, 'bikini': 984, 'swimming_trunks': 944, 'poncho': 854, 'coat': -311, 'kimono': 769, 'abaya': 852, 'cardigan': 835, 'sweatshirt': 836, 'bow_tie': 816, 'bolo_tie': 939, 'windsor_tie': 934, 'rotisserie': 662, 'dutch_oven': 661, 'confectionery': 707, 'shoe_shop': 708, 'toyshop': 710, 'bookshop': 705, 'tobacco_shop': 709, 'bakery': 703, 'barbershop': 704, 'butcher_shop': 706, 'bassinet': 295, 'crib': 297, 'cradle': 296, 'desk': 312, 'medicine_chest': 301, 'china_cabinet': 300, 'chair': -312, 'toilet_seat': 311, 'park_bench': 305, 'studio_couch': 310, 'wok': 671, 'frying_pan': 670, 'caldron': 672, 'teapot': 674, 'coffeepot': 673, 'crutch': 855, 'flagpole': 994, 'shovel': 379, 'opener': -313, 'plunger': 377, 'hammer': 374, 'screwdriver': 378, 'hatchet': 368, 'knife': -314, 'plane': 371, 'jack-o-lantern': -315, 'spotlight': 592, 'candle': 590, 'sunglasses': 534, 'projector': 533, 'binoculars': 532, 'gun': -316, 'projectile': 541, 'bow': 537, 'rule': 518, 'scale': 520, 'odometer': 519, 'barometer': 517, 'timepiece': -317, 'radio_telescope': 536, 'loupe': 535, 'stethoscope': 529, 'syringe': 530, 'buckle': 579, 'nail': 584, 'safety_pin': 585, 'knot': 581, 'screw': 586, 'lock': -318, 'hair_slide': 580, 'switch': 560, 'joystick': 559, 'hook': 561, 'gas_pump': 566, 'carousel': 567, 'reel': 569, 'wheel': -319, 'swing': 568, 'harp': 343, 'banjo': 340, 'bowed_stringed_instrument': -320, 'guitar': -321, 'steel_drum': 339, 'maraca': 337, 'marimba': 338, 'gong': 336, 'chime': 334, 'drum': 335, 'piano': -322, 'organ': 332, 'panpipe': 351, 'brass_1': -323, 'woodwind': -324, 'ocarina': 350, 'free-reed_instrument': -325, 'chain_saw': 381, 'harvester': 553, 'thresher': 554, 'personal_computer': -326, 'slide_rule': 548, 'web_site': 552, 'vending_machine': 557, 'slot': 556, 'warship': -327, 'half_track': 248, 'bobsled': 251, 'dogsled': 252, 'vessel_1': -328, 'aircraft': -329, 'space_shuttle': 233, 'minibus': 919, 'trolleybus': 881, 'school_bus': 961, 'horizontal_bar': 923, 'parallel_bars': 993, 'balance_beam': 766, 'golfcart': 275, 'dumbbell': 999, 'barbell': 915, 'dial_telephone': 958, 'pay-phone': 842, 'cellular_telephone': 913, 'data_input_device': -330, 'printer': 555, 'jigsaw_puzzle': 962, 'crossword_puzzle': 790, 'tennis_ball': 969, 'volleyball': 935, 'golf_ball': 791, 'punching_bag': 845, 'ping-pong_ball': 840, 'soccer_ball': 221, 'rugby_ball': 875, 'baseball': 806, 'basketball': 907, 'croquet_ball': 755, 'beaker': 990, 'vase': 873, 'water_tower': 794, 'jug': -331, 'water_bottle': 957, 'pill_bottle': 900, 'beer_bottle': 776, 'wine_bottle': 830, 'pop_bottle': 787, 'freight_car': 255, 'passenger_car': 256, 'jinrikisha': 293, 'oxcart': 294, 'horse_cart': 292, 'shopping_cart': 258, 'barrow': 257, 'bicycle-built-for-two': 253, 'mountain_bike': 254, 'streetcar': 286, 'forklift': 260, 'tank': 249, 'tractor': 288, 'recreational_vehicle': 285, 'tracked_vehicle': -332, 'locomotive': -333, 'motor_vehicle': -334, 'mixing_bowl': 828, 'soup_bowl': 821, 'theater_curtain': 902, 'shower_curtain': 746, 'bulletproof_vest': 832, 'chain_mail': 901, 'cuirass': 864, 'grey_whale': 5, 'killer_whale': 21, 'bovid': -335, 'arabian_camel': 120, 'swine': -336, 'llama': 185, 'hippopotamus': 166, 'sorrel': 38, 'zebra': 79, 'cat': -337, 'big_cat': -338, 'hyena': 33, 'wild_dog': -339, 'dog': -340, 'fox': -341, 'wolf': -342, 'meerkat': 34, 'mongoose': 73, 'american_black_bear': 162, 'brown_bear': 60, 'ice_bear': 102, 'sloth_bear': 208, 'mink': 22, 'badger': 15, 'otter': 211, 'weasel': 47, 'skunk': 43, 'polecat': 181, 'black-footed_ferret': 39, 'giant_panda': 168, 'lesser_panda': 6, 'wood_rabbit': 187, 'angora': 163, 'lesser_ape': -343, 'great_ape': -344, 'new_world_monkey': -345, 'old_world_monkey': -346, 'indri': 74, 'madagascar_cat': 198, 'ruddy_turnstone': 432, 'sandpiper': -347, 'dowitcher': 435, 'oystercatcher': 436, 'little_blue_heron': 426, 'bittern': 427, 'american_egret': 425, 'black_stork': 422, 'white_stork': 421, 'red-breasted_merganser': 418, 'drake': 417, 'rock_beauty': 451, 'anemone_fish': 452, 'goldfish': 449, 'tench': 448, 'horned_viper': 490, 'rattlesnake': -348, 'thunder_snake': 476, 'ringneck_snake': 477, 'garter_snake': 481, 'water_snake': 482, 'vine_snake': 483, 'king_snake': 480, 'green_snake': 479, 'hognose_snake': 478, 'night_snake': 484, 'boa_constrictor': 485, 'rock_python': 486, 'indian_cobra': 487, 'green_mamba': 488, 'frilled_lizard': 467, 'agama': 466, 'common_iguana': 463, 'american_chameleon': 464, 'king_crab': 615, 'rock_crab': 613, 'fiddler_crab': 614, 'dungeness_crab': 612, 'american_lobster': 616, 'spiny_lobster': 617, 'rhinoceros_beetle': 626, 'dung_beetle': 625, 'trench_coat': 824, 'fur_coat': 756, 'lab_coat': 917, 'folding_chair': 308, 'barber_chair': 306, 'throne': 307, 'rocking_chair': 309, 'can_opener': 376, 'corkscrew': 375, 'cleaver': 369, 'letter_opener': 370, 'cannon': 538, 'firearm': -349, 'timer': -350, 'sundial': 525, 'digital_watch': 528, 'hourglass': 524, 'clock': -351, 'combination_lock': 582, 'padlock': 583, 'car_wheel': 562, 'paddlewheel': 563, 'potters_wheel': -352, 'pinwheel': 564, 'violin': 342, 'cello': 341, 'electric_guitar': 345, 'acoustic_guitar': 344, 'upright': 333, 'grand_piano': 226, 'french_horn': 347, 'cornet': 346, 'trombone': 348, 'beating-reed_instrument': -353, 'flute': 355, 'accordion': 222, 'harmonica': 349, 'desktop_computer': 549, 'portable_computer': -354, 'aircraft_carrier': 245, 'submarine': 246, 'boat': -355, 'ship': -356, 'sailing_vessel': -357, 'heavier-than-air_craft': -358, 'lighter-than-air_craft': -359, 'computer_keyboard': 542, 'whiskey_jug': 771, 'water_jug': 818, 'snowmobile': 287, 'steam_locomotive': 262, 'electric_locomotive': 261, 'amphibian': 263, 'car': -360, 'go-kart': 274, 'truck': -361, 'snowplow': 277, 'moped': 276, 'ox': 107, 'ibex': 8, 'bison': 164, 'ram': 80, 'antelope': -362, 'bighorn': 51, 'water_buffalo': 161, 'hog': 146, 'warthog': 119, 'wild_boar': 77, 'wildcat': -363, 'domestic_cat': -364, 'tiger': 75, 'cheetah': 205, 'lion': 189, 'snow_leopard': 152, 'leopard': 84, 'jaguar': 29, 'dhole': 135, 'dingo': 154, 'african_hunting_dog': 201, 'mexican_hairless': 45, 'basenji': 83, 'spitz': -365, 'pug': 142, 'newfoundland': 59, 'great_pyrenees': 171, 'brabancon_griffon': 69, 'poodle': -366, 'corgi': -367, 'dalmatian': 40, 'working_dog': -368, 'leonberg': 132, 'toy_dog': -369, 'hunting_dog': -370, 'arctic_fox': 158, 'grey_fox': 66, 'red_fox': 61, 'kit_fox': 0, 'timber_wolf': 204, 'red_wolf': 27, 'coyote': 57, 'white_wolf': 101, 'gibbon': 184, 'siamang': 121, 'gorilla': 103, 'orangutan': 82, 'chimpanzee': 95, 'howler_monkey': 165, 'marmoset': 174, 'spider_monkey': 110, 'titi': 36, 'capuchin': 141, 'squirrel_monkey': 91, 'proboscis_monkey': 99, 'macaque': 137, 'colobus': 53, 'langur': 202, 'baboon': 136, 'guenon': 72, 'patas': 134, 'red-backed_sandpiper': 433, 'redshank': 434, 'sidewinder': 492, 'diamondback': 491, 'assault_rifle': 539, 'rifle': 540, 'revolver': 218, 'parking_meter': 526, 'stopwatch': 527, 'digital_clock': 522, 'wall_clock': 523, 'analog_clock': 521, 'sax': 354, 'double-reed_instrument': -371, 'laptop': 227, 'notebook': 551, 'hand-held_computer': 550, 'gondola': 235, 'fireboat': 234, 'lifeboat': 237, 'small_boat': -372, 'speedboat': 236, 'container_ship': 242, 'liner': 243, 'wreck': 247, 'pirate': 244, 'sailboat': -373, 'schooner': 220, 'airliner': 229, 'warplane': 230, 'balloon': 232, 'airship': 231, 'beach_wagon': 265, 'cab': 266, 'jeep': 268, 'minivan': 270, 'ambulance': 264, 'limousine': 269, 'sports_car': 273, 'model_t': 271, 'racer': 272, 'convertible': 267, 'fire_engine': 278, 'van': -374, 'garbage_truck': 279, 'trailer_truck': 282, 'pickup': 280, 'tow_truck': 281, 'gazelle': 11, 'hartebeest': 64, 'impala': 56, 'cougar': 10, 'lynx': 200, 'siamese_cat': 94, 'persian_cat': 9, 'tabby': 173, 'egyptian_cat': 7, 'tiger_cat': 54, 'chow': 167, 'keeshond': 147, 'pomeranian': 117, 'samoyed': 178, 'toy_poodle': 105, 'standard_poodle': 150, 'miniature_poodle': 48, 'pembroke': 196, 'cardigan_1': 49, 'bull_mastiff': 139, 'sennenhunde': -376, 'watchdog': -377, 'shepherd_dog': -378, 'boxer': 129, 'tibetan_mastiff': 109, 'saint_bernard': 176, 'french_bulldog': 81, 'toy_spaniel': -379, 'eskimo_dog': 148, 'great_dane': 16, 'hound': -380, 'toy_terrier': 88, 'pekinese': 68, 'japanese_spaniel': 98, 'maltese_dog': 86, 'shih-tzu': 115, 'chihuahua': 172, 'rhodesian_ridgeback': 199, 'sporting_dog': -381, 'terrier': -382, 'pinscher': -383, 'bassoon': 352, 'oboe': 353, 'yawl': 239, 'canoe': 238, 'trimaran': 241, 'catamaran': 240, 'moving_van': 283, 'police_van': 284, 'malamute': 14, 'siberian_husky': 2, 'appenzeller': 114, 'greater_swiss_mountain_dog': 113, 'entlebucher': 78, 'bernese_mountain_dog': 85, 'belgian_sheepdog': -384, 'schipperke': 67, 'kuvasz': 140, 'shetland_sheepdog': 170, 'kelpie': 183, 'bouvier_des_flandres': 46, 'komondor': 96, 'old_english_sheepdog': 28, 'border_collie': 127, 'collie': 123, 'briard': 207, 'german_shepherd': 210, 'rottweiler': 63, 'greyhound': -385, 'papillon': 42, 'blenheim_spaniel': 197, 'bluetick': 179, 'scottish_deerhound': 20, 'ibizan_hound': 203, 'wolfhound': -386, 'bloodhound': 31, 'beagle': 131, 'otterhound': 30, 'weimaraner': 24, 'redbone': 180, 'foxhound': -387, 'afghan_hound': 97, 'saluki': 65, 'basset': 160, 'spaniel': -388, 'norwegian_elkhound': 62, 'black-and-tan_coonhound': 41, 'setter': -389, 'pointer': -390, 'retriever': -391, 'schnauzer': -392, 'kerry_blue_terrier': 106, 'scotch_terrier': 108, 'tibetan_terrier': 157, 'west_highland_white_terrier': 70, 'australian_terrier': 3, 'dandie_dinmont': 26, 'lhasa': 55, 'bedlington_terrier': 118, 'airedale': 32, 'norwich_terrier': 144, 'silky_terrier': 130, 'norfolk_terrier': 87, 'yorkshire_terrier': 58, 'wirehair': -393, 'bullterrier': -394, 'cairn': 90, 'boston_bull': 112, 'soft-coated_wheaten_terrier': 25, 'springer_spaniel': -396, 'irish_terrier': 125, 'border_terrier': 209, 'wire-haired_fox_terrier': 159, 'affenpinscher': 126, 'doberman': 111, 'miniature_pinscher': 186, 'groenendael': 92, 'malinois': 50, 'whippet': 19, 'italian_greyhound': 188, 'irish_wolfhound': 76, 'borzoi': 104, 'english_foxhound': 206, 'walker_hound': 17, 'clumber': 93, 'sussex_spaniel': 195, 'cocker_spaniel': 190, 'brittany_spaniel': 149, 'irish_water_spaniel': 116, 'english_setter': 1, 'gordon_setter': 153, 'irish_setter': 191, 'german_short-haired_pointer': 133, 'vizsla': 89, 'golden_retriever': 124, 'flat-coated_retriever': 145, 'labrador_retriever': 175, 'chesapeake_bay_retriever': 138, 'curly-coated_retriever': 143, 'standard_schnauzer': 155, 'miniature_schnauzer': 122, 'giant_schnauzer': 35, 'lakeland_terrier': 151, 'sealyham_terrier': 71, 'american_staffordshire_terrier': 169, 'staffordshire_bullterrier': 44, 'english_springer': 4, 'welsh_springer_spaniel': 18, 'sled_dog': -375}
    data_label_index_dict['other_placentals'] = -400
    data_label_index_dict['other_carnivores'] = -401
    data_label_index_dict['other_canines'] = -402
    data_label_index_dict['other_dogs'] = -403
    data_label_index_dict['other_working_dog'] = -404
    taxonomy_tree = Tree('imagenet1000', data_name_hier_dict_100, data_label_index_dict)
    # data_label_index_dict_inv = {}
    # for x in data_label_index_dict.keys():
    #     data_label_index_dict_inv[data_label_index_dict[x]] = x
    # names = []
    # for x in range(1, 101):
    #     names.append(data_label_index_dict_inv[x])
    # new_root = taxonomy_tree.root.copy()
    # new_tree = Tree('imagenet100')
    # taxonomy_tree.expand_tree(new_tree, names)

    used_nodes, leaf_id, node_labels = taxonomy_tree.prepro()

    def __init__(self, data_folder, train, is_fine_label=False):
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
                ['red_wolf', 'coyote', 'hyena', 'red_fox', 'grey_fox'], # other canines

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
