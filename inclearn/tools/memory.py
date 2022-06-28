import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F

from inclearn.tools.utils import get_class_loss
from inclearn.convnet.utils import extract_features


class MemorySize:
    def __init__(self, mode, inc_dataset, total_memory=None, fixed_memory_per_cls=None):
        self.mode = mode
        assert mode.lower() in ["uniform_fixed_per_cls", "uniform_fixed_total_mem", "dynamic_fixed_per_cls"]
        self.total_memory = total_memory
        self.fixed_memory_per_cls = fixed_memory_per_cls
        self._n_classes = 0
        self.mem_per_cls = []
        self._inc_dataset = inc_dataset

    def update_n_classes(self, n_classes):
        # self._n_classes = n_classes
        self._n_classes = n_classes

    def update_memory_per_cls_uniform(self, n_classes):
        if "fixed_per_cls" in self.mode:
            self.mem_per_cls = [self.fixed_memory_per_cls for i in range(n_classes)]
        elif "fixed_total_mem" in self.mode:
            # self.mem_per_cls = [self.total_memory // n_classes for i in range(n_classes)]
            self.mem_per_cls = [self.total_memory // n_classes for i in range(n_classes)]
        return self.mem_per_cls

    def update_memory_per_cls(self, network, n_classes, task_size):
        if "uniform" in self.mode:
            self.update_memory_per_cls_uniform(n_classes)
        else:
            if n_classes == task_size:
                self.update_memory_per_cls_uniform(n_classes)

    @property
    def memsize(self):
        if self.mode == "fixed_total_mem":
            return self.total_memory
        elif self.mode == "fixed_per_cls":
            return self.fixed_memory_per_cls * self._n_classes


def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
    EPSILON = 1e-8
    D = feat_norm.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.0

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    # mean = np.dot(D, alph_mean)
    mean /= np.linalg.norm(mean) + EPSILON

    return mean, alph


def select_examplars(features, nb_max):
    EPSILON = 1e-8
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0], ))
    idxes = []
    w_t = mu

    iter_herding, iter_herding_eff = 0, 0

    while not (np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        # tmp_t = -np.linalg.norm(w_t[:,np.newaxis]-D, axis=0)
        # tmp_t = np.linalg.norm(w_t[:,np.newaxis]-D, axis=0)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        try:
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                idxes.append(ind_max)
                iter_herding += 1
        except:
            raise('Error')

        w_t = w_t + mu - D[:, ind_max]

    return herding_matrix, idxes


def random_selection(n_classes, task_size, network, logger, inc_dataset, memory_per_class: list):
    # TODO: Move data_memroy,targets_memory into IncDataset
    logger.info("Building & updating memory.(Random Selection)")
    tmp_data_memory, tmp_targets_memory = [], []
    assert len(memory_per_class) == n_classes
    for class_idx in range(n_classes):
        if class_idx < n_classes - task_size:
            inputs, targets, loader = inc_dataset.get_custom_loader_from_memory([class_idx])
        else:
            inputs, targets, loader = inc_dataset.get_custom_loader(class_idx, mode="test")

        memory_this_cls = min(memory_per_class[class_idx], inputs.shape[0])
        idxs = np.random.choice(inputs.shape[0], memory_this_cls, replace=False)
        tmp_data_memory.append(inputs[idxs])
        tmp_targets_memory.append(targets[idxs])
    tmp_data_memory = np.concatenate(tmp_data_memory)
    tmp_targets_memory = np.concatenate(tmp_targets_memory)

    return tmp_data_memory, tmp_targets_memory


def herding(n_classes, task_size, network, herding_matrix, inc_dataset, x_train, y_train_parent_level, shared_data_inc, memory_per_class,
            logger, curr_new_y_train_label):
    """Herding matrix: list
    """
    # final_inputs = np.array([])
    # for i, data in enumerate(train_loader, start=1):
    #
    #     inputs, targets = data
    #
    #     if final_inputs.shape[0] != 0:
    #         final_inputs = np.vstack((final_inputs, inputs))
    #         final_targets = np.vstack((final_targets, targets))
    #     else:
    #         final_inputs = inputs
    #         final_targets = targets
    # final_targets = final_targets.reshape((final_targets.shape[0] * final_targets.shape[1]))
    # final_inputs = np.delete(final_inputs, np.where(final_targets == int((20 - n_classes) / 4)), 0)
    # final_targets = np.delete(final_targets, np.where(final_targets == int((20 - n_classes) / 4)), 0)

    final_inputs = np.delete(x_train, np.where(y_train_parent_level == int((20-n_classes)/4)-1), 0)
    final_targets = np.delete(y_train_parent_level, np.where(y_train_parent_level == int((20-n_classes)/4)-1), 0)
    curr_new_y_train_label = np.array(curr_new_y_train_label)
    curr_new_y_train_label = np.delete(curr_new_y_train_label, np.where(curr_new_y_train_label == int((20-n_classes)/4)-1))

    logger.info("Building & updating memory.(iCaRL)")
    tmp_data_memory, tmp_targets_memory, tmp_data_memory_ori_label = [], [], []

    # The following two blocks maps final_targets and curr_new_y_train_label into leaf_ids
    leaf_id_index_list = []
    leaf_id_keys = network.module.leaf_id.keys()
    for target_i in list(np.array(final_targets)):
        if target_i in leaf_id_keys:
            leaf_id_index_list.append(network.module.leaf_id[target_i])
        else:
            leaf_id_index_list.append(target_i)
    # leaf_id_indexes = torch.tensor(leaf_id_index_list)

    curr_new_y_train_id_indexes = []
    leaf_id_keys = network.module.leaf_id.keys()
    for target_i in list(curr_new_y_train_label):
        if target_i in leaf_id_keys:
            curr_new_y_train_id_indexes.append(network.module.leaf_id[target_i])
        else:
            curr_new_y_train_id_indexes.append(target_i)

    leaf_id_indexes = np.array(leaf_id_index_list)

    for class_idx in curr_new_y_train_id_indexes:
        inputs = final_inputs[leaf_id_indexes == class_idx]
        targets = final_targets[leaf_id_indexes == class_idx]
        if len(shared_data_inc) > len(final_targets):
            share_memory = [shared_data_inc[i] for i in np.where(leaf_id_indexes == class_idx)[0].tolist()]
        else:
            share_memory = []
            for i in np.where(leaf_id_indexes == class_idx)[0].tolist():
                if i < len(shared_data_inc):
                    share_memory.append(shared_data_inc[i])

        loader = inc_dataset._get_loader(final_inputs[leaf_id_indexes == class_idx],
                                        final_targets[leaf_id_indexes == class_idx],
                                        share_memory=share_memory,
                                        batch_size=1,
                                        shuffle=False,
                                         mode="test")
        features, _ = extract_features(network, loader)  # order



            # herding_matrix.append(select_examplars(features, memory_per_class[n_class_idx])[0])


#
# 0: -2 - -20 19*5
# 1: 1-5 -3- -20 18*5+5

        alph = select_examplars(features, memory_per_class[0])[0]

        alph = (alph > 0) * (alph < memory_per_class[0] + 1) * 1.0
        try:
            tmp_data_memory.append(inputs[np.where(alph == 1)[0]])
        except:
            raise('Error')

        tmp_targets_memory.append(targets[np.where(alph == 1)[0]])

        leaf_id_inv = {v: u for u, v in network.module.leaf_id.items()}
        new_array = np.array(
            [leaf_id_inv[i] if i in leaf_id_inv.keys() else i for i in list(targets[np.where(alph == 1)[0]])])
        tmp_data_memory_ori_label.append(new_array)

    tmp_data_memory = np.concatenate(tmp_data_memory)
    tmp_targets_memory = np.concatenate(tmp_targets_memory)
    tmp_data_memory_ori_label = np.concatenate(tmp_data_memory_ori_label)

    return tmp_data_memory, tmp_targets_memory, herding_matrix
