import numpy as np
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
        if "fixed_total_mem" in self.mode:
            return self.total_memory
        elif "fixed_per_cls" in self.mode:
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
    if len(features.shape) == 3:
        features = features[0, :, :]
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
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            idxes.append(ind_max)
            iter_herding += 1

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


def herding(n_classes, network, inc_dataset, shared_data_inc, memory_per_class, logger, device):
    """Herding matrix: list
    """
    logger.info("Building & updating memory.(iCaRL)")
    new_memory_dict = {}
    x_train = inc_dataset.data_inc
    y_train = inc_dataset.targets_inc

    for class_i in set(y_train):
        # if class_i != int((20-n_classes)/4)-1:
        inputs = x_train[y_train == class_i]
        if len(shared_data_inc) > len(y_train):
            share_memory = [shared_data_inc[i] for i in np.where(y_train == class_i)[0].tolist()]
        else:
            share_memory = []
            for i in np.where(y_train == class_i)[0].tolist():
                if i < len(shared_data_inc):
                    share_memory.append(shared_data_inc[i])

        bs = 128 if str(device) == 'cuda:0' else 1
        loader = inc_dataset._get_loader(inputs, [class_i] * inputs.shape[0], share_memory=share_memory,
                                         batch_size=bs, shuffle=False, mode="test")

        features, _ = extract_features(network, loader, device)  # order

        alph = select_examplars(features, memory_per_class[0])[0]
        alph_ranked = list(enumerate([i for i in alph if (memory_per_class[0] + 1 > i > 0)]))
        alph_ranked.sort(key=lambda x: x[1])
        new_memory_dict[class_i] = inputs[[i[0] for i in alph_ranked]]

    return new_memory_dict

    # data_memory = []
    # target_memory = []
    # for i in new_memory_dict:
    #     data_memory += [new_memory_dict[i]]
    #     target_memory += [i] * new_memory_dict[i].shape[0]
    # data_memory = np.concatenate(data_memory)
    # target_memory = np.array(target_memory)
    # return new_memory_dict, data_memory, target_memory