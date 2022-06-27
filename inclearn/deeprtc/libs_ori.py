import numpy as np
import os
import copy

from collections import defaultdict


# train scripts

# train
# with
# "./configs/2_2.yaml"
# exp.name="10scifar100_trial0_debug"
# exp.savedir="./logs/"
# exp.ckptdir="./logs/"
# exp.tensorboard_dir="./tensorboard/"
# trial=0
# --name="10scifar100_trial0_debug"
# -D
# -p
# -c
# "None"
# --force

class TreeNode():
    def __init__(self, name, label_index, depth, node_id, child_idx=-1, parent=None, codeword=None, cond=None,
                 children_unid=None, mask=None):
        self.name = name
        self.label_index = label_index
        self.depth = depth
        self.node_id = node_id
        self.children = {}
        self.child_idx = child_idx
        self.parent = parent
        self.codeword = codeword
        self.cond = cond
        self.children_unid = children_unid
        self.mask = mask

    def add_child(self, child):
        self.children[len(self.children)] = child

    def init_codeword(self, cw_size):
        self.codeword = np.zeros([cw_size])

    def set_codeword(self, idx):
        self.codeword[idx] = 1

    def set_cond(self, parent_idx):
        self.cond = [parent_idx, self.child_idx]

    def __str__(self):
        attr = 'name={}, node_id={}, depth={}, children={}'.format(
            self.name, self.node_id, self.depth,
            ','.join([chd for chd in self.children.values()])
        )
        return attr

    def copy(self):
        new_node = TreeNode(self.name, self.label_index, self.depth, self.node_id, self.child_idx, self.parent)
        new_node.children = self.children.copy()
        if self.cond:
            new_node.cond = self.cond.copy()
        return new_node


class Tree():
    def __init__(self, dataset_name, data_name_hier_dict, data_label_index_dict, dict_depth):
        self.dataset_name = dataset_name
        self.data_name_hier_dict = data_name_hier_dict
        self.data_label_index_dict = data_label_index_dict
        self.root = TreeNode('root', 'root', 0, 0)
        # self.max_depth = 0              # max depth
        self.max_depth = dict_depth + 1  # including root(0) and datasets(1)
        self.dict_depth = dict_depth
        self.nodes = {'root': self.root}
        self.depth_dict = {}
        self._buildTree(self.root, data_name_hier_dict, depth=0)
        self.used_nodes = {}        # inter nodes pair {id: Treenode}
        self.leaf_nodes = {}        # leaf nodes pair {id: Treenode}
        self.id2name = {}
        self.label2name = {}

        self.gen_codeword()
        self.gen_rel_path()
        self.gen_Id2name()

    def prepro(self, save_path=None):
        # find nodes we want, and get codewords under these nodes
        used_nodes = {}
        for n_id, name in self.used_nodes.items():
            used_nodes[n_id] = self.nodes.get(name).copy()
            used_nodes[n_id].codeword = self.get_codeword(name)
            # generate mask for internal nodes other than root node
            if n_id > 0:
                n_cw = self.nodes.get(name).codeword
                idx = n_cw.tolist().index(1)
                used_nodes[n_id].mask = 1 - n_cw
                assert used_nodes[n_id].mask[idx] == 0
                used_nodes[n_id].mask[idx] = 1
        # print('number of used nodes: {}'.format(len(used_nodes)))

        # save leaf nodes
        leaf_id = {self.nodes.get(v).label_index: k for k, v in self.leaf_nodes.items()}  # node_name: id

        # save label at each node for each class
        node_labels = defaultdict(list)
        for k in self.leaf_nodes.keys():
            for n_id in used_nodes.keys():
                chd_idx = np.where(used_nodes[n_id].codeword[:, k] == 1)[0]
                if len(chd_idx) > 0:
                    node_labels[k].append([n_id, chd_idx[0]])
                    # node_labels[self.nodes.get(self.leaf_nodes[k]).label_index].append([n_id, chd_idx[0]])

        if save_path:
            # np.save(os.path.join(save_path, 'tree.npy'), self)
            np.save(os.path.join(save_path, 'used_nodes.npy'), used_nodes)
            np.save(os.path.join(save_path, 'leaf_nodes.npy'), leaf_id)
            np.save(os.path.join(save_path, 'node_labels.npy'), node_labels)

        return used_nodes, leaf_id, node_labels

    def _buildTree(self, root, partial_data_name_hier_dir, depth=0):
        if depth == 0: # dataset_level
            child_idx = len(root.children)
            root.add_child(self.dataset_name)
            node_id = len(self.nodes)
            child = TreeNode(self.dataset_name, self.dataset_name, depth + 1, node_id, child_idx, root.name)
            self.nodes[self.dataset_name] = child

            self._buildTree(child, partial_data_name_hier_dir, depth + 1)

        elif depth == self.max_depth-1:
            for chd_label_index in partial_data_name_hier_dir:
                # chd_label_index = int(chd_label_index)
                child_idx = len(root.children)
                root.add_child(chd_label_index)
                node_id = len(self.nodes)
                child = TreeNode(chd_label_index, self.data_label_index_dict[chd_label_index], depth + 1, node_id, child_idx, root.name)
                self.nodes[chd_label_index] = child

                self._buildTree(child, [chd_label_index], depth + 1)

        elif depth != 0 and depth != self.max_depth - 1 and depth != self.max_depth:
            for chd_label_index in partial_data_name_hier_dir.keys():
                # chd_label_index = int(chd_label_index)

                child_idx = len(root.children)
                root.add_child(chd_label_index)
                node_id = len(self.nodes)
                child = TreeNode(chd_label_index, self.data_label_index_dict[chd_label_index], depth + 1, node_id, child_idx, root.name)
                self.nodes[chd_label_index] = child

                self._buildTree(child, partial_data_name_hier_dir[chd_label_index], depth + 1)

        if depth in self.depth_dict.keys():
            self.depth_dict[depth].append(root.name)
        else:
            self.depth_dict[depth] = [root.name]

    def show(self, node_name='root', root_depth=-1, max_depth=np.Inf, show_label_index=False):
        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        if root_depth == -1:
            print(root.name)
            root_depth = root.depth
            max_depth = min(self.max_depth, max_depth)

        if root.depth - root_depth < max_depth:
            for chd in root.children.values():
                child = self.nodes[chd]
                print('--' * (child.depth - root_depth), end='')
                if show_label_index:
                    print(child.label_index)
                else:
                    print(child.name)
                self.show(chd, root_depth, max_depth, show_label_index)

    def gen_codeword(self, max_depth=np.Inf):
        if max_depth == np.Inf:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if len(x.children) == 0])
        elif max_depth <= self.max_depth:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if x.depth == max_depth])
        else:
            raise ValueError('max_depth should be equal or smaller than {}'.format(self.max_depth))

        used_nodes = [x for x in self.nodes.values() if x.depth < max_depth and x.name not in leaf_nodes]
        used_nodes = sorted(used_nodes, key=lambda x: x.node_id)
        self.used_nodes = dict(enumerate([x.name for x in used_nodes]))

        self.leaf_nodes = dict(enumerate(leaf_nodes))
        node_list = [x for x in self.used_nodes.values()] + [x for x in self.leaf_nodes.values()]
        for n in node_list:
            node = self.nodes.get(n)
            node.init_codeword(len(self.leaf_nodes))

        for idx, n in self.leaf_nodes.items():
            node = self.nodes.get(n)
            node.set_codeword(idx)
            parent = self.nodes.get(node.parent)
            # reverse traversal
            while parent.name != 'root':
                parent.set_codeword(idx)
                parent = self.nodes.get(parent.parent)
            parent.set_codeword(idx)

    def gen_rel_path(self):
        name2Id = {v: k for k, v in self.used_nodes.items()}
        for idx, n in self.used_nodes.items():
            node = self.nodes.get(n)
            parent = node.parent
            if parent:
                idx = name2Id.get(parent)
                node.set_cond(idx)

    def get_codeword(self, node_name=None):
        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        codeword = []
        for i in range(len(node.children)):
            chd = node.children[i]
            child = self.nodes.get(chd)
            codeword.append(child.codeword)
        codeword = np.array(codeword)
        return codeword

    def gen_Id2name(self):
        for node_i in self.nodes.values():
            self.id2name[node_i.node_id] = node_i.name
            self.label2name[node_i.label_index] = node_i.name

    def get_nodeId(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.node_id

    def get_parent(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.parent

    def get_coarse_node_list(self):
        return [self.nodes.get(i) for i in self.used_nodes.values() if i != 'root']

    def get_finest_label(self, node):
        if node.depth == 0 or node.depth == 1:
            finest_nodes_name = list(self.leaf_nodes.values())
        elif node.depth == self.max_depth:
            finest_nodes_name = [node.name]
        else:
            finest_nodes_id = []
            children_nodes_list = [self.nodes.get(i) for i in node.children.values()]
            for children_node_i in children_nodes_list:
                finest_nodes_id += self.get_finest_label(children_node_i)[1]
            # finest_nodes_label = list(node.children.values())
            finest_nodes_name = [self.id2name[i] for i in finest_nodes_id]

        finest_nodes_label = [self.nodes.get(i).label_index for i in finest_nodes_name]
        finest_nodes_id = [self.get_nodeId(i) for i in finest_nodes_name]
        return finest_nodes_label, finest_nodes_id

    def get_children_label(self, node_name):
        node = self.nodes.get(node_name, None)
        return list(node.children.values()), [self.get_nodeId(i) for i in node.children.values()]

    def get_parent_n_layer(self, node_label_list=None, n_layer=0):

        parent_name_list = []
        parent_label_index_list = []
        for node_label_i in node_label_list:
            node_name_i = self.label2name[node_label_i]
            node_i = self.nodes.get(node_name_i, None)
            if n_layer < 0 or n_layer > self.max_depth:
                raise 'n_layer error'

            for i in range(node_i.depth - n_layer):
                node_i = self.nodes.get(node_i.parent, None)
            parent_name_list.append(node_i.name)
            parent_label_index_list.append(node_i.label_index)

        return parent_name_list, parent_label_index_list

    def gen_partial_tree(self, node_id_parent_list):
        if 0 in node_id_parent_list:
            raise 'node 0 in node_id_parent_list'

        for node_id_i in node_id_parent_list:

            if self.id2name[node_id_i] in self.depth_dict[self.max_depth]:
                raise 'node is leaf node'
        node_id_list = []
        for node_id_parent_i in node_id_parent_list:
            for children_node_j in self.nodes.get(self.id2name[node_id_parent_i]).children.values():
                node_id_list.append(self.get_nodeId(children_node_j))

        for node_id_i in node_id_parent_list:
            if node_id_i in node_id_list:
                node_id_list.remove(node_id_i)

        node_list = []
        inter_name_list = []
        full_name_list = []
        for node_id_i in node_id_list:
            name_i = self.id2name[node_id_i]
            node_i = self.nodes.get(name_i)
            node_list.append(node_i)

        for node_i in node_list:
            # inter_name_list.append(node_i.name)
            full_name_list.append(node_i.name)
            node_i_parent = node_i.parent
            while node_i_parent:
                inter_name_list.append(node_i_parent)
                node_i_parent = self.nodes.get(node_i_parent).parent

        inter_name_list = list(set(inter_name_list))
        inter_name_list.remove('root')
        inter_name_list.remove(self.dataset_name)
        full_name_list += inter_name_list
        full_name_list = list(set(full_name_list))
        full_name_list_with_depth = [(i, self.nodes.get(i).depth) for i in full_name_list]
        full_name_list_with_depth.sort(key=lambda x: x[1], reverse=True)

        def build_dict(full_name_list_with_depth, curr_depth, temp_dict):
            if curr_depth == 2:
                for node_i in full_name_list_with_depth:
                    if node_i[1] == 2:
                        if node_i[0] not in temp_dict.keys():
                            temp_dict[node_i[0]] = {}
                return temp_dict
            else:

                for node_i in full_name_list_with_depth:
                    if node_i[1] == curr_depth:
                        node_i_parent = self.nodes.get(node_i[0]).parent
                        if node_i_parent in temp_dict.keys():
                            if type(temp_dict[node_i_parent]) == list:
                                temp_dict[node_i_parent].append(node_i[0])
                            else:
                                if node_i[0] in temp_dict:
                                    temp_dict[node_i_parent][node_i[0]] = temp_dict[node_i[0]]
                                    temp_dict.pop(node_i[0])
                                else:
                                    temp_dict[node_i_parent][node_i[0]] = []
                        else:
                            if node_i[0] in temp_dict:
                                temp_dict[node_i_parent] = {node_i[0]: temp_dict[node_i[0]]}
                                temp_dict.pop(node_i[0])
                            else:
                                if curr_depth != self.max_depth:
                                    temp_dict[node_i_parent] = {node_i[0]: []}
                                else:
                                    temp_dict[node_i_parent] = [node_i[0]]

                return build_dict(full_name_list_with_depth, curr_depth - 1, temp_dict)

        partial_dict = build_dict(full_name_list_with_depth, full_name_list_with_depth[0][1] + 1, {})
        tree = Tree('cifar100', partial_dict, self.data_label_index_dict, full_name_list_with_depth[0][1] - 1)
        return tree

def write_file(file_name, data_list):
    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))


if __name__ == '__main__':
    data_name_hier_dict = {'vehicles_1': ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
     'trees': ['palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree'],
     'large_man-made_outdoor_things': ['bridge', 'road', 'skyscraper', 'house', 'castle'],
     'food_containers': ['can', 'cup', 'plate', 'bowl', 'bottle'],
     'small_mammals': ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],
     'large_omnivores_and_herbivores': ['cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant'],
     'flowers': ['rose', 'tulip', 'poppy', 'orchid', 'sunflower'],
     'large_natural_outdoor_scenes': ['forest', 'plain', 'cloud', 'mountain', 'sea'],
     'reptiles': ['turtle', 'crocodile', 'dinosaur', 'lizard', 'snake'],
     'household_furniture': ['wardrobe', 'bed', 'couch', 'chair', 'table'],
     'fruit_and_vegetables': ['apple', 'pear', 'mushroom', 'sweet_pepper', 'orange'],
     'large_carnivores': ['bear', 'leopard', 'tiger', 'wolf', 'lion'],
     'vehicles_2': ['streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket'],
     'people': ['man', 'boy', 'girl', 'baby', 'woman'],
     'insects': ['butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach'],
     'household_electrical_devices': ['lamp', 'television', 'telephone', 'keyboard', 'clock'],
     'non-insect_invertebrates': ['crab', 'snail', 'lobster', 'worm', 'spider'],
     'aquatic_mammals': ['dolphin', 'whale', 'otter', 'seal', 'beaver'],
     'fish': ['aquarium_fish', 'flatfish', 'ray', 'trout', 'shark'],
     'medium_mammals': ['raccoon', 'fox', 'porcupine', 'skunk', 'possum']}

    data_label_index_dict = {'medium_mammals': -20, 'fish': -19, 'aquatic_mammals': -18, 'non-insect_invertebrates': -17,
     'household_electrical_devices': -16, 'insects': -15, 'people': -14, 'vehicles_2': -13, 'large_carnivores': -12,
     'fruit_and_vegetables': -11, 'household_furniture': -10, 'reptiles': -9, 'large_natural_outdoor_scenes': -8,
     'flowers': -7, 'large_omnivores_and_herbivores': -6, 'small_mammals': -5, 'food_containers': -4,
     'large_man-made_outdoor_things': -3, 'trees': -2, 'vehicles_1': -1, 'apple': 0, 'aquarium_fish': 1, 'baby': 2,
     'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11,
     'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
     'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27,
     'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35,
     'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42,
     'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49,
     'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57,
     'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64,
     'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73,
     'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80,
     'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87,
     'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95,
     'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

    data_name_hier_dir_2 = {'vehicles_1': {'motorcycle':['bus', 'train', 'bicycle', 'pickup_truck'], 'small_mammals': ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],},
                          }

    tree = Tree('cifar100', data_name_hier_dict, data_label_index_dict, dict_depth = 2)
    # tree = Tree('cifar100', data_name_hier_dir_2, data_label_index_dict, dict_depth = 3)
    tree.show()
    used_nodes, leaf_id, node_labels = tree.prepro()

    print(node_labels)
    # used_nodes, leaf_id, node_labels = tree.prepro()
    test_list = [2]
    # for i in test_list:
    #     print(tree.nodes.get(tree.id2name[i]).name)
    #
    tree_2 = tree.gen_partial_tree(test_list)
    used_nodes, leaf_id, node_labels = tree_2.prepro()
    print(tree_2.leaf_nodes)
    tree_2.show()
