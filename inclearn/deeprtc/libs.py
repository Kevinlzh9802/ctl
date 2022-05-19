
import numpy as np
import os


class TreeNode():
    def __init__(self, name, label_index, depth, node_id, child_idx=-1, parent=None, codeword=None,
                 cond=None, children_unid=None, mask=None):
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

    def get_all_info(self):
        return [self.name, self.label_index, self.depth, self.node_id, self.children, self.child_idx,
                self.parent, self.codeword, self.cond, self.children_unid, self.mask]

    def set_all_info(self, all_info):
        self.name = all_info[0]
        self.label_index = all_info[1]
        self.depth = all_info[2]
        self.node_id = all_info[3]
        self.children = all_info[4]
        self.child_idx = all_info[5]
        self.parent = all_info[6]
        self.codeword = all_info[7]
        self.cond = all_info[8]
        self.children_unid = all_info[9]
        self.mask = all_info[10]

    # def copy(self):
    #     new_node = TreeNode(self.name, self.label_index, self.depth, self.node_id, self.child_idx,
    #     self.parent, codeword=self.codeword, cond=self.cond, children_unid=self.children_unid, mask=self.mask)
    #     new_node.children = self.children.copy()
    #
    #     return new_node


# class TreeNode():
#
#     def __init__(self, name, label_index, depth, node_id, child_idx=-1, parent=None):
#         self.name = name
#         self.label_index = label_index
#         self.depth = depth
#         self.node_id = node_id
#         self.children = {}
#         self.child_idx = child_idx
#         self.parent = parent
#         self.codeword = None
#         self.cond = None
#         self.children_unid = None
#         self.mask = None
#
#     def add_child(self, child):
#         self.children[len(self.children)] = child
#
#     def init_codeword(self, cw_size):
#         self.codeword = np.zeros([cw_size])
#
#     def set_codeword(self, idx):
#         self.codeword[idx] = 1
#
#     def set_cond(self, parent_idx):
#         self.cond = [parent_idx, self.child_idx]
#
#     def __str__(self):
#         attr = 'name={}, node_id={}, depth={}, children={}'.format(
#                     self.name, self.node_id, self.depth,
#                     ','.join([chd for chd in self.children.values()])
#                 )
#         return  attr
#
#     # def copy(self):
#     #     new_node = TreeNode(self.name, self.label_index, self.depth, self.node_id, self.child_idx, self.parent)
#     #     new_node.children = self.children.copy()
#     #     if self.cond:
#     #         new_node.cond = self.cond.copy()
#     #     return new_node
#
#     def copy(self):
#         new_node = TreeNode(self.name, self.label_index, self.depth, self.node_id, -1, self.parent)
#         # new_node.children = self.children.copy()
#         # if self.cond:
#         #     new_node.cond = self.cond.copy()
#         return new_node


class Tree():
    def __init__(self, dataset_name, data_root_dir):
        self.dataset_name = dataset_name
        self.data_root_dir = data_root_dir
        self.root = TreeNode('root', data_root_dir, 0, 0)
        self.depth = 0
        self.nodes = {'root': self.root}
        self._buildTree(self.root)
        self.used_nodes = {}
        self.leaf_nodes = {}

    def _buildTree(self, root, depth=0):
        if depth == 0:
            child_idx = len(root.children)
            root.add_child(self.dataset_name)
            node_id = len(self.nodes)
            child = TreeNode(self.dataset_name, self.dataset_name, depth + 1, node_id, child_idx, root.name)
            self.nodes[self.dataset_name] = child
            self._buildTree(child, depth + 1)

        elif depth == 1:
            for chd_label_index in self.data_root_dir.keys():
                # chd_label_index = int(chd_label_index)
                child_idx = len(root.children)
                root.add_child(chd_label_index)
                node_id = len(self.nodes)
                child = TreeNode(chd_label_index, chd_label_index, depth + 1, node_id, child_idx, root.name)
                self.nodes[chd_label_index] = child
                self._buildTree(child, depth + 1)

        elif depth == 2:
            # for chd_label_index in self.data_root_dir[str(root.label_index)]:
            for chd_label_index in self.data_root_dir[root.label_index]:
                child_idx = len(root.children)
                root.add_child(chd_label_index)
                node_id = len(self.nodes)
                child = TreeNode(chd_label_index, chd_label_index, depth + 1, node_id, child_idx, root.name)
                self.nodes[chd_label_index] = child
                self._buildTree(child, depth + 1)
        elif depth == 3:
            pass
        self.depth = max(self.depth, depth)

    def show(self, node_name='root', root_depth=-1, max_depth=np.Inf):
        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        if root_depth == -1:
            print(root.name)
            root_depth = root.depth
            max_depth = min(self.depth, max_depth)

        if root.depth - root_depth < max_depth:
            for chd in root.children.values():
                child = self.nodes[chd]
                print('--' * (child.depth - root_depth), end='')
                print(child.name)
                self.show(chd, root_depth, max_depth)

    def gen_codeword(self, max_depth=np.Inf):

        if max_depth == np.Inf:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if len(x.children) == 0])
        elif max_depth <= self.depth:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if x.depth == max_depth])
        else:
            raise ValueError('max_depth should be equal or smaller than {}'.format(self.depth))

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
        nodes_label = [i for i in self.nodes]
        nodes_label.remove('root')
        nodes_label.remove('cifar100')
        # inner_nodes_label = [i for i in nodes_label if type(i) != int]
        inner_nodes_label = [i for i in nodes_label if i < 0]
        if 'root' in inner_nodes_label:
            inner_nodes_label.remove('root')

        inner_nodes_label.insert(0, 'cifar100')
        inner_nodes_list = [self.nodes.get(i, None) for i in inner_nodes_label]
        return inner_nodes_list

    def get_finest_label(self, node):
        if node.depth == 0 or node.depth == 1:
            nodes_label = [i for i in self.nodes]
            nodes_label.remove('root')
            nodes_label.remove('cifar100')
            # finest_nodes_label = [i for i in nodes_label if type(i) == int]
            finest_nodes_label = [i for i in nodes_label if i >= 0]
        elif node.depth == 2:
            finest_nodes_label = list(node.children.values())
        elif node.depth == 3:
            finest_nodes_label = [node.name]
        else:
            raise 'node depth error'
        finest_nodes_label_id = [self.get_nodeId(i) for i in finest_nodes_label]
        return finest_nodes_label, finest_nodes_label_id

    def get_children_label(self, node_name):
        node = self.nodes.get(node_name, None)
        return list(node.children.values()), [self.get_nodeId(i) for i in node.children.values()]

    def get_parent_n_layer(self, node_name_list=None, n_layer=0):
        parent_name_list = []
        parent_name_id_list = []
        for node_name_i in node_name_list:

            node_i = self.nodes.get(node_name_i, None)
            if n_layer < 0 or n_layer > 3:
                raise 'n_layer error'

            for i in range(node_i.depth - n_layer):
                node_i = self.nodes.get(node_i.parent, None)
            parent_name_list.append(node_i.name)
            parent_name_id_list.append(node_i.node_id)

        return parent_name_list, parent_name_id_list

    def gen_partial_tree(self, node_id_list):
        partial_dic = {}
        for node_id_i in node_id_list:
            node_i = list(self.nodes.values())[node_id_i]
            if node_i.depth == 1:
                node_i_children_name = list(node_i.children.values())
                for child_node_i in node_i_children_name:
                    partial_dic[child_node_i] = []

            elif node_i.depth == 2:
                node_i_children_name = list(node_i.children.values())
                partial_dic[node_i.name] = node_i_children_name
            else:
                raise 'partial_tree node depth error'
        tree = Tree('cifar100', partial_dic)

        for node_i in tree.nodes.values():
            if node_i.name != 'root' and node_i.name != 'cifar100':

                if node_i.node_id in node_id_list:
                    ori_node_i = self.nodes.get(node_i.name, None)
                    node_i.set_all_info(ori_node_i.get_all_info())
                else:
                    ori_node_i = self.nodes.get(node_i.name, None)
                    ori_node_i_all_info = ori_node_i.get_all_info()

                    ori_node_i_all_info[4] = {}
                    for ind in [7, 8, 9, 10]:
                        ori_node_i_all_info[ind] = None

                    node_i.set_all_info(ori_node_i_all_info)

        return tree


def write_file(file_name, data_list):

    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))


if __name__ == '__main__':
    hiera_dic = {'vehicles_1': ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
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
    hiera_index_dic = {'vehicles_1': [48, 13, 90, 8, 58], 'trees': [56, 96, 47, 52, 59], 'large_man-made_outdoor_things': [12, 68, 76, 37, 17], 'food_containers': [16, 28, 61, 10, 9], 'small_mammals': [36, 50, 74, 65, 80], 'large_omnivores_and_herbivores': [19, 15, 21, 38, 31], 'flowers': [70, 92, 62, 54, 82], 'large_natural_outdoor_scenes': [33, 60, 23, 49, 71], 'reptiles': [93, 27, 29, 44, 78], 'household_furniture': [94, 5, 25, 20, 84], 'fruit_and_vegetables': [0, 57, 51, 83, 53], 'large_carnivores': [3, 42, 88, 97, 43], 'vehicles_2': [81, 89, 85, 41, 69], 'people': [46, 11, 35, 2, 98], 'insects': [14, 6, 7, 18, 24], 'household_electrical_devices': [40, 87, 86, 39, 22], 'non-insect_invertebrates': [26, 77, 45, 99, 79], 'aquatic_mammals': [30, 95, 55, 72, 4], 'fish': [1, 32, 67, 91, 73], 'medium_mammals': [66, 34, 63, 75, 64]}

    # print(hiera_dic)
    # print(hiera_index_dic)

    res = Tree('cifar100', hiera_index_dic)
    # # res.show()
    node_id = 1
    node_i = list(res.nodes.values())[node_id]
    res_2 = res.gen_partial_tree([2, 8, 14, 20, 116]) # depth=2: 6k+2 [2, 116]
    res_2.show()

    node_id = 2
    node_ori_tree = list(res.nodes.values())[node_id]
    node_copied_tree = list(res_2.nodes.values())[node_id]

    print(node_ori_tree.get_all_info())
    print(node_copied_tree.get_all_info())
    #
    # parent_name_list, parent_name_id_list  = res.get_parent_n_layer(node_name_list=[1, 12, 3], n_layer=2)
