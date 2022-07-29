import numpy as np
import os

from collections import defaultdict


class TreeNode:
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


class Tree:
    def __init__(self, dataset_name, label_dict_hier=None, label_dict_index=None):
        if label_dict_index is None:
            label_dict_index = {}
        if label_dict_hier is None:
            label_dict_hier = {}
        self.dataset_name = dataset_name
        self.label_dict_hier = label_dict_hier
        self.label_dict_index = label_dict_index
        """For the root node, attach a child node with name of the dataset. root.depth=0, dataset.depth=1."""
        self._setup_root_nodes()
        self._buildTree(self.data_root, label_dict_hier, label_dict_index)
        self.max_depth = max(n.depth for n in self.nodes.values())  # including root(0) and datasets(1)
        self.dict_depth = self.max_depth - 1

        self.depth_dict = {}

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
                # which column of the codeword matrix has at least one non-zero element?
                # in such column, determine the row index of the element 1, which is the child index
                chd_idx = np.where(used_nodes[n_id].codeword[:, k] == 1)[0]

                # k is the index in leaf_nodes, n_id is the index in used_nodes
                # node_labels determines all intermediate nodes that a particular leaf_node belongs to
                # and the corresponding child index
                if len(chd_idx) > 0:
                    node_labels[k].append([n_id, chd_idx[0]])

        if save_path:
            np.save(os.path.join(save_path, 'used_nodes.npy'), used_nodes)
            np.save(os.path.join(save_path, 'leaf_nodes.npy'), leaf_id)
            np.save(os.path.join(save_path, 'node_labels.npy'), node_labels)

        return used_nodes, leaf_id, node_labels

    def _setup_root_nodes(self):
        self.root = TreeNode('root', 'root', 0, 0)
        self.root.add_child('data_root')
        self.data_root = TreeNode('data_root', self.dataset_name, 1, 1, child_idx=1, parent='root')
        self.nodes = {'root': self.root, 'data_root': self.data_root}

    def _buildTree(self, root, label_dict_hier, label_dict_index):
        for child_name in sorted(list(label_dict_hier.keys())):
            root.add_child(child_name)
            child = TreeNode(child_name, label_dict_index[child_name], root.depth + 1, len(self.nodes),
                             len(root.children), root.name)
            self.nodes[child_name] = child
            self._buildTree(child, label_dict_hier[child_name], label_dict_index)

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
        self.leaf_nodes = dict(enumerate(leaf_nodes))

        # used_nodes are actually intermediate nodes
        used_nodes = [x for x in self.nodes.values() if x.depth < max_depth and x.name not in leaf_nodes]
        used_nodes = sorted(used_nodes, key=lambda x: x.node_id)
        self.used_nodes = dict(enumerate([x.name for x in used_nodes]))

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
            if parent:  # if parent is not None
                idx = name2Id.get(parent)
                node.set_cond(idx)

    def get_codeword(self, node_name=None):
        # concatenate all codewords of the parent node
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

    def get_finest(self, node_name):
        node = self.nodes.get(node_name)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))
        if node.depth == 0 or node.depth == 1:
            finest_nodes_name = list(self.leaf_nodes.values())
        elif len(node.children) == 0:
            finest_nodes_name = [node.name]
        else:
            finest_nodes_name = []
            children_nodes_list = [self.nodes.get(i) for i in node.children.values()]
            for children_node_i in children_nodes_list:
                finest_nodes_name += self.get_finest(children_node_i.name)
        return finest_nodes_name

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

    def gen_partial_tree(self, name_list):
        partial_dict = partial_copy_dict(self.label_dict_hier, name_list)
        tree = Tree(self.dataset_name, partial_dict, self.label_dict_index)
        return tree


def write_file(file_name, data_list):
    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))


def partial_copy_dict(dict_full, name_list):
    dict_part = {}
    for name in dict_full.keys():
        if name in name_list:
            dict_part[name] = partial_copy_dict(dict_full[name], name_list)
    return dict_part
