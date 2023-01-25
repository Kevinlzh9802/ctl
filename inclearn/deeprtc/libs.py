import numpy as np
import os

from collections import defaultdict, OrderedDict


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
    def __init__(self, dataset_name, label_dict_hier=None, label_dict_index=None, node_order=None):
        if label_dict_index is None:
            label_dict_index = {}
        if label_dict_hier is None:
            label_dict_hier = {}
        self.dataset_name = dataset_name
        self.label_dict_hier = label_dict_hier
        self.label_dict_index = label_dict_index
        """For the root node, attach a child node with name of the dataset. root.depth=0, dataset.depth=1."""
        self._setup_root_nodes()
        # self._buildTree(self.data_root, label_dict_hier, label_dict_index)
        self._buildTree(self.root, label_dict_hier, label_dict_index, node_order)
        self.max_depth = max(n.depth for n in self.nodes.values())  # including root(0) and datasets(1)
        self.dict_depth = self.max_depth - 1

        self.depth_dict = {}

        self.used_nodes = {}        # inter nodes pair {id: Treenode}
        self.leaf_nodes = {}        # leaf nodes pair {id: Treenode}
        self.id2name = {}
        self.label2name = {}

        if len(label_dict_hier) > 0:
            self.gen_codeword()
            self.gen_rel_path()
            self.gen_Id2name()
            self.init_depth_dict()

    def prepro(self, save_path=None):
        # find nodes we want, and get codewords under these nodes
        used_nodes = {}
        for n_id, name in self.used_nodes.items():
            used_nodes[n_id] = self.nodes.get(name).copy()
            used_nodes[n_id].codeword = self.get_codeword(name)
            # generate mask for internal nodes other than root node
            # if n_id > 0:
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
        # self.root.add_child('data_root')
        # self.data_root = TreeNode('data_root', self.dataset_name, 1, 1, child_idx=1, parent='root')
        # self.nodes = {'root': self.root, 'data_root': self.data_root}
        self.nodes = {'root': self.root}

    def _buildTree(self, root, label_dict_hier, label_dict_index, node_order=None):
        for child_name in label_dict_hier.keys():
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

    def get_ancestor_list(self, node_name=None):
        parent_list = []
        node = self.nodes.get(node_name, None)
        while node.name != 'root':
            parent_list.append(node.parent)
            node = self.nodes.get(node.parent, None)
        return parent_list

    def get_coarse_node_list(self):
        return [self.nodes.get(i) for i in self.used_nodes.values() if i != 'root']

    def get_finest(self, node_name):
        node = self.nodes.get(node_name)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))
        # if node.depth == 0 or node.depth == 1:
        if node.depth == 0:
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

    def gen_partial_tree(self, task_until_now):
        name_list = list(np.array(task_until_now).flatten())
        parent_node_order = [self.get_task_parent(x) for x in task_until_now]
        parent_node_order.pop(0)
        if len(parent_node_order) == 0:
            partial_dict = partial_copy_dict(self.label_dict_hier, name_list)
        else:
            partial_dict = partial_copy_dict(self.label_dict_hier, name_list, parent_node_order)
        tree = Tree(self.dataset_name, partial_dict, self.label_dict_index, task_until_now)
        return tree

    def expand_tree(self, existing_tree, node_names):
        if len(existing_tree.label_dict_index) == 0:
            existing_tree.label_dict_index = self.label_dict_index
        for x in node_names:
            self.connect_node(existing_tree, x)
        # existing_tree.max_depth = max(n.depth for n in existing_tree.nodes.values())
        # existing_tree.show()

    def connect_node(self, existing_tree, node_name):
        node = self.nodes.get(node_name).copy()
        node.children = {}
        node.cond = []
        node_parent = existing_tree.nodes.get(node.parent)
        if node_parent is None:
            self.connect_node(existing_tree, node.parent)
        node_parent = existing_tree.nodes.get(node.parent)
        assert node_parent is not None

        node.node_id = len(existing_tree.nodes)
        existing_tree.nodes[node_name] = node
        node.child_idx = len(node_parent.children)
        node_parent.add_child(node_name)

    def init_depth_dict(self):
        for node_i_ind in self.nodes:
            node_i = self.nodes[node_i_ind]
            if node_i.depth in self.depth_dict:
                self.depth_dict[node_i.depth].append(node_i.name)
            else:
                self.depth_dict[node_i.depth] = [node_i.name]

    def reset_params(self):
        self.label_dict_hier = self.tree_node_to_dict(self.root)
        self.max_depth = max(n.depth for n in self.nodes.values())  # including root(0) and datasets(1)
        self.dict_depth = self.max_depth - 1
        self.depth_dict = {}

        self.used_nodes = {}
        self.leaf_nodes = {}
        self.id2name = {}
        self.label2name = {}

        if len(self.label_dict_hier) > 0:
            self.gen_codeword()
            self.gen_rel_path()
            self.gen_Id2name()
            self.init_depth_dict()


    def tree_node_to_dict(self, node):
        child_dict = OrderedDict()
        for x in node.children.values():
            child_node = self.nodes.get(x)
            child_dict[x] = self.tree_node_to_dict(child_node)
        return child_dict

    # def flatten_children(self, node_name):
    #     all_children = self.get_finest(node_name)
    #     leaf_nodes = [self.nodes.get(x) for x in all_children]
    #     for x in leaf_nodes:
    #         self.merge
    #     return

    def get_task_parent(self, name_list):
        return [self.nodes.get(x).parent for x in name_list][0]


def write_file(file_name, data_list):
    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))


def partial_copy_dict(dict_full, name_list, key_order=None):
    dict_part = OrderedDict()
    if key_order is None:
        for name in dict_full.keys():
            if name in name_list:
                dict_part[name] = partial_copy_dict(dict_full[name], name_list)
    else:
        for x in key_order:
            dict_part[x] = dict_full[x]
        for x in dict_full.keys():
            if x not in dict_part.keys():
                dict_part[x] = {}
    return dict_part


if __name__ == '__main__':
    def imagenet1000_label_dict_index_trial3():
        return {'physical_entity': -1, 'abstraction': -2, 'matter': -3, 'causal_agent': -4, 'object': -5, 'bubble': 971,
                'communication': -6, 'food': -7, 'food_1': -8, 'substance': -9, 'person': -10, 'whole': -11,
                'geological_formation': -12, 'traffic_light': 920, 'street_sign': 919, 'nutriment': -13,
                'foodstuff': -14,
                'hay': 958, 'menu': 922, 'beverage': -15, 'bread': -16, 'produce': -17, 'toilet_tissue': 999,
                'ballplayer': 981, 'scuba_diver': 983, 'groom': 982, 'animal': -18, 'natural_object': -19,
                'organism': -20,
                'artifact': -21, 'valley': 979, 'geyser': 974, 'shore': -22, 'cliff': 972, 'natural_elevation': -23,
                'course': -24, 'dish': -25, 'condiment': -26, 'dough': 961, 'espresso': 967, 'alcohol': -27,
                'loaf_of_bread': -28, 'pretzel': 932, 'bagel': 931, 'vegetable': -29, 'vertebrate': -30,
                'invertebrate': -31, 'fruit': -32, 'sandbar': 977, 'flower': -33, 'fungus': -34, 'fabric': -35,
                'pillow': 721, 'sheet': -36, 'consumer_goods': -37, 'rain_barrel': 756, 'plaything': -38,
                'structure': -39,
                'instrumentality': -40, 'stage': 819, 'decoration': -41, 'commodity': -42, 'covering': -43,
                'lakeside': 975,
                'seashore': 978, 'mountain': -44, 'ridge': -45, 'promontory': 976, 'plate': 923, 'dessert': -46,
                'pizza': 963, 'consomme': 925, 'burrito': 965, 'meat_loaf': 962, 'hot_pot': 926, 'sandwich': -47,
                'potpie': 964, 'sauce': -48, 'guacamole': 924, 'red_wine': 966, 'punch': -49, 'french_loaf': 930,
                'solanaceous_vegetable': -50, 'mashed_potato': 935, 'mushroom': 947, 'artichoke': 944,
                'cruciferous_vegetable': -51, 'cucumber': 943, 'cardoon': 946, 'squash': -52, 'mammal': -53,
                'bird': -54,
                'fish': -55, 'reptile': -56, 'amphibian_1': -57, 'worm': -58, 'echinoderm': -59, 'arthropod': -60,
                'coelenterate': -61, 'mollusk': -62, 'ear': 998, 'seed': -63, 'hip': 989, 'acorn': 988,
                'edible_fruit': -64,
                'yellow_ladys_slipper': 986, 'daisy': 985, 'agaric': 992, 'stinkhorn': 994, 'hen-of-the-woods': 996,
                'coral_fungus': 991, 'earthstar': 995, 'bolete': 997, 'gyromitra': 993, 'piece_of_cloth': -66,
                'velvet': 885, 'wool': 911, 'scoreboard': 781, 'clothing': -67, 'home_appliance': -68, 'teddy': 850,
                'coil': 506, 'bridge': -69, 'mountain_tent': 672, 'fountain': 562, 'patio': 706, 'housing': -70,
                'dock': 536, 'establishment': -71, 'barrier': -72, 'beacon': 437, 'triumphal_arch': 873, 'altar': 406,
                'castle': 483, 'lumbermill': 634, 'memorial': -73, 'building': -74, 'column': -75,
                'supporting_structure': -76, 'furnishing': -77, 'comic_book': 917, 'system': -78, 'implement': -79,
                'toiletry': -80, 'chain': 488, 'device': -81, 'conveyance': -82, 'equipment': -83, 'container': -84,
                'necklace': 679, 'bath_towel': 434, 'protective_covering': -85, 'mask': 643, 'shoji': 789,
                'footwear': -86,
                'cloak': 501, 'top': -87, 'book_jacket': 921, 'floor_cover': -88, 'cloth_covering': -89, 'alp': 970,
                'volcano': 980, 'coral_reef': 973, 'frozen_dessert': -90, 'trifle': 927, 'hotdog': 934,
                'cheeseburger': 933,
                'chocolate_sauce': 960, 'carbonara': 959, 'eggnog': 969, 'cup': 968, 'bell_pepper': 945,
                'cauliflower': 938,
                'head_cabbage': 936, 'broccoli': 937, 'summer_squash': -91, 'winter_squash': -92, 'monotreme': -93,
                'placental': -94, 'marsupial': -95, 'tusker': 101, 'oscine': -96, 'piciform_bird': -97, 'ostrich': 9,
                'cock': 7, 'hummingbird': 94, 'game_bird': -98, 'coraciiform_bird': -99, 'aquatic_bird': -100,
                'parrot': -101, 'coucal': 91, 'hen': 8, 'bird_of_prey': -102, 'teleost_fish': -103,
                'elasmobranch': -104,
                'food_fish': -105, 'diapsid': -106, 'turtle': -107, 'salamander': -108, 'frog': -109, 'flatworm': 110,
                'nematode': 111, 'starfish': 327, 'sea_urchin': 328, 'sea_cucumber': 329, 'arachnid': -110,
                'trilobite': 69,
                'crustacean': -111, 'insect': -112, 'centipede': 79, 'jellyfish': 107, 'anthozoan': -113,
                'chambered_nautilus': 117, 'gastropod': -114, 'chiton': 116, 'corn': 987, 'buckeye': 990,
                'rapeseed': 984,
                'custard_apple': 956, 'fig': 952, 'strawberry': 949, 'pineapple': 953, 'citrus': -115,
                'pomegranate': 957,
                'jackfruit': 955, 'granny_smith': 948, 'banana': 954, 'bib': 443, 'towel': -116, 'handkerchief': 591,
                'dishrag': 533, 'seat_belt': 785, 'mitten': 658, 'protective_garment': -117, 'headdress': -118,
                'gown': 910,
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
                'heater': -145, 'paintbrush': 696, 'hard_disc': 592, 'remote_control': 761, 'pick': 714,
                'restraint': -146,
                'electric_fan': 545, 'neck_brace': 678, 'electro-acoustic_transducer': -147, 'ski': 795,
                'sunglass': 836,
                'trap': -148, 'mechanism': -149, 'crane': 517, 'breathing_device': -150, 'wing': 908,
                'electronic_device': -151, 'reflector': -152, 'support': -153, 'hand_blower': 589, 'keyboard': -154,
                'musical_instrument': -155, 'machine': -156, 'stretcher': 830, 'vehicle': -157,
                'public_transport': -158,
                'sports_equipment': -159, 'electronic_equipment': -160, 'parachute': 701, 'camera': -161,
                'game_equipment': -162, 'photocopier': 713, 'gear': -163, 'packet': 692, 'pot_1': 738, 'vessel': -165,
                'cassette': 481, 'box': -166, 'shaker': -167, 'tray': 868, 'soap_dispenser': 804, 'milk_can': 653,
                'wooden_spoon': 910, 'envelope': 549, 'ashcan': 412, 'wheeled_vehicle': -168, 'glass': -169,
                'dish_1': -170,
                'wallet': 893, 'bag': -171, 'piggy_bank': 719, 'basket': -172, 'measuring_cup': 647,
                'armor_plate': -173,
                'roof': -174, 'sheath': -175, 'cap_1': -176, 'blind': -177, 'screen': 509, 'lampshade': 619,
                'shelter': -178, 'mask_1': -179, 'binder': 446, 'armor': -180, 'cowboy_boot': 514, 'clog': 502,
                'shoe': -181, 'manhole_cover': 640, 'cap': -182, 'doormat': 539, 'prayer_rug': 741, 'quilt': 750,
                'band_aid': 419, 'ice_lolly': 929, 'ice_cream': 928, 'spaghetti_squash': 940, 'zucchini': 939,
                'butternut_squash': 942, 'acorn_squash': 941, 'platypus': 103, 'echidna': 102, 'edentate': -183,
                'aquatic_mammal': -184, 'ungulate': -185, 'rodent': -186, 'carnivore': -187, 'leporid': -188,
                'elephant': -189, 'primate': -190, 'koala': 105, 'wallaby': 104, 'wombat': 106, 'corvine_bird': -191,
                'water_ouzel': 20, 'finch': -192, 'chickadee': 19, 'thrush': -193, 'toucan': 96, 'jacamar': 95,
                'phasianid': -194, 'grouse': -195, 'hornbill': 93, 'bee_eater': 92, 'wading_bird': -196,
                'black_swan': 100,
                'european_gallinule': 136, 'seabird': -197, 'anseriform_bird': -198, 'sulphur-crested_cockatoo': 89,
                'lorikeet': 90, 'african_grey': 87, 'macaw': 88, 'bald_eagle': 22, 'vulture': 23, 'kite': 21,
                'great_grey_owl': 24, 'spiny-finned_fish': -199, 'soft-finned_fish': -200, 'ganoid': -201, 'ray': -202,
                'shark': -203, 'barracouta': 389, 'coho': 391, 'snake': -204, 'lizard': -205, 'triceratops': 51,
                'crocodilian_reptile': -206, 'sea_turtle': -207, 'box_turtle': 37, 'terrapin': 36, 'mud_turtle': 35,
                'european_fire_salamander': 25, 'newt': -208, 'ambystomid': -209, 'tailed_frog': 32, 'bullfrog': 30,
                'tree_frog': 31, 'tick': 78, 'spider': -210, 'scorpion': 71, 'harvestman': 70,
                'decapod_crustacean': -211,
                'isopod': 126, 'homopterous_insect': -212, 'butterfly': -213, 'beetle': -214,
                'orthopterous_insect': -215,
                'walking_stick': 313, 'fly': 308, 'hymenopterous_insect': -216, 'dictyopterous_insect': -217,
                'odonate': -218, 'lacewing': 318, 'sea_anemone': 108, 'brain_coral': 109, 'slug': 114, 'sea_slug': 115,
                'conch': 112, 'snail': 113, 'orange': 950, 'lemon': 951, 'paper_towel': 700, 'knee_pad': 615,
                'apron': 411,
                'helmet': -219, 'hat': -220, 'cap_2': -221, 'vestment': 887, 'academic_gown': 400,
                'christmas_stocking': 496, 'sock': 806, 'maillot_1': 639, 'gown_1': 578, 'wig': 903, 'scarf': -224,
                'diaper': 529, 'skirt': -225, 'suit': 834, 'swimsuit': -226, 'jean': 608, 'brassiere': 459,
                'overgarment': -227, 'robe': -228, 'sweater': -229, 'necktie': -230, 'jersey': 610, 'refrigerator': 760,
                'washer': 897, 'dishwasher': 534, 'espresso_maker': 550, 'waffle_iron': 891, 'microwave': 651,
                'oven': -231,
                'toaster': 859, 'cliff_dwelling': 500, 'yurt': 915, 'shop': -232, 'grocery_store': 582,
                'sliding_door': 799,
                'turnstile': 877, 'chainlink_fence': 489, 'picket_fence': 716, 'stone_wall': 825, 'worm_fence': 912,
                'cinema': 498, 'home_theater': 598, 'stupa': 832, 'mosque': 668, 'church': 497, 'apiary': 410,
                'boathouse': 449, 'palace': 698, 'monastery': 663, 'honeycomb': 599, 'plate_rack': 729,
                'four-poster': 564,
                'baby_bed': -233, 'table': -234, 'cabinet': -235, 'file': 553, 'entertainment_center': 548,
                'dining_table': 532, 'bookcase': 453, 'table_lamp': 846, 'wardrobe': 894, 'seat': -236,
                'chiffonier': 493,
                'radio': 754, 'television': 851, 'fountain_pen': 563, 'quill': 749, 'ballpoint': 418, 'pan': -237,
                'spatula': 813, 'crock_pot': 521, 'pot': 837, 'drumstick': 542, 'spindle': 816, 'staff': -238,
                'matchstick': 644, 'hand_tool': -239, 'edge_tool': -240, 'plow': 730, 'power_drill': 740,
                'lawn_mower': 621,
                'swab': 840, 'broom': 462, 'face_powder': 551, 'lipstick': 629, 'torch': 862, 'lamp': -241,
                'optical_instrument': -242, 'weapon': -243, 'measuring_instrument': -244, 'magnifier': -245,
                'guillotine': 583, 'medical_instrument': -246, 'magnetic_compass': 635, 'oil_filter': 686,
                'strainer': 828,
                'space_heater': 811, 'stove': 827, 'disk_brake': 535, 'fastener': -247, 'muzzle': 676,
                'loudspeaker': 632,
                'microphone': 650, 'mousetrap': 674, 'spider_web': 815, 'radiator': 753, 'puck': 746, 'control': -248,
                'mechanical_device': -249, 'snorkel': 801, 'oxygen_mask': 691, 'mouse': 673, 'screen_1': 782,
                'solar_dish': 807, 'car_mirror': 475, 'pier': 718, 'tripod': 872, 'maypole': 645,
                'typewriter_keyboard': 878, 'stringed_instrument': -251, 'percussion_instrument': -252,
                'keyboard_instrument': -253, 'wind_instrument': -254, 'power_tool': -255, 'cash_machine': 480,
                'abacus': 398, 'farm_machine': -256, 'computer': -257, 'slot_machine': -258, 'military_vehicle': -259,
                'sled': -260, 'craft': -261, 'missile': 657, 'bus': -262, 'bullet_train': 466,
                'gymnastic_apparatus': -263,
                'golf_equipment': -264, 'weight': -265, 'cassette_player': 482, 'modem': 662, 'telephone': -266,
                'tape_player': 848, 'ipod': 605, 'cd_player': 485, 'monitor': 664, 'oscilloscope': 688,
                'peripheral': -267,
                'polaroid_camera': 732, 'reflex_camera': 759, 'puzzle': -268, 'ball': -269, 'pool_table': 736,
                'carpenters_kit': 477, 'drilling_platform': 540, 'mortar': 666, 'ladle': 618, 'tub': 876,
                'pitcher': 725,
                'jar': -271, 'coffee_mug': 504, 'bucket': 463, 'barrel': 427, 'bathtub': 435, 'reservoir': -272,
                'washbasin': 896, 'bottle': -273, 'safe': 771, 'pencil_box': 709, 'mailbox': 637, 'crate': 519,
                'chest': 492, 'carton': 478, 'saltshaker': 773, 'cocktail_shaker': 503, 'unicycle': 880, 'car_1': -274,
                'motor_scooter': 670, 'cart': -275, 'handcart': -276, 'bicycle': -277, 'tricycle': 870,
                'self-propelled_vehicle': -278, 'beer_glass': 441, 'goblet': 572, 'bowl': -279, 'petri_dish': 712,
                'backpack': 414, 'sleeping_bag': 797, 'mailbag': 636, 'purse': 748, 'plastic_bag': 728,
                'shopping_basket': 790, 'hamper': 588, 'breastplate': 461, 'pickelhaube': 715, 'vault': 884,
                'dome': 538,
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
                'goose': 99, 'lionfish': 396, 'percoid_fish': -300, 'puffer': 397, 'eel': 390, 'cyprinid': -301,
                'gar': 395,
                'sturgeon': 394, 'electric_ray': 5, 'stingray': 6, 'great_white_shark': 2, 'hammerhead': 4,
                'tiger_shark': 3, 'viper': -302, 'colubrid_snake': -303, 'boa': -304, 'sea_snake': 65, 'elapid': -305,
                'agamid': -306, 'banded_gecko': 38, 'green_lizard': 46, 'komodo_dragon': 48, 'gila_monster': 45,
                'whiptail': 41, 'iguanid': -307, 'alligator_lizard': 44, 'african_chameleon': 47,
                'african_crocodile': 49,
                'american_alligator': 50, 'leatherback_turtle': 34, 'loggerhead': 33, 'eft': 27, 'common_newt': 26,
                'axolotl': 29, 'spotted_salamander': 28, 'garden_spider': 74, 'barn_spider': 73, 'black_widow': 75,
                'tarantula': 76, 'wolf_spider': 77, 'black_and_gold_garden_spider': 72, 'hermit_crab': 125,
                'crab': -308,
                'lobster': -309, 'crayfish': 124, 'cicada': 316, 'leafhopper': 317, 'admiral': 321, 'lycaenid': 326,
                'sulphur_butterfly': 325, 'ringlet': 322, 'cabbage_butterfly': 324, 'monarch': 323, 'leaf_beetle': 304,
                'tiger_beetle': 300, 'long-horned_beetle': 303, 'scarabaeid_beetle': -310, 'ground_beetle': 302,
                'weevil': 307, 'ladybug': 301, 'cricket': 312, 'grasshopper': 311, 'ant': 310, 'bee': 309,
                'cockroach': 314,
                'mantis': 315, 'dragonfly': 319, 'damselfly': 320, 'crash_helmet': 518, 'football_helmet': 560,
                'bearskin': 439, 'bonnet': 452, 'cowboy_hat': 515, 'sombrero': 808, 'bathing_cap': 433,
                'shower_cap': 793,
                'mortarboard': 667, 'stole': 824, 'feather_boa': 552, 'hoopskirt': 601, 'sarong': 775, 'overskirt': 689,
                'miniskirt': 655, 'maillot': 638, 'bikini': 445, 'swimming_trunks': 842, 'poncho': 735, 'coat': -311,
                'kimono': 614, 'abaya': 399, 'cardigan': 474, 'sweatshirt': 841, 'bow_tie': 457, 'bolo_tie': 451,
                'windsor_tie': 906, 'rotisserie': 766, 'dutch_oven': 544, 'confectionery': 509, 'shoe_shop': 788,
                'toyshop': 865, 'bookshop': 454, 'tobacco_shop': 860, 'bakery': 415, 'barbershop': 424,
                'butcher_shop': 467,
                'bassinet': 431, 'crib': 520, 'cradle': 516, 'desk': 526, 'medicine_chest': 648, 'china_cabinet': 495,
                'chair': -312, 'toilet_seat': 861, 'park_bench': 703, 'studio_couch': 831, 'wok': 909,
                'frying_pan': 567,
                'caldron': 469, 'teapot': 849, 'coffeepot': 505, 'crutch': 523, 'flagpole': 557, 'shovel': 792,
                'opener': -313, 'plunger': 731, 'hammer': 587, 'screwdriver': 784, 'hatchet': 596, 'knife': -314,
                'plane': 726, 'jack-o-lantern': 607, 'spotlight': 818, 'candle': 470, 'sunglasses': 837,
                'projector': 745,
                'binoculars': 447, 'gun': -316, 'projectile': 744, 'bow': 456, 'rule': 769, 'scale': 778,
                'odometer': 685,
                'barometer': 426, 'timepiece': -317, 'radio_telescope': 755, 'loupe': 633, 'stethoscope': 823,
                'syringe': 845, 'buckle': 464, 'nail': 677, 'safety_pin': 772, 'knot': 616, 'screw': 783, 'lock': -318,
                'hair_slide': 584, 'switch': 844, 'joystick': 613, 'hook': 600, 'gas_pump': 571, 'carousel': 476,
                'reel': 758, 'wheel': -319, 'swing': 843, 'harp': 594, 'banjo': 420, 'bowed_stringed_instrument': -320,
                'guitar': -321, 'steel_drum': 822, 'maraca': 641, 'marimba': 642, 'gong': 577, 'chime': 494,
                'drum': 541,
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
                'horse_cart': 603, 'shopping_cart': 791, 'barrow': 428, 'bicycle-built-for-two': 444,
                'mountain_bike': 671,
                'streetcar': 829, 'forklift': 561, 'tank': 847, 'tractor': 866, 'recreational_vehicle': 757,
                'tracked_vehicle': -332, 'locomotive': -333, 'motor_vehicle': -334, 'mixing_bowl': 659,
                'soup_bowl': 809,
                'theater_curtain': 854, 'shower_curtain': 794, 'bulletproof_vest': 465, 'chain_mail': 490,
                'cuirass': 524,
                'grey_whale': 147, 'killer_whale': 148, 'bovid': -335, 'arabian_camel': 354, 'swine': -336,
                'llama': 355,
                'hippopotamus': 344, 'sorrel': 339, 'zebra': 340, 'cat': -337, 'big_cat': -338, 'hyena': 276,
                'wild_dog': -339, 'dog': -340, 'fox': -341, 'wolf': -342, 'meerkat': 299, 'mongoose': 298,
                'american_black_bear': 295, 'brown_bear': 294, 'ice_bear': 296, 'sloth_bear': 297, 'mink': 357,
                'badger': 362, 'otter': 360, 'weasel': 356, 'skunk': 361, 'polecat': 358, 'black-footed_ferret': 359,
                'giant_panda': 388, 'lesser_panda': 387, 'wood_rabbit': 330, 'angora': 332, 'lesser_ape': -343,
                'great_ape': -344, 'new_world_monkey': -345, 'old_world_monkey': -346, 'indri': 384,
                'madagascar_cat': 383,
                'ruddy_turnstone': 139, 'sandpiper': -347, 'dowitcher': 142, 'oystercatcher': 143,
                'little_blue_heron': 131,
                'bittern': 133, 'american_egret': 132, 'black_stork': 128, 'white_stork': 127,
                'red-breasted_merganser': 98,
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
                'padlock': 695, 'car_wheel': 479, 'paddlewheel': 694, 'potters_wheel': 739, 'pinwheel': 723,
                'violin': 889,
                'cello': 486, 'electric_guitar': 546, 'acoustic_guitar': 402, 'upright': 881, 'grand_piano': 579,
                'french_horn': 566, 'cornet': 513, 'trombone': 875, 'beating-reed_instrument': -353, 'flute': 558,
                'accordion': 401, 'harmonica': 593, 'desktop_computer': 527, 'portable_computer': -354,
                'aircraft_carrier': 403, 'submarine': 833, 'boat': -355, 'ship': -356, 'sailing_vessel': -357,
                'heavier-than-air_craft': -358, 'lighter-than-air_craft': -359, 'computer_keyboard': 508,
                'whiskey_jug': 901, 'water_jug': 899, 'snowmobile': 802, 'steam_locomotive': 820,
                'electric_locomotive': 547, 'amphibian': 408, 'car': -360, 'go-kart': 573, 'truck': -361,
                'snowplow': 803,
                'moped': 665, 'ox': 345, 'ibex': 350, 'bison': 347, 'ram': 348, 'antelope': -362, 'bighorn': 349,
                'water_buffalo': 346, 'hog': 341, 'warthog': 343, 'wild_boar': 342, 'wildcat': -363,
                'domestic_cat': -364,
                'tiger': 292, 'cheetah': 293, 'lion': 291, 'snow_leopard': 289, 'leopard': 288, 'jaguar': 290,
                'dhole': 274,
                'dingo': 273, 'african_hunting_dog': 275, 'mexican_hairless': 268, 'basenji': 253, 'spitz': -365,
                'pug': 254, 'newfoundland': 256, 'great_pyrenees': 257, 'brabancon_griffon': 262, 'poodle': -366,
                'corgi': -367, 'dalmatian': 251, 'working_dog': -368, 'leonberg': 255, 'toy_dog': -369,
                'hunting_dog': -370,
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
                'standard_poodle': 267, 'miniature_poodle': 266, 'pembroke': 263, 'cardigan_1': 264,
                'bull_mastiff': 243,
                'sennenhunde': -376, 'watchdog': -377, 'shepherd_dog': -378, 'boxer': 242, 'tibetan_mastiff': 244,
                'saint_bernard': 247, 'french_bulldog': 245, 'toy_spaniel': -379, 'eskimo_dog': 248, 'great_dane': 246,
                'hound': -380, 'toy_terrier': 158, 'pekinese': 154, 'japanese_spaniel': 152, 'maltese_dog': 153,
                'shih-tzu': 155, 'chihuahua': 151, 'rhodesian_ridgeback': 159, 'sporting_dog': -381, 'terrier': -382,
                'pinscher': -383, 'bassoon': 432, 'oboe': 683, 'yawl': 914, 'canoe': 472, 'trimaran': 871,
                'catamaran': 484,
                'moving_van': 675, 'police_van': 734, 'malamute': 249, 'siberian_husky': 250, 'appenzeller': 240,
                'greater_swiss_mountain_dog': 238, 'entlebucher': 241, 'bernese_mountain_dog': 239,
                'belgian_sheepdog': -384, 'schipperke': 223, 'kuvasz': 222, 'shetland_sheepdog': 230, 'kelpie': 227,
                'bouvier_des_flandres': 233, 'komondor': 228, 'old_english_sheepdog': 229, 'border_collie': 232,
                'collie': 231, 'briard': 226, 'german_shepherd': 235, 'rottweiler': 234, 'greyhound': -385,
                'papillon': 157,
                'blenheim_spaniel': 156, 'bluetick': 164, 'scottish_deerhound': 177, 'ibizan_hound': 173,
                'wolfhound': -386,
                'bloodhound': 163, 'beagle': 162, 'otterhound': 175, 'weimaraner': 178, 'redbone': 168,
                'foxhound': -387,
                'afghan_hound': 160, 'saluki': 176, 'basset': 161, 'spaniel': -388, 'norwegian_elkhound': 174,
                'black-and-tan_coonhound': 165, 'setter': -389, 'pointer': -390, 'retriever': -391, 'schnauzer': -392,
                'kerry_blue_terrier': 183, 'scotch_terrier': 199, 'tibetan_terrier': 200,
                'west_highland_white_terrier': 203, 'australian_terrier': 193, 'dandie_dinmont': 194, 'lhasa': 204,
                'bedlington_terrier': 181, 'airedale': 191, 'norwich_terrier': 186, 'silky_terrier': 201,
                'norfolk_terrier': 185, 'yorkshire_terrier': 187, 'wirehair': -393, 'bullterrier': -394, 'cairn': 192,
                'boston_bull': 195, 'soft-coated_wheaten_terrier': 202, 'springer_spaniel': -396, 'irish_terrier': 184,
                'border_terrier': 182, 'wire-haired_fox_terrier': 188, 'affenpinscher': 252, 'doberman': 236,
                'miniature_pinscher': 237, 'groenendael': 224, 'malinois': 225, 'whippet': 172,
                'italian_greyhound': 171,
                'irish_wolfhound': 170, 'borzoi': 169, 'english_foxhound': 167, 'walker_hound': 166, 'clumber': 216,
                'sussex_spaniel': 220, 'cocker_spaniel': 219, 'brittany_spaniel': 215, 'irish_water_spaniel': 221,
                'english_setter': 212, 'gordon_setter': 214, 'irish_setter': 213, 'german_short-haired_pointer': 210,
                'vizsla': 211, 'golden_retriever': 207, 'flat-coated_retriever': 205, 'labrador_retriever': 208,
                'chesapeake_bay_retriever': 209, 'curly-coated_retriever': 206, 'standard_schnauzer': 198,
                'miniature_schnauzer': 196, 'giant_schnauzer': 197, 'lakeland_terrier': 189, 'sealyham_terrier': 190,
                'american_staffordshire_terrier': 180, 'staffordshire_bullterrier': 179, 'english_springer': 217,
                'welsh_springer_spaniel': 218, 'sled_dog': -375, 'other_oscine': -450, 'other_aquatic_bird': -451,
                'other_wheeled_vehicle': -452}


    index_list_trial3 = [398, 146, 279, 414, 428, 438, 337, 100, 10, 464, 16, 138, 471, 378, 479, 480, 491, 293, 492,
                         19, 513,
                         519, 527, 274, 136, 561, 565, 368, 11, 99, 583, 338, 584, 333, 594, 351, 595, 344, 12, 379,
                         276, 352,
                         14, 17, 612, 13, 145, 618, 135, 291, 131, 18, 636, 637, 336, 268, 666, 670, 676, 683, 345, 694,
                         695,
                         86, 371, 84, 709, 283, 725, 728, 334, 739, 81, 746, 748, 85, 755, 757, 139, 82, 769, 771, 772,
                         776,
                         797, 129, 829, 837, 844, 847, 292, 282, 269, 380, 866, 875, 876, 880, 20, 340]

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

    imagenet1000_label_dict_index = imagenet1000_label_dict_index_trial3
    index_list = index_list_trial3
    data_name_hier_dict_100 = data_name_hier_dict_100_trial3

    data_label_index_dict = imagenet1000_label_dict_index()
    data_label_index_dict['other_oscine'] = -450
    data_label_index_dict['other_aquatic_bird'] = -451
    data_label_index_dict['other_wheeled_vehicle'] = -452

    def class_order():

        return [
                ['mammal', 'bird', 'device', 'container'],  # init

                ['ungulate', 'rodent', 'primate', 'feline', 'canine'],  # mammal
                ['game_bird', 'finch', 'wading_bird', 'other_oscine', 'other_aquatic_bird'],  # bird
                ['instrument', 'restraint', 'mechanism', 'musical_instrument', 'machine'],  # device
                ['vessel', 'box', 'bag', 'self-propelled_vehicle', 'other_wheeled_vehicle'],  # container


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




    data_label_index_dict = imagenet1000_label_dict_index()
    data_label_index_dict['other_working_dog'] = -404
    taxonomy_tree = Tree('imagenet1000', data_name_hier_dict_100, data_label_index_dict)

    used_nodes, leaf_id, node_labels = taxonomy_tree.prepro()
    # taxonomy_tree.show()
    import pandas as pd
    import numpy as np

    df = pd.read_csv('/Users/chenyuzhao/Downloads/ctl_rtc_imagenet100_trial3_BFS_seed500_with_imagenet_config_check_noutput/nout_details_task24.csv')
    # df = pd.read_csv('/Users/chenyuzhao/Downloads/test_12112312.csv')

    df['max_index']  = df[['p_root_c_mammal', 'p_root_c_bird', 'p_root_c_device', 'p_root_c_container']].apply(lambda x: np.argmax(x), axis=1)
    pred_list = np.array(df['max_index'])

    targets_list = list(df['targets'])

    gt_list = []
    index_1 = list
    # for i in targets_list:
    #     top1_list.append()
    index2child = taxonomy_tree.nodes.get('root').children
    child2index = {index2child[i]:i for i in index2child}
    for i in targets_list:
        gt_list.append(child2index[taxonomy_tree.get_parent_n_layer(node_label_list=[i], n_layer=1)[0][0]])
    gt_list = np.array(gt_list)
    total_acc = np.sum(gt_list == pred_list)/len(pred_list)
    print(total_acc)
    partial_list = {}
    for i in range(4):
        index = np.where(gt_list==i)
        partial_res_i = np.sum(pred_list[index] == i)/len(index[0])
        partial_list[i] = partial_res_i
    print(partial_list)


    # df['tart']

    # data_label_index_dict_inv = {data_label_index_dict[i]:i for i in data_label_index_dict}
    # df_total = pd.read_csv('/Users/chenyuzhao/Downloads/imagenet100_trial3_acc_total.csv')
    # class_index = list(df_total['class_index'])
    # BFS_diff = list(df_total['BFS_DER_diff'])
    # DFS_diff = list(df_total['DFS_DER_diff'])
    # stat_dict = {}
    # class_to_check = []
    # for i in range(len(class_index)):
    #     stat_dict[class_index[i]] = [BFS_diff[i], DFS_diff[i]]
    # for i in stat_dict:
    #     if stat_dict[i][0] <-0.1 and stat_dict[i][1] <-0.1:
    #         class_to_check.append(i)
    # # class_to_check = [283, 464, 471, 480, 527, 561, 594, 636, 670, 676, 683, 694, 725, 739, 748, 769, 772, 844, 875]
    # name_list = [data_label_index_dict_inv[i] for i in class_to_check]
    # node_label_list = [taxonomy_tree.nodes.get(i).label_index for i in name_list]
    #
    # parent_list = taxonomy_tree.get_parent_n_layer(node_label_list, 1)
    #
    # print(class_to_check)
    # print(name_list)
    # print(parent_list)
    # print({i: parent_list[0].count(i) for i in set(parent_list[0])})

    # import pandas as pd
    #
    # taxonomy_tree.show()
    #
    # task_order = class_order()
    #
    # data_label_index_dict_inv = {data_label_index_dict[i]:i for i in data_label_index_dict}
    #
    # prob_list =[153, 158, 185, 193, 212, 225, 246, 256, 264, 266, 268, 282, 298, 338, 342, 348, 356, 380]
    #
    #
    # acc_csv = pd.read_csv('/Users/chenyuzhao/Downloads/_task_16-3.csv')
    # avg_acc_list = list(acc_csv['avg_acc'][1:])
    # class_index_list = list(acc_csv['class_index'][1:])
    # count_list = list(acc_csv['count'][1:])
    #
    # acc_dict = {}
    # for i in range(len(class_index_list)):
    #     acc_dict[class_index_list[i]] = avg_acc_list[i]
    #
    # count_dict = {}
    # for i in range(len(class_index_list)):
    #     count_dict[class_index_list[i]] = count_list[i]
    #
    #
    # res_list = []
    # for i in prob_list:
    #     for j in task_order:
    #
    #         if data_label_index_dict_inv[i] in j:
    #             res_list.append(task_order.index(j))
    #
    # finest_dict = {}
    # for i in range(len(task_order)):
    #     for j in task_order[i]:
    #         if j in taxonomy_tree.leaf_nodes.values():
    #             if i not in finest_dict:
    #                 finest_dict[i] = [j]
    #             else:
    #                 finest_dict[i].append(j)
    #
    # finest_acc_dict = {i: [acc_dict[str(data_label_index_dict[j])] for j in finest_dict[i]] for i in finest_dict}
    #
    #
    # count_dict = {i: [count_dict[str(data_label_index_dict[j])] for j in finest_dict[i]] for i in finest_dict}
    #
    #
    # for i in finest_acc_dict:
    #     print(f'task {i+1}, with len {len(finest_acc_dict[i])}, mean {np.round(np.mean(finest_acc_dict[i]), 3)}, avg count {np.round(np.mean(count_dict[i]), 3)}')
    #
    # print(count_dict[10])


    # print(finest_acc_dict)


    # print(res_list)
    # print([(i, len(task_order[i])) for i in range(17)])


