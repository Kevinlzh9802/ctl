import copy
import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier, RealTaxonomicClassifier
from inclearn.deeprtc import get_model
from inclearn.deeprtc.pivot import Pivot
from inclearn.datasets.data import tgt_to_tgt0


class TaxonomicDer(nn.Module):  # used in incmodel.py
    def __init__(self, convnet_type, cfg, nf=64, use_bias=False, init="kaiming", device=None, dataset="cifar100",
                 current_tax_tree=None, current_task=0):
        super(TaxonomicDer, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.use_aux_cls = cfg['use_aux_cls']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.module_cls = cfg['model_cls']
        self.module_pivot = cfg['model_pivot']
        self.current_tax_tree = current_tax_tree
        self.current_task = current_task

        if self.der:
            print("Enable dynamical representation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.taxonomy = cfg['taxonomy']
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.device = device
        # self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" , index = 0)

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        if self.taxonomy is not None:
            gate = self.model_pivot(torch.ones([x.size(0), len(self.used_nodes)]))
            # gate[:, 0] = 1
            # print(features)
            output, nout, sfmx_base = self.classifier(x=features, gate=gate)
            # logits = self.classifier(features)
        else:
            output = self.classifier(features)
            nout, sfmx_base = None, None

        if self.use_aux_cls:
            aux_logits = self.aux_classifier(features[:, -self.out_dim:]) \
                if features.shape[1] > self.out_dim else None
        else:
            aux_logits = None
        return {'feature': features, 'output': output, 'nout': nout, 'sfmx_base': sfmx_base, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        # self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.taxonomy:
            all_classes = len(self.current_tax_tree.leaf_nodes)
        else:
            all_classes = self.n_classes + n_classes

        if self.current_task > 0:
            new_net = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_net.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_net)

        new_clf = self._gen_classifier(self.out_dim * len(self.convnets), all_classes)
        if self.taxonomy:
            if self.classifier is not None and self.reuse_oldfc:
                old_clf = self.classifier
                for k in range(old_clf.num_nodes):
                    for j in range(old_clf.cur_task):
                        fc_name = old_clf.nodes[k].name + f'_TF{j}'
                        fc_old = getattr(old_clf, fc_name, None)
                        fc_new = getattr(new_clf, fc_name, None)
                        assert fc_old is not None
                        assert fc_new is not None
                        # weight = copy.deepcopy(fc_old.weight.data)
                        fc_new.weight.data = copy.deepcopy(fc_old.weight.data)
                        fc_new.bias.data = copy.deepcopy(fc_old.bias.data)
                        for param in fc_new.parameters():
                            param.requires_grad = False
                        fc_new.eval()
        else:
            if self.classifier is not None and self.reuse_oldfc:
                weight = copy.deepcopy(self.classifier.weight.data)
                new_clf.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight

        del self.classifier
        self.classifier = new_clf

        if self.aux_nplus1:
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
            aux_fc = nn.Linear(self.out_dim, n_classes + 1, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(aux_fc.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(aux_fc.bias, 0.0)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc
        c = 9

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.taxonomy is not None:
            self._update_tree_info()
            if self.taxonomy == 'rtc':
                # classifier
                # used_nodes = setup_tree(self.current_task, self.current_tax_tree)
                model_dict = {'arch': self.module_cls, 'feat_size': in_features}
                if self.device.type == 'cuda':
                    model_cls = get_model(model_dict, self.used_nodes).cuda()
                    # model_cls = nn.DataParallel(model_cls, device_ids=range(torch.cuda.device_count()))
                else:
                    model_cls = get_model(model_dict, self.used_nodes)
                    # model_cls = nn.DataParallel(model_cls, device_ids=range(0))
                classifier = model_cls

                # pivot
                self._gen_pivot()
            else:
                raise NotImplementedError('')
        else:
            if self.weight_normalization:
                classifier = CosineClassifier(in_features, n_classes).to(self.device)
            else:
                classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
                if self.init == "kaiming":
                    nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
                if self.use_bias:
                    nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_pivot(self):
        if self.device.type == 'cuda':
            model_pivot = get_model(self.module_pivot).cuda()
            model_pivot = nn.DataParallel(model_pivot, device_ids=range(torch.cuda.device_count()))
        else:
            model_pivot = get_model(self.module_pivot)
        self.model_pivot = model_pivot

    def _update_tree_info(self):
        used_nodes, leaf_id, node_labels = self.current_tax_tree.prepro()
        self.used_nodes = used_nodes
        self.node_labels = node_labels
        self.leaf_id = leaf_id

    def cal_score_tree(self, inputs):
        if self.der:
            features = [convnet(inputs) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(inputs)

        leaf_id_inv = {self.leaf_id[x]: x for x in self.leaf_id.keys()}
        score_info = []
        for node in self.used_nodes.values():
            prod = 0.0
            for j in range(self.current_task + 1):
                fc_name = node.name + f'_TF{j}'
                fc_layers = getattr(self.classifier, fc_name)
                prod += fc_layers(features[:, 512 * j: 512 * (j + 1)])
            score_info.append({'name': node.name,
                               'depth': node.depth,
                               'children': node.children,
                               'score': torch.mean(prod, 0)})
        for i in range(len(score_info)):
            print(score_info[i])
