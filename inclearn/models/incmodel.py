import numpy as np
import pandas as pd
import os
from copy import deepcopy
from scipy.spatial.distance import cdist

import torch
from torch.nn import DataParallel
from torch.nn import functional as F

from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer
from inclearn.deeprtc.metrics import averageMeter
from inclearn.deeprtc.utils import deep_rtc_nloss, deep_rtc_sts_loss, leaf_id_indices

# Constants
EPSILON = 1e-8


class IncModel(IncrementalLearner):
    def __init__(self, cfg, _run, ex, tensorboard, inc_dataset):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device']
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self._trial_i = cfg['trial']  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]

        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.TaxonomicDer(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )
        self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]

        # Memory
        self._memory_enable = cfg["memory_enable"]
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        # Task info
        self._task = 0
        self._task_size = 0
        self._current_tax_tree = None
        self._n_train_data = []
        self._n_test_data = []
        self._n_tasks = self._inc_dataset.n_tasks

        # Loaders
        self._train_loader, self._val_loader, self._test_loader = None, None, None

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
        self.curr_acc_list = None
        self.acc_detail_path = cfg["acc_detail_path"]
        if not os.path.exists(self.acc_detail_path):
            os.makedirs(self.acc_detail_path)

    def eval(self):
        self._parallel_network.eval()

    def set_task_info(self, task_info):
        self._task = task_info["task"]
        self._task_size = task_info["task_size"]
        self._current_tax_tree = task_info["partial_tree"]
        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]
        self._n_classes = len(self._current_tax_tree.leaf_nodes)

    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            self._parallel_network.train()

    def _before_task(self):
        # Update Task info
        task_info, train_loader, val_loader, test_loader = self._inc_dataset.new_task(self._memory_enable)
        self.set_task_info(task_info)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

        # log out task info
        self._ex.logger.info(f"Begin step {self._task}")
        print(self._n_classes)

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes-1, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.current_tax_tree = self._current_tax_tree
        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        self._network.current_task = self._task
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        # In DER model, freeze the previous network parameters
        # only updates parameters for the current network
        if self._der and self._task > 0:
            for i in range(self._task):
                for p in self._parallel_network.module.convnets[i].parameters():
                    p.requires_grad = False

        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr, weight_decay)

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)

    def _train_task(self):
        self._ex.logger.info(f"nb {len(self._train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size

        self._optimizer.zero_grad()
        self._optimizer.step()

        acc_list = []

        for epoch in range(self._n_epochs):
            _loss, _loss_aux = 0.0, 0.0

            nlosses = averageMeter()
            stslosses = averageMeter()
            losses = averageMeter()
            acc = averageMeter()

            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()
            for i, data in enumerate(self._train_loader, start=1):
                inputs, targets = data
                self.train()
                self._optimizer.zero_grad()
                nloss, stsloss, loss, acc = self._forward_loss(inputs, targets, nlosses, stslosses, losses, acc)

                # if self._cfg["use_aux_cls"] and self._task > 0:
                #     loss = loss_ce + loss_aux
                # else:
                #     loss = loss_ce

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)

                # _loss += loss_ce
                # _loss_aux += loss_aux
                _loss += loss
            _loss = _loss.item()
            # _loss_aux = _loss_aux.item()
            _loss_aux = 0
            if not self._warmup:
                self._scheduler.step()
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf Avg Loss: {}, Avg Acc: {}".format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss / i, 3),
                    round(acc.avg, 3),
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(self._val_loader)
            acc_list.append(acc)

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = self._train_loader.dataset.share_memory
        self.curr_acc_list = acc_list
        # self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, inputs, targets, nlosses, stslosses, losses, acc):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
        outputs = self._parallel_network(inputs)
        # since self._parallel_network = DataParallel(self._network)
        # this is equivalent to self._network.forward(inputs)

        output = outputs['output']
        nout = outputs['nout']
        sfmx_base = outputs['sfmx_base']

        nloss = deep_rtc_nloss(nout, targets, self._network.leaf_id, self._network.node_labels, self._device)
        nlosses.update(nloss.item(), inputs.size(0))

        # compute stochastic tree sampling loss
        leaf_id_indexes = leaf_id_indices(targets, self._network.leaf_id, self._device)
        gt_z = torch.gather(output, 1, leaf_id_indexes.view(-1, 1))
        stsloss = torch.mean(-gt_z + torch.log(torch.clamp(sfmx_base.view(-1, 1), 1e-17, 1e17)))
        stslosses.update(stsloss.item(), inputs.size(0))

        loss = nloss + stsloss * 1
        losses.update(loss.item(), inputs.size(0))

        # measure accuracy
        max_z = torch.max(output, dim=1)[0]
        preds = torch.eq(output, max_z.view(-1, 1))
        iscorrect = torch.gather(preds, 1, leaf_id_indexes.view(-1, 1)).flatten().float()
        acc.update(torch.mean(iscorrect).item(), inputs.size(0))

        acc_update_info = self.update_acc_detail(list(np.array(targets.cpu())), list(np.array(iscorrect.cpu())),
                                                 list((np.sum(np.array(preds.cpu()), 1) > 1) * 1))

        acc.update(torch.mean(iscorrect).item(), inputs.size(0))
        acc.update_detail(acc_update_info)
        return nloss, stsloss, loss, acc

    def update_acc_detail(self, targets, pred, multi_pred_list):
        res_dict = {i: {'avg': 0, 'multi_rate': 0, 'sum': 0, 'count': 0, 'multi_num': 0} for i in targets}

        for i in range(len(targets)):
            gt_label = targets[i]
            pred_iscorrect = pred[i]
            multi_pred = multi_pred_list[i]

            res_dict[gt_label]['count'] += 1
            res_dict[gt_label]['sum'] += pred_iscorrect
            res_dict[gt_label]['multi_num'] += np.array(multi_pred)

        for i in res_dict:
            if res_dict[i]['count'] != 0:
                res_dict[i]['avg'] = round(res_dict[i]['sum'] / res_dict[i]['count'], 3)
                res_dict[i]['multi_rate'] = round(res_dict[i]['multi_num'] / res_dict[i]['count'], 3)

        return res_dict

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes):
        loss = F.cross_entropy(outputs['logit'], targets)

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            if self._cfg["aux_n+1"]:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
        else:
            if str(self._device) == 'cuda:0':
                aux_loss = torch.zeros([1]).cuda()
            else:
                aux_loss = torch.zeros([1])

        return loss, aux_loss

    def _after_task(self):
        inc_dataset = self._inc_dataset
        network = deepcopy(self._parallel_network)
        network.eval()
        self._ex.logger.info("save model")
        if self._cfg["save_ckpt"] and self._task >= self._cfg["start_task"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if self._cfg["decouple"]['enable'] and self._task > 0:
            if self._cfg["decouple"]["fullset"]:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc,
                                                       mode="balanced_train")

            # finetuning
            # print(self._parallel_network.module.classifier.num_nodes)
            self._parallel_network.module.classifier.module.reset_parameters()
            finetune_last_layer(self._ex.logger,
                                self._parallel_network,
                                train_loader,
                                self._n_classes,
                                device=self._device,
                                nepoch=self._decouple["epochs"],
                                lr=self._decouple["lr"],
                                scheduling=self._decouple["scheduling"],
                                lr_decay=self._decouple["lr_decay"],
                                weight_decay=self._decouple["weight_decay"],
                                loss_type="ce",
                                temperature=self._decouple["temperature"])
            network = deepcopy(self._parallel_network)
            if self._cfg["save_ckpt"]:
                save_path = os.path.join(os.getcwd(), "ckpts")
                torch.save(network.cpu().state_dict(), "{}/decouple_step{}.ckpt".format(save_path, self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_enable and self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        acc = averageMeter()
        preds, targets = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                # _preds = self._parallel_network(inputs)['logit']
                _output = self._parallel_network(inputs)['output']
                max_z = torch.max(_output, dim=1)[0]
                _preds = torch.eq(_output, max_z.view(-1, 1))

                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())

        preds = torch.tensor(np.concatenate(preds, axis=0))
        targets = torch.tensor(np.concatenate(targets, axis=0))

        leaf_id_keys = self._network.leaf_id.keys()
        leaf_id_index_list = []
        for target_i in list(np.array(targets.cpu())):
            if target_i in leaf_id_keys:
                leaf_id_index_list.append(self._network.leaf_id[target_i])

        if str(self._device) == 'cuda:0':
            preds = preds.cuda()
            leaf_id_indexes = torch.tensor(leaf_id_index_list).cuda()
        else:
            leaf_id_indexes = torch.tensor(leaf_id_index_list)

        iscorrect = torch.gather(preds, 1, leaf_id_indexes.view(-1, 1)).flatten().float()

        acc_update_info = self.update_acc_detail(list(np.array(targets.cpu())), list(np.array(iscorrect.cpu())),
                                                 list((np.sum(np.array(preds.cpu()), 1) > 1) * 1))

        acc.update_detail(acc_update_info)
        self.curr_acc_list = [acc]

        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._ex.logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(os.getcwd(), f"ckpts/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None \
                else self._inc_dataset.data_inc

            self._inc_dataset.memory_dict = herding(
                self._parallel_network,
                self._inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
                self._device
            )
            self._inc_dataset.update_memory_array()
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")

    def save_acc_detail_info(self):
        class_index = []
        sum_list = []
        count_list = []
        multi_num_list = []
        avg_acc_list = []
        multi_rate_list = []

        for epoch_i in range(len(self.curr_acc_list)):
            class_index.append(f'epoch_{epoch_i}')
            sum_list.append('')
            count_list.append('')
            multi_num_list.append('')
            avg_acc_list.append('')
            multi_rate_list.append('')

            acc_epoch_i_info = self.curr_acc_list[epoch_i].info_detail

            for i in sorted(acc_epoch_i_info.keys()):
                class_index.append(i)
                sum_list.append(acc_epoch_i_info[i]['sum'])
                count_list.append(acc_epoch_i_info[i]['count'])
                multi_num_list.append(acc_epoch_i_info[i]['multi_num'])
                avg_acc_list.append(acc_epoch_i_info[i]['avg'])
                multi_rate_list.append(acc_epoch_i_info[i]['multi_rate'])

        df = pd.DataFrame({'class_index': class_index, 'avg_acc': avg_acc_list, 'multi_rate': multi_rate_list,
                           'count': count_list, 'acc_sum': sum_list, 'multi_num': multi_num_list})
        df.to_csv(f'{self.acc_detail_path}/task_{self._task}.csv')
