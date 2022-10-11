import numpy as np

import os
from copy import deepcopy
from scipy.spatial.distance import cdist
import pynvml

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer
from inclearn.deeprtc.metrics import averageMeter
from inclearn.deeprtc.utils import deep_rtc_nloss, deep_rtc_sts_loss
from inclearn.datasets.data import tgt_to_tgt0, tgt0_to_tgt, tgt_to_aux_tgt, aux_tgt_to_tgt, tgt_to_tgt0_no_tax

import pandas as pd

# Constants
EPSILON = 1e-8


class IncModel(IncrementalLearner):
    def __init__(self, cfg, logger, inc_dataset):
        super().__init__()
        self.mode_train = True
        self._cfg = cfg
        self._device = cfg['device']
        if cfg["is_distributed"]:
            self._device = torch.device("cuda:{}".format(cfg["rank"]))
            self._rank = self._cfg["rank"]
        self._logger = logger
        # self._run = _run  # the sacred _run object.

        # Data
        self._train_from_task = cfg['retrain_from_task']
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
        # self._tensorboard = tensorboard
        # if f"trial{self._trial_i}" not in self._run.info:
        #     self._run.info[f"trial{self._trial_i}"] = {}
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
        if self._cfg["is_distributed"]:
            self._parallel_network = DDP(self._network, device_ids=[self._rank], output_device=self._rank)
            self._parallel_network.to(self._device)
        else:
            self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
        self.curr_acc_list = []
        self.curr_acc_list_aux = []
        self.curr_preds = torch.tensor([])
        self.curr_preds_aux = torch.tensor([])
        self.curr_targets = torch.tensor([])
        self.curr_targets_aux = torch.tensor([])

        # Task info
        self._task = 0
        self._task_size = 0
        self._current_tax_tree = None
        self._n_train_data = 0
        self._n_test_data = 0
        self._n_tasks = self._inc_dataset.n_tasks

        # Loaders
        self._cur_train_loader = None
        self._cur_test_loader = None
        self._cur_val_loader = None

        # Save paths
        self.train_save_option = cfg['train_save_option']
        self.sp = cfg['sp']

    def eval(self):
        # self._parallel_network.eval()
        self._parallel_network.eval()

    def set_task_info(self, task_info):
        if self._cfg["is_distributed"]:
            self._logger.info(f'process {self._cfg["rank"]} begin set task info')
        self._task = task_info["task"]
        # task size for current task
        # n_classes for total number of classes (history + present)
        self._task_size = task_info["task_size"]

        if self._cfg["taxonomy"] is None:
            self._current_tax_tree = None
            self._n_classes += self._task_size
        elif self._cfg["taxonomy"] == 'rtc':
            self._current_tax_tree = task_info["partial_tree"]
            self._n_classes = len(self._current_tax_tree.leaf_nodes)

        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]

    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            # self._parallel_network.train()
            self._parallel_network.train()

    def _new_task(self):
        self._logger.info('begin new task')
        task_info, train_loader, val_loader, test_loader = self._inc_dataset.new_task()
        self.set_task_info(task_info)
        self._cur_train_loader = train_loader
        self._cur_val_loader = val_loader
        self._cur_test_loader = test_loader

    def _before_task(self, inc_dataset):
        self._logger.info(f"Begin step {self._task}")

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._logger.info("Now {} examplars per class.".format(self._memory_per_class))

        # Network
        self._network.current_tax_tree = self._current_tax_tree
        self._network.task_size = self._task_size
        self._network.current_task = self._task

        self._network.add_classes(self._task_size)
        self._network.n_classes = self._n_classes
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr
        # if self._cfg["is_distributed"]:
        #     lr = lr * self._cfg["world_size"]

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

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
        train_loader = self._cur_train_loader
        val_loader = self._cur_val_loader
        self._logger.info(f"nb {len(train_loader.dataset)}")

        # utils.display_weight_norm(self._logger, self._parallel_network, self._increments, "Initial trainset")
        # utils.display_feature_norm(self._logger, self._parallel_network, train_loader, self._n_classes,
        #                            self._increments, "Initial trainset")

        self._optimizer.zero_grad()
        self._optimizer.step()

        acc_list, acc_list_aux = [], []
        self.curr_preds, self.curr_preds_aux = self._to_device(torch.tensor([])), self._to_device(torch.tensor([]))
        self.curr_targets, self.curr_targets_aux = self._to_device(torch.tensor([])), self._to_device(torch.tensor([]))

        for epoch in range(self._n_epochs):
            _ce_loss, _joint_ce_loss, _loss_aux, _total_loss = 0.0, 0.0, 0.0, 0.0

            nlosses = averageMeter()
            stslosses = averageMeter()
            jointcelosses = averageMeter()
            losses = averageMeter()
            acc = averageMeter()
            acc_5 = averageMeter()
            acc_aux = averageMeter()

            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()

                    self._parallel_network.to(self._device)
            count = 0
            for i, data in enumerate(train_loader, start=1):
                inputs, targets = data
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                self.train()
                self._optimizer.zero_grad()

                outputs = self._parallel_network(inputs)
                self.record_details(outputs, targets, acc, acc_5, acc_aux, self.train_save_option)
                joint_ce_loss, ce_loss, loss_aux = \
                    self._compute_loss(outputs, targets, nlosses, jointcelosses, stslosses, losses)
                # ce_loss, loss_aux, acc, acc_aux = \
                #     self._forward_loss(inputs, targets, nlosses, stslosses, losses, acc, acc_aux)

                if self._cfg["use_aux_cls"] and self._task > 0:
                    total_loss = ce_loss + loss_aux
                else:
                    total_loss = ce_loss

                if self._cfg["use_joint_ce_loss"]:
                    total_loss += joint_ce_loss

                # print(total_loss)
                total_loss.backward()

                # a = self._optimizer.param_groups[0]['params']
                # print(a[0].grad)

                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)

                # _loss += loss_ce
                _loss_aux += loss_aux
                _ce_loss += ce_loss
                _joint_ce_loss += joint_ce_loss
                _total_loss += total_loss
                count += 1

                self._network.cal_score_tree(inputs)

            _ce_loss = _ce_loss.item()
            _loss_aux = _loss_aux.item()
            _joint_ce_loss = _joint_ce_loss.item()
            _total_loss = _total_loss.item()
            if not self._warmup:
                self._scheduler.step()
            self._logger.info(
                "Task {}/{}, Epoch {}/{} => Clf Avg Total Loss: {}, Clf Avg CE Loss: {}, Avg Aux Loss: {}, "
                "Avg Joint CE Loss: {}, Avg Acc: {}, Avg Aux Acc: {}".format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_total_loss / i, 3),
                    round(_ce_loss / i, 3),
                    round(_loss_aux / i, 3),
                    round(_joint_ce_loss / i, 3),
                    round(acc.avg, 3),
                    round(acc_aux.avg, 3)
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

            acc_list.append(acc)
            acc_list_aux.append(acc_aux)

            self.check_gpu_info(f'epoch {epoch + 1}')
            if epoch % 40 == 0:
                epoch_net = deepcopy(self._parallel_network)
                epoch_net.eval()
                self._logger.info("save model")
                torch.save(epoch_net.cpu().state_dict(), "{}/epoch{}.ckpt".format(self.sp['model'], epoch))

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        # save accuracy and preds info into files
        self.curr_acc_list = acc_list
        self.curr_acc_list_aux = acc_list_aux
        self.save_details(self.sp['acc_detail']['train'], self.train_save_option)

    def record_details(self, outputs, targets, acc, acc_5, acc_aux, save_option=None):
        if self._cfg["taxonomy"] is not None:
            targets_0 = tgt_to_tgt0(targets, self._network.leaf_id, self._device)
        else:
            targets_0 = tgt_to_tgt0_no_tax(targets, self._inc_dataset.targets_all_unique, self._device)

        # if self._cfg["taxonomy"] is not None:
        output = outputs['output']
        aux_output = outputs['aux_logit']

        self.record_accuracy(output, targets_0, acc, acc_5)
        # record to self.curr_acc_list
        if save_option["acc_details"]:
            self.record_acc_details(output, targets, targets_0, acc)
        if save_option["preds_details"]:
            self.curr_preds = torch.cat((self.curr_preds, output), 0)
            self.curr_targets = torch.cat((self.curr_targets, targets), 0)

        if aux_output is not None:
            cur_labels = self._inc_dataset.targets_cur_unique  # it should be sorted
            aux_targets = tgt_to_aux_tgt(targets, cur_labels, self._device)
            self.record_accuracy(aux_output, aux_targets, acc_aux, acc_5)
            if save_option["acc_aux_details"]:
                self.record_acc_details(aux_output, aux_targets, aux_targets, acc_aux)
            if save_option["preds_aux_details"]:
                self.curr_preds_aux = torch.cat((self.curr_preds_aux, aux_output), 0)
                self.curr_targets_aux = torch.cat((self.curr_targets_aux, aux_targets), 0)

    def save_details(self, acc_dir, save_option):
        if not os.path.exists(acc_dir):
            os.makedirs(acc_dir)
        if save_option["acc_details"]:
            self.save_acc_details('train', acc_dir)
        if save_option["acc_aux_details"]:
            self.save_acc_aux_details('train', acc_dir)

        preds_dir = acc_dir + 'preds/'
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        if save_option["preds_details"]:
            self.save_preds_details(self.curr_preds, self.curr_targets, preds_dir)
        if save_option["preds_aux_details"]:
            self.save_preds_aux_details(self.curr_preds_aux, self.curr_targets_aux, preds_dir)

    def _compute_loss(self, outputs, targets, nlosses, jointcelosses, stslosses, losses):
        batch_size = targets.size(0)
        if self._cfg["taxonomy"] is not None:
            output = outputs['output']
            aux_output = outputs['aux_logit']
            nout = outputs['nout']
            sfmx_base = outputs['sfmx_base']
            targets_0 = tgt_to_tgt0(targets, self._network.leaf_id, self._device)
            aux_loss = self._compute_aux_loss(targets, aux_output)

            nloss = deep_rtc_nloss(nout, targets, self._network.leaf_id, self._network.node_labels, self._device)
            nlosses.update(nloss.item(), batch_size)

            gt_z = torch.gather(output, 1, targets_0.view(-1, 1))
            stsloss = torch.mean(-gt_z + torch.log(torch.clamp(sfmx_base.view(-1, 1), 1e-17, 1e17)))
            stslosses.update(stsloss.item(), batch_size)

            if self._cfg["use_joint_ce_loss"]:
                joint_ce_loss = self._compute_joint_ce_loss(targets_0, output)
                jointcelosses.update(joint_ce_loss.item(), batch_size)
            else:
                joint_ce_loss = torch.tensor([0])

            loss = nloss + stsloss * 0
            losses.update(loss.item(), batch_size)

        else:
            output = outputs['output']
            aux_output = outputs['aux_logit']
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            targets_0 = tgt_to_tgt0_no_tax(targets, self._inc_dataset.targets_all_unique, self._device)
            loss = torch.mean(criterion(output, targets_0.long()))
            losses.update(loss.item(), batch_size)
            aux_loss = self._compute_aux_loss(targets, aux_output)
            joint_ce_loss = torch.tensor([0])
        return joint_ce_loss, loss, aux_loss

    def _compute_joint_ce_loss(self, targets_0, output):
        joint_ce_loss = torch.mean(F.cross_entropy(output, targets_0.long()))
        return joint_ce_loss

    def update_acc_detail(self, leaf_id_index_list, pred, multi_pred_list):
        res_dict = {i: {'avg': 0, 'multi_rate': 0, 'sum': 0, 'count': 0, 'multi_num': 0} for i in leaf_id_index_list}

        for i in range(len(leaf_id_index_list)):
            gt_label = leaf_id_index_list[i]
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

    def _compute_aux_loss(self, targets, aux_output):
        if aux_output is not None:
            cur_labels = self._inc_dataset.targets_cur_unique  # it should be sorted
            aux_targets = tgt_to_aux_tgt(targets, cur_labels, self._device)
            aux_loss = F.cross_entropy(aux_output, aux_targets)
        else:
            if self._device.type == 'cuda':
                aux_loss = torch.zeros([1]).cuda()
            else:
                aux_loss = torch.zeros([1])
        return aux_loss

    def _after_task(self, inc_dataset):
        taski = self._task
        # network = deepcopy(self._parallel_network)
        network = deepcopy(self._parallel_network)
        network.eval()
        self._logger.info("save model")
        if taski >= self._train_from_task and taski in self._cfg["save_ckpt"]:
            # save_path = os.path.join(os.getcwd(), "ckpts")
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(self.sp['model'], self._task))

        if self._cfg["decouple"]['enable'] and taski > 0 and taski >= self._train_from_task:
            if self._cfg["decouple"]["fullset"]:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                       inc_dataset.targets_inc,
                                                       mode="balanced_train")

            # finetuning
            # only hiernet on cuda needs .module
            self._network.classifier.reset_parameters()
            finetune_last_layer(self._logger,
                                # self._parallel_network,
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
                                temperature=self._decouple["temperature"],
                                save_path=f"{self.sp['acc_detail']['train']}/task_{self._task}_decouple",
                                index_map=self._inc_dataset.targets_all_unique)
            network = deepcopy(self._parallel_network)
            if taski in self._cfg["save_ckpt"]:
                # save_path = os.path.join(os.getcwd(), "ckpts")
                torch.save(network.cpu().state_dict(),
                           "{}/decouple_step{}.ckpt".format(self.sp['model'], self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._logger.info("compute prototype")
            self.update_prototype()

        if self._cfg['memory_enable'] and self._memory_size.memsize != 0:
            self._logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                # save_path = os.path.join(os.getcwd(), "ckpts/mem")
                save_path = self.sp['model'] + 'mem'
                data_memory, targets_memory = self._inc_dataset.gen_memory_array_from_dict()
                memory = {
                    'x': data_memory,
                    'y': targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def _eval_task(self, data_loader, save_path='', name='default', save_option=None):
        if self._infer_head == "softmax":
            self._compute_accuracy_by_netout(data_loader, save_path=save_path, name=name, save_option=save_option)
        elif self._infer_head == "NCM":
            # ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
            pass
        else:
            raise ValueError()

    def _compute_accuracy_by_netout(self, data_loader, name='default', save_path='', save_option=None):
        self._logger.info(f"Begin evaluation: {name}")
        factory.print_dataset_info(data_loader)

        acc = averageMeter()
        acc_5 = averageMeter()
        acc_aux = averageMeter()
        self.curr_preds, self.curr_preds_aux = self._to_device(torch.tensor([])), self._to_device(torch.tensor([]))
        self.curr_targets, self.curr_targets_aux = self._to_device(torch.tensor([])), self._to_device(torch.tensor([]))

        self._parallel_network.eval()

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                outputs = self._parallel_network(inputs)
                # print(inputs)
                # self._logger.info(inputs)
                # self._logger.info(targets)
                self.record_details(outputs, targets, acc, acc_5, acc_aux, save_option)

        self._logger.info(f"Evaluation {name} acc: {acc.avg}, aux_acc: {acc_aux.avg}")

        self._network.cal_score_tree(inputs[:30])
        # save accuracy and preds info into files
        self.curr_acc_list = [acc]
        self.curr_acc_list_aux = [acc_aux]

        sp = save_path + name + '/acc_details/'
        self.save_details(sp, save_option)

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader, self._device)
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
            self._network.postprocessor.update(self._logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._logger.info("Post processor wa update !")
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
        save_path = self.sp['model'] + f'mem/mem_step{self._task}.ckpt'
        # save_path = os.path.join(os.getcwd(), f"ckpts/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            data_memory = memory_states['x']
            targets_memory = memory_states['y']
            memory_dict = {}
            for class_i in set(targets_memory):
                memory_dict[class_i] = data_memory[targets_memory == class_i]
            self._inc_dataset.memory_dict = memory_dict
            self._herding_matrix = memory_states['herding']
            self._logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None \
                else self._inc_dataset.data_inc
            self._inc_dataset.memory_dict = herding(
                self._n_classes,
                # self._parallel_network,
                self._parallel_network,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._logger,
                self._device
            )
            # self._inc_dataset.update_memory_array()
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._logger.info(f"test top1acc:{test_acc_stats['top1']}")

    def save_acc_details(self, save_name, save_path):
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
                           'count': count_list, 'acc_sum': sum_list, 'multi_num': multi_num_list,
                           })
        df.to_csv(f'{save_path}_task_{self._task}.csv', index=False)

    def save_acc_aux_details(self, save_name, save_path):
        class_index_aux = []
        sum_list_aux = []
        count_list_aux = []
        multi_num_list_aux = []
        avg_acc_list_aux = []
        multi_rate_list_aux = []

        for epoch_i in range(len(self.curr_acc_list_aux)):
            class_index_aux.append(f'epoch_{epoch_i}')
            sum_list_aux.append('')
            count_list_aux.append('')
            multi_num_list_aux.append('')
            avg_acc_list_aux.append('')
            multi_rate_list_aux.append('')

            acc_epoch_i_info_aux = self.curr_acc_list_aux[epoch_i].info_detail

            for i in sorted(acc_epoch_i_info_aux.keys()):
                class_index_aux.append(i)
                count_list_aux.append(acc_epoch_i_info_aux[i]['count'])
                sum_list_aux.append(acc_epoch_i_info_aux[i]['sum'])
                multi_num_list_aux.append(acc_epoch_i_info_aux[i]['multi_num'])
                avg_acc_list_aux.append(acc_epoch_i_info_aux[i]['avg'])
                multi_rate_list_aux.append(acc_epoch_i_info_aux[i]['multi_rate'])

        df_aux = pd.DataFrame(
            {'class_index_aux': class_index_aux, 'avg_acc_aux': avg_acc_list_aux, 'multi_rate_aux': multi_rate_list_aux,
             'count_aux': count_list_aux, 'acc_sum_aux': sum_list_aux, 'multi_num_aux': multi_num_list_aux,
             })
        df_aux.to_csv(f'{save_path}_task_{self._task}_aux.csv', index=False)

    @staticmethod
    def record_accuracy(output, targets, acc, acc_5=None):
        iscorrect = (output.argmax(1) == targets)
        acc.update(float(iscorrect.count_nonzero() / iscorrect.size(0)), iscorrect.size(0))
        # if acc_5 is not None:
        #     preds_sort = output.argsort(1)
        #     ind_range = min(output.size(1), 5)
        #     iscorrect_5 = torch.zeros(size=preds_sort[:, 0].size())
        #     for x in range(ind_range):
        #         pred = preds_sort[:, x]
        #         iscorrect_5 = torch.logical_or(iscorrect_5, pred == targets)
        #     acc_5.update(float(iscorrect_5.count_nonzero() / iscorrect_5.size(0)), iscorrect_5.size(0))

    def record_acc_details(self, output, targets, targets_0, acc):
        # targets is the real label
        # targets_0 is the re-indexed label that starts from 0, it still can be the same with targets
        max_z = torch.max(output, dim=1)[0]
        preds = torch.eq(output, max_z.view(-1, 1))
        iscorrect = torch.gather(preds.cpu(), 1, targets_0.type(torch.LongTensor).view(-1, 1)).flatten().float()
        acc_update_info = self.update_acc_detail(list(np.array(targets.cpu())), list(np.array(iscorrect.cpu())),
                                                 list((np.sum(np.array(preds.cpu()), 1) > 1) * 1))
        acc.update_detail(acc_update_info)

    def save_preds_details(self, output, targets, save_path):
        # TODO: fix the output!
        preds_ori = output.argmax(1)
        if self._cfg['taxonomy']:
            preds = tgt0_to_tgt(preds_ori, self._network.leaf_id)
        elif preds_ori.device.type == 'cuda':
            preds = preds_ori.cpu()
        else:
            preds = preds_ori

        np.save(save_path + 'preds_res.npy', preds)
        np.save(save_path + 'targets_res.npy', targets.cpu())

    def save_preds_aux_details(self, output_aux, targets_aux, save_path):
        # TODO: fix the output!
        if len(output_aux) > 0:
            preds_aux_ori = output_aux.argmax(1)
            preds_aux = aux_tgt_to_tgt(preds_aux_ori, self._inc_dataset.targets_cur_unique, self._device)

            np.save(save_path + 'preds_aux_res.npy', preds_aux.cpu())
            np.save(save_path + 'targets_aux_res.npy', targets_aux.cpu())

    def _to_device(self, x):
        if self._device.type == 'cuda':
            return x.cuda()
        else:
            return x

    def nvidia_info(self):
        nvidia_dict = {
            "state": True,
            "nvidia_version": "",
            "nvidia_count": 4,
            "gpus": []
        }
        try:
            pynvml.nvmlInit()
            nvidia_dict["nvidia_version"] = pynvml.nvmlSystemGetDriverVersion()
            nvidia_dict["nvidia_count"] = pynvml.nvmlDeviceGetCount()
            for i in range(nvidia_dict["nvidia_count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = {
                    "gpu_name": pynvml.nvmlDeviceGetName(handle),
                    "total": memory_info.total,
                    "free": memory_info.free,
                    "used": memory_info.used,
                    "temperature": f"{pynvml.nvmlDeviceGetTemperature(handle, 0)}℃",
                    "powerStatus": pynvml.nvmlDeviceGetPowerState(handle)
                }
                nvidia_dict['gpus'].append(gpu)
        except pynvml.NVMLError as _:
            nvidia_dict["state"] = False
        except Exception as _:
            nvidia_dict["state"] = False
        finally:
            try:
                pynvml.vmlShutdown()
            except:
                pass
        return nvidia_dict

    def check_gpu_info(self, pos_name):
        gpus_info = self.nvidia_info()['gpus']
        logger_mesg = ''
        for i in range(len(gpus_info)):
            gpu_info_i = gpus_info[i]
            logger_mesg += f"GPU {i} at {pos_name} Name {gpu_info_i['gpu_name']}, " + \
                           f"Usage {np.round(gpu_info_i['used'] / gpu_info_i['total'] * 100, 3)}%\n"

        self._logger.info(logger_mesg)
