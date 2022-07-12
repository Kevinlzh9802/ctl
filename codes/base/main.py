'''
@Author : Yan Shipeng, Xie Jiangwei
@Contact: yanshp@shanghaitech.edu.cn, xiejw@shanghaitech.edu.cn
'''

import sys
import os
import os.path as osp
import copy
import time
import shutil
import cProfile
import logging
from pathlib import Path
import numpy as np
import random
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torchinfo import summary

repo_name = 'ctl'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.append(base_dir)

from sacred import Experiment
ex = Experiment(base_dir=base_dir, save_git_info=False)

# Save which files
# ex.add_source_file(osp.join(base_dir, "inclearn/models/icarl.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/lib/data.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/lib/network.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/convnet/resnet.py"))
# ex.add_source_file(osp.join(os.getcwd(), "icarl.py"))
# ex.add_source_file(osp.join(os.getcwd(), "network.py"))
# ex.add_source_file(osp.join(os.getcwd(), "resnet.py"))

# MongoDB Observer
# ex.observers.append(MongoObserver.create(url='xx.xx.xx.xx:port', db_name='classil'))

import torch

from inclearn.tools import factory, results_utils, utils
from inclearn.learn.pretrain import pretrain
from inclearn.tools.metrics import IncConfusionMeter

def initialization(config, seed, mode, exp_id):
    # Add it if your input size is fixed
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True  # This will result in non-deterministic results.
    # ex.captured_out_filter = lambda text: 'Output capturing turned off.'
    cfg = edict(config)
    utils.set_seed(cfg['seed'])
    if exp_id is None:
        exp_id = -1
        cfg.exp.savedir = "./logs"
    logger = utils.make_logger(f"exp{exp_id}_{cfg.exp.name}_{mode}", savedir=cfg.exp.savedir)

    # Tensorboard
    exp_name = f'{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else f'../inbox/{cfg["exp"]["name"]}'
    tensorboard_dir = cfg["exp"]["tensorboard_dir"] + f"/{exp_name}"

    # If not only save latest tensorboard log.
    # if Path(tensorboard_dir).exists():
    #     shutil.move(tensorboard_dir, cfg["exp"]["tensorboard_dir"] + f"/../inbox/{time.time()}_{exp_name}")

    tensorboard = SummaryWriter(tensorboard_dir)
    return cfg, logger, tensorboard


@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
    ex.logger.info(cfg)
    cfg.data_folder = osp.join(base_dir, "data")

    start_time = time.time()
    _train(cfg, _run, ex, tensorboard)
    ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(cfg, _run, ex, tensorboard):
    if cfg["device_auto_detect"]:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    else:
        factory.set_device(cfg)
    trial_i = cfg['trial']

    inc_dataset = factory.get_data(cfg)
    ex.logger.info("curriculum")
    ex.logger.info(inc_dataset.curriculum)

    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)

    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg["exp"]["ckptdir"]

    results = results_utils.get_template_results(cfg)

    for task_i in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader, x_train, y_train = inc_dataset.new_task(cfg['sample_rate'])

        model.set_task_info(
            task=task_info["task"],
            task_size=task_info["task_size"],
            tax_tree=task_info["partial_tree"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
            acc_detail_path=cfg.acc_detail_path
        )
        model.before_task(task_i, inc_dataset)
        # TODO: Move to incmodel.py
        if 'min_class' in task_info:
            ex.logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))

        if cfg['retrain_from_task0']:
            model.train_task(train_loader, val_loader)
        else:
            # state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
            state_dict = torch.load(f"{cfg['pre_train_model_path']}/step{task_i}.ckpt")
            model._parallel_network.load_state_dict(state_dict)

        # model.after_task(task_i, inc_dataset)
        model.save_acc_detail_info('train_with_step')

        ypred, ytrue = model.eval_task(test_loader)
        model.after_task(task_i, inc_dataset, x_train, y_train)
        model.save_acc_detail_info('after_train')

    top1_avg_acc, top5_avg_acc = results_utils.compute_avg_inc_acc(results["results"])

    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"] = top1_avg_acc
    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"] = top5_avg_acc
    ex.logger.info("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"],
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"],
    ))
    if cfg["exp"]["name"]:
        results_utils.save_results(results, cfg["exp"]["name"])


def do_pretrain(cfg, ex, model, device, train_loader, test_loader):
    if not os.path.exists(osp.join(ex.base_dir, 'pretrain/')):
        os.makedirs(osp.join(ex.base_dir, 'pretrain/'))
    model_path = osp.join(
        ex.base_dir,
        "pretrain/{}_{}_cosine_{}_multi_{}_aux{}_nplus1_{}_{}_trial_{}_{}_seed_{}_start_{}_epoch_{}.pth".format(
            cfg["model"],
            cfg["convnet"],
            cfg["weight_normalization"],
            cfg["der"],
            cfg["use_aux_cls"],
            cfg["aux_n+1"],
            cfg["dataset"],
            cfg["trial"],
            cfg["train_head"],
            cfg['seed'],
            cfg["start_class"],
            cfg["pretrain"]["epochs"],
        ),
    )
    if osp.exists(model_path):
        print("Load pretrain model")
        if hasattr(model._network, "module"):
            model._network.module.load_state_dict(torch.load(model_path))
        else:
            model._network.load_state_dict(torch.load(model_path))
    else:
        pretrain(cfg, ex, model, device, train_loader, test_loader, model_path)

@ex.command
def test(_run, _rnd, _seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    ex.logger.info(cfg)

    trial_i = cfg['trial']
    cfg.data_folder = osp.join(base_dir, "data")
    inc_dataset = factory.get_data(cfg, trial_i)
    # inc_dataset._current_task = task_i
    # train_loader = inc_dataset._get_loader(inc_dataset.data_cur, inc_dataset.targets_cur)
    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset, device)
    model._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)
    for task_i in range(inc_dataset.n_tasks):

        task_info, train_loader, val_loader, test_loader, x_train, y_train = inc_dataset.new_task(cfg['sample_rate'],
                                                                                                  device)

        model.set_task_info(
            task=task_info["task"],
            # total_n_classes=task_info["max_class"],
            # increment=task_info["increment"],
            task_size=task_info["task_size"],
            tax_tree=task_info["partial_tree"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
            acc_detail_path=cfg.acc_detail_path
        )


        model.before_task(task_i, inc_dataset)
        state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
        model._parallel_network.load_state_dict(state_dict)
        classifier_parameter = model._parallel_network.module.classifier.state_dict()
        print(classifier_parameter.keys())
        for i in classifier_parameter:
            print(i)
            print(classifier_parameter[i])
            print(classifier_parameter[i].size())
        model.eval()

        ypred, ytrue = model.eval_task(test_loader)
        model.save_acc_detail_info('test')


if __name__ == "__main__":
    # ex.add_config('./codes/base/configs/default.yaml')
    ex.add_config("./configs/default.yaml")
    ex.run_commandline()
