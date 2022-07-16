"""
@Author : Yan Shipeng, Xie Jiangwei
@Contact: yanshp@shanghaitech.edu.cn, xiejw@shanghaitech.edu.cn
"""

import sys
import os
import os.path as osp
import copy
import time
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
    utils.set_save_paths(cfg)

    if exp_id is None:
        exp_id = -1
    #     cfg.exp.savedir = "./logs"
    logger = utils.make_logger(f"{mode}", savedir=cfg['log_path'])

    # Tensorboard
    # exp_name = f'{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else f'../inbox/{cfg["exp"]["name"]}'
    # tensorboard_dir = cfg["exp"]["tensorboard_dir"] + f"/{exp_name}"

    tensorboard = SummaryWriter(cfg['tensorboard_path'])
    return cfg, logger, tensorboard


@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
    ex.logger.info(cfg)

    # adjust config
    cfg.data_folder = osp.join(base_dir, "data")
    if cfg["device_auto_detect"]:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    else:
        factory.set_device(cfg)
    cfg["exp"]["mode_train"] = True

    start_time = time.time()
    _train(cfg, _run, ex, tensorboard)
    ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))
    with open('delete_warning.txt', 'w') as dw:
        dw.write('This is a fully conducted experiment without errors and interruptions. Please be careful as deleting'
                 'it may lose important data and results. See log file for configuration details.')


def _train(cfg, _run, ex, tensorboard):

    inc_dataset = factory.get_data(cfg)
    ex.logger.info("curriculum")
    ex.logger.info(inc_dataset.curriculum)

    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)

    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg['model_path']

    results = results_utils.get_template_results(cfg)

    # for task_i in range(inc_dataset.n_tasks):
    for task_i in range(1):
        model.new_task()
        model.before_task(inc_dataset)

        if cfg['retrain_from_task0']:
            model.train_task()
        else:
            # state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
            state_dict = torch.load(f"{cfg['pre_train_model_path']}/step{task_i}.ckpt")
            model._parallel_network.load_state_dict(state_dict)

        model.save_acc_detail_info('train_with_step')

        model.eval_task(model._cur_test_loader)
        model.after_task(inc_dataset)
        model.save_acc_detail_info('after_train')

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
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    if cfg["device_auto_detect"]:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    else:
        factory.set_device(cfg)
    cfg["exp"]["mode_train"] = False
    ex.logger.info(cfg)

    cfg.data_folder = osp.join(base_dir, "data")
    inc_dataset = factory.get_data(cfg)
    # inc_dataset._current_task = task_i
    # train_loader = inc_dataset._get_loader(inc_dataset.data_cur, inc_dataset.targets_cur)
    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)
    # model._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)
    for task_i in range(inc_dataset.n_tasks):

        model.new_task()
        model.before_task(inc_dataset)
        state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
        # state_dict = torch.load(f'../../../cyz_codes/ctl/codes/base/ckpts/step{task_i}.ckpt')
        model._parallel_network.load_state_dict(state_dict)
        model.eval()
        model.eval_task(model._cur_test_loader)
        # model.save_acc_detail_info('test')


if __name__ == "__main__":
    # ex.add_config('./codes/base/configs/default.yaml')
    ex.add_config("./configs/default.yaml")
    ex.run_commandline()
