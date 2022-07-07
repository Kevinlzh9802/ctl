"""
@Author : Yan Shipeng, Xie Jiangwei
@Contact: yanshp@shanghaitech.edu.cn, xiejw@shanghaitech.edu.cn
"""

import sys
import os
import os.path as osp
import copy
import time
import logging
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import torch

from sacred import Experiment

repo_name = 'ctl'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.append(base_dir)
ex = Experiment(base_dir=base_dir, save_git_info=False)

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
    factory.set_acc_detail_path(cfg, 'train')
    trial_i = cfg['trial']

    inc_dataset = factory.get_data(cfg)
    inc_dataset.train = True
    ex.logger.info("curriculum")
    ex.logger.info(inc_dataset.curriculum)

    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)

    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg["exp"]["ckptdir"]

    results = results_utils.get_template_results(cfg)

    for ti in range(inc_dataset.n_tasks):
        model.before_task()
        print(f'task {model._task}')
        model.train_task()
        model.after_task()
        # TODO: fix the name
        model.save_acc_detail_info('tbd')
        ypred, ytrue = model.eval_task(model._test_loader)
        # TODO: fix the name
        model.save_acc_detail_info('tbd2')

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
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    if cfg["device_auto_detect"]:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    else:
        factory.set_device(cfg)
    factory.set_acc_detail_path(cfg, 'test')
    cfg.data_folder = osp.join(base_dir, "data")
    ex.logger.info(cfg)

    inc_dataset = factory.get_data(cfg)
    inc_dataset.test = True
    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)

    test_results = results_utils.get_template_results(cfg)
    for task_i in range(inc_dataset.n_tasks):
        model.before_task()
        state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
        model._parallel_network.load_state_dict(state_dict)
        model.eval()

        # Build exemplars
        ypred, ytrue = model.eval_task(model._test_loader)
        # TODO: fix the name
        model.save_acc_detail_info('test1')


if __name__ == "__main__":
    # ex.add_config('./codes/base/configs/default.yaml')
    ex.add_config("./configs/default.yaml")
    ex.run_commandline()
