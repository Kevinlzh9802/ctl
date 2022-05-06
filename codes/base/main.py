import sys
import os
import os.path as osp
import torch
import torchvision

repo_name = 'ctl'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.append(base_dir)

from inclearn.tools import factory, results_utils, utils
from inclearn.tools.metrics import IncConfusionMeter
from easydict import EasyDict as edict


from sacred import Experiment
ex = Experiment(base_dir=base_dir)

tasks = [1, 2, 3]
trial = 2

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


def train():
#     cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
#     model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    
    for ti in range(len(tasks)):
        print('task ' + str(ti) + ':, dataset: ')
    
#         model.before_task(ti, inc_dataset)
#         model.train_task(train_loader, val_loader)
#         model.after_task(task_i, inc_dataset)
train()
print('666')
