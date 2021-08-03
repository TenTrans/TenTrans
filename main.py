

from src.tasks import task_builder
import yaml
import logging
import os
import argparse
import torch
from src.utils.utility import make_logger, set_seed, load_config, parase_config, str2bool, log_config
from src.task_manger import TaskManger
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--local_rank")
parser.add_argument("--multi_gpu", type=str2bool, nargs='?', default=False)

args = parser.parse_args()
config = load_config(args.config)
config['multi_gpu'] = args.multi_gpu


def init_exp():
    parase_config(config)
    os.makedirs(config['dumpdir'], exist_ok=True)


def main():
    local_rank = None
    init_exp()
    make_logger(config['dumpdir'], "train")

    if config['multi_gpu']:
        torch.distributed.init_process_group(backend='nccl', init_method="env://")
        if torch.distributed.get_rank() == 0:
            log_config(config)
        local_rank = dist.get_rank()
    else:
        log_config(config)
    task_manger = TaskManger(config, local_rank=local_rank)
    task_manger.train()

main()