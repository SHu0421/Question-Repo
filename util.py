import logging
import os
import torch.distributed as dist
import pickle
import time

import torch



        
def get_log(file_name):
    logger = logging.getLogger('train_CIFAR10')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()