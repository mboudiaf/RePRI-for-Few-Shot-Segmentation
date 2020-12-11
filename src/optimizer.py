import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from typing import List
import torch.nn as nn


def get_optimizer(args: argparse.Namespace,
                  parameters: List[nn.Module]) -> torch.optim.Optimizer:
    if args.main_optim == 'SGD':
        return torch.optim.SGD(parameters,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay,
                               nesterov=args.nesterov)
    elif args.main_optim == 'Adam':
        return torch.optim.Adam(parameters,
                                weight_decay=args.weight_decay)


def get_scheduler(args: argparse.Namespace,
                  optimizer: torch.optim.Optimizer,
                  batches: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULERS = {'step': StepLR(optimizer, args.lr_stepsize, args.gamma),
                  'multi_step': MultiStepLR(optimizer, milestones=args.milestones,
                                            gamma=args.gamma),
                  'cosine': CosineAnnealingLR(optimizer, batches * args.epochs, eta_min=1e-6),
                  None: None}
    return SCHEDULERS[args.scheduler]