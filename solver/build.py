from __future__ import annotations

import logging

import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    logging.getLogger("IRRA.solver").info(
        "Using %.3fx learning rate for the ID classifier",
        args.lr_factor,
    )
    parameter_groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = args.lr
        weight_decay = args.weight_decay
        if "bias" in name:
            learning_rate *= args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in name:
            learning_rate = args.lr * args.lr_factor
        parameter_groups.append(
            {
                "params": [parameter],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )

    if args.optimizer == "SGD":
        return torch.optim.SGD(
            parameter_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    if args.optimizer == "Adam":
        return torch.optim.Adam(
            parameter_groups,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(
            parameter_groups,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
