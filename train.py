from __future__ import annotations

import os
import os.path as path
import random
import time

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.experiment import resolve_device, snapshot_annotation
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_args()
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = num_gpus > 1
    device = resolve_device(args.device)
    args.resolved_device = str(device)
    if args.distributed:
        if device.type != "cuda":
            raise RuntimeError("Distributed baseline training requires CUDA")
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    set_seed(args.seed + get_rank())

    timestamp_values = [
        time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if get_rank() == 0
        else None
    ]
    if args.distributed:
        torch.distributed.broadcast_object_list(timestamp_values, src=0)
    timestamp = timestamp_values[0]
    if timestamp is None:
        raise RuntimeError("Failed to synchronize the distributed run timestamp")
    args.output_dir = path.join(
        args.output_dir, args.dataset_name, f"{timestamp}_{args.name}"
    )
    if get_rank() == 0:
        snapshot_annotation(args, args.output_dir, repo_dir=path.dirname(__file__))
    synchronize()
    if get_rank() != 0:
        snapshot_annotation(args, args.output_dir, repo_dir=path.dirname(__file__))
    synchronize()
    logger = setup_logger(
        "IRRA",
        save_dir=args.output_dir,
        if_train=True,
        distributed_rank=get_rank(),
    )
    logger.info("Using %d GPU process(es), device=%s", num_gpus, device)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    if device.type == "cpu":
        model.float()
    model.to(device)
    args.total_parameters = sum(parameter.numel() for parameter in model.parameters())
    args.trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if get_rank() == 0:
        save_train_configs(args.output_dir, args)
    synchronize()
    logger.info("Configuration: %s", args)
    logger.info(
        "Total parameters: %.3fM",
        args.total_parameters / 1_000_000,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    is_master = get_rank() == 0
    checkpointer = Checkpointer(
        model, optimizer, scheduler, args.output_dir, is_master
    )
    evaluator = Evaluator(val_img_loader, val_txt_loader, gallery_mode="mixed")

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = int(checkpoint["epoch"]) + 1
    do_train(
        start_epoch,
        args,
        model,
        train_loader,
        evaluator,
        optimizer,
        scheduler,
        checkpointer,
    )
