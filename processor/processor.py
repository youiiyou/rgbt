from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.iotools import write_json
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
):
    logger = logging.getLogger("IRRA.train")
    device = torch.device(args.resolved_device)
    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }
    writer = SummaryWriter(log_dir=args.output_dir) if get_rank() == 0 else None
    best_r1 = float("-inf")
    best_epoch = None
    training_started = time.time()
    last_completed_epoch = start_epoch - 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, args.num_epoch + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for meter in meters.values():
            meter.reset()
        model.train()
        start_time = time.time()
        iteration_count = 0

        for iteration_count, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            losses = [value for key, value in outputs.items() if key.endswith("_loss")]
            if not losses:
                raise RuntimeError("Model returned no losses")
            total_loss = sum(losses)
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss: {total_loss}")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = batch["images"].shape[0]
            meters["loss"].update(float(total_loss.item()), batch_size)
            for name in ("sdm_loss", "id_loss", "img_acc", "txt_acc"):
                value = outputs.get(name)
                if value is not None:
                    meters[name].update(float(value.detach().item()), batch_size)

            if iteration_count % args.log_period == 0:
                values = ", ".join(
                    f"{name}={meter.avg:.4f}" for name, meter in meters.items()
                    if meter.count
                )
                logger.info(
                    "Epoch[%d] Iteration[%d/%d] %s lr=%.2e",
                    epoch,
                    iteration_count,
                    len(train_loader),
                    values,
                    optimizer.param_groups[0]["lr"],
                )

        if iteration_count == 0:
            raise RuntimeError("Training loader produced no batches")
        scheduler.step()
        last_completed_epoch = epoch
        if get_rank() == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d done: %.3fs/batch, %.1f samples/s",
                epoch,
                elapsed / iteration_count,
                meters["loss"].count / elapsed,
            )
            if writer is not None:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("temperature", float(outputs["temperature"]), epoch)
                for name, meter in meters.items():
                    if meter.count:
                        writer.add_scalar(name, meter.avg, epoch)

        if epoch % args.eval_period == 0:
            if get_rank() == 0:
                eval_model = model.module if hasattr(model, "module") else model
                metrics = evaluator.eval(
                    eval_model.eval(), include_reverse=args.bidirectional_eval
                )
                r1 = float(metrics["R1"])
                if math.isfinite(r1) and r1 > best_r1:
                    best_r1 = r1
                    best_epoch = epoch
                    checkpointer.save("best", epoch=epoch, best_r1=best_r1)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            synchronize()

    if writer is not None:
        writer.close()
    if get_rank() == 0:
        checkpointer.save(
            "last",
            epoch=args.num_epoch,
            best_r1=best_r1,
            best_epoch=best_epoch,
        )
        training_summary = {
            "epochs_requested": int(args.num_epoch),
            "last_completed_epoch": int(last_completed_epoch),
            "elapsed_seconds": float(time.time() - training_started),
            "best_mixed_gallery_r1": best_r1 if math.isfinite(best_r1) else None,
            "best_epoch": best_epoch,
            "num_train_samples": int(args.num_train_samples),
            "batch_size_per_process": int(args.batch_size),
            "world_size": int(torch.distributed.get_world_size())
            if torch.distributed.is_initialized()
            else 1,
            "total_parameters": int(args.total_parameters),
            "trainable_parameters": int(args.trainable_parameters),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda"
                else None
            ),
        }
        write_json(training_summary, Path(args.output_dir) / "training_summary.json")
        logger.info("Best mixed-gallery R1 %.3f at epoch %s", best_r1, best_epoch)


def do_inference(
    model,
    test_img_loader,
    test_txt_loader,
    gallery_mode="mixed",
    include_reverse=True,
):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inference")
    evaluator = Evaluator(
        test_img_loader, test_txt_loader, gallery_mode=gallery_mode
    )
    return evaluator.eval(model.eval(), include_reverse=include_reverse)
