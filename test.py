from __future__ import annotations

import argparse
from pathlib import Path

from datasets import build_dataloader
from model import build_model
from processor import do_inference
from utils.checkpoint import Checkpointer
from utils.experiment import (
    normalize_loaded_config,
    resolve_device,
    result_metadata,
    verify_annotation_snapshot,
)
from utils.iotools import load_train_configs, write_json
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the video baseline")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--gallery_mode", choices=["rgb", "ir", "mixed"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    cli_args = parser.parse_args()

    args = load_train_configs(cli_args.config_file)
    normalize_loaded_config(args)
    args.training = False
    args.gallery_mode = cli_args.gallery_mode or args.gallery_mode
    if cli_args.device:
        args.device = cli_args.device
    device = resolve_device(args.device)
    args.resolved_device = str(device)
    if args.caption_source == "json":
        verify_annotation_snapshot(args)
    logger = setup_logger("IRRA", args.output_dir, if_train=False)
    logger.info("Evaluation configuration: %s", args)

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    if device.type == "cpu":
        model.float()
    checkpoint_path = Path(
        cli_args.checkpoint or Path(args.output_dir) / "best.pth"
    ).resolve()
    Checkpointer(model).load(str(checkpoint_path))
    model.to(device)
    metrics = do_inference(
        model,
        image_loader,
        text_loader,
        gallery_mode=args.gallery_mode,
        include_reverse=args.bidirectional_eval,
    )
    metrics.update(result_metadata(args, checkpoint_path))
    result_path = Path(args.output_dir) / f"eval_{args.gallery_mode}.json"
    write_json(metrics, result_path)
    logger.info("Saved metrics to %s", result_path)
