from __future__ import annotations

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="NuanceID shared-text video baseline")

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--name", default="shared_text_baseline")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--val_dataset", choices=["val", "test"], default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt_file", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--img_aug", action="store_true")
    parser.add_argument("--img_size", type=int, nargs=2, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)

    parser.add_argument("--loss_names", default="sdm+id")
    parser.add_argument("--id_loss_weight", type=float, default=1.0)

    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="Adam")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr_factor", type=float, default=5.0)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--milestones", type=int, nargs="+", default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_method", choices=["constant", "linear"], default="linear")
    parser.add_argument(
        "--lrscheduler",
        choices=["step", "exp", "poly", "cosine", "linear"],
        default="cosine",
    )
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9)

    parser.add_argument("--dataset_name", choices=["VCM", "BUPT"], required=True)
    parser.add_argument("--root_dir", default="/data/ydl/datasets")
    parser.add_argument(
        "--annotation_file",
        default="",
        help="Caption JSON. Defaults to <root>/<dataset>/<VCM|BUPT>.json.",
    )
    parser.add_argument(
        "--caption_source",
        choices=["json", "legacy"],
        default="json",
        help="Formal training uses json; legacy is reserved for old saved test configs.",
    )
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--train_caption_mode", choices=["single", "double"], default="single")
    parser.add_argument("--gallery_mode", choices=["rgb", "ir", "mixed"], default="mixed")
    parser.add_argument("--sampler", choices=["identity", "random"], default="random")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.set_defaults(training=True, bidirectional_eval=True)
    parser.add_argument(
        "--no_reverse_eval",
        dest="bidirectional_eval",
        action="store_false",
        help="Disable video-to-text metrics.",
    )

    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    if args.num_frames < 1:
        parser.error("--num_frames must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.num_epoch < 1:
        parser.error("--num_epoch must be at least 1")
    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs cannot be negative")
    if args.batch_size < 1 or args.test_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.log_period < 1 or args.eval_period < 1:
        parser.error("log/evaluation periods must be positive")
    if args.caption_source != "json":
        parser.error("Training requires --caption_source json")
    if args.dataset_name == "BUPT" and args.train_caption_mode != "single":
        parser.error("BUPT currently provides one caption per RGB camera")
    return args
