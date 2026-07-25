from __future__ import annotations

import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, ImageTextDataset, TextDataset
from .bupt import BUPT
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from .vcm import VCM


DATASET_FACTORY = {"VCM": VCM, "BUPT": BUPT}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    if not is_train:
        return transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    operations = [
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(0.5),
    ]
    if aug:
        operations.extend(
            [
                transforms.Pad(10),
                transforms.RandomCrop((height, width)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if aug:
        operations.append(transforms.RandomErasing(scale=(0.02, 0.4), value=mean))
    return transforms.Compose(operations)


def collate(batch):
    keys = set().union(*(sample.keys() for sample in batch))
    tensor_batch = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], int):
            tensor_batch[key] = torch.tensor(values, dtype=torch.long)
        elif torch.is_tensor(values[0]):
            tensor_batch[key] = torch.stack(values)
        else:
            raise TypeError(f"Unsupported batch value for {key}: {type(values[0])}")
    return tensor_batch


def build_dataset(args):
    if args.dataset_name not in DATASET_FACTORY:
        raise ValueError(
            f"Unsupported dataset {args.dataset_name!r}; choose from {sorted(DATASET_FACTORY)}"
        )
    common = {
        "root": args.root_dir,
        "annotation_file": getattr(args, "annotation_file", ""),
        "num_frames": args.num_frames,
        "train_caption_mode": args.train_caption_mode,
    }
    if args.dataset_name == "VCM":
        common["caption_source"] = getattr(args, "caption_source", "legacy")
    elif getattr(args, "caption_source", "json") != "json":
        raise ValueError("BUPT supports only JSON captions")
    if args.dataset_name == "BUPT":
        common["protocol_dir"] = getattr(args, "protocol_dir", "") or None
    return DATASET_FACTORY[args.dataset_name](**common)


def _eval_split(dataset, split_name, gallery_mode):
    return dataset.get_eval_split(split_name, gallery_mode)


def _build_eval_loaders(args, dataset, split_name, gallery_mode, transform):
    split = _eval_split(dataset, split_name, gallery_mode)
    image_set = ImageDataset(
        split["image_pids"],
        split["img_paths"],
        split["image_modalities"],
        transform=transform,
    )
    text_set = TextDataset(
        split["caption_pids"],
        split["captions"],
        text_length=args.text_length,
    )
    image_loader = DataLoader(
        image_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.resolved_device).startswith("cuda"),
    )
    text_loader = DataLoader(
        text_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.resolved_device).startswith("cuda"),
    )
    return image_loader, text_loader


def build_dataloader(args, custom_transforms=None):
    logger = logging.getLogger("IRRA.dataset")
    dataset = build_dataset(args)
    num_classes = len(dataset.train_id_container)
    args.dataset_metadata = dict(dataset.metadata)
    args.num_train_ids = len(dataset.train_id_container)
    args.num_train_tracklets = len(dataset.train_tracklets)
    args.num_train_samples = len(dataset.train)
    args.num_queries = len(dataset.queries)
    args.num_gallery_rgb = len(dataset.gallery_rgb)
    args.num_gallery_ir = len(dataset.gallery_ir)
    args.num_gallery_mixed = len(dataset.gallery_mixed)
    if getattr(args, "caption_source", "json") == "legacy":
        args.annotation_source_path = "legacy-directory-captions"
        args.annotation_snapshot_path = ""
        args.annotation_sha256 = dataset.metadata["annotation_sha256"]
    logger.info("Dataset metadata: %s", dataset.metadata)

    if args.training:
        train_transform = build_transforms(args.img_size, args.img_aug, is_train=True)
        eval_transform = build_transforms(args.img_size, is_train=False)
        train_set = ImageTextDataset(
            dataset.train,
            transform=train_transform,
            text_length=args.text_length,
        )

        sampler = None
        shuffle = False
        if args.sampler == "identity":
            if args.distributed:
                sampler = RandomIdentitySampler_DDP(
                    dataset.train,
                    args.batch_size * get_world_size(),
                    args.num_instance,
                    seed=args.seed,
                )
            else:
                sampler = RandomIdentitySampler(
                    dataset.train,
                    args.batch_size,
                    args.num_instance,
                    seed=args.seed,
                )
        elif args.sampler == "random":
            if args.distributed:
                sampler = DistributedSampler(train_set, shuffle=True)
            else:
                shuffle = True
        else:
            raise ValueError(f"Unsupported sampler: {args.sampler}")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=str(args.resolved_device).startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )
        split_name = "val" if args.val_dataset == "val" else "test"
        val_img_loader, val_txt_loader = _build_eval_loaders(
            args, dataset, split_name, "mixed", eval_transform
        )
        return train_loader, val_img_loader, val_txt_loader, num_classes

    test_transform = custom_transforms or build_transforms(args.img_size, is_train=False)
    test_img_loader, test_txt_loader = _build_eval_loaders(
        args, dataset, "test", args.gallery_mode, test_transform
    )
    return test_img_loader, test_txt_loader, num_classes
