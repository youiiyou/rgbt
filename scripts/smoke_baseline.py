#!/usr/bin/env python3
"""Run one real CLIP forward/backward batch for either baseline dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate
from datasets.bupt import BUPT
from datasets.vcm import VCM
from model import build_model
from utils.experiment import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["VCM", "BUPT"])
    parser.add_argument("--root", default="/data/ydl/datasets")
    parser.add_argument("--annotation-file", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    device = resolve_device(cli.device)
    common = {
        "root": cli.root,
        "annotation_file": cli.annotation_file or None,
        "verbose": False,
        "num_frames": 6,
        "train_caption_mode": "single",
    }
    dataset = (
        VCM(caption_source="json", **common)
        if cli.dataset == "VCM"
        else BUPT(**common)
    )
    train_set = ImageTextDataset(
        dataset.train,
        transform=build_transforms((384, 128), is_train=False),
    )
    modality_indices = {}
    for index, sample in enumerate(dataset.train):
        modality_indices.setdefault(int(sample[4]), index)
    required_modalities = {0, 1}
    if set(modality_indices) != required_modalities:
        raise RuntimeError(
            f"Smoke data must contain RGB and IR samples, got {set(modality_indices)}"
        )
    selected_indices = [modality_indices[0]]
    if cli.batch_size > 1:
        selected_indices.append(modality_indices[1])
    remaining = [
        index for index in range(len(dataset.train)) if index not in selected_indices
    ]
    selected_indices.extend(remaining[: max(0, cli.batch_size - len(selected_indices))])
    loader = DataLoader(
        Subset(train_set, selected_indices),
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}

    model_args = SimpleNamespace(
        loss_names="sdm+id",
        pretrain_choice="ViT-B/16",
        img_size=(384, 128),
        stride_size=16,
        num_frames=6,
        temperature=0.02,
        id_loss_weight=1.0,
    )
    model = build_model(model_args, len(dataset.train_id_container))
    if device.type == "cpu":
        model.float()
    model.to(device).train()
    outputs = model(batch)
    losses = [value for key, value in outputs.items() if key.endswith("_loss")]
    total_loss = sum(losses)
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Non-finite smoke loss: {total_loss}")
    total_loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise FloatingPointError("Smoke test produced missing or non-finite gradients")

    report = {
        "dataset": cli.dataset,
        "device": str(device),
        "image_shape": list(batch["images"].shape),
        "caption_shape": list(batch["caption_ids"].shape),
        "modalities": batch["modalities"].detach().cpu().tolist(),
        "losses": {
            key: float(value.detach().cpu().item())
            for key, value in outputs.items()
            if key.endswith("_loss")
        },
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
