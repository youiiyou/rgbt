#!/usr/bin/env python3
"""Validate the fixed NuanceID baseline contract against an external dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate
from datasets.bupt import BUPT
from datasets.vcm import VCM


EXPECTED = {
    "VCM": {
        "annotation_sha256": "db2d5c9eec7f4c79911ae892bb4e7125e8b94f9ab56f60087f7f7517efccd3fe",
        "train_ids": 500,
        "test_ids": 427,
        "train_tracklets": 2961,
        "train_samples": 2961,
        "queries": 1261,
        "gallery_rgb": 1261,
        "gallery_ir": 1261,
        "gallery_mixed": 2522,
    },
    "BUPT": {
        "annotation_sha256": "81e46435b916f361b00a4976736983622fd29b660a89d84569304e55a74180e5",
        "train_ids": 2004,
        "test_ids": 1076,
        "train_tracklets": 9008,
        "train_samples": 9008,
        "queries": 1076,
        "gallery_rgb": 2422,
        "gallery_ir": 2422,
        "gallery_mixed": 4844,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(EXPECTED))
    parser.add_argument("--root", default="/data/ydl/datasets")
    parser.add_argument("--annotation-file", default="")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--load-batch", action="store_true")
    parser.add_argument(
        "--allow-count-changes",
        action="store_true",
        help="Report observed values without enforcing the current official counts/hash.",
    )
    return parser.parse_args()


def observed_contract(dataset) -> dict[str, object]:
    return {
        "annotation_sha256": dataset.metadata["annotation_sha256"],
        "train_ids": len(dataset.train_id_container),
        "test_ids": len(dataset.test_id_container),
        "train_tracklets": len(dataset.train_tracklets),
        "train_samples": len(dataset.train),
        "queries": len(dataset.queries),
        "gallery_rgb": len(dataset.gallery_rgb),
        "gallery_ir": len(dataset.gallery_ir),
        "gallery_mixed": len(dataset.gallery_mixed),
    }


def validate_shared_queries(dataset) -> None:
    reference_pids = dataset.test_mixed["caption_pids"]
    reference_captions = dataset.test_mixed["captions"]
    for mode in ("rgb", "ir"):
        split = getattr(dataset, f"test_{mode}")
        if split["caption_pids"] != reference_pids:
            raise RuntimeError(f"{mode} gallery changed the query identity list")
        if split["captions"] != reference_captions:
            raise RuntimeError(f"{mode} gallery changed the query caption list")


def validate_paths(dataset, num_frames: int) -> None:
    tracklets = dataset.train_tracklets + dataset.gallery_tracklets
    for tracklet in tracklets:
        paths = list(tracklet["img_paths"])
        if len(paths) != num_frames:
            raise RuntimeError(
                f"Expected {num_frames} sampled frames, got {len(paths)}: {tracklet}"
            )
        if dataset.metadata["dataset_name"] == "BUPT":
            if any("/FakeIR/" in Path(path).as_posix() for path in paths):
                raise RuntimeError("BUPT FakeIR leaked into the baseline")


def load_one_batch(dataset, num_frames: int) -> list[int]:
    train_set = ImageTextDataset(
        dataset.train,
        transform=build_transforms((384, 128), is_train=False),
    )
    loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    expected = [2, num_frames, 3, 384, 128]
    shape = list(batch["images"].shape)
    if shape != expected:
        raise RuntimeError(f"Unexpected image tensor shape: {shape} != {expected}")
    if list(batch["caption_ids"].shape) != [2, 77]:
        raise RuntimeError("Unexpected caption tensor shape")
    if list(batch["modalities"].shape) != [2]:
        raise RuntimeError("Unexpected modality tensor shape")
    return shape


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    common = {
        "root": args.root,
        "annotation_file": args.annotation_file or None,
        "verbose": False,
        "num_frames": args.num_frames,
        "train_caption_mode": "single",
    }
    if args.dataset == "VCM":
        dataset = VCM(caption_source="json", **common)
    else:
        dataset = BUPT(**common)

    observed = observed_contract(dataset)
    if not args.allow_count_changes and observed != EXPECTED[args.dataset]:
        raise RuntimeError(
            "Official dataset contract changed:\n"
            f"expected={json.dumps(EXPECTED[args.dataset], indent=2)}\n"
            f"observed={json.dumps(observed, indent=2)}"
        )
    validate_shared_queries(dataset)
    validate_paths(dataset, args.num_frames)

    report = {
        "dataset": args.dataset,
        "root": str(Path(args.root).expanduser().resolve()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": observed,
        "metadata": dataset.metadata,
    }
    if args.load_batch:
        report["batch_image_shape"] = load_one_batch(dataset, args.num_frames)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
