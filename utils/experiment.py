from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch

from datasets.video_utils import SAMPLING_POLICY, sha256_file


DEFAULT_ANNOTATIONS = {
    "VCM": ("vcm", "VCM.json"),
    "BUPT": ("bupt", "BUPT.json"),
}
PROTOCOL_FILES = ("train.txt", "train_auxiliary.txt", "query.txt", "gallery.txt")


def _count_test_queries(records: object) -> int:
    if not isinstance(records, list):
        raise RuntimeError("Caption annotation must contain a JSON list")
    count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Caption annotation record {index} is not an object")
        count += int(record.get("split") == "test" and record.get("is_query") is True)
    return count


def normalize_loaded_config(args) -> None:
    """Fill options absent from historical IRRA/VCM configuration files."""
    if not hasattr(args, "num_frames"):
        args.num_frames = 6
    num_frames = int(args.num_frames)
    defaults = {
        "annotation_file": "",
        "annotation_source_path": "",
        "annotation_snapshot_path": "",
        "annotation_sha256": "",
        "bidirectional_eval": True,
        "caption_source": "legacy",
        "device": "auto",
        "fake_ir_policy": "not_applicable",
        "gallery_mode": "mixed",
        "protocol_dir": "",
        "protocol_sha256": {},
        "sampling_policy": SAMPLING_POLICY,
        "seed": 1,
        "sequence_length": num_frames,
        "source_commit": "unknown",
        "source_dirty": True,
        "train_modalities": "rgb+real_ir",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, "img_size"):
        args.img_size = (384, 128)
    args.img_size = tuple(args.img_size)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _git_state(repo_dir: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _annotation_source(args) -> Path:
    if args.dataset_name not in DEFAULT_ANNOTATIONS:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    if args.annotation_file:
        return Path(args.annotation_file).expanduser().resolve()
    dataset_dir, filename = DEFAULT_ANNOTATIONS[args.dataset_name]
    root = Path(args.root_dir).expanduser().resolve()
    if root.name.casefold() == dataset_dir.casefold():
        return (root / filename).resolve()
    return (root / dataset_dir / filename).resolve()


def snapshot_annotation(args, output_dir: str | Path, repo_dir: str | Path) -> None:
    if getattr(args, "caption_source", "json") != "json":
        raise ValueError(
            "Formal training requires --caption_source json so its query set can be snapshotted"
        )
    source = _annotation_source(args)
    if not source.is_file():
        raise FileNotFoundError(f"Caption annotation not found: {source}")
    destination_dir = Path(output_dir) / "annotations"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    source_hash = sha256_file(source)
    if not destination.is_file():
        shutil.copy2(source, destination)
    snapshot_hash = sha256_file(destination)
    if source_hash != snapshot_hash:
        raise RuntimeError("Annotation snapshot checksum differs from its source")
    records = json.loads(destination.read_text(encoding="utf-8"))
    num_queries = _count_test_queries(records)
    if num_queries == 0:
        raise RuntimeError("Annotation snapshot contains no query records")

    source_commit, source_dirty = _git_state(Path(repo_dir).resolve())
    args.annotation_source_path = str(source)
    args.annotation_snapshot_path = str(destination.resolve())
    args.annotation_file = args.annotation_snapshot_path
    args.annotation_sha256 = snapshot_hash
    args.num_queries = int(num_queries)
    args.source_commit = source_commit
    args.source_dirty = source_dirty
    args.sampling_policy = SAMPLING_POLICY
    args.sequence_length = int(args.num_frames)
    args.train_modalities = "rgb+real_ir"
    args.fake_ir_policy = "excluded" if args.dataset_name == "BUPT" else "not_applicable"
    args.checkpoint_gallery = "mixed"
    args.protocol_dir = ""
    args.protocol_sha256 = {}

    if args.dataset_name == "BUPT":
        dataset_dir, _filename = DEFAULT_ANNOTATIONS[args.dataset_name]
        root = Path(args.root_dir).expanduser().resolve()
        dataset_root = root if root.name.casefold() == dataset_dir else root / dataset_dir
        protocol_snapshot_dir = Path(output_dir) / "protocols"
        protocol_snapshot_dir.mkdir(parents=True, exist_ok=True)
        protocol_hashes = {}
        for filename in PROTOCOL_FILES:
            protocol_source = dataset_root / filename
            if not protocol_source.is_file():
                raise FileNotFoundError(f"Dataset protocol not found: {protocol_source}")
            protocol_destination = protocol_snapshot_dir / filename
            source_protocol_hash = sha256_file(protocol_source)
            if not protocol_destination.is_file():
                shutil.copy2(protocol_source, protocol_destination)
            snapshot_protocol_hash = sha256_file(protocol_destination)
            if source_protocol_hash != snapshot_protocol_hash:
                raise RuntimeError(
                    f"Protocol snapshot checksum differs for {filename}"
                )
            protocol_hashes[filename] = snapshot_protocol_hash
        args.protocol_dir = str(protocol_snapshot_dir.resolve())
        args.protocol_sha256 = protocol_hashes


def verify_annotation_snapshot(args) -> None:
    snapshot = Path(args.annotation_file).expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"Saved annotation snapshot is missing: {snapshot}")
    observed_hash = sha256_file(snapshot)
    expected_hash = str(args.annotation_sha256)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Annotation snapshot checksum changed: {observed_hash} != {expected_hash}"
        )
    records = json.loads(snapshot.read_text(encoding="utf-8"))
    observed_queries = _count_test_queries(records)
    if observed_queries != int(args.num_queries):
        raise RuntimeError(
            f"Annotation query count changed: {observed_queries} != {args.num_queries}"
        )
    protocol_hashes = dict(getattr(args, "protocol_sha256", {}))
    if args.dataset_name == "BUPT":
        if set(protocol_hashes) != set(PROTOCOL_FILES):
            raise RuntimeError("Saved BUPT configuration has incomplete protocol hashes")
        protocol_dir = Path(args.protocol_dir).expanduser().resolve()
        for filename, expected_protocol_hash in protocol_hashes.items():
            protocol_path = protocol_dir / filename
            if not protocol_path.is_file():
                raise FileNotFoundError(
                    f"Saved BUPT protocol snapshot is missing: {protocol_path}"
                )
            observed_protocol_hash = sha256_file(protocol_path)
            if observed_protocol_hash != str(expected_protocol_hash):
                raise RuntimeError(
                    f"BUPT protocol snapshot changed for {filename}: "
                    f"{observed_protocol_hash} != {expected_protocol_hash}"
                )


def result_metadata(args, checkpoint_path: str | Path) -> dict[str, object]:
    return {
        "dataset_name": str(args.dataset_name),
        "data_root": str(args.root_dir),
        "annotation_source_path": str(args.annotation_source_path),
        "annotation_snapshot_path": str(args.annotation_snapshot_path),
        "annotation_sha256": str(args.annotation_sha256),
        "configured_num_queries": int(args.num_queries),
        "source_commit": str(args.source_commit),
        "source_dirty_at_training_start": bool(args.source_dirty),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "num_frames": int(args.num_frames),
        "sequence_length": int(getattr(args, "sequence_length", args.num_frames)),
        "sampling_policy": str(args.sampling_policy),
        "checkpoint_selection_gallery": str(
            getattr(args, "checkpoint_gallery", "mixed")
        ),
        "train_caption_mode": str(args.train_caption_mode),
        "train_modalities": str(args.train_modalities),
        "fake_ir_policy": str(args.fake_ir_policy),
        "caption_source": str(args.caption_source),
        "protocol_dir": str(getattr(args, "protocol_dir", "")),
        "protocol_sha256": dict(getattr(args, "protocol_sha256", {})),
        "loss_names": str(args.loss_names),
        "seed": int(args.seed),
        "total_parameters": int(getattr(args, "total_parameters", 0)),
        "trainable_parameters": int(getattr(args, "trainable_parameters", 0)),
    }
