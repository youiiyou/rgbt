from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence


RGB_MODALITY = 0
IR_MODALITY = 1
MODALITY_NAME_TO_ID = {"rgb": RGB_MODALITY, "ir": IR_MODALITY}
MODALITY_ID_TO_NAME = {value: key for key, value in MODALITY_NAME_TO_ID.items()}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDED_IMAGE_NAMES = {
    "mosaic_for_caption.jpg",
}
SAMPLING_POLICY = "uniform_segment_midpoint_repeat_short"


class DatasetContractError(RuntimeError):
    """Raised when an external dataset violates the baseline protocol."""


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
        if part
    )


def sorted_directories(path: Path) -> list[Path]:
    if not path.is_dir():
        raise DatasetContractError(f"Missing directory: {path}")
    try:
        with os.scandir(path) as entries:
            directories = [Path(entry.path) for entry in entries if entry.is_dir()]
    except OSError as error:
        raise DatasetContractError(f"Cannot scan directory {path}: {error}") from error
    return sorted(directories, key=lambda entry: natural_key(entry.name))


def resolve_dataset_root(root: str | Path, dataset_dir: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    if root_path.name.casefold() == dataset_dir.casefold():
        return root_path
    return (root_path / dataset_dir).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_caption_records(path: str | Path) -> list[dict[str, object]]:
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise DatasetContractError(f"Missing caption annotation: {annotation_path}")

    try:
        records = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetContractError(
            f"Cannot read caption annotation {annotation_path}: {error}"
        ) from error

    if not isinstance(records, list) or not records:
        raise DatasetContractError(
            f"Caption annotation must be a non-empty list: {annotation_path}"
        )

    required = {
        "split",
        "id",
        "file_path",
        "captions",
        "camera",
        "modality",
        "media_type",
        "is_query",
    }
    seen: set[tuple[str, int, str]] = set()
    normalized: list[dict[str, object]] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise DatasetContractError(f"Annotation record {index} is not an object")
        missing = required - set(raw_record)
        if missing:
            raise DatasetContractError(
                f"Annotation record {index} is missing fields: {sorted(missing)}"
            )

        if not isinstance(raw_record["split"], str):
            raise DatasetContractError(f"Annotation record {index} has a non-string split")
        if not isinstance(raw_record["id"], int) or isinstance(raw_record["id"], bool):
            raise DatasetContractError(f"Annotation record {index} has a non-integer ID")
        if not isinstance(raw_record["file_path"], str) or not raw_record["file_path"]:
            raise DatasetContractError(f"Annotation record {index} has an invalid file_path")
        if not isinstance(raw_record["camera"], str):
            raise DatasetContractError(f"Annotation record {index} has a non-string camera")
        if not isinstance(raw_record["is_query"], bool):
            raise DatasetContractError(f"Annotation record {index} has a non-boolean is_query")
        split = raw_record["split"]
        pid = raw_record["id"]
        camera = raw_record["camera"]
        captions = raw_record["captions"]
        if split not in {"train", "train_auxiliary", "test"}:
            raise DatasetContractError(f"Invalid annotation split {split!r}")
        if not camera:
            raise DatasetContractError(f"Empty camera in annotation record {index}")
        if raw_record["modality"] != "rgb":
            raise DatasetContractError(
                f"Captions must be RGB-derived, got {raw_record['modality']!r}"
            )
        if raw_record["media_type"] != "video":
            raise DatasetContractError(
                f"Expected video annotation, got {raw_record['media_type']!r}"
            )
        if (
            not isinstance(captions, list)
            or not captions
            or any(not isinstance(caption, str) for caption in captions)
        ):
            raise DatasetContractError(
                f"Annotation record {index} must contain string captions"
            )
        clean_captions = [caption.strip() for caption in captions]
        if any(not caption for caption in clean_captions):
            raise DatasetContractError(
                f"Annotation record {index} contains an empty caption"
            )

        key = (split, pid, camera)
        if key in seen:
            raise DatasetContractError(f"Duplicate annotation key: {key}")
        seen.add(key)
        normalized.append(
            {
                "split": split,
                "id": pid,
                "file_path": raw_record["file_path"],
                "captions": clean_captions,
                "camera": camera,
                "modality": "rgb",
                "media_type": "video",
                "is_query": raw_record["is_query"],
            }
        )
    return normalized


def select_captions(record: dict[str, object], mode: str) -> list[str]:
    captions = list(record["captions"])
    if mode == "single":
        return captions[:1]
    if mode == "double":
        if len(captions) < 2:
            raise DatasetContractError(
                f"double caption mode requested but only one caption exists for "
                f"{record['split']}/{record['id']}/{record['camera']}"
            )
        return captions[:2]
    raise ValueError(f"Unsupported caption mode: {mode}")


def collect_frames(camera_dir: Path, filename_prefix: str | None = None) -> list[str]:
    if not camera_dir.is_dir():
        raise DatasetContractError(f"Missing camera directory: {camera_dir}")
    paths: list[Path] = []
    try:
        entries = os.scandir(camera_dir)
        with entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if name.casefold() in EXCLUDED_IMAGE_NAMES:
                    continue
                if Path(name).suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                if filename_prefix is not None and not name.startswith(filename_prefix):
                    continue
                paths.append(Path(entry.path))
    except OSError as error:
        raise DatasetContractError(
            f"Cannot scan camera directory {camera_dir}: {error}"
        ) from error
    paths.sort(key=lambda path: natural_key(path.name))
    return [str(path) for path in paths]


def uniform_sample_frames(frame_paths: Sequence[str], num_frames: int) -> list[str]:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    total = len(frame_paths)
    if total == 0:
        raise DatasetContractError("Cannot sample an empty tracklet")
    if num_frames == 1:
        return [frame_paths[total // 2]]

    if total >= num_frames:
        indices = []
        for index in range(num_frames):
            start = index * total // num_frames
            end = (index + 1) * total // num_frames
            indices.append((start + end - 1) // 2)
    else:
        indices = [
            round(index * (total - 1) / (num_frames - 1))
            for index in range(num_frames)
        ]
    return [frame_paths[index] for index in indices]


def build_eval_dict(
    gallery: Iterable[dict[str, object]],
    queries: Sequence[dict[str, object]],
) -> dict[str, list[object]]:
    gallery_list = list(gallery)
    gallery_pids = {int(item["pid"]) for item in gallery_list}
    missing_pids = sorted(
        {int(item["pid"]) for item in queries} - gallery_pids
    )
    if missing_pids:
        raise DatasetContractError(
            f"Gallery is missing positive tracklets for query PIDs: {missing_pids[:10]}"
        )
    unsupported_modalities = {
        int(item["modality"])
        for item in gallery_list
        if int(item["modality"]) not in MODALITY_ID_TO_NAME
    }
    if unsupported_modalities:
        raise DatasetContractError(
            f"Gallery contains unsupported modalities: {sorted(unsupported_modalities)}"
        )
    return {
        "image_pids": [int(item["pid"]) for item in gallery_list],
        "img_paths": [list(item["img_paths"]) for item in gallery_list],
        "image_modalities": [int(item["modality"]) for item in gallery_list],
        "caption_pids": [int(item["pid"]) for item in queries],
        "captions": [str(item["caption"]) for item in queries],
    }
