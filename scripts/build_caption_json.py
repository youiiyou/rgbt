#!/usr/bin/env python3
"""Build normalized caption annotations for the VCM and BUPT datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDED_IMAGE_NAMES = {"mosaic_for_caption.jpg"}
DEFAULT_ROOTS = {
    "vcm": Path("/data/ydl/datasets/vcm"),
    "bupt": Path("/data/ydl/datasets/bupt"),
}
DEFAULT_OUTPUT_NAMES = {
    "vcm": "VCM.json",
    "bupt": "BUPT.json",
}
SPLIT_ORDER = ("train", "train_auxiliary", "test")


class BuildError(RuntimeError):
    """Raised when the source data cannot produce a valid annotation file."""


def natural_key(value: str) -> tuple[object, ...]:
    """Sort strings containing numbers in human order (D2 before D10)."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
        if part
    )


def sorted_directories(path: Path) -> list[Path]:
    return sorted((entry for entry in path.iterdir() if entry.is_dir()), key=lambda item: natural_key(item.name))


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise BuildError(f"Missing {description}: {path}")


def read_caption(path: Path) -> str:
    if not path.is_file():
        raise BuildError(f"Missing caption file: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise BuildError(f"Caption is not valid UTF-8: {path}: {error}") from error
    except OSError as error:
        raise BuildError(f"Cannot read caption file: {path}: {error}") from error

    caption = raw_text.strip()
    if not caption:
        raise BuildError(f"Caption is empty: {path}")
    if len(caption.splitlines()) != 1:
        raise BuildError(f"Caption must contain exactly one non-empty line: {path}")
    if "\x00" in caption:
        raise BuildError(f"Caption contains a NUL character: {path}")
    return caption


def require_video_frames(camera_dir: Path) -> None:
    try:
        has_frame = any(
            entry.is_file()
            and entry.name not in EXCLUDED_IMAGE_NAMES
            and entry.suffix.lower() in IMAGE_SUFFIXES
            for entry in camera_dir.iterdir()
        )
    except OSError as error:
        raise BuildError(f"Cannot inspect video directory: {camera_dir}: {error}") from error

    if not has_frame:
        raise BuildError(f"Video directory contains no usable frames: {camera_dir}")


def make_record(
    *,
    split: str,
    pid: int,
    file_path: str,
    captions: list[str],
    camera: str,
    is_query: bool,
) -> dict[str, object]:
    # Dict insertion order is intentional: it defines the serialized schema order.
    return {
        "split": split,
        "id": pid,
        "file_path": file_path,
        "captions": captions,
        "camera": camera,
        "modality": "rgb",
        "media_type": "video",
        "is_query": is_query,
    }


def build_vcm(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    root = root.resolve()

    for source_split, split, is_query in (
        ("Train", "train", False),
        ("Test", "test", True),
    ):
        split_dir = root / source_split
        require_directory(split_dir, f"VCM {source_split} directory")

        for pid_dir in sorted_directories(split_dir):
            if not pid_dir.name.isdigit():
                raise BuildError(f"VCM identity directory is not numeric: {pid_dir}")

            rgb_dir = pid_dir / "rgb"
            require_directory(rgb_dir, "VCM RGB directory")
            camera_dirs = sorted_directories(rgb_dir)
            if not camera_dirs:
                raise BuildError(f"VCM RGB directory contains no camera directories: {rgb_dir}")

            for camera_dir in camera_dirs:
                caption = read_caption(camera_dir / "caption.txt")
                caption_aug = read_caption(camera_dir / "caption_aug.txt")
                if caption == caption_aug:
                    raise BuildError(
                        "VCM caption.txt and caption_aug.txt must be distinct: "
                        f"{camera_dir}"
                    )
                require_video_frames(camera_dir)
                records.append(
                    make_record(
                        split=split,
                        pid=int(pid_dir.name),
                        file_path=camera_dir.relative_to(root).as_posix(),
                        captions=[caption, caption_aug],
                        camera=camera_dir.name,
                        is_query=is_query,
                    )
                )

    if not records:
        raise BuildError(f"VCM contains no caption records: {root}")
    return records


def parse_protocol(path: Path) -> list[tuple[int, str, str, int]]:
    if not path.is_file():
        raise BuildError(f"Missing BUPT protocol file: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise BuildError(f"Protocol file is not valid UTF-8: {path}: {error}") from error
    except OSError as error:
        raise BuildError(f"Cannot read protocol file: {path}: {error}") from error

    rows: list[tuple[int, str, str, int]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            raise BuildError(f"Blank protocol line: {path}:{line_number}")
        fields = line.split()
        if len(fields) != 4:
            raise BuildError(
                f"Expected 4 protocol fields at {path}:{line_number}, got {len(fields)}"
            )
        pid_text, modality, camera, tracklet_text = fields
        if not pid_text.isdigit() or not tracklet_text.isdigit():
            raise BuildError(f"Non-numeric identity or tracklet at {path}:{line_number}")
        if modality not in {"RGB", "IR", "RGB/IR"}:
            raise BuildError(f"Unknown modality {modality!r} at {path}:{line_number}")
        if not camera:
            raise BuildError(f"Empty camera at {path}:{line_number}")
        rows.append((int(pid_text), modality, camera, int(tracklet_text)))

    if not rows:
        raise BuildError(f"BUPT protocol file is empty: {path}")
    return rows


def format_id_sample(ids: Iterable[int]) -> str:
    ordered = sorted(ids)
    sample = ", ".join(str(pid) for pid in ordered[:10])
    if len(ordered) > 10:
        sample += ", ..."
    return sample


def require_disjoint(named_sets: Sequence[tuple[str, set[int]]]) -> None:
    for index, (left_name, left_ids) in enumerate(named_sets):
        for right_name, right_ids in named_sets[index + 1 :]:
            overlap = left_ids & right_ids
            if overlap:
                raise BuildError(
                    f"BUPT identity overlap between {left_name} and {right_name}: "
                    f"{format_id_sample(overlap)}"
                )


def numeric_identity_directories(data_dir: Path) -> dict[int, Path]:
    identity_dirs: dict[int, Path] = {}
    for entry in data_dir.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in identity_dirs:
            raise BuildError(
                f"BUPT has duplicate numeric identity directories: {identity_dirs[pid]} and {entry}"
            )
        identity_dirs[pid] = entry
    if not identity_dirs:
        raise BuildError(f"BUPT DATA contains no numeric identity directories: {data_dir}")
    return identity_dirs


def build_bupt(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    data_dir = root / "DATA"
    require_directory(data_dir, "BUPT DATA directory")

    train_rows = parse_protocol(root / "train.txt")
    auxiliary_rows = parse_protocol(root / "train_auxiliary.txt")
    query_rows = parse_protocol(root / "query.txt")
    gallery_rows = parse_protocol(root / "gallery.txt")

    for split_name, rows, allowed_modalities in (
        ("train", train_rows, {"RGB/IR"}),
        ("train_auxiliary", auxiliary_rows, {"RGB/IR"}),
        ("query", query_rows, {"RGB", "IR"}),
        ("gallery", gallery_rows, {"RGB", "IR"}),
    ):
        invalid = sorted({row[1] for row in rows} - allowed_modalities)
        if invalid:
            raise BuildError(
                f"BUPT {split_name} has unsupported modalities {invalid}; "
                f"expected {sorted(allowed_modalities)}"
            )

    train_ids = {row[0] for row in train_rows}
    auxiliary_ids = {row[0] for row in auxiliary_rows}
    query_ids = {row[0] for row in query_rows}
    gallery_ids = {row[0] for row in gallery_rows}
    if query_ids != gallery_ids:
        only_query = query_ids - gallery_ids
        only_gallery = gallery_ids - query_ids
        raise BuildError(
            "BUPT query/gallery identity sets differ; "
            f"query-only=[{format_id_sample(only_query)}], "
            f"gallery-only=[{format_id_sample(only_gallery)}]"
        )

    require_disjoint(
        (
            ("train", train_ids),
            ("train_auxiliary", auxiliary_ids),
            ("test", query_ids),
        )
    )

    identity_dirs = numeric_identity_directories(data_dir)
    protocol_ids = train_ids | auxiliary_ids | query_ids
    data_ids = set(identity_dirs)
    if data_ids != protocol_ids:
        data_only = data_ids - protocol_ids
        protocol_only = protocol_ids - data_ids
        raise BuildError(
            "BUPT DATA/protocol identity sets differ; "
            f"data-only=[{format_id_sample(data_only)}], "
            f"protocol-only=[{format_id_sample(protocol_only)}]"
        )

    query_keys = [(pid, camera) for pid, _modality, camera, _tracklet in query_rows]
    if len(set(query_keys)) != len(query_keys):
        raise BuildError(
            "BUPT query.txt contains duplicate (identity, camera) pairs that cannot be "
            "represented by a camera-level is_query flag"
        )
    query_key_set = set(query_keys)
    observed_camera_keys: set[tuple[int, str]] = set()
    records: list[dict[str, object]] = []

    split_ids = (
        ("train", train_ids),
        ("train_auxiliary", auxiliary_ids),
        ("test", query_ids),
    )
    for split, ids in split_ids:
        for pid in sorted(ids):
            pid_dir = identity_dirs[pid]
            rgb_dir = pid_dir / "RGB"
            require_directory(rgb_dir, "BUPT RGB directory")
            camera_dirs = sorted_directories(rgb_dir)
            if not camera_dirs:
                raise BuildError(f"BUPT RGB directory contains no camera directories: {rgb_dir}")

            for camera_dir in camera_dirs:
                camera_key = (pid, camera_dir.name)
                observed_camera_keys.add(camera_key)
                caption = read_caption(camera_dir / "caption.txt")
                require_video_frames(camera_dir)
                records.append(
                    make_record(
                        split=split,
                        pid=pid,
                        file_path=camera_dir.relative_to(root).as_posix(),
                        captions=[caption],
                        camera=camera_dir.name,
                        is_query=camera_key in query_key_set,
                    )
                )

    unmatched_queries = query_key_set - observed_camera_keys
    if unmatched_queries:
        sample = ", ".join(
            f"({pid}, {camera})"
            for pid, camera in sorted(unmatched_queries, key=lambda item: (item[0], natural_key(item[1])))[:10]
        )
        raise BuildError(f"BUPT query cameras are missing from DATA: {sample}")

    if not records:
        raise BuildError(f"BUPT contains no caption records: {root}")
    return records


def write_json_atomic(records: list[dict[str, object]], output_path: Path) -> None:
    output_path = output_path.resolve()
    require_directory(output_path.parent, "JSON output parent directory")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o644)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def print_summary(dataset: str, root: Path, output_path: Path, records: list[dict[str, object]]) -> None:
    identities: dict[str, set[int]] = defaultdict(set)
    record_counts: dict[str, int] = defaultdict(int)
    caption_counts: dict[str, int] = defaultdict(int)
    query_counts: dict[str, int] = defaultdict(int)

    for record in records:
        split = str(record["split"])
        identities[split].add(int(record["id"]))
        record_counts[split] += 1
        caption_counts[split] += len(record["captions"])
        query_counts[split] += int(bool(record["is_query"]))

    print(f"dataset: {dataset}")
    print(f"root: {root.resolve()}")
    print(f"output: {output_path.resolve()}")
    print("split             identities  records  captions  queries")
    print("----------------------------------------------------------")
    present_splits = [split for split in SPLIT_ORDER if split in record_counts]
    for split in present_splits:
        print(
            f"{split:<17} {len(identities[split]):>10}  {record_counts[split]:>7}  "
            f"{caption_counts[split]:>8}  {query_counts[split]:>7}"
        )
    print("----------------------------------------------------------")
    print(
        f"{'total':<17} {sum(len(ids) for ids in identities.values()):>10}  "
        f"{len(records):>7}  {sum(caption_counts.values()):>8}  "
        f"{sum(query_counts.values()):>7}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized camera-level caption JSON for VCM or BUPT."
    )
    parser.add_argument("dataset", choices=tuple(DEFAULT_ROOTS))
    parser.add_argument(
        "--root",
        type=Path,
        help="Dataset root (defaults to the configured /data/ydl/datasets path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (defaults to <root>/VCM.json or <root>/BUPT.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = (args.root or DEFAULT_ROOTS[args.dataset]).resolve()
    output_path = (args.output or root / DEFAULT_OUTPUT_NAMES[args.dataset]).resolve()

    try:
        if args.dataset == "vcm":
            records = build_vcm(root)
        else:
            records = build_bupt(root)
        write_json_atomic(records, output_path)
    except (BuildError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print_summary(args.dataset, root, output_path, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
