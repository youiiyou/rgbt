from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from .bases import BaseDataset
from .video_utils import (
    DatasetContractError,
    EXCLUDED_IMAGE_NAMES,
    IMAGE_SUFFIXES,
    MODALITY_NAME_TO_ID,
    SAMPLING_POLICY,
    build_eval_dict,
    load_caption_records,
    natural_key,
    resolve_dataset_root,
    select_captions,
    sha256_file,
    sorted_directories,
    uniform_sample_frames,
)


class BUPT(BaseDataset):
    """Official BUPTCampus protocol using real RGB and real IR only."""

    dataset_dir = "bupt"
    annotation_name = "BUPT.json"

    def __init__(
        self,
        root: str = "",
        annotation_file: str | None = None,
        verbose: bool = True,
        num_frames: int = 6,
        train_caption_mode: str = "single",
        protocol_dir: str | None = None,
    ):
        super().__init__()
        self.dataset_root = resolve_dataset_root(root, self.dataset_dir)
        self.data_dir = self.dataset_root / "DATA"
        self.annotation_path = Path(
            annotation_file or self.dataset_root / self.annotation_name
        ).expanduser().resolve()
        self.protocol_dir = Path(protocol_dir or self.dataset_root).expanduser().resolve()
        self.num_frames = int(num_frames)
        self.train_caption_mode = train_caption_mode
        self._camera_cache: dict[tuple[int, str, str], dict[int, list[str]]] = {}

        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if train_caption_mode not in {"single", "double"}:
            raise ValueError(f"Invalid train_caption_mode: {train_caption_mode}")
        if not self.data_dir.is_dir():
            raise DatasetContractError(f"Missing BUPT DATA directory: {self.data_dir}")

        records = load_caption_records(self.annotation_path)
        self._records = {
            (str(record["split"]), int(record["id"]), str(record["camera"])): record
            for record in records
        }
        protocols = {
            name: self._read_protocol(name)
            for name in ("train", "train_auxiliary", "query", "gallery")
        }
        self._validate_protocol_ids(protocols)
        self._validate_annotation_coverage(records, protocols)

        train_raw_ids = {
            row[0] for name in ("train", "train_auxiliary") for row in protocols[name]
        }
        test_raw_ids = {row[0] for row in protocols["query"]}
        self.train_pid2label = {
            pid: label for label, pid in enumerate(sorted(train_raw_ids))
        }
        self.test_pid2label = {
            pid: label for label, pid in enumerate(sorted(test_raw_ids))
        }
        self.train_id_container = set(self.train_pid2label.values())
        self.test_id_container = set(self.test_pid2label.values())
        self.val_id_container = set(self.test_id_container)

        self.train_tracklets = self._build_train_tracklets(protocols)
        self.gallery_tracklets = self._build_gallery_tracklets(protocols["gallery"])
        self.queries = self._build_queries(protocols["query"])
        self.train = self._build_train_samples()
        self._build_eval_splits()
        self._camera_cache.clear()

        self.train_annos = self.train_tracklets
        self.test_annos = self.gallery_tracklets
        self.val_annos = self.gallery_tracklets
        query_modalities = Counter(row[1].casefold() for row in protocols["query"])
        self.metadata = {
            "dataset_name": "BUPT",
            "dataset_root": str(self.dataset_root),
            "annotation_file": str(self.annotation_path),
            "annotation_sha256": sha256_file(self.annotation_path),
            "protocol_dir": str(self.protocol_dir),
            "num_frames": self.num_frames,
            "sampling_policy": SAMPLING_POLICY,
            "train_modalities": "rgb+real_ir",
            "fake_ir_policy": "excluded",
            "train_protocol_splits": "train+train_auxiliary",
            "num_queries": len(self.queries),
            "query_protocol_rgb": query_modalities["rgb"],
            "query_protocol_ir": query_modalities["ir"],
        }

        if verbose:
            self.logger.info(
                "BUPT: train_ids=%d train_tracklets=%d queries=%d "
                "gallery_rgb=%d gallery_ir=%d num_frames=%d FakeIR=excluded",
                len(self.train_id_container),
                len(self.train_tracklets),
                len(self.queries),
                len(self.gallery_rgb),
                len(self.gallery_ir),
                self.num_frames,
            )

    def _read_protocol(self, name: str) -> list[tuple[int, str, str, int]]:
        path = self.protocol_dir / f"{name}.txt"
        if not path.is_file():
            raise DatasetContractError(f"Missing BUPT protocol: {path}")
        rows = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.split()
            if len(fields) != 4:
                raise DatasetContractError(
                    f"Expected four fields at {path}:{line_number}"
                )
            pid_text, modality, camera, tracklet_text = fields
            if not pid_text.isdigit() or not tracklet_text.isdigit():
                raise DatasetContractError(
                    f"Invalid PID/tracklet at {path}:{line_number}"
                )
            if name in {"train", "train_auxiliary"}:
                allowed_modalities = {"RGB/IR"}
            else:
                allowed_modalities = {"RGB", "IR"}
            if modality not in allowed_modalities:
                raise DatasetContractError(
                    f"Invalid modality {modality!r} for {name} at "
                    f"{path}:{line_number}; expected {sorted(allowed_modalities)}"
                )
            rows.append((int(pid_text), modality, camera, int(tracklet_text)))
        if not rows:
            raise DatasetContractError(f"Empty BUPT protocol: {path}")
        return rows

    def _validate_protocol_ids(
        self, protocols: dict[str, list[tuple[int, str, str, int]]]
    ) -> None:
        train_ids = {row[0] for row in protocols["train"]}
        auxiliary_ids = {row[0] for row in protocols["train_auxiliary"]}
        query_ids = {row[0] for row in protocols["query"]}
        gallery_ids = {row[0] for row in protocols["gallery"]}
        if query_ids != gallery_ids:
            raise DatasetContractError("BUPT query/gallery identity sets differ")
        named = (
            ("train", train_ids),
            ("train_auxiliary", auxiliary_ids),
            ("test", query_ids),
        )
        for index, (left_name, left_ids) in enumerate(named):
            for right_name, right_ids in named[index + 1 :]:
                overlap = left_ids & right_ids
                if overlap:
                    raise DatasetContractError(
                        f"BUPT identity overlap between {left_name} and {right_name}: "
                        f"{sorted(overlap)[:10]}"
                    )
        query_keys = [(row[0], row[2]) for row in protocols["query"]]
        if len(query_keys) != len(set(query_keys)):
            raise DatasetContractError("BUPT query PID/camera pairs are not unique")

    def _validate_annotation_coverage(
        self,
        records: list[dict[str, object]],
        protocols: dict[str, list[tuple[int, str, str, int]]],
    ) -> None:
        split_by_pid: dict[int, str] = {}
        for split in ("train", "train_auxiliary"):
            for pid, _modality, _camera, _tracklet in protocols[split]:
                split_by_pid[pid] = split
        for pid, _modality, _camera, _tracklet in protocols["query"]:
            split_by_pid[pid] = "test"

        tree_keys: set[tuple[str, int, str]] = set()
        tree_paths: dict[tuple[str, int, str], str] = {}
        identity_dirs = {
            int(path.name): path
            for path in sorted_directories(self.data_dir)
            if path.name.isdigit()
        }
        if set(identity_dirs) != set(split_by_pid):
            raise DatasetContractError("BUPT DATA/protocol identity sets differ")
        for pid, pid_dir in identity_dirs.items():
            rgb_dir = pid_dir / "RGB"
            for camera_dir in sorted_directories(rgb_dir):
                key = (split_by_pid[pid], pid, camera_dir.name)
                tree_keys.add(key)
                tree_paths[key] = camera_dir.relative_to(self.dataset_root).as_posix()

        json_keys = {
            (str(record["split"]), int(record["id"]), str(record["camera"]))
            for record in records
        }
        if tree_keys != json_keys:
            missing = sorted(tree_keys - json_keys)[:10]
            extra = sorted(json_keys - tree_keys)[:10]
            raise DatasetContractError(
                f"BUPT JSON/tree RGB coverage differs; missing={missing}, extra={extra}"
            )
        for record in records:
            key = (str(record["split"]), int(record["id"]), str(record["camera"]))
            if str(record["file_path"]) != tree_paths[key]:
                raise DatasetContractError(
                    f"BUPT JSON path mismatch for {key}: "
                    f"{record['file_path']} != {tree_paths[key]}"
                )

        protocol_query_keys = {(row[0], row[2]) for row in protocols["query"]}
        json_query_keys = {
            (int(record["id"]), str(record["camera"]))
            for record in records
            if record["split"] == "test" and record["is_query"]
        }
        if protocol_query_keys != json_query_keys:
            raise DatasetContractError("BUPT JSON query flags differ from query.txt")

    def _camera_tracklets(self, pid: int, modality: str, camera: str) -> dict[int, list[str]]:
        cache_key = (pid, modality, camera)
        if cache_key in self._camera_cache:
            return self._camera_cache[cache_key]
        camera_dir = self.data_dir / str(pid) / modality / camera
        if not camera_dir.is_dir():
            raise DatasetContractError(f"Missing BUPT camera directory: {camera_dir}")
        pattern = re.compile(
            rf"^{pid}_{re.escape(modality)}_{re.escape(camera)}_(\d+)_(\d+)"
            rf"\.(?:jpg|jpeg|png|bmp)$",
            re.IGNORECASE,
        )
        grouped: dict[int, list[tuple[int, str]]] = {}
        seen_frame_keys: set[tuple[int, int]] = set()
        try:
            entries = os.scandir(camera_dir)
            with entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    match = pattern.fullmatch(entry.name)
                    if match is None:
                        suffix = Path(entry.name).suffix.casefold()
                        if (
                            suffix in IMAGE_SUFFIXES
                            and entry.name.casefold() not in EXCLUDED_IMAGE_NAMES
                        ):
                            raise DatasetContractError(
                                f"Unexpected BUPT image filename: {entry.path}"
                            )
                        continue
                    tracklet_id = int(match.group(1))
                    frame_id = int(match.group(2))
                    frame_key = (tracklet_id, frame_id)
                    if frame_key in seen_frame_keys:
                        raise DatasetContractError(
                            f"Duplicate BUPT frame ID in {camera_dir}: "
                            f"tracklet={tracklet_id} frame={frame_id}"
                        )
                    seen_frame_keys.add(frame_key)
                    grouped.setdefault(tracklet_id, []).append(
                        (frame_id, entry.path)
                    )
        except OSError as error:
            raise DatasetContractError(
                f"Cannot scan BUPT camera directory {camera_dir}: {error}"
            ) from error
        result = {
            tracklet_id: [path for _frame_id, path in sorted(frames)]
            for tracklet_id, frames in grouped.items()
        }
        self._camera_cache[cache_key] = result
        return result

    def _make_tracklet(
        self,
        split: str,
        pid: int,
        modality: str,
        camera: str,
        tracklet_id: int,
        pid2label: dict[int, int],
    ) -> dict[str, object]:
        available = self._camera_tracklets(pid, modality, camera)
        if tracklet_id not in available or not available[tracklet_id]:
            raise DatasetContractError(
                f"Missing BUPT tracklet: split={split} pid={pid} modality={modality} "
                f"camera={camera} tracklet={tracklet_id}"
            )
        frame_paths = available[tracklet_id]
        return {
            "split": split,
            "raw_pid": pid,
            "pid": pid2label[pid],
            "camera": camera,
            "tracklet_id": tracklet_id,
            "modality_name": modality.casefold(),
            "modality": MODALITY_NAME_TO_ID[modality.casefold()],
            "all_frame_paths": frame_paths,
            "img_paths": uniform_sample_frames(frame_paths, self.num_frames),
        }

    def _build_train_tracklets(
        self, protocols: dict[str, list[tuple[int, str, str, int]]]
    ) -> list[dict[str, object]]:
        tracklets = []
        for split in ("train", "train_auxiliary"):
            for pid, modality, camera, tracklet_id in protocols[split]:
                modalities = ("RGB", "IR") if modality == "RGB/IR" else (modality,)
                for real_modality in modalities:
                    tracklets.append(
                        self._make_tracklet(
                            split,
                            pid,
                            real_modality,
                            camera,
                            tracklet_id,
                            self.train_pid2label,
                        )
                    )
        return tracklets

    def _build_gallery_tracklets(
        self, rows: list[tuple[int, str, str, int]]
    ) -> list[dict[str, object]]:
        gallery = []
        for pid, modality, camera, tracklet_id in rows:
            if modality not in {"RGB", "IR"}:
                raise DatasetContractError(
                    f"BUPT gallery must specify RGB or IR, got {modality}"
                )
            gallery.append(
                self._make_tracklet(
                    "test",
                    pid,
                    modality,
                    camera,
                    tracklet_id,
                    self.test_pid2label,
                )
            )
        return gallery

    def _build_queries(
        self, rows: list[tuple[int, str, str, int]]
    ) -> list[dict[str, object]]:
        queries = []
        for pid, source_modality, camera, tracklet_id in rows:
            available = self._camera_tracklets(pid, source_modality, camera)
            if tracklet_id not in available or not available[tracklet_id]:
                raise DatasetContractError(
                    f"Missing BUPT query tracklet: {pid} {source_modality} "
                    f"{camera} {tracklet_id}"
                )
            record = self._records[("test", pid, camera)]
            queries.append(
                {
                    "pid": self.test_pid2label[pid],
                    "caption": list(record["captions"])[0],
                    "raw_pid": pid,
                    "camera": camera,
                    "source_modality": source_modality.casefold(),
                    "tracklet_id": tracklet_id,
                }
            )
        return queries

    def _build_train_samples(self) -> list[tuple[int, int, list[str], str, int]]:
        samples = []
        for image_id, tracklet in enumerate(self.train_tracklets):
            key = (
                str(tracklet["split"]),
                int(tracklet["raw_pid"]),
                str(tracklet["camera"]),
            )
            record = self._records[key]
            for caption in select_captions(record, self.train_caption_mode):
                samples.append(
                    (
                        int(tracklet["pid"]),
                        image_id,
                        list(tracklet["img_paths"]),
                        caption,
                        int(tracklet["modality"]),
                    )
                )
        return samples

    def _build_eval_splits(self) -> None:
        self.gallery_rgb = [
            item for item in self.gallery_tracklets if item["modality_name"] == "rgb"
        ]
        self.gallery_ir = [
            item for item in self.gallery_tracklets if item["modality_name"] == "ir"
        ]
        self.gallery_mixed = list(self.gallery_tracklets)
        self.test_rgb = build_eval_dict(self.gallery_rgb, self.queries)
        self.test_ir = build_eval_dict(self.gallery_ir, self.queries)
        self.test_mixed = build_eval_dict(self.gallery_mixed, self.queries)
        self.val_rgb = {key: list(value) for key, value in self.test_rgb.items()}
        self.val_ir = {key: list(value) for key, value in self.test_ir.items()}
        self.val_mixed = {key: list(value) for key, value in self.test_mixed.items()}
        self.test = self.test_mixed
        self.val = self.val_mixed

    def get_eval_split(self, split_name: str = "test", gallery_mode: str = "mixed"):
        if split_name not in {"test", "val"}:
            raise ValueError(f"Invalid evaluation split: {split_name}")
        if gallery_mode not in {"rgb", "ir", "mixed"}:
            raise ValueError(f"Invalid gallery mode: {gallery_mode}")
        return getattr(self, f"{split_name}_{gallery_mode}")
