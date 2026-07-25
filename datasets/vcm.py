from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .bases import BaseDataset
from .video_utils import (
    DatasetContractError,
    MODALITY_NAME_TO_ID,
    SAMPLING_POLICY,
    build_eval_dict,
    collect_frames,
    load_caption_records,
    natural_key,
    resolve_dataset_root,
    select_captions,
    sha256_file,
    sorted_directories,
    uniform_sample_frames,
)


class VCM(BaseDataset):
    """VCM RGB/IR video-text baseline with JSON captions."""

    dataset_dir = "vcm"
    annotation_name = "VCM.json"

    def __init__(
        self,
        root: str = "",
        annotation_file: str | None = None,
        verbose: bool = True,
        num_frames: int = 6,
        train_caption_mode: str = "single",
        caption_source: str = "json",
    ):
        super().__init__()
        self.dataset_root = resolve_dataset_root(root, self.dataset_dir)
        self.train_dir = self.dataset_root / "Train"
        self.test_dir = self.dataset_root / "Test"
        self.annotation_path = Path(
            annotation_file or self.dataset_root / self.annotation_name
        ).expanduser().resolve()
        self.num_frames = int(num_frames)
        self.train_caption_mode = train_caption_mode
        self.caption_source = str(caption_source)

        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if train_caption_mode not in {"single", "double"}:
            raise ValueError(f"Invalid train_caption_mode: {train_caption_mode}")
        if self.caption_source not in {"legacy", "json"}:
            raise ValueError(f"Invalid caption_source: {self.caption_source}")
        if not self.train_dir.is_dir() or not self.test_dir.is_dir():
            raise DatasetContractError(
                f"VCM requires Train and Test directories under {self.dataset_root}"
            )

        train_tracklets = self._scan_split(self.train_dir, "train")
        test_tracklets = self._scan_split(self.test_dir, "test")
        self._validate_identity_modalities(train_tracklets, "train")
        self._validate_identity_modalities(test_tracklets, "test")
        if self.caption_source == "json":
            records = load_caption_records(self.annotation_path)
            self._validate_annotation_coverage(
                records, train_tracklets, test_tracklets
            )
            annotation_file = str(self.annotation_path)
            annotation_sha256 = sha256_file(self.annotation_path)
        else:
            records = self._load_legacy_records(train_tracklets, test_tracklets)
            annotation_file = ""
            annotation_sha256 = self._legacy_caption_sha256(records)
        self._records = {
            (record["split"], int(record["id"]), str(record["camera"])): record
            for record in records
        }

        train_raw_ids = {int(item["raw_pid"]) for item in train_tracklets}
        test_raw_ids = {int(item["raw_pid"]) for item in test_tracklets}
        overlap = train_raw_ids & test_raw_ids
        if overlap:
            raise DatasetContractError(
                f"VCM train/test identities overlap: {sorted(overlap)[:10]}"
            )
        self.train_pid2label = {
            pid: label for label, pid in enumerate(sorted(train_raw_ids))
        }
        self.test_pid2label = {
            pid: label for label, pid in enumerate(sorted(test_raw_ids))
        }
        self.train_id_container = set(self.train_pid2label.values())
        self.test_id_container = set(self.test_pid2label.values())
        self.val_id_container = set(self.test_id_container)

        self.train_tracklets = self._label_and_sample(
            train_tracklets, self.train_pid2label
        )
        self.gallery_tracklets = self._label_and_sample(
            test_tracklets, self.test_pid2label
        )
        self.queries = self._build_queries(records)
        self.train = self._build_train_samples()
        self._build_eval_splits()

        self.train_annos = self.train_tracklets
        self.test_annos = self.gallery_tracklets
        self.val_annos = self.gallery_tracklets
        self.metadata = {
            "dataset_name": "VCM",
            "dataset_root": str(self.dataset_root),
            "caption_source": self.caption_source,
            "annotation_file": annotation_file,
            "annotation_sha256": annotation_sha256,
            "num_frames": self.num_frames,
            "sampling_policy": SAMPLING_POLICY,
            "train_modalities": "rgb+real_ir",
            "fake_ir_policy": "not_applicable",
            "num_queries": len(self.queries),
        }

        if verbose:
            self.logger.info(
                "VCM: train_ids=%d train_tracklets=%d queries=%d "
                "gallery_rgb=%d gallery_ir=%d num_frames=%d",
                len(self.train_id_container),
                len(self.train_tracklets),
                len(self.queries),
                len(self.gallery_rgb),
                len(self.gallery_ir),
                self.num_frames,
            )

    @staticmethod
    def _read_legacy_caption(path: Path) -> str:
        if not path.is_file():
            raise DatasetContractError(f"Missing legacy VCM caption: {path}")
        try:
            caption = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise DatasetContractError(
                f"Cannot read legacy VCM caption {path}: {error}"
            ) from error
        if not caption or len(caption.splitlines()) != 1:
            raise DatasetContractError(
                f"Legacy VCM caption must be one non-empty line: {path}"
            )
        return caption

    def _load_legacy_records(
        self,
        train_tracklets: list[dict[str, object]],
        test_tracklets: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        records = []
        for item in train_tracklets + test_tracklets:
            if item["modality_name"] != "rgb":
                continue
            camera_dir = self.dataset_root / str(item["relative_path"])
            caption_path = camera_dir / "caption.txt"
            if not caption_path.is_file() and item["split"] == "test":
                # Historical VCM snapshots may contain a gallery camera without
                # a caption. It remains a gallery candidate but is not a query.
                continue
            captions = [self._read_legacy_caption(caption_path)]
            augmented_path = camera_dir / "caption_aug.txt"
            if augmented_path.is_file():
                captions.append(self._read_legacy_caption(augmented_path))
            records.append(
                {
                    "split": str(item["split"]),
                    "id": int(item["raw_pid"]),
                    "file_path": str(item["relative_path"]),
                    "captions": captions,
                    "camera": str(item["camera"]),
                    "modality": "rgb",
                    "media_type": "video",
                    "is_query": item["split"] == "test",
                }
            )
        return records

    @staticmethod
    def _legacy_caption_sha256(records: list[dict[str, object]]) -> str:
        payload = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _scan_split(self, split_dir: Path, split: str) -> list[dict[str, object]]:
        tracklets: list[dict[str, object]] = []
        for pid_dir in sorted_directories(split_dir):
            if not pid_dir.name.isdigit():
                raise DatasetContractError(f"Non-numeric VCM identity: {pid_dir}")
            raw_pid = int(pid_dir.name)
            for modality_name in ("rgb", "ir"):
                modality_dir = pid_dir / modality_name
                if not modality_dir.is_dir():
                    continue
                for camera_dir in sorted_directories(modality_dir):
                    frame_paths = collect_frames(camera_dir)
                    if not frame_paths:
                        raise DatasetContractError(
                            f"VCM tracklet has no frames: {camera_dir}"
                        )
                    tracklets.append(
                        {
                            "split": split,
                            "raw_pid": raw_pid,
                            "camera": camera_dir.name,
                            "modality_name": modality_name,
                            "all_frame_paths": frame_paths,
                            "relative_path": camera_dir.relative_to(
                                self.dataset_root
                            ).as_posix(),
                        }
                    )
        if not tracklets:
            raise DatasetContractError(f"VCM split contains no tracklets: {split_dir}")
        return tracklets

    @staticmethod
    def _validate_identity_modalities(
        tracklets: list[dict[str, object]], split: str
    ) -> None:
        modalities_by_pid: dict[int, set[str]] = defaultdict(set)
        for item in tracklets:
            modalities_by_pid[int(item["raw_pid"])].add(
                str(item["modality_name"])
            )
        incomplete = sorted(
            pid
            for pid, modalities in modalities_by_pid.items()
            if modalities != {"rgb", "ir"}
        )
        if incomplete:
            raise DatasetContractError(
                f"VCM {split} identities missing RGB or IR tracklets: {incomplete[:10]}"
            )

    def _validate_annotation_coverage(
        self,
        records: list[dict[str, object]],
        train_tracklets: list[dict[str, object]],
        test_tracklets: list[dict[str, object]],
    ) -> None:
        unsupported_splits = sorted(
            {str(record["split"]) for record in records} - {"train", "test"}
        )
        if unsupported_splits:
            raise DatasetContractError(
                "VCM annotations support only train/test splits, got "
                f"{unsupported_splits}"
            )
        tree_keys: set[tuple[str, int, str]] = set()
        tree_paths: dict[tuple[str, int, str], str] = {}
        for item in train_tracklets + test_tracklets:
            if item["modality_name"] != "rgb":
                continue
            key = (str(item["split"]), int(item["raw_pid"]), str(item["camera"]))
            tree_keys.add(key)
            tree_paths[key] = str(item["relative_path"])

        json_keys = {
            (str(record["split"]), int(record["id"]), str(record["camera"]))
            for record in records
        }
        if tree_keys != json_keys:
            missing = sorted(tree_keys - json_keys)[:10]
            extra = sorted(json_keys - tree_keys)[:10]
            raise DatasetContractError(
                f"VCM JSON/tree RGB coverage differs; missing={missing}, extra={extra}"
            )
        for record in records:
            key = (str(record["split"]), int(record["id"]), str(record["camera"]))
            if str(record["file_path"]) != tree_paths[key]:
                raise DatasetContractError(
                    f"VCM JSON path mismatch for {key}: {record['file_path']} != {tree_paths[key]}"
                )

    def _label_and_sample(
        self,
        tracklets: list[dict[str, object]],
        pid2label: dict[int, int],
    ) -> list[dict[str, object]]:
        labeled = []
        for item in tracklets:
            copied = dict(item)
            copied["pid"] = pid2label[int(item["raw_pid"])]
            copied["modality"] = MODALITY_NAME_TO_ID[str(item["modality_name"])]
            copied["img_paths"] = uniform_sample_frames(
                item["all_frame_paths"], self.num_frames
            )
            labeled.append(copied)
        return labeled

    def _build_train_samples(self) -> list[tuple[int, int, list[str], str, int]]:
        rgb_records_by_pid: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in self._records.values():
            if record["split"] == "train":
                rgb_records_by_pid[int(record["id"])].append(record)
        for records in rgb_records_by_pid.values():
            records.sort(key=lambda record: natural_key(str(record["camera"])))

        samples = []
        for image_id, tracklet in enumerate(self.train_tracklets):
            raw_pid = int(tracklet["raw_pid"])
            if tracklet["modality_name"] == "rgb":
                key = ("train", raw_pid, str(tracklet["camera"]))
                caption_record = self._records[key]
            else:
                if not rgb_records_by_pid[raw_pid]:
                    raise DatasetContractError(
                        f"VCM IR identity {raw_pid} has no RGB caption source"
                    )
                caption_record = rgb_records_by_pid[raw_pid][0]
            for caption in select_captions(caption_record, self.train_caption_mode):
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

    def _build_queries(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        queries = []
        for record in records:
            if record["split"] != "test" or not record["is_query"]:
                continue
            raw_pid = int(record["id"])
            if raw_pid not in self.test_pid2label:
                raise DatasetContractError(
                    f"VCM query identity is absent from Test tree: {raw_pid}"
                )
            queries.append(
                {
                    "pid": self.test_pid2label[raw_pid],
                    "caption": list(record["captions"])[0],
                    "raw_pid": raw_pid,
                    "camera": str(record["camera"]),
                }
            )
        if not queries:
            raise DatasetContractError("VCM annotation contains no test queries")
        return queries

    def _build_eval_splits(self) -> None:
        self.gallery_rgb = [
            item for item in self.gallery_tracklets if item["modality_name"] == "rgb"
        ]
        self.gallery_ir = [
            item for item in self.gallery_tracklets if item["modality_name"] == "ir"
        ]
        self.gallery_mixed = self.gallery_rgb + self.gallery_ir
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
