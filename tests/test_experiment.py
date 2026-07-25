import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.checkpoint import Checkpointer
from utils.experiment import (
    normalize_loaded_config,
    snapshot_annotation,
    verify_annotation_snapshot,
)
from utils.iotools import load_train_configs


class ExperimentSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "bupt"
        self.dataset_root.mkdir()
        annotation = [
            {
                "split": "test",
                "id": 1,
                "file_path": "DATA/1/RGB/C1",
                "captions": ["query"],
                "camera": "C1",
                "modality": "rgb",
                "media_type": "video",
                "is_query": True,
            }
        ]
        self.annotation_path = self.dataset_root / "BUPT.json"
        self.annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        for filename in ("train.txt", "train_auxiliary.txt", "query.txt", "gallery.txt"):
            (self.dataset_root / filename).write_text(
                f"1 RGB C1 1 # {filename}\n", encoding="utf-8"
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_bupt_annotation_and_protocols_are_snapshotted_and_verified(self):
        args = SimpleNamespace(
            dataset_name="BUPT",
            root_dir=str(self.root),
            annotation_file=str(self.annotation_path),
            caption_source="json",
            num_frames=6,
        )
        output_dir = self.root / "run"
        snapshot_annotation(args, output_dir, repo_dir=self.root)
        self.assertEqual(args.num_queries, 1)
        self.assertEqual(set(args.protocol_sha256), {
            "train.txt",
            "train_auxiliary.txt",
            "query.txt",
            "gallery.txt",
        })
        verify_annotation_snapshot(args)

        query_snapshot = Path(args.protocol_dir) / "query.txt"
        query_snapshot.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "protocol snapshot changed"):
            verify_annotation_snapshot(args)

    def test_snapshot_query_count_uses_only_flagged_test_records(self):
        records = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        records.append(
            {
                "split": "train",
                "id": 2,
                "file_path": "DATA/2/RGB/C2",
                "captions": ["not an evaluation query"],
                "camera": "C2",
                "modality": "rgb",
                "media_type": "video",
                "is_query": True,
            }
        )
        self.annotation_path.write_text(json.dumps(records), encoding="utf-8")
        args = SimpleNamespace(
            dataset_name="BUPT",
            root_dir=str(self.root),
            annotation_file=str(self.annotation_path),
            caption_source="json",
            num_frames=6,
        )

        snapshot_annotation(args, self.root / "test-only-query-run", repo_dir=self.root)

        self.assertEqual(args.num_queries, 1)
        verify_annotation_snapshot(args)

    def test_historical_config_defaults_to_legacy_vcm(self):
        args = SimpleNamespace(num_frames=6, img_size=[384, 128])
        normalize_loaded_config(args)
        self.assertEqual(args.caption_source, "legacy")
        self.assertEqual(args.gallery_mode, "mixed")
        self.assertEqual(args.img_size, (384, 128))

    def test_historical_python_tuple_config_is_loaded_safely(self):
        config_path = self.root / "legacy.yaml"
        config_path.write_text(
            "img_size: !!python/tuple\n- 384\n- 128\nnum_frames: 6\n",
            encoding="utf-8",
        )
        args = load_train_configs(config_path)
        normalize_loaded_config(args)
        self.assertEqual(tuple(args.img_size), (384, 128))


class CheckpointerTests(unittest.TestCase):
    def test_historical_checkpoint_may_omit_fixed_logit_scale_only(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.register_buffer("logit_scale", torch.tensor(50.0))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.pth"
            source = TinyModel()
            state = source.state_dict()
            state.pop("logit_scale")
            torch.save({"model": state, "epoch": 1}, path)
            target = TinyModel()
            checkpoint = Checkpointer(target).load(path)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(torch.equal(target.linear.weight, source.linear.weight))


if __name__ == "__main__":
    unittest.main()
