import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.bupt import BUPT
from datasets.vcm import VCM
from datasets.video_utils import DatasetContractError


def write_image(path: Path, color: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 12), color=(color, color, color)).save(path)


def write_json(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def record(
    split: str,
    pid: int,
    file_path: str,
    camera: str,
    captions: list[str],
    is_query: bool,
) -> dict[str, object]:
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


class VCMContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "vcm"
        records = []
        for camera, caption in (("D10", "train D10"), ("D2", "train D2")):
            camera_dir = self.root / "Train" / "0001" / "rgb" / camera
            for frame in ("10.jpg", "2.jpg", "1.jpg"):
                write_image(camera_dir / frame)
            records.append(
                record(
                    "train",
                    1,
                    camera_dir.relative_to(self.root).as_posix(),
                    camera,
                    [caption, f"{caption} augmented"],
                    False,
                )
            )
        ir_dir = self.root / "Train" / "0001" / "ir" / "D1"
        write_image(ir_dir / "1.jpg")
        write_image(ir_dir / "2.jpg")

        test_rgb = self.root / "Test" / "0501" / "rgb" / "D1"
        test_ir = self.root / "Test" / "0501" / "ir" / "D1"
        write_image(test_rgb / "1.jpg")
        write_image(test_ir / "1.jpg")
        records.append(
            record(
                "test",
                501,
                test_rgb.relative_to(self.root).as_posix(),
                "D1",
                ["test query", "test query augmented"],
                True,
            )
        )
        self.annotation = self.root / "VCM.json"
        write_json(self.annotation, records)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_json_mode_uses_own_rgb_and_canonical_ir_captions(self):
        dataset = VCM(
            root=self.root,
            caption_source="json",
            num_frames=6,
            train_caption_mode="single",
            verbose=False,
        )
        self.assertEqual(len(dataset.train_tracklets), 3)
        self.assertEqual(len(dataset.train), 3)
        self.assertEqual(len(dataset.queries), 1)
        self.assertEqual(len(dataset.gallery_rgb), 1)
        self.assertEqual(len(dataset.gallery_ir), 1)
        ir_samples = [sample for sample in dataset.train if sample[4] == 1]
        self.assertEqual(ir_samples[0][3], "train D2")
        self.assertEqual(
            [Path(path).name for path in dataset.train[0][2]],
            ["1.jpg", "1.jpg", "2.jpg", "2.jpg", "10.jpg", "10.jpg"],
        )
        for mode in ("rgb", "ir", "mixed"):
            split = dataset.get_eval_split("test", mode)
            self.assertEqual(split["captions"], ["test query"])
            self.assertEqual(split["caption_pids"], [0])

    def test_double_caption_mode_expands_every_training_tracklet(self):
        dataset = VCM(
            root=self.root,
            caption_source="json",
            num_frames=6,
            train_caption_mode="double",
            verbose=False,
        )
        self.assertEqual(len(dataset.train), 6)

    def test_json_tree_coverage_mismatch_is_an_error(self):
        records = json.loads(self.annotation.read_text(encoding="utf-8"))
        records.append(
            record("test", 501, "Test/0501/rgb/D9", "D9", ["extra"], True)
        )
        write_json(self.annotation, records)
        with self.assertRaisesRegex(DatasetContractError, "coverage differs"):
            VCM(root=self.root, caption_source="json", verbose=False)

    def test_json_mode_rejects_train_auxiliary_records(self):
        records = json.loads(self.annotation.read_text(encoding="utf-8"))
        records.append(
            record(
                "train_auxiliary",
                2,
                "Train/0002/rgb/D1",
                "D1",
                ["unsupported split"],
                False,
            )
        )
        write_json(self.annotation, records)

        with self.assertRaisesRegex(DatasetContractError, "only train/test"):
            VCM(root=self.root, caption_source="json", verbose=False)


class BUPTContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "bupt"
        specifications = (
            (1, "C1", "train"),
            (2, "C2", "train_auxiliary"),
            (3, "C3", "test"),
        )
        records = []
        for pid, camera, split in specifications:
            for modality in ("RGB", "IR"):
                camera_dir = self.root / "DATA" / str(pid) / modality / camera
                for frame in (1, 2):
                    write_image(
                        camera_dir
                        / f"{pid}_{modality}_{camera}_1_{frame}.jpg",
                        color=pid * 20,
                    )
            fake_dir = self.root / "DATA" / str(pid) / "FakeIR" / camera
            write_image(fake_dir / f"{pid}_FakeIR_{camera}_1_1.jpg")
            rgb_dir = self.root / "DATA" / str(pid) / "RGB" / camera
            records.append(
                record(
                    split,
                    pid,
                    rgb_dir.relative_to(self.root).as_posix(),
                    camera,
                    [f"caption {pid}"],
                    split == "test",
                )
            )
        write_json(self.root / "BUPT.json", records)
        protocols = {
            "train.txt": "1 RGB/IR C1 1\n",
            "train_auxiliary.txt": "2 RGB/IR C2 1\n",
            "query.txt": "3 IR C3 1\n",
            "gallery.txt": "3 RGB C3 1\n3 IR C3 1\n",
        }
        for filename, content in protocols.items():
            (self.root / filename).write_text(content, encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_official_protocol_expands_real_modalities_and_excludes_fake_ir(self):
        dataset = BUPT(root=self.root, num_frames=6, verbose=False)
        self.assertEqual(len(dataset.train_id_container), 2)
        self.assertEqual(len(dataset.train_tracklets), 4)
        self.assertEqual(len(dataset.train), 4)
        self.assertEqual(len(dataset.queries), 1)
        self.assertEqual(len(dataset.gallery_rgb), 1)
        self.assertEqual(len(dataset.gallery_ir), 1)
        self.assertEqual(dataset.queries[0]["source_modality"], "ir")
        self.assertEqual(dataset.test_rgb["captions"], ["caption 3"])
        all_tracklets = dataset.train_tracklets + dataset.gallery_tracklets
        all_paths = [
            path for tracklet in all_tracklets for path in tracklet["img_paths"]
        ]
        self.assertFalse(any("FakeIR" in path for path in all_paths))
        self.assertEqual({tracklet["modality"] for tracklet in dataset.train_tracklets}, {0, 1})

    def test_protocol_identity_overlap_is_an_error(self):
        (self.root / "train_auxiliary.txt").write_text(
            "1 RGB/IR C1 1\n2 RGB/IR C2 1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(DatasetContractError, "identity overlap"):
            BUPT(root=self.root, verbose=False)


if __name__ == "__main__":
    unittest.main()
