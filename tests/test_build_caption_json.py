import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_caption_json import BuildError, build_bupt, build_vcm, write_json_atomic


class CaptionJsonBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def add_frame(camera_dir: Path) -> None:
        camera_dir.mkdir(parents=True, exist_ok=True)
        (camera_dir / "1.jpg").write_bytes(b"fixture")

    def add_vcm_camera(
        self,
        split: str,
        pid: str,
        camera: str,
        caption: str = "Original caption.",
        caption_aug: str = "Augmented caption.",
        add_frame: bool = True,
    ) -> Path:
        camera_dir = self.root / split / pid / "rgb" / camera
        self.write_text(camera_dir / "caption.txt", caption)
        self.write_text(camera_dir / "caption_aug.txt", caption_aug)
        if add_frame:
            self.add_frame(camera_dir)
        return camera_dir

    def test_vcm_builds_ordered_camera_records_and_writes_json(self):
        self.add_vcm_camera("Train", "0001", "D10")
        self.add_vcm_camera("Train", "0001", "D2")
        self.add_vcm_camera("Test", "0501", "D1")

        records = build_vcm(self.root)

        self.assertEqual(
            [record["file_path"] for record in records],
            [
                "Train/0001/rgb/D2",
                "Train/0001/rgb/D10",
                "Test/0501/rgb/D1",
            ],
        )
        self.assertEqual(
            list(records[0]),
            [
                "split",
                "id",
                "file_path",
                "captions",
                "camera",
                "modality",
                "media_type",
                "is_query",
            ],
        )
        self.assertEqual(records[0]["captions"], ["Original caption.", "Augmented caption."])
        self.assertFalse(records[0]["is_query"])
        self.assertTrue(records[-1]["is_query"])

        output_path = self.root / "VCM.json"
        write_json_atomic(records, output_path)
        self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), records)
        self.assertTrue(output_path.read_bytes().endswith(b"\n"))

    def test_vcm_rejects_missing_caption(self):
        camera_dir = self.add_vcm_camera("Train", "0001", "D1")
        (camera_dir / "caption_aug.txt").unlink()
        (self.root / "Test").mkdir()

        with self.assertRaisesRegex(BuildError, "Missing caption file"):
            build_vcm(self.root)

    def test_vcm_rejects_empty_caption(self):
        self.add_vcm_camera("Train", "0001", "D1", caption="\n")
        (self.root / "Test").mkdir()

        with self.assertRaisesRegex(BuildError, "Caption is empty"):
            build_vcm(self.root)

    def test_vcm_rejects_duplicate_captions(self):
        self.add_vcm_camera(
            "Train",
            "0001",
            "D1",
            caption="Same caption.",
            caption_aug="Same caption.",
        )
        (self.root / "Test").mkdir()

        with self.assertRaisesRegex(BuildError, "must be distinct"):
            build_vcm(self.root)

    def test_vcm_rejects_camera_without_frames(self):
        self.add_vcm_camera("Train", "0001", "D1", add_frame=False)
        (self.root / "Test").mkdir()

        with self.assertRaisesRegex(BuildError, "no usable frames"):
            build_vcm(self.root)

    def write_protocol(self, name: str, lines: list[str]) -> None:
        self.write_text(self.root / name, "\n".join(lines) + "\n")

    def add_bupt_camera(self, pid: int, camera: str) -> None:
        camera_dir = self.root / "DATA" / str(pid) / "RGB" / camera
        self.write_text(camera_dir / "caption.txt", f"Caption for identity {pid}.")
        self.add_frame(camera_dir)

    def make_valid_bupt_fixture(self):
        self.add_bupt_camera(1, "C1")
        self.add_bupt_camera(2, "C2")
        self.add_bupt_camera(3, "C3")
        self.write_protocol("train.txt", ["1 RGB/IR C1 1"])
        self.write_protocol("train_auxiliary.txt", ["2 RGB/IR C2 1"])
        self.write_protocol("query.txt", ["3 IR C3 1"])
        self.write_protocol("gallery.txt", ["3 RGB C3 1"])

    def test_bupt_preserves_auxiliary_and_marks_official_query(self):
        self.make_valid_bupt_fixture()

        records = build_bupt(self.root)

        self.assertEqual([record["split"] for record in records], ["train", "train_auxiliary", "test"])
        self.assertEqual([record["id"] for record in records], [1, 2, 3])
        self.assertEqual([record["is_query"] for record in records], [False, False, True])
        self.assertEqual(records[-1]["file_path"], "DATA/3/RGB/C3")
        self.assertEqual(records[-1]["captions"], ["Caption for identity 3."])

    def test_bupt_rejects_protocol_identity_overlap(self):
        self.make_valid_bupt_fixture()
        self.write_protocol("train_auxiliary.txt", ["1 RGB/IR C1 1", "2 RGB/IR C2 1"])

        with self.assertRaisesRegex(BuildError, "identity overlap"):
            build_bupt(self.root)


if __name__ == "__main__":
    unittest.main()
