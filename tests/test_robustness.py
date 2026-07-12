"""Robustness guarantees: format round-trips survive intact, hostile or broken
input degrades into clean per-file failures or FormatError (never a raw
traceback), and packed archives contain only safe relative member paths."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import zstandard as zstd
from PIL import Image

from visionpack.core.errors import FormatError
from visionpack.core.project import Project
from visionpack.formats.coco import CocoImporter, export_coco
from visionpack.formats.yolo import YoloImporter, export_yolo
from visionpack.packing import pack_archive


def _image(path: Path, size: tuple[int, int] = (200, 100), seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _yolo_fixture(root: Path) -> Path:
    """Two images with one box each, plus classes.txt."""
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    _image(raw / "images" / "a.png", (200, 100), seed=1)
    (raw / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    _image(raw / "images" / "b.png", (100, 100), seed=2)
    (raw / "labels" / "b.txt").write_text("1 0.25 0.25 0.1 0.2\n", encoding="utf-8")
    return raw


def _boxes_by_image(project: Project) -> dict[str, list[tuple[str, float, float, float, float]]]:
    """Absolute-coordinate boxes keyed by image content hash (stable across
    round-trips, where exports rename files to asset ids)."""
    result: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for asset in project.index.assets():
        annotation = project.index.annotation_for_asset(asset.id)
        boxes = []
        for obj in annotation.objects if annotation else []:
            bbox = obj.bbox
            if bbox is not None:
                boxes.append((obj.class_id, bbox.x, bbox.y, bbox.width, bbox.height))
        result[asset.sha256] = sorted(boxes)
    return result


def _assert_boxes_close(test: unittest.TestCase, left: dict, right: dict) -> None:
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertEqual(len(left[key]), len(right[key]), key)
        for (cls_a, *coords_a), (cls_b, *coords_b) in zip(left[key], right[key]):
            test.assertEqual(cls_a, cls_b, key)
            for value_a, value_b in zip(coords_a, coords_b):
                test.assertAlmostEqual(value_a, value_b, places=3, msg=key)


class RoundTripTest(unittest.TestCase):
    def test_yolo_to_coco_and_back_preserves_classes_and_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Project.init(root / "src", name="src")
            YoloImporter(source, _yolo_fixture(root / "src")).run()
            original = _boxes_by_image(Project.open(root / "src"))

            coco_dir = root / "coco-export"
            export_coco(Project.open(root / "src"), coco_dir)

            reimported = Project.init(root / "dst", name="dst")
            summary = CocoImporter(reimported, coco_dir / "annotations.json", coco_dir / "images").run()
            self.assertEqual(summary.assets, 2)
            self.assertEqual(summary.failures, [])

            reopened = Project.open(root / "dst")
            self.assertEqual(
                sorted(item.name for item in reopened.manifest.classes),
                ["cat", "dog"],
            )
            _assert_boxes_close(self, original, _boxes_by_image(reopened))

    def test_yolo_export_reimports_identically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Project.init(root / "src", name="src")
            YoloImporter(source, _yolo_fixture(root / "src")).run()
            original = _boxes_by_image(Project.open(root / "src"))

            yolo_dir = root / "yolo-export"
            export_yolo(Project.open(root / "src"), yolo_dir)

            reimported = Project.init(root / "dst", name="dst")
            summary = YoloImporter(reimported, yolo_dir).run()
            self.assertEqual(summary.assets, 2)
            self.assertEqual(summary.failures, [])
            _assert_boxes_close(self, original, _boxes_by_image(Project.open(root / "dst")))


class MalformedInputTest(unittest.TestCase):
    def test_coco_invalid_json_raises_format_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="t")
            images = root / "images"
            images.mkdir()
            bad = root / "instances.json"
            bad.write_text("{not json at all", encoding="utf-8")
            with self.assertRaises(FormatError) as ctx:
                CocoImporter(project, bad, images).run()
            self.assertIn("not valid JSON", str(ctx.exception))

    def test_coco_non_object_top_level_raises_format_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="t")
            images = root / "images"
            images.mkdir()
            bad = root / "instances.json"
            bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(FormatError) as ctx:
                CocoImporter(project, bad, images).run()
            self.assertIn("JSON object", str(ctx.exception))

    def test_bad_yolo_label_line_is_per_file_failure(self) -> None:
        """A malformed label file fails that one image; the rest still import."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="t")
            raw = _yolo_fixture(root)
            _image(raw / "images" / "broken.png", seed=3)
            # 3 values: neither a detection box (4) nor a seg polygon (>=6, even).
            (raw / "labels" / "broken.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
            _image(raw / "images" / "garbage.png", seed=4)
            (raw / "labels" / "garbage.txt").write_text("zero point five nonsense\n", encoding="utf-8")

            summary = YoloImporter(project, raw).run()
            self.assertEqual(summary.assets, 2)  # a.png and b.png survived
            failed = sorted(Path(failure.path).name for failure in summary.failures)
            self.assertEqual(failed, ["broken.png", "garbage.png"])
            for failure in summary.failures:
                self.assertIn("YOLO label", failure.error)

    def test_truncated_image_is_per_file_failure(self) -> None:
        """Valid header + truncated body must not abort the batch."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="t")
            raw = _yolo_fixture(root)
            truncated = raw / "images" / "cut.png"
            _image(truncated, (100, 100), seed=5)
            payload = truncated.read_bytes()
            truncated.write_bytes(payload[: len(payload) // 2])
            (raw / "labels" / "cut.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            summary = YoloImporter(project, raw).run()
            self.assertEqual(summary.assets, 2)
            self.assertEqual([Path(f.path).name for f in summary.failures], ["cut.png"])


class ArchiveSafetyTest(unittest.TestCase):
    def test_archive_members_are_relative_and_traversal_free(self) -> None:
        """No member of a packed archive may escape the extraction directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="t")
            YoloImporter(project, _yolo_fixture(root)).run()
            summary = pack_archive(Project.open(root), output=root / "out.tar.zst")

            with summary.path.open("rb") as handle:
                reader = zstd.ZstdDecompressor().stream_reader(handle)
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    names = [member.name for member in tar]
            self.assertGreater(len(names), 0)
            for name in names:
                pure = PurePosixPath(name)
                self.assertFalse(pure.is_absolute(), name)
                self.assertNotIn("..", pure.parts, name)


if __name__ == "__main__":
    unittest.main()
