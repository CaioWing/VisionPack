from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.errors import VisionPackError
from visionpack.drift import drift_from_stats
from visionpack.formats.detect import coco_json_in, detect_import_format
from visionpack.sdk import VisionPackClient


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed_yolo(root: Path, spec: dict[str, str]) -> Path:
    data = root / "raw"
    for index, (name, label) in enumerate(spec.items(), start=1):
        _png(data / f"{name}.png", index)
        (data / f"{name}.txt").write_text(label, encoding="utf-8")
    (data / "classes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    return data


class DriftMathTest(unittest.TestCase):
    def test_identical_distributions_have_near_zero_divergence(self) -> None:
        stats = {"class_distribution": {"a": 10, "b": 10}, "assets": 20}
        drift = drift_from_stats(stats, stats)
        self.assertAlmostEqual(drift["kl_divergence"], 0.0, places=6)
        self.assertAlmostEqual(drift["js_divergence"], 0.0, places=6)
        self.assertTrue(all(item["delta"] == 0 for item in drift["classes"]))

    def test_shift_reports_deltas_and_positive_divergence(self) -> None:
        before = {"class_distribution": {"a": 10, "b": 10}, "assets": 20}
        after = {"class_distribution": {"a": 30, "b": 5}, "assets": 35}
        drift = drift_from_stats(before, after)
        self.assertGreater(drift["kl_divergence"], 0.0)
        self.assertGreater(drift["js_divergence"], 0.0)
        by_class = {item["class_id"]: item for item in drift["classes"]}
        self.assertEqual(by_class["a"]["delta"], 20)
        self.assertEqual(by_class["b"]["delta"], -5)
        self.assertGreater(by_class["a"]["share_delta"], 0)
        self.assertLess(by_class["b"]["share_delta"], 0)
        # sorted by |share_delta| descending
        self.assertEqual(drift["classes"][0]["class_id"], "a")

    def test_new_class_stays_finite(self) -> None:
        before = {"class_distribution": {"a": 10}, "assets": 10}
        after = {"class_distribution": {"a": 10, "b": 10}, "assets": 20}
        drift = drift_from_stats(before, after)
        self.assertIsNotNone(drift["kl_divergence"])
        self.assertLess(drift["kl_divergence"], 100)  # smoothing keeps it finite

    def test_empty_side_yields_none_divergence(self) -> None:
        drift = drift_from_stats({"class_distribution": {}}, {"class_distribution": {"a": 5}})
        self.assertIsNone(drift["kl_divergence"])
        self.assertIsNone(drift["js_divergence"])


class DriftEndToEndTest(unittest.TestCase):
    def test_sdk_drift_and_cli_diff_drift(self) -> None:
        from visionpack.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _seed_yolo(root, {"a": "0 0.5 0.5 0.5 0.5\n", "b": "0 0.5 0.5 0.5 0.5\n"})
            ds = VisionPackClient.init(root, name="drift", task="detection")
            ds.import_dir(data, format="yolo")
            ds.snapshot("v1: two alpha")

            for i in range(3):
                _png(data / f"new{i}.png", 50 + i)
                (data / f"new{i}.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            ds.import_dir(data, format="yolo")
            ds.snapshot("v2: beta arrives")

            drift = ds.drift("v1", "v2")
            self.assertEqual(drift["from"], "v1")
            by_class = {item["class_id"]: item for item in drift["classes"]}
            self.assertEqual(by_class["beta"]["before"], 0)
            self.assertEqual(by_class["beta"]["after"], 3)
            self.assertGreater(drift["js_divergence"], 0.0)

            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["diff", "v1", "v2", "--drift", "--json"]), 0)
                envelope = json.loads(buffer.getvalue())
                self.assertIn("drift", envelope["data"])
                self.assertEqual(envelope["data"]["drift"]["to"], "v2")
            finally:
                os.chdir(cwd)


class SnapshotLineageTest(unittest.TestCase):
    def _dataset(self, root: Path) -> VisionPackClient:
        data = _seed_yolo(root, {"a": "0 0.5 0.5 0.5 0.5\n"})
        ds = VisionPackClient.init(root, name="lineage", task="detection")
        ds.import_dir(data, format="yolo")
        ds.snapshot("baseline")
        return ds

    def test_tag_untag_and_find(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._dataset(Path(tmp))
            snap = ds.tag_snapshot("v1", "trained:run-812")
            self.assertEqual(snap["tags"], ["trained:run-812"])
            # idempotent
            snap = ds.tag_snapshot("v1", "trained:run-812")
            self.assertEqual(snap["tags"], ["trained:run-812"])
            ds.tag_snapshot("v1", "release:2026-07")

            self.assertEqual(len(ds.snapshots_by_tag("trained:run-812")), 1)
            self.assertEqual(len(ds.snapshots_by_tag("trained:")), 1)  # bare key: prefix
            self.assertEqual(ds.snapshots_by_tag("trained:other"), [])

            snap = ds.untag_snapshot("v1", "trained:run-812")
            self.assertEqual(snap["tags"], ["release:2026-07"])
            # removing a missing tag is a no-op
            snap = ds.untag_snapshot("v1", "nope")
            self.assertEqual(snap["tags"], ["release:2026-07"])

    def test_empty_tag_is_rejected_and_readonly_view_cannot_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._dataset(Path(tmp))
            with self.assertRaises(VisionPackError):
                ds.tag_snapshot("v1", "   ")
            view = ds.checkout("v1")
            with self.assertRaises(VisionPackError):
                view.tag_snapshot("v1", "trained:x")

    def test_cli_tag_and_list(self) -> None:
        from visionpack.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            self._dataset(Path(tmp))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["snapshot", "tag", "v1", "trained:run-9", "--json"]), 0)
                envelope = json.loads(buffer.getvalue())
                self.assertEqual(envelope["data"]["tags"], ["trained:run-9"])

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["snapshot", "list"]), 0)
                self.assertIn("[trained:run-9]", buffer.getvalue())

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["snapshot", "tag", "v1", "trained:run-9", "--remove", "--json"]), 0)
                envelope = json.loads(buffer.getvalue())
                self.assertEqual(envelope["data"]["tags"], [])
            finally:
                os.chdir(cwd)


class FormatDetectionTest(unittest.TestCase):
    def test_yolo_by_labels_and_furniture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _png(root / "img.png", 1)
            (root / "img.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            self.assertEqual(detect_import_format(root), "yolo")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _png(root / "images" / "img.png", 1)
            (root / "classes.txt").write_text("a\n", encoding="utf-8")
            self.assertEqual(detect_import_format(root), "yolo")

    def test_imagefolder_by_class_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _png(root / "cat" / "c0.png", 1)
            _png(root / "dog" / "d0.png", 2)
            self.assertEqual(detect_import_format(root), "imagefolder")

    def test_coco_json_file_and_directory(self) -> None:
        document = {"images": [], "annotations": [], "categories": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "instances.json"
            annotations.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(detect_import_format(annotations), "coco")
            _png(root / "img.png", 1)
            self.assertEqual(detect_import_format(root), "coco")
            self.assertEqual(coco_json_in(root), annotations)

    def test_plain_images_fall_back_to_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _png(root / "img.png", 1)
            self.assertEqual(detect_import_format(root), "yolo")

    def test_undetectable_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(VisionPackError):
                detect_import_format(Path(tmp))  # empty dir
        with tempfile.TemporaryDirectory() as tmp:
            stray = Path(tmp) / "notes.md"
            stray.write_text("hi", encoding="utf-8")
            with self.assertRaises(VisionPackError):
                detect_import_format(stray)

    def test_sdk_import_auto_detects_coco_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _png(dataset / "img0.png", 3)
            document = {
                "images": [{"id": 1, "file_name": "img0.png", "width": 40, "height": 40}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0}],
                "categories": [{"id": 1, "name": "thing"}],
            }
            (dataset / "instances.json").write_text(json.dumps(document), encoding="utf-8")
            ds = VisionPackClient.init(root, name="auto", task="detection")
            summary = ds.import_dir(dataset)  # format defaults to auto
            self.assertEqual(summary["format"], "coco")
            self.assertEqual(summary["assets"], 1)
            self.assertEqual([c.name for c in ds.classes], ["thing"])

    def test_cli_import_auto_detects_yolo(self) -> None:
        from visionpack.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _seed_yolo(root, {"a": "0 0.5 0.5 0.5 0.5\n"})
            VisionPackClient.init(root, name="cli-auto", task="detection")
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["import", str(data), "--json"]), 0)
                envelope = json.loads(buffer.getvalue())
                self.assertEqual(envelope["data"]["format"], "yolo")
                self.assertEqual(envelope["data"]["assets"], 1)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
