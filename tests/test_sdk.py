from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack import sdk
from visionpack.core.errors import VisionPackError
from visionpack.sdk import VisionPackClient


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed_yolo(root: Path, count: int = 6) -> Path:
    data = root / "raw"
    for i in range(count):
        _png(data / f"img{i}.png", i + 1)
        (data / f"img{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (data / "classes.txt").write_text("widget\n", encoding="utf-8")
    return data


class SdkLifecycleTest(unittest.TestCase):
    def test_full_dataset_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _seed_yolo(root)

            ds = VisionPackClient.init(root, name="lifecycle", task="detection")
            self.assertEqual(ds.name, "lifecycle")
            self.assertEqual(ds.task, "detection")

            summary = ds.import_dir(data, format="yolo")
            self.assertEqual(summary["assets"], 6)
            self.assertEqual(summary["failures"], [])
            self.assertEqual(len(ds), 6)
            self.assertEqual([c.name for c in ds.classes], ["widget"])

            report = ds.validate()
            self.assertTrue(report.ok)
            audit = ds.audit(min_class_count=1)
            self.assertTrue(audit.ok, [f.message for f in audit.findings])
            self.assertEqual(ds.stats()["assets"], 6)

            split = ds.create_split(train=0.5, val=0.25, test=0.25, strategy="random")
            self.assertEqual(sum(len(ids) for ids in split.sets.values()), 6)
            ds.lock_split()
            self.assertTrue(ds.split().locked)

            snap = ds.snapshot("baseline")
            self.assertEqual(snap["version"], "v1")
            self.assertEqual(len(ds.snapshots()), 1)

            out = root / "exports" / "yolo"
            result = ds.export(out, format="yolo", split="default")
            self.assertEqual(result["images"], 6)
            self.assertTrue((out / "data.yaml").exists())

            # streaming access
            pairs = list(ds.samples())
            self.assertEqual(len(pairs), 6)
            self.assertTrue(all(ann is not None for _, ann in pairs))

    def test_module_level_init_and_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdk.init(root, name="mod", task="classification")
            ds = sdk.open(root)
            self.assertEqual(ds.name, "mod")
            self.assertEqual(ds.task, "classification")

    def test_import_coco_requires_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = VisionPackClient.init(Path(tmp), name="x")
            with self.assertRaises(VisionPackError):
                ds.import_dir("annotations.json", format="coco")

    def test_unknown_formats_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = VisionPackClient.init(Path(tmp), name="x")
            with self.assertRaises(VisionPackError):
                ds.import_dir("nowhere", format="voc")
            with self.assertRaises(VisionPackError):
                ds.export(Path(tmp) / "out", format="tfrecord")


class SdkModelLoopTest(unittest.TestCase):
    def _dataset(self, root: Path) -> VisionPackClient:
        data = _seed_yolo(root, count=4)
        ds = VisionPackClient.init(root, name="loop", task="detection")
        ds.import_dir(data, format="yolo")
        return ds

    def _predictions_file(self, ds: VisionPackClient, root: Path) -> Path:
        items = [
            {
                "image": asset.id,
                "objects": [{"class": "widget", "confidence": 0.9, "bbox": [10, 10, 20, 20]}],
            }
            for asset in ds.assets()
        ]
        path = root / "preds.json"
        path.write_text(json.dumps({"predictions": items}), encoding="utf-8")
        return path

    def test_evaluate_autolabel_and_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = self._dataset(root)
            ds.create_split(train=0.5, val=0.25, test=0.25, strategy="random")
            preds = self._predictions_file(ds, root)

            metrics = ds.evaluate(preds, split="default", set_name="test")
            self.assertEqual(metrics["task"], "detection")
            self.assertIn("map50", json.dumps(metrics).lower())

            queue = ds.annotation_queue(preds, include_labeled=True)
            self.assertIsInstance(queue, list)

            result = ds.autolabel(preds, min_confidence=0.5)
            self.assertEqual(result["skipped_existing"], 4)  # all already labeled

    def test_predictions_accepts_loaded_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = self._dataset(root)
            loaded = ds.load_predictions(self._predictions_file(ds, root))
            self.assertIs(ds.load_predictions(loaded), loaded)


class SdkSnapshotViewTest(unittest.TestCase):
    def test_checkout_is_readonly_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _seed_yolo(root, count=3)
            ds = VisionPackClient.init(root, name="ver", task="detection")
            ds.import_dir(data, format="yolo")
            ds.snapshot("three images")

            # grow the live dataset past the snapshot
            _png(data / "extra.png", 99)
            (data / "extra.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            ds.import_dir(data, format="yolo")
            self.assertEqual(len(ds), 4)

            view = ds.checkout("v1")
            self.assertTrue(view.readonly)
            self.assertEqual(len(view), 3)
            with self.assertRaises(VisionPackError):
                view.snapshot("nope")
            with self.assertRaises(VisionPackError):
                view.create_split()

            out = root / "exports" / "v1"
            result = view.export(out, format="yolo")
            self.assertEqual(result["images"], 3)


class SdkLockTest(unittest.TestCase):
    def test_mutations_take_the_project_lock(self) -> None:
        from visionpack.core.lock import project_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _seed_yolo(root, count=2)
            ds = VisionPackClient.init(root, name="locked")
            with project_lock(ds.root):
                with self.assertRaises(VisionPackError):
                    ds.import_dir(data, format="yolo")


if __name__ == "__main__":
    unittest.main()
