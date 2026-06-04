from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.coco import CocoImporter, export_coco
from visionpack.stats import collect_stats
from visionpack.validation import validate_project


def _write_coco_fixture(root: Path) -> tuple[Path, Path]:
    images_dir = root / "coco" / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), color=(40, 40, 40)).save(images_dir / "img001.jpg", format="JPEG")
    document = {
        "images": [{"id": 1, "file_name": "img001.jpg", "width": 200, "height": 100}],
        "categories": [
            {"id": 1, "name": "scratch"},
            {"id": 2, "name": "dent"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [20, 30, 60, 40], "area": 2400, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [100, 10, 50, 50], "area": 2500, "iscrowd": 0},
        ],
    }
    annotations_path = root / "coco" / "instances.json"
    annotations_path.write_text(json.dumps(document), encoding="utf-8")
    return annotations_path, images_dir


class CocoFlowTest(unittest.TestCase):
    def test_import_discovers_classes_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="coco-demo")
            annotations_path, images_dir = _write_coco_fixture(root)

            summary = CocoImporter(project, annotations_path, images_dir).run()
            self.assertEqual(summary.assets, 1)
            self.assertEqual(summary.annotations, 1)
            self.assertEqual(summary.objects, 2)

            reopened = Project.open(root)
            self.assertEqual([item.id for item in reopened.manifest.classes], ["scratch", "dent"])
            self.assertEqual(collect_stats(reopened)["class_distribution"], {"dent": 1, "scratch": 1})
            self.assertTrue(validate_project(reopened, strict=True).ok)

    def test_export_round_trips_bboxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="coco-demo")
            annotations_path, images_dir = _write_coco_fixture(root)
            CocoImporter(project, annotations_path, images_dir).run()

            output = root / "exports" / "coco-v1"
            summary = export_coco(Project.open(root), output)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["annotations"], 2)

            document = json.loads((output / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(len(document["categories"]), 2)
            self.assertEqual(len(document["images"]), 1)
            bboxes = sorted(ann["bbox"] for ann in document["annotations"])
            self.assertEqual(bboxes, [[20.0, 30.0, 60.0, 40.0], [100.0, 10.0, 50.0, 50.0]])

    def test_missing_image_file_is_skipped_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="coco-demo")
            annotations_path, images_dir = _write_coco_fixture(root)
            (images_dir / "img001.jpg").unlink()  # the only image is now missing
            summary = CocoImporter(project, annotations_path, images_dir).run()
            # Reported as a skipped failure rather than raising and aborting.
            self.assertEqual(summary.assets, 0)
            self.assertEqual(len(summary.failures), 1)
            self.assertIn("img001.jpg", summary.failures[0].error)


if __name__ == "__main__":
    unittest.main()
