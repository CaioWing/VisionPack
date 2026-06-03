from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.models import BBox, Keypoints, ObjectAnnotation, Polygon, parse_geometry
from visionpack.core.project import Project
from visionpack.formats.classification import ImageFolderImporter, export_imagefolder
from visionpack.formats.coco import CocoImporter, export_coco
from visionpack.split import create_split
from visionpack.stats import collect_stats, split_breakdown
from visionpack.validation import validate_project


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


class GeometryModelTest(unittest.TestCase):
    def test_legacy_bbox_schema_still_loads(self) -> None:
        # Old persisted form: a bare "bbox" with no "geometry"/"kind".
        obj = ObjectAnnotation.from_dict({"bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "class_id": "cat"})
        self.assertIsInstance(obj.geometry, BBox)
        self.assertEqual(obj.bbox.width, 3)

    def test_geometry_roundtrips_through_dict(self) -> None:
        for geometry in (
            BBox(1, 2, 3, 4),
            Polygon(rings=[[0, 0, 10, 0, 10, 10, 0, 10]]),
            Keypoints(points=[5, 5, 2, 8, 8, 0]),
        ):
            restored = parse_geometry(geometry.to_dict())
            self.assertEqual(restored.to_dict(), geometry.to_dict())

    def test_derived_bbox_from_polygon_and_keypoints(self) -> None:
        poly = Polygon(rings=[[0, 0, 10, 0, 10, 20, 0, 20]])
        self.assertEqual((poly.bounding_box().width, poly.bounding_box().height), (10, 20))
        # Only visible keypoints (v > 0) bound the box.
        kp = Keypoints(points=[5, 5, 2, 100, 100, 0, 15, 25, 1])
        box = kp.bounding_box()
        self.assertEqual((box.x, box.y, box.width, box.height), (5, 5, 10, 20))

    def test_classification_object_has_no_geometry(self) -> None:
        obj = ObjectAnnotation(class_id="cat", geometry=None)
        self.assertIsNone(obj.bbox)
        self.assertIsNone(obj.to_dict()["geometry"])


class ClassificationTest(unittest.TestCase):
    def _seed(self, root: Path) -> Project:
        project = Project.init(root, name="cls", task="classification")
        data = root / "data"
        for i in range(4):
            _png(data / "cat" / f"c{i}.png", i)
        for i in range(4):
            _png(data / "dog" / f"d{i}.png", i + 50)
        ImageFolderImporter(project, data).run()
        return Project.open(root)

    def test_imagefolder_import_labels_by_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._seed(root)
            self.assertEqual({c.name for c in project.manifest.classes}, {"cat", "dog"})
            stats = collect_stats(project)
            self.assertEqual(stats["assets"], 8)
            self.assertEqual(stats["images_without_annotations"], 0)
            self.assertEqual(stats["class_distribution"], {"cat": 4, "dog": 4})

    def test_validation_clean_for_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._seed(root)
            report = validate_project(project)
            # No bbox/annotation-missing errors for a fully-labeled classification set.
            self.assertEqual(report.errors, [])

    def test_stratified_split_balances_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            create_split(Project.open(root), train=0.5, val=0.25, test=0.25, strategy="stratified")
            breakdown = split_breakdown(Project.open(root))
            self.assertEqual(breakdown["sets"]["train"]["class_distribution"], {"cat": 2, "dog": 2})

    def test_imagefolder_export_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed(root)
            out = root / "exp"
            summary = export_imagefolder(Project.open(root), out)
            self.assertEqual(summary["images"], 8)
            self.assertEqual(len(list((out / "cat").glob("*.png"))), 4)
            self.assertEqual(len(list((out / "dog").glob("*.png"))), 4)


class SegmentationTest(unittest.TestCase):
    def _coco_seg(self, root: Path) -> Project:
        images = root / "images"
        _png(images / "a.png", 1)
        document = {
            "images": [{"id": 1, "file_name": "a.png", "width": 40, "height": 40}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [5, 5, 20, 20],
                    "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "lesion"}],
        }
        ann = root / "instances.json"
        ann.write_text(json.dumps(document), encoding="utf-8")
        project = Project.init(root, name="seg", task="segmentation")
        CocoImporter(project, ann, images).run()
        return Project.open(root)

    def test_polygon_imported_and_bbox_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._coco_seg(root)
            obj = project.index.annotations()[0].objects[0]
            self.assertIsInstance(obj.geometry, Polygon)
            # Derived bbox spans the polygon: x5..25 -> w20.
            self.assertEqual((obj.bbox.x, obj.bbox.width), (5, 20))

    def test_segmentation_survives_coco_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._coco_seg(root)
            out = root / "exp"
            export_coco(Project.open(root), out)
            doc = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
            self.assertIn("segmentation", doc["annotations"][0])
            self.assertEqual(doc["annotations"][0]["segmentation"], [[5, 5, 25, 5, 25, 25, 5, 25]])


if __name__ == "__main__":
    unittest.main()
