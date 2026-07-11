from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.models import Polygon
from visionpack.core.project import Project
from visionpack.formats.masks import export_masks
from visionpack.formats.yolo import YoloImporter, export_yolo


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed_yolo_seg(root: Path) -> Project:
    """One image with a YOLO-seg square: (0.25, 0.25) .. (0.75, 0.75) -> abs 10..30."""
    data = root / "raw"
    _png(data / "a.png", 1)
    (data / "a.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    (data / "classes.txt").write_text("lesion\n", encoding="utf-8")
    project = Project.init(root, name="seg", task="segmentation")
    YoloImporter(project, data).run()
    return Project.open(root)


class YoloSegTest(unittest.TestCase):
    def test_seg_label_imports_as_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_yolo_seg(root)
            obj = project.index.annotations()[0].objects[0]
            self.assertIsInstance(obj.geometry, Polygon)
            self.assertEqual(obj.geometry.rings, [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]])
            # The derived bbox spans the polygon.
            self.assertEqual((obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height), (10.0, 10.0, 20.0, 20.0))

    def test_seg_export_roundtrips_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_yolo_seg(root)
            out = root / "exp"
            summary = export_yolo(project, out)  # task=segmentation -> seg lines by default
            self.assertEqual(summary["objects"], 1)
            label = next((out / "labels").glob("*.txt")).read_text(encoding="utf-8").strip()
            parts = label.split()
            self.assertEqual(parts[0], "0")
            self.assertEqual([float(v) for v in parts[1:]], [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75])

    def test_bbox_project_still_exports_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "raw"
            _png(data / "a.png", 1)
            (data / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (data / "classes.txt").write_text("obj\n", encoding="utf-8")
            project = Project.init(root, name="det", task="detection")
            YoloImporter(project, data).run()
            out = root / "exp"
            export_yolo(Project.open(root), out)
            label = next((out / "labels").glob("*.txt")).read_text(encoding="utf-8").strip()
            self.assertEqual(len(label.split()), 5)

    def test_detection_export_can_opt_into_seg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "raw"
            _png(data / "a.png", 1)
            (data / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (data / "classes.txt").write_text("obj\n", encoding="utf-8")
            project = Project.init(root, name="det", task="detection")
            YoloImporter(project, data).run()
            out = root / "exp"
            export_yolo(Project.open(root), out, seg=True)
            label = next((out / "labels").glob("*.txt")).read_text(encoding="utf-8").strip()
            # The box degrades to its four corners: class + 8 coordinates.
            self.assertEqual(len(label.split()), 9)


class MasksExportTest(unittest.TestCase):
    def test_polygon_rasterizes_to_class_index_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_yolo_seg(root)
            out = root / "exp"
            summary = export_masks(project, out)
            self.assertEqual(summary["masks"], 1)
            self.assertEqual(summary["objects"], 1)

            mask_path = next((out / "masks").glob("*.png"))
            mask = Image.open(mask_path)
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.getpixel((20, 20)), 1)  # inside the square: class index 1
            self.assertEqual(mask.getpixel((2, 2)), 0)  # outside: background
            self.assertEqual(
                (out / "classes.txt").read_text(encoding="utf-8"),
                "0 __background__\n1 lesion\n",
            )

    def test_images_are_exported_alongside_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_yolo_seg(root)
            out = root / "exp"
            export_masks(project, out)
            self.assertEqual(len(list((out / "images").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
