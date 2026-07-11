from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.audit import AuditThresholds, audit_project
from visionpack.core.errors import FormatError
from visionpack.core.project import Project
from visionpack.formats.base import safe_path_component
from visionpack.formats.classification import ImageFolderImporter, export_imagefolder
from visionpack.formats.yolo import YoloImporter
from visionpack.media import image_info_from_bytes


def _png(path: Path, seed: int, size: tuple[int, int] = (40, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed_detection(root: Path, labels: dict[str, str]) -> Project:
    data = root / "raw"
    for index, (name, text) in enumerate(labels.items(), start=1):
        _png(data / f"{name}.png", index)
        (data / f"{name}.txt").write_text(text, encoding="utf-8")
    (data / "classes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    project = Project.init(root, name="audit", task="detection")
    YoloImporter(project, data).run()
    return Project.open(root)


class AuditBoxChecksTest(unittest.TestCase):
    def _codes(self, project: Project, thresholds: AuditThresholds | None = None) -> dict[str, int]:
        return audit_project(project, thresholds or AuditThresholds()).counts_by_code()

    def test_duplicate_boxes_same_class_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.5 0.5\n"})
            codes = self._codes(project)
            self.assertEqual(codes.get("box.duplicate"), 1)

    def test_overlapping_boxes_of_different_classes_are_not_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.5 0.5\n"})
            codes = self._codes(project)
            self.assertNotIn("box.duplicate", codes)

    def test_tiny_box_is_degenerate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 4x4 px on a 40x40 image, below the 8 px default.
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.1 0.1\n"})
            codes = self._codes(project)
            self.assertEqual(codes.get("box.degenerate"), 1)

    def test_aspect_ratio_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 36x9 px: ratio 4:1 — an outlier once the threshold is lowered.
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.9 0.225\n"})
            self.assertNotIn("box.aspect_outlier", self._codes(project))
            codes = self._codes(project, AuditThresholds(max_aspect_ratio=3.0))
            self.assertEqual(codes.get("box.aspect_outlier"), 1)

    def test_edge_pinned_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Full-width band at the top: pinned to three borders, covers 25%.
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.125 1.0 0.25\n"})
            codes = self._codes(project)
            self.assertEqual(codes.get("box.edge_pinned"), 1)
            self.assertNotIn("box.covers_image", codes)

    def test_full_image_box_reports_coverage_not_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 1.0 1.0\n"})
            codes = self._codes(project)
            self.assertEqual(codes.get("box.covers_image"), 1)
            self.assertNotIn("box.edge_pinned", codes)

    def test_box_touching_one_border_is_fine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 20x20 box flush with the left edge only.
            project = _seed_detection(Path(tmp), {"a": "0 0.25 0.5 0.5 0.5\n"})
            codes = self._codes(project)
            self.assertNotIn("box.edge_pinned", codes)


class AuditClassBalanceTest(unittest.TestCase):
    def test_rare_class_and_imbalance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = {f"img{i}": "0 0.5 0.5 0.5 0.5\n" for i in range(12)}
            labels["odd"] = "1 0.5 0.5 0.5 0.5\n"
            project = _seed_detection(Path(tmp), labels)
            report = audit_project(project, AuditThresholds(min_class_count=5, imbalance_ratio=10.0))
            codes = report.counts_by_code()
            self.assertEqual(codes.get("class.rare"), 1)  # beta has 1 < 5
            self.assertEqual(codes.get("class.imbalance"), 1)  # 12:1 > 10:1
            rare = next(f for f in report.findings if f.code == "class.rare")
            self.assertEqual(rare.class_id, "beta")

    def test_balanced_dataset_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = {f"a{i}": "0 0.5 0.5 0.5 0.5\n" for i in range(3)}
            labels.update({f"b{i}": "1 0.5 0.5 0.5 0.5\n" for i in range(3)})
            project = _seed_detection(Path(tmp), labels)
            report = audit_project(project, AuditThresholds(min_class_count=2, imbalance_ratio=10.0))
            self.assertTrue(report.ok, [f.message for f in report.findings])
            self.assertEqual(report.images_audited, 6)
            self.assertEqual(report.objects_audited, 6)


class AuditThresholdsConfigTest(unittest.TestCase):
    def test_manifest_and_overrides_layering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.5 0.5\n"})
            project.manifest.validation["audit"] = {"min_box_px": 12, "unknown_key": True}
            thresholds = AuditThresholds.from_project(project, duplicate_iou=0.5, max_aspect_ratio=None)
            self.assertEqual(thresholds.min_box_px, 12)  # from manifest
            self.assertEqual(thresholds.duplicate_iou, 0.5)  # explicit override
            self.assertEqual(thresholds.max_aspect_ratio, 20.0)  # None override ignored


class AuditCliTest(unittest.TestCase):
    def test_cli_json_envelope_and_exit_codes(self) -> None:
        import contextlib
        import io as _io
        import os

        from visionpack.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_detection(Path(tmp), {"a": "0 0.5 0.5 0.1 0.1\n"})
            self.assertIsNotNone(project)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                buffer = _io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["audit", "--json"]), 0)  # advisory by default
                envelope = json.loads(buffer.getvalue())
                self.assertEqual(envelope["command"], "audit")
                self.assertGreaterEqual(envelope["data"]["by_code"]["box.degenerate"], 1)

                buffer = _io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self.assertEqual(main(["audit", "--json", "--fail-on-findings"]), 1)
            finally:
                os.chdir(cwd)


class SafePathComponentTest(unittest.TestCase):
    def test_traversal_and_separators_are_neutralized(self) -> None:
        self.assertEqual(safe_path_component(".."), "unnamed")
        self.assertEqual(safe_path_component("."), "unnamed")
        self.assertEqual(safe_path_component(""), "unnamed")
        self.assertEqual(safe_path_component("../../etc"), ".._.._etc")  # separators replaced, no traversal
        self.assertNotIn("/", safe_path_component("a/b"))
        self.assertNotIn("\\", safe_path_component("a\\b"))
        self.assertEqual(safe_path_component("dog"), "dog")

    def test_export_with_hostile_class_name_stays_inside_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            for i in range(2):
                _png(data / "cat" / f"c{i}.png", i)
            project = Project.init(root, name="cls", task="classification")
            ImageFolderImporter(project, data).run()
            project = Project.open(root)
            # Simulate a hostile class name arriving from imported metadata.
            project.manifest.classes[0].name = "../../escape"
            output = root / "exports" / "out"
            export_imagefolder(project, output)
            written = [p for p in output.rglob("*") if p.is_file()]
            self.assertTrue(written)
            for path in written:
                self.assertTrue(path.resolve().is_relative_to(output.resolve()), path)
            self.assertFalse((root / "escape").exists())


class DecompressionBombTest(unittest.TestCase):
    def test_bomb_is_reported_as_format_error(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (40, 40)).save(buffer, format="PNG")
        original = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 10  # force the bomb check to trip on a tiny image
        try:
            with self.assertRaises(FormatError):
                image_info_from_bytes(buffer.getvalue(), Path("bomb.png"))
        finally:
            Image.MAX_IMAGE_PIXELS = original


if __name__ == "__main__":
    unittest.main()
