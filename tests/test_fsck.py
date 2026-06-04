from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.fsck import run_fsck


def _seed(root: Path) -> Project:
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    Project.init(root, name="fsck")
    for i in range(3):
        Image.new("RGB", (30, 30), (i * 30, i * 10, i * 50)).save(raw / "images" / f"i{i}.png", format="PNG")
        (raw / "labels" / f"i{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    YoloImporter(Project.open(root), raw).run()
    return Project.open(root)


class FsckTest(unittest.TestCase):
    def test_clean_project_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed(root)
            report = run_fsck(Project.open(root))
            self.assertTrue(report.ok)
            self.assertEqual(report.checked_assets, 3)

    def test_missing_object_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            victim = project.index.assets()[0].resolved_path(project.root)
            victim.unlink()
            report = run_fsck(Project.open(root))
            self.assertFalse(report.ok)
            self.assertIn("object.missing", [i.code for i in report.errors])

    def test_deep_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            victim = project.index.assets()[0].resolved_path(project.root)
            victim.write_bytes(b"corrupted bytes that won't hash the same")
            quick = run_fsck(Project.open(root), deep=False)
            self.assertTrue(quick.ok)  # the file still exists -> quick check passes
            deep = run_fsck(Project.open(root), deep=True)
            self.assertIn("object.hash_mismatch", [i.code for i in deep.errors])

    def test_orphan_object_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed(root)
            stray = root / ".vp" / "objects" / "sha256" / "ab" / "cd" / ("ab" + "0" * 62)
            stray.parent.mkdir(parents=True, exist_ok=True)
            stray.write_bytes(b"orphan")
            report = run_fsck(Project.open(root))
            self.assertTrue(report.ok)  # orphan is a warning, not an error
            self.assertIn("object.orphan", [i.code for i in report.warnings])


if __name__ == "__main__":
    unittest.main()
