from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visionpack.cli.commands.import_ import _record_source
from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.sources import sync_sources


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _yolo_dataset(root: Path) -> Path:
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")  # at the ROOT, not labels/
    for i in range(4):
        _png(raw / "images" / f"i{i}.png", i)
        (raw / "labels" / f"i{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return raw


def _args(**kw) -> SimpleNamespace:
    base = {"format": "yolo", "source": None, "images": None, "copy": "ingest", "name": None}
    base.update(kw)
    return SimpleNamespace(**base)


class ImportRecordTest(unittest.TestCase):
    def test_import_records_source_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="rec")
            raw = _yolo_dataset(root)
            project = Project.open(root)
            YoloImporter(project, raw).run()

            name = _record_source(project, _args(source=str(raw)))
            self.assertEqual(name, "raw")
            reopened = Project.open(root)
            self.assertEqual(len(reopened.manifest.sources), 1)
            entry = reopened.manifest.sources[0]
            self.assertEqual(entry["format"], "yolo")
            self.assertEqual(entry["root"], "./raw")

    def test_reimport_same_path_does_not_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="rec")
            raw = _yolo_dataset(root)
            project = Project.open(root)
            YoloImporter(project, raw).run()
            _record_source(project, _args(source=str(raw)))
            # Re-record through a freshly-reloaded manifest (carries pydantic defaults).
            again = _record_source(Project.open(root), _args(source=str(raw)))
            self.assertIsNone(again)
            self.assertEqual(len(Project.open(root).manifest.sources), 1)

    def test_recorded_yolo_source_syncs_without_inventing_classes(self) -> None:
        # Regression: classes.txt lives at the source root, but root-shorthand
        # expands labels to root/labels — sync must still find it and not invent
        # class_0/class_1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="rec")
            raw = _yolo_dataset(root)
            project = Project.open(root)
            YoloImporter(project, raw).run()
            _record_source(project, _args(source=str(raw)))

            summaries = sync_sources(Project.open(root))
            self.assertEqual(summaries[0].classes_added, 0)
            self.assertEqual(summaries[0].assets_added, 0)  # already present (idempotent)
            self.assertEqual({c.name for c in Project.open(root).manifest.classes}, {"cat", "dog"})

    def test_imagefolder_source_syncs_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="rec", task="classification")
            data = root / "data"
            for i in range(3):
                _png(data / "cat" / f"c{i}.png", i)
                _png(data / "dog" / f"d{i}.png", i + 30)
            project = Project.open(root)
            project.manifest.sources = [{"name": "folder", "format": "imagefolder", "root": "./data", "copy": "ingest"}]
            project.save_manifest()

            summaries = sync_sources(Project.open(root))
            self.assertEqual(summaries[0].assets_added, 6)
            opened = Project.open(root)
            self.assertEqual({c.name for c in opened.manifest.classes}, {"cat", "dog"})
            self.assertTrue(all(a.source == "folder" for a in opened.index.assets()))

    def test_manifest_is_written_with_section_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="rec")
            text = (root / "visionpack.yaml").read_text(encoding="utf-8")
            self.assertIn("# VisionPack dataset manifest", text)
            self.assertIn("# Validation policy", text)


if __name__ == "__main__":
    unittest.main()
