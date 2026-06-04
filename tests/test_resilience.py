from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.sources import sync_sources


def _good(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 11 % 256, seed * 19 % 256)).save(path, format="PNG")


def _build(root: Path, n_good: int = 5) -> Path:
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    for i in range(n_good):
        _good(raw / "images" / f"g{i}.png", i)
        (raw / "labels" / f"g{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    # One corrupt file with a valid image extension but garbage bytes.
    (raw / "images" / "bad.png").write_bytes(b"not really a png \x00\x01\x02")
    (raw / "labels" / "bad.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return raw


class ResilienceTest(unittest.TestCase):
    def test_import_skips_corrupt_and_keeps_good(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="r")
            raw = _build(root, n_good=5)
            summary = YoloImporter(Project.open(root), raw).run()
            self.assertEqual(summary.assets, 5)  # the 5 good images survived
            self.assertEqual(len(summary.failures), 1)
            self.assertIn("bad.png", summary.failures[0].path)
            # And the good ones are actually queryable.
            self.assertEqual(len(Project.open(root).index.assets()), 5)

    def test_sync_skips_corrupt_and_keeps_good(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="r")
            _build(root, n_good=4)
            project.manifest.sources = [{"name": "cam", "format": "yolo", "root": "./raw", "copy": "ingest"}]
            project.save_manifest()
            summaries = sync_sources(Project.open(root))
            self.assertEqual(summaries[0].assets_added, 4)
            self.assertEqual(len(summaries[0].failures), 1)
            self.assertIn("bad.png", summaries[0].failures[0].path)


if __name__ == "__main__":
    unittest.main()
