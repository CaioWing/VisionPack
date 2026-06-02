from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.snapshot import create_snapshot, load_snapshot


def _seed_project(root: Path) -> Project:
    project = Project.init(root, name="dedup-demo")
    source = root / "raw"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir(parents=True)
    (source / "classes.txt").write_text("scratch\n", encoding="utf-8")
    Image.new("RGB", (100, 50), color=(0, 0, 0)).save(source / "images" / "img001.png", format="PNG")
    (source / "labels" / "img001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    YoloImporter(project, source).run()
    return Project.open(root)


class SnapshotDedupTest(unittest.TestCase):
    def test_inventory_is_stored_as_blob_and_rehydrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_project(root)

            snapshot = create_snapshot(Project.open(root), "initial")
            version_file = root / ".vp" / "snapshots" / "v1.json"
            on_disk = json.loads(version_file.read_text(encoding="utf-8"))

            # The version file references the inventory by hash, not inline.
            self.assertIn("inventory_hash", on_disk)
            self.assertNotIn("inventory", on_disk)

            # load_snapshot rehydrates it transparently.
            self.assertIn("inventory", snapshot)
            loaded = load_snapshot(Project.open(root), "v1")
            self.assertEqual(loaded["inventory"], snapshot["inventory"])

    def test_unchanged_inventory_is_shared_across_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_project(root)

            first = create_snapshot(Project.open(root), "one")
            second = create_snapshot(Project.open(root), "two")

            # No data changed, so both versions point at the same blob...
            self.assertEqual(first["inventory_hash"], second["inventory_hash"])
            # ...and that blob exists exactly once.
            blobs = list((root / ".vp" / "snapshots" / "blobs").glob("*.json"))
            self.assertEqual(len(blobs), 1)


if __name__ == "__main__":
    unittest.main()
