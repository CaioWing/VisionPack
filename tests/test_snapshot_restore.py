from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter, export_yolo
from visionpack.snapshot import create_snapshot, open_snapshot


def _import_batch(root: Path, names: list[str], start: int) -> None:
    """Import a fresh batch of YOLO images (distinct content) into the project."""
    raw = root / f"raw_{start}"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    for offset, name in enumerate(names):
        seed = start + offset
        Image.new("RGB", (32, 32), (seed * 9 % 256, seed * 17 % 256, seed * 29 % 256)).save(
            raw / "images" / f"{name}.png", format="PNG"
        )
        (raw / "labels" / f"{name}.txt").write_text(f"{seed % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    YoloImporter(Project.open(root), raw).run()


class SnapshotRestoreTest(unittest.TestCase):
    def test_export_from_snapshot_reflects_that_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="ss", task="detection")

            _import_batch(root, ["a", "b", "c"], start=0)
            v1 = create_snapshot(Project.open(root), "first three")
            self.assertEqual(v1["version"], "v1")

            _import_batch(root, ["d", "e"], start=100)
            create_snapshot(Project.open(root), "two more")

            # The frozen v1 view sees 3 assets; the live project sees 5.
            view_v1 = open_snapshot(Project.open(root), "v1")
            self.assertEqual(len(view_v1.index.assets()), 3)
            self.assertEqual(len(Project.open(root).index.assets()), 5)

            # Exporting from v1 materializes exactly that version (images live in CAS).
            out_v1 = root / "exp_v1"
            summary_v1 = export_yolo(view_v1, out_v1)
            self.assertEqual(summary_v1["images"], 3)
            self.assertEqual(len(list((out_v1 / "images").glob("*.png"))), 3)

            out_cur = root / "exp_cur"
            summary_cur = export_yolo(Project.open(root), out_cur)
            self.assertEqual(summary_cur["images"], 5)

    def test_identical_state_shares_one_frozen_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="ss")
            _import_batch(root, ["a", "b"], start=0)
            a = create_snapshot(Project.open(root), "snap a")
            b = create_snapshot(Project.open(root), "snap b")  # nothing changed
            # Content-addressed: an unchanged index is frozen once and shared.
            self.assertEqual(a["index_db_hash"], b["index_db_hash"])
            frozen = list((root / ".vp" / "snapshots" / "dbs").glob("*.db"))
            self.assertEqual(len(frozen), 1)


if __name__ == "__main__":
    unittest.main()
