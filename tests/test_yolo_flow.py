from __future__ import annotations

import struct
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import zstandard as zstd

from visionpack.core.project import Project
from visionpack.diff import diff_snapshots
from visionpack.formats.yolo import YoloImporter, export_yolo
from visionpack.packing import pack_archive
from visionpack.snapshot import create_snapshot
from visionpack.stats import collect_stats
from visionpack.validation import validate_project


class YoloFlowTest(unittest.TestCase):
    def test_import_validate_snapshot_diff_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="factory-defects")
            source = root / "raw"
            (source / "images").mkdir(parents=True)
            (source / "labels").mkdir(parents=True)
            (source / "classes.txt").write_text("scratch\ndent\n", encoding="utf-8")
            _write_png_header(source / "images" / "img001.png", width=100, height=50)
            (source / "labels" / "img001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

            summary = YoloImporter(project, source).run()
            self.assertEqual(summary.assets, 1)
            self.assertEqual(summary.objects, 1)

            reopened = Project.open(root)
            stats = collect_stats(reopened)
            self.assertEqual(stats["assets"], 1)
            self.assertEqual(stats["class_distribution"], {"scratch": 1})
            self.assertTrue(validate_project(reopened, strict=True).ok)

            first = create_snapshot(reopened, "initial import")
            self.assertEqual(first["version"], "v1")

            # Modify the annotation and verify snapshots capture the change.
            label = source / "labels" / "img001.txt"
            label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            YoloImporter(reopened, source).run()
            second = create_snapshot(Project.open(root), "wider box")
            self.assertEqual(second["version"], "v2")
            diff = diff_snapshots(Project.open(root), "v1", "v2")
            self.assertEqual(len(diff["annotations_modified"]), 1)

            output = root / "exports" / "yolo-v2"
            export_summary = export_yolo(Project.open(root), output)
            self.assertEqual(export_summary["images"], 1)
            self.assertTrue((output / "classes.txt").exists())
            self.assertEqual(len(list((output / "labels").glob("*.txt"))), 1)

            pack_summary = pack_archive(Project.open(root))
            self.assertEqual(pack_summary.format, "tar.zst")
            self.assertEqual(pack_summary.assets, 1)
            archive_names = _read_tar_zst_names(pack_summary.path)
            self.assertIn("visionpack.yaml", archive_names)
            self.assertIn(".vp/db/index.json", archive_names)
            self.assertIn(".vp/snapshots/v1.json", archive_names)
            self.assertIn(".vp/snapshots/v2.json", archive_names)
            self.assertIn("pack.json", archive_names)
            self.assertTrue(any(name.startswith(".vp/objects/sha256/") for name in archive_names))

    def test_import_infers_classes_when_yolo_names_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="unnamed-classes")
            source = root / "raw"
            source.mkdir()
            _write_png_header(source / "img001.png", width=64, height=64)
            (source / "img001.txt").write_text("\ufeff1 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            YoloImporter(project, source).run()
            reopened = Project.open(root)

            self.assertEqual([item.id for item in reopened.manifest.classes], ["class_0", "class_1"])
            self.assertEqual(collect_stats(reopened)["class_distribution"], {"class_1": 1})


def _write_png_header(path: Path, width: int, height: int) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"
    path.write_bytes(signature + ihdr + b"\x00" * 16)


def _read_tar_zst_names(path: Path) -> set[str]:
    with path.open("rb") as compressed:
        payload = zstd.ZstdDecompressor().stream_reader(compressed).read()
    with tarfile.open(fileobj=BytesIO(payload), mode="r:") as tar:
        return set(tar.getnames())


if __name__ == "__main__":
    unittest.main()
