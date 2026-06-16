from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import fsspec
from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import export_yolo
from visionpack.sources import sync_sources


def _png_bytes(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


class LocalHardlinkExportTest(unittest.TestCase):
    def test_export_hardlinks_from_cas_instead_of_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="hl")
            imgs, lbls = root / "imgs", root / "lbls"
            imgs.mkdir()
            lbls.mkdir()
            (imgs / "a.png").write_bytes(_png_bytes(1))
            (lbls / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")
            project = Project.open(root)
            project.manifest.sources = [
                {"name": "s1", "images": imgs.as_posix(), "labels": lbls.as_posix(), "copy": "copy"}
            ]
            project.save_manifest()
            project = Project.open(root)
            sync_sources(project)

            out = root / "export"
            export_yolo(Project.open(root), out)

            asset = next(iter(Project.open(root).index.assets()))
            cas_object = asset.resolved_path(root)
            exported = next((out / "images").glob("*.png"))
            # The export shares the CAS object's inode — zero extra bytes.
            self.assertTrue(os.path.samefile(cas_object, exported))


class CloudManifestExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = fsspec.filesystem("memory")
        for path in list(self.fs.find("/src")) + list(self.fs.find("/dst")):
            self.fs.rm(path)
        self.fs.pipe("/src/imgs/a.png", _png_bytes(1))
        self.fs.pipe("/src/imgs/b.png", _png_bytes(2))
        self.fs.pipe("/src/lbls/a.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/src/lbls/b.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/src/lbls/classes.txt", b"cat\n")

    def test_remote_assets_become_a_streaming_manifest_with_no_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="cloud")
            project = Project.open(root)
            project.manifest.sources = [
                {"name": "s1", "images": "memory://src/imgs", "labels": "memory://src/lbls", "copy": "copy"}
            ]
            project.manifest.target = "memory://dst"
            project.save_manifest()
            sync_sources(Project.open(root))

            out = root / "export"
            summary = export_yolo(Project.open(root), out)

            self.assertEqual(summary["streamed"], 2)
            # No image bytes were materialized locally...
            self.assertEqual(list((out / "images").glob("*")) if (out / "images").exists() else [], [])
            # ...but labels (generated from the index) and the manifest are written.
            self.assertEqual(len(list((out / "labels").glob("*.txt"))), 2)
            entries = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertTrue(entry["uri"].startswith("memory://dst/objects/sha256/"))
                self.assertIn("image", entry)
                self.assertIn("label", entry)


if __name__ == "__main__":
    unittest.main()
