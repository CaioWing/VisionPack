from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import fsspec
from PIL import Image

from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project
from visionpack.sources import sync_sources
from visionpack.sources.resolver import FsspecResolver


def _png_bytes(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _target_objects(fs: fsspec.AbstractFileSystem) -> list[str]:
    return [p for p in fs.find("/dst") if "/objects/sha256/" in p]


class CloudTargetSyncTest(unittest.TestCase):
    """End-to-end cloud-internal sync over fsspec's in-memory filesystem: a
    memory:// source feeding a memory:// content-addressed target, server-side."""

    def setUp(self) -> None:
        self.fs = fsspec.filesystem("memory")
        for path in list(self.fs.find("/src")) + list(self.fs.find("/dst")):
            self.fs.rm(path)
        self.fs.pipe("/src/imgs/a.png", _png_bytes(1))
        self.fs.pipe("/src/imgs/b.png", _png_bytes(2))
        self.fs.pipe("/src/lbls/a.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/src/lbls/b.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/src/lbls/classes.txt", b"cat\n")

    def _project(self, tmp: str, copy: str, target: str | None = "memory://dst") -> Project:
        root = Path(tmp)
        Project.init(root, name="cloud")
        project = Project.open(root)
        project.manifest.sources = [
            {"name": "s1", "images": "memory://src/imgs", "labels": "memory://src/lbls", "copy": copy}
        ]
        project.manifest.target = target
        project.save_manifest()
        return Project.open(root)

    def test_copy_lands_objects_in_target_cas_server_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = sync_sources(self._project(tmp, "copy"))[0]
            self.assertEqual(summary.assets_added, 2)
            # Both images copied into the target's content-addressed layout...
            self.assertEqual(len(_target_objects(self.fs)), 2)
            # ...without moving the source (copy is non-destructive).
            self.assertTrue(self.fs.exists("/src/imgs/a.png"))
            # The index records the remote object URI as the asset path.
            for asset in Project.open(Path(tmp)).index.assets():
                self.assertTrue(asset.path.startswith("memory://dst/objects/sha256/"))
                self.assertTrue(asset.is_remote)

    def test_resync_is_idempotent_and_does_not_recopy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp, "copy"))

            copies: list[str] = []
            original = FsspecResolver.server_copy

            def counting(self: FsspecResolver, src: str, dst: str) -> None:
                copies.append(dst)
                return original(self, src, dst)

            FsspecResolver.server_copy = counting  # type: ignore[method-assign]
            try:
                summary = sync_sources(Project.open(Path(tmp)))[0]
            finally:
                FsspecResolver.server_copy = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(summary.assets_existing, 2)
            self.assertEqual(copies, [], "unchanged re-sync re-copied objects to the target")

    def test_identical_content_from_two_paths_dedups_to_one_object(self) -> None:
        # Same bytes under a different key must land on the same target object.
        self.fs.pipe("/src/imgs/dup.png", _png_bytes(1))
        self.fs.pipe("/src/lbls/dup.txt", b"0 0.5 0.5 0.4 0.4\n")
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp, "copy"))
            # 3 source images, but a.png and dup.png share content -> 2 objects.
            self.assertEqual(len(_target_objects(self.fs)), 2)

    def test_reference_points_at_source_and_writes_no_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp, "reference"))
            self.assertEqual(_target_objects(self.fs), [])
            for asset in Project.open(Path(tmp)).index.assets():
                self.assertTrue(asset.is_remote)
                self.assertIn("/src/imgs/", asset.path)

    def test_remote_asset_cannot_be_materialized_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp, "copy"))
            asset = next(iter(Project.open(Path(tmp)).index.assets()))
            with self.assertRaises(VisionPackError):
                asset.resolved_path(Path(tmp))

    def test_cross_provider_copy_relays_already_read_bytes(self) -> None:
        # A local source feeding a remote target can't be a server-side copy, so
        # the sync relays the bytes it already read for hashing: one upload per
        # object, no server_copy call, target still content-addressed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="x")
            imgs = root / "imgs"
            imgs.mkdir()
            (imgs / "a.png").write_bytes(_png_bytes(1))
            project = Project.open(root)
            project.manifest.sources = [{"name": "s1", "images": imgs.as_posix(), "copy": "copy"}]
            project.manifest.target = "memory://dst"
            project.save_manifest()

            copies: list[str] = []
            original = FsspecResolver.server_copy

            def counting(self: FsspecResolver, src: str, dst: str) -> None:
                copies.append(dst)
                return original(self, src, dst)

            FsspecResolver.server_copy = counting  # type: ignore[method-assign]
            try:
                summary = sync_sources(Project.open(root))[0]
            finally:
                FsspecResolver.server_copy = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 1)
            self.assertEqual(copies, [], "cross-provider transfer must not attempt a server-side copy")
            self.assertEqual(len(_target_objects(self.fs)), 1)
            for asset in Project.open(root).index.assets():
                self.assertTrue(asset.path.startswith("memory://dst/objects/sha256/"))

    def test_cross_provider_resync_is_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="x")
            imgs = root / "imgs"
            imgs.mkdir()
            (imgs / "a.png").write_bytes(_png_bytes(1))
            project = Project.open(root)
            project.manifest.sources = [{"name": "s1", "images": imgs.as_posix(), "copy": "copy"}]
            project.manifest.target = "memory://dst"
            project.save_manifest()
            sync_sources(Project.open(root))

            writes: list[str] = []
            original = FsspecResolver.write_bytes

            def counting(self: FsspecResolver, uri: str, data: bytes) -> None:
                writes.append(uri)
                return original(self, uri, data)

            FsspecResolver.write_bytes = counting  # type: ignore[method-assign]
            try:
                summary = sync_sources(Project.open(root))[0]
            finally:
                FsspecResolver.write_bytes = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(summary.assets_existing, 1)
            self.assertEqual(writes, [], "unchanged cross-provider re-sync re-uploaded objects")

    def test_remote_source_relays_into_local_target(self) -> None:
        # memory:// source, local-directory target: the opposite relay direction.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dst:
            project = self._project(tmp, "copy", target=Path(dst).as_posix())
            summary = sync_sources(project)[0]
            self.assertEqual(summary.assets_added, 2)
            landed = sorted(p for p in Path(dst).rglob("*") if p.is_file())
            self.assertEqual(len(landed), 2)
            for path in landed:
                self.assertIn("objects/sha256/", path.as_posix())


class ManifestTargetTest(unittest.TestCase):
    def test_target_round_trips_through_save_and_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="t")
            project = Project.open(root)
            project.manifest.target = {"uri": "s3://bucket/ds", "region": "us-east-1"}
            project.save_manifest()
            reopened = Project.open(root)
            self.assertEqual(reopened.manifest.target["uri"], "s3://bucket/ds")
            self.assertEqual(reopened.manifest.target["region"], "us-east-1")


if __name__ == "__main__":
    unittest.main()
