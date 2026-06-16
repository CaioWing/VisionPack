from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fsspec
from PIL import Image

from visionpack.core.project import Project
from visionpack.sources import sync_sources
from visionpack.sources.resolver import FsspecResolver, LocalResolver, get_resolver


def _png_bytes(seed: int) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(seed))


def _label(path: Path, class_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")


class FsspecResolverTest(unittest.TestCase):
    """Exercises the cloud resolver against fsspec's in-memory filesystem, so the
    cloud code path is covered with no network and no provider library."""

    def setUp(self) -> None:
        self.fs = fsspec.filesystem("memory")
        # A clean tree per test: memory:// is process-global.
        for path in self.fs.find("/imgs"):
            self.fs.rm(path)
        self.fs.pipe("/imgs/a.png", _png_bytes(1))
        self.fs.pipe("/imgs/sub/b.jpg", _png_bytes(2))

    def test_list_files_returns_round_trippable_uris(self) -> None:
        resolver = FsspecResolver("memory")
        refs = resolver.list_files("memory://imgs", {".png", ".jpg"})
        self.assertEqual({ref.relkey for ref in refs}, {"a", "sub/b"})
        # The uri each ref carries must read back the same bytes.
        for ref in refs:
            self.assertEqual(resolver.read_bytes(ref.uri), self.fs.cat_file(ref.uri.split("://", 1)[1]))

    def test_suffix_filter(self) -> None:
        resolver = FsspecResolver("memory")
        refs = resolver.list_files("memory://imgs", {".png"})
        self.assertEqual({ref.suffix for ref in refs}, {".png"})

    def test_stat_reports_size_without_reading_body(self) -> None:
        resolver = FsspecResolver("memory")
        stat = resolver.stat("memory://imgs/a.png")
        self.assertEqual(stat.size, len(_png_bytes(1)))

    def test_list_files_carries_stat_from_a_single_listing(self) -> None:
        # The listing must already carry size + a change-detector, so a sync
        # never needs a per-object HEAD afterwards.
        resolver = FsspecResolver("memory")
        refs = resolver.list_files("memory://imgs", {".png", ".jpg"})
        self.assertTrue(refs)
        for ref in refs:
            self.assertIsNotNone(ref.stat)
            self.assertGreater(ref.stat.size, 0)

    def test_storage_options_are_forwarded_to_fsspec(self) -> None:
        # Credentials/region from the YAML must reach the provider filesystem.
        import fsspec.core

        captured: dict = {}
        original = fsspec.core.url_to_fs

        def spy(uri: str, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return original(uri)

        fsspec.core.url_to_fs = spy  # type: ignore[assignment]
        try:
            FsspecResolver("memory", {"key": "AKIA", "secret": "shh"}).stat("memory://imgs/a.png")
        finally:
            fsspec.core.url_to_fs = original  # type: ignore[assignment]
        self.assertEqual(captured, {"key": "AKIA", "secret": "shh"})

    def test_get_resolver_unsupported_scheme_is_clear(self) -> None:
        with self.assertRaises(Exception) as ctx:
            get_resolver("ftp://nope/x")
        self.assertIn("Unsupported source scheme", str(ctx.exception))


class LocalStatTest(unittest.TestCase):
    def test_local_stat_has_size_and_etag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.png"
            _png(path, 1)
            stat = LocalResolver().stat(path.as_posix())
            self.assertEqual(stat.size, path.stat().st_size)
            self.assertIsNotNone(stat.etag)


class ResyncCacheTest(unittest.TestCase):
    def test_unchanged_resync_skips_body_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="cache")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")
            sources = [{"name": "s1", "images": imgs.as_posix(), "labels": lbls.as_posix()}]

            project = Project.open(root)
            project.manifest.sources = sources
            project.save_manifest()
            sync_sources(Project.open(root))

            # Second sync: count how many image bodies are read. With the cache
            # warm and the object unchanged, the image must not be re-read.
            project = Project.open(root)
            reads: list[str] = []
            original = LocalResolver.read_bytes

            def counting_read(self: LocalResolver, uri: str) -> bytes:
                if uri.endswith(".png"):
                    reads.append(uri)
                return original(self, uri)

            LocalResolver.read_bytes = counting_read  # type: ignore[method-assign]
            try:
                summary = sync_sources(project)[0]
            finally:
                LocalResolver.read_bytes = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(summary.assets_existing, 1)
            self.assertEqual(reads, [], "unchanged image was re-read despite a warm cache")

    def test_reference_mode_resync_also_skips_body_read(self) -> None:
        # `reference` stores no CAS copy, but its probe is still cacheable: an
        # unchanged re-sync must not re-hash the cheapest storage mode.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="ref")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")
            sources = [{"name": "s1", "images": imgs.as_posix(), "labels": lbls.as_posix(), "copy": "reference"}]

            project = Project.open(root)
            project.manifest.sources = sources
            project.save_manifest()
            sync_sources(Project.open(root))

            project = Project.open(root)
            reads: list[str] = []
            original = LocalResolver.read_bytes

            def counting_read(self: LocalResolver, uri: str) -> bytes:
                if uri.endswith(".png"):
                    reads.append(uri)
                return original(self, uri)

            LocalResolver.read_bytes = counting_read  # type: ignore[method-assign]
            try:
                summary = sync_sources(project)[0]
            finally:
                LocalResolver.read_bytes = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_existing, 1)
            self.assertEqual(reads, [], "reference re-sync re-read an unchanged image")

    def test_changed_file_is_reread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="cache")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")
            sources = [{"name": "s1", "images": imgs.as_posix(), "labels": lbls.as_posix()}]
            project = Project.open(root)
            project.manifest.sources = sources
            project.save_manifest()
            sync_sources(Project.open(root))

            # Overwrite with different content (new size + mtime) → must re-read
            # and produce a new asset.
            _png(imgs / "a.png", 9)
            summary = sync_sources(Project.open(root))[0]
            self.assertEqual(summary.assets_added, 1)


if __name__ == "__main__":
    unittest.main()
