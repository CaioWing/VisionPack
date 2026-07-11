from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import fsspec
from PIL import Image

from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project
from visionpack.sources import retry, sync_sources
from visionpack.sources.importer import SourceSyncer
from visionpack.sources.resolver import FsspecResolver
from visionpack.sources.retry import with_retries
from visionpack.sources.schema import Source


def _png_bytes(seed: int, size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _clear_memory_fs(*prefixes: str) -> fsspec.AbstractFileSystem:
    fs = fsspec.filesystem("memory")
    for prefix in prefixes:
        for path in list(fs.find(prefix)):
            fs.rm(path)
    return fs


class RetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._delay = retry.BASE_DELAY_SECONDS
        retry.BASE_DELAY_SECONDS = 0.0

    def tearDown(self) -> None:
        retry.BASE_DELAY_SECONDS = self._delay

    def test_transient_failure_is_retried_until_success(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError("throttled")
            return "ok"

        self.assertEqual(with_retries("read(x)", flaky), "ok")
        self.assertEqual(calls["n"], 3)

    def test_exhaustion_becomes_a_visionpack_error(self) -> None:
        calls = {"n": 0}

        def always_down() -> None:
            calls["n"] += 1
            raise TimeoutError("gateway timeout")

        with self.assertRaises(VisionPackError) as ctx:
            with_retries("read(s3://b/k)", always_down)
        self.assertIn("read(s3://b/k)", str(ctx.exception))
        self.assertEqual(calls["n"], retry.MAX_ATTEMPTS)

    def test_permanent_errors_are_not_retried(self) -> None:
        calls = {"n": 0}

        def missing() -> None:
            calls["n"] += 1
            raise FileNotFoundError("no such key")

        with self.assertRaises(FileNotFoundError):
            with_retries("read(x)", missing)
        self.assertEqual(calls["n"], 1)

    def test_resolver_read_rides_out_transient_provider_errors(self) -> None:
        fs = _clear_memory_fs("/retrysrc")
        fs.pipe("/retrysrc/a.bin", b"payload")
        original = type(fs).cat_file
        calls = {"n": 0}

        def flaky(self, path, *args, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError("blip")
            return original(self, path, *args, **kwargs)

        type(fs).cat_file = flaky  # type: ignore[method-assign]
        try:
            data = FsspecResolver("memory").read_bytes("memory://retrysrc/a.bin")
        finally:
            type(fs).cat_file = original  # type: ignore[method-assign]
        self.assertEqual(data, b"payload")
        self.assertEqual(calls["n"], 3)


class TargetFastListTest(unittest.TestCase):
    """The target CAS membership comes from one prefix listing, not per-object
    existence checks — and relayed uploads are size-verified."""

    def setUp(self) -> None:
        self.fs = _clear_memory_fs("/flsrc", "/fldst")
        self.fs.pipe("/flsrc/imgs/a.png", _png_bytes(1))
        self.fs.pipe("/flsrc/imgs/b.png", _png_bytes(2))
        self.fs.pipe("/flsrc/lbls/a.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/flsrc/lbls/b.txt", b"0 0.5 0.5 0.4 0.4\n")
        self.fs.pipe("/flsrc/lbls/classes.txt", b"cat\n")
        self._delay = retry.BASE_DELAY_SECONDS
        retry.BASE_DELAY_SECONDS = 0.0

    def tearDown(self) -> None:
        retry.BASE_DELAY_SECONDS = self._delay

    def _project(self, tmp: str) -> Project:
        root = Path(tmp)
        Project.init(root, name="fastlist")
        project = Project.open(root)
        project.manifest.sources = [
            {"name": "s1", "images": "memory://flsrc/imgs", "labels": "memory://flsrc/lbls", "copy": "copy"}
        ]
        project.manifest.target = "memory://fldst"
        project.save_manifest()
        return Project.open(root)

    def test_no_per_object_existence_checks_against_the_target_cas(self) -> None:
        checked: list[str] = []
        original = FsspecResolver.exists

        def counting(self: FsspecResolver, uri: str) -> bool:
            checked.append(uri)
            return original(self, uri)

        FsspecResolver.exists = counting  # type: ignore[method-assign]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary = sync_sources(self._project(tmp))[0]
        finally:
            FsspecResolver.exists = original  # type: ignore[method-assign]

        self.assertEqual(summary.assets_added, 2)
        object_heads = [uri for uri in checked if "/objects/sha256/" in uri]
        self.assertEqual(object_heads, [], "membership must come from the prefix listing, not per-object checks")

    def test_truncated_relay_upload_is_caught_and_recorded_as_failure(self) -> None:
        # Local source -> memory target forces the relay path; corrupt it.
        original = FsspecResolver.write_bytes

        def truncating(self: FsspecResolver, uri: str, data: bytes) -> None:
            return original(self, uri, data[:-1])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="verify")
            imgs = root / "imgs"
            imgs.mkdir()
            (imgs / "a.png").write_bytes(_png_bytes(3))
            project = Project.open(root)
            project.manifest.sources = [{"name": "s1", "images": imgs.as_posix(), "copy": "copy"}]
            project.manifest.target = "memory://fldst"
            project.save_manifest()

            FsspecResolver.write_bytes = truncating  # type: ignore[method-assign]
            try:
                summary = sync_sources(Project.open(root))[0]
            finally:
                FsspecResolver.write_bytes = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(len(summary.failures), 1)
            self.assertIn("landed", summary.failures[0].error)
            self.assertEqual(Project.open(root).index.count_assets(), 0)


class LabelSingleReadTest(unittest.TestCase):
    def test_each_remote_label_is_fetched_exactly_once_when_inferring_classes(self) -> None:
        _clear_memory_fs("/lblsrc")
        fs = fsspec.filesystem("memory")
        fs.pipe("/lblsrc/imgs/a.png", _png_bytes(1))
        fs.pipe("/lblsrc/imgs/b.png", _png_bytes(2))
        # No classes.txt anywhere: class names must be inferred from the labels.
        fs.pipe("/lblsrc/lbls/a.txt", b"1 0.5 0.5 0.4 0.4\n")
        fs.pipe("/lblsrc/lbls/b.txt", b"0 0.5 0.5 0.4 0.4\n")

        reads: list[str] = []
        original = FsspecResolver.read_bytes

        def counting(self: FsspecResolver, uri: str) -> bytes:
            reads.append(uri)
            return original(self, uri)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="labels")
            project = Project.open(root)
            project.manifest.sources = [
                {"name": "s1", "images": "memory://lblsrc/imgs", "labels": "memory://lblsrc/lbls"}
            ]
            project.save_manifest()

            FsspecResolver.read_bytes = counting  # type: ignore[method-assign]
            try:
                summary = sync_sources(Project.open(root))[0]
            finally:
                FsspecResolver.read_bytes = original  # type: ignore[method-assign]

            self.assertEqual(summary.assets_added, 2)
            self.assertEqual(summary.objects, 2)
            label_reads = [uri for uri in reads if uri.endswith(".txt")]
            self.assertEqual(len(label_reads), 2, f"labels must be read once each, got: {label_reads}")
            # Inferred names cover the highest class index seen (0 and 1).
            names = {item.name for item in Project.open(root).manifest.classes}
            self.assertEqual(names, {"class_0", "class_1"})


class RemoteImageFolderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = _clear_memory_fs("/ifsrc")
        for index, class_name in ((1, "ok"), (2, "ok"), (3, "defect")):
            self.fs.pipe(f"/ifsrc/{class_name}/img{index}.png", _png_bytes(index))

    def _project(self, tmp: str) -> Project:
        root = Path(tmp)
        Project.init(root, name="cls", task="classification")
        project = Project.open(root)
        project.manifest.sources = [{"name": "folder-a", "format": "imagefolder", "root": "memory://ifsrc"}]
        project.save_manifest()
        return Project.open(root)

    def test_remote_imagefolder_syncs_with_whole_image_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = sync_sources(self._project(tmp))[0]
            self.assertEqual(summary.assets_added, 3)
            self.assertEqual(summary.annotations, 3)
            self.assertEqual(summary.classes_added, 2)

            project = Project.open(Path(tmp))
            names = {item.name for item in project.manifest.classes}
            self.assertEqual(names, {"ok", "defect"})
            for asset in project.index.assets():
                self.assertEqual(asset.source, "folder-a")
                annotation = project.index.annotation_for_asset(asset.id)
                assert annotation is not None
                self.assertEqual(len(annotation.objects), 1)
                self.assertIsNone(annotation.objects[0].geometry)

    def test_remote_imagefolder_resync_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp))
            summary = sync_sources(Project.open(Path(tmp)))[0]
            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(summary.assets_existing, 3)


class RemoteCocoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = _clear_memory_fs("/cocosrc")
        self.fs.pipe("/cocosrc/imgs/a.png", _png_bytes(1))
        self.fs.pipe("/cocosrc/imgs/sub/b.png", _png_bytes(2))
        document = {
            "images": [
                {"id": 1, "file_name": "a.png", "width": 32, "height": 24},
                {"id": 2, "file_name": "sub/b.png", "width": 32, "height": 24},
                {"id": 3, "file_name": "missing.png", "width": 32, "height": 24},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 7, "bbox": [2, 3, 10, 8], "iscrowd": 0},
                {"id": 11, "image_id": 1, "category_id": 8, "bbox": [1, 1, 5, 5], "iscrowd": 0},
                {"id": 12, "image_id": 2, "category_id": 8, "bbox": [4, 4, 6, 6], "iscrowd": 0},
            ],
            "categories": [{"id": 7, "name": "scratch"}, {"id": 8, "name": "dent"}],
        }
        self.fs.pipe("/cocosrc/instances.json", json.dumps(document).encode("utf-8"))

    def _project(self, tmp: str) -> Project:
        root = Path(tmp)
        Project.init(root, name="cocor")
        project = Project.open(root)
        project.manifest.sources = [
            {
                "name": "coco-a",
                "format": "coco",
                "images": "memory://cocosrc/imgs",
                "labels": "memory://cocosrc/instances.json",
            }
        ]
        project.save_manifest()
        return Project.open(root)

    def test_remote_coco_syncs_images_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = sync_sources(self._project(tmp))[0]
            self.assertEqual(summary.assets_added, 2)
            self.assertEqual(summary.annotations, 2)
            self.assertEqual(summary.objects, 3)
            self.assertEqual(summary.classes_added, 2)
            # The image declared in the JSON but absent from the listing is a
            # clean per-image failure, not an aborted sync.
            self.assertEqual(len(summary.failures), 1)
            self.assertIn("missing.png", summary.failures[0].path)

            project = Project.open(Path(tmp))
            names = {item.name for item in project.manifest.classes}
            self.assertEqual(names, {"scratch", "dent"})
            class_by_name = {item.name: item.id for item in project.manifest.classes}
            annotated = [
                project.index.annotation_for_asset(asset.id)
                for asset in project.index.assets()
                if asset.source == "coco-a"
            ]
            all_objects = [obj for ann in annotated if ann for obj in ann.objects]
            self.assertEqual(len(all_objects), 3)
            self.assertEqual(
                sorted(obj.class_id for obj in all_objects),
                sorted([class_by_name["scratch"], class_by_name["dent"], class_by_name["dent"]]),
            )
            bboxes = {(obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height) for obj in all_objects}
            self.assertIn((2.0, 3.0, 10.0, 8.0), bboxes)

    def test_remote_coco_resync_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sync_sources(self._project(tmp))
            summary = sync_sources(Project.open(Path(tmp)))[0]
            self.assertEqual(summary.assets_added, 0)
            self.assertEqual(summary.assets_existing, 2)


class PoolSizeTest(unittest.TestCase):
    def _syncer(self, tmp: str, images_uri: str, max_workers: int | None) -> SourceSyncer:
        root = Path(tmp)
        Project.init(root, name="jobs")
        project = Project.open(root)
        source = Source.from_dict({"name": "s1", "images": images_uri})
        return SourceSyncer(project, source, max_workers=max_workers)

    def test_jobs_flag_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._syncer(tmp, "memory://x/imgs", 3)._pool_size(), 3)

    def test_remote_default_has_a_floor_of_16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            size = self._syncer(tmp, "memory://x/imgs", None)._pool_size()
            assert size is not None
            self.assertGreaterEqual(size, 16)
            self.assertLessEqual(size, 32)

    def test_local_default_is_the_executor_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self._syncer(tmp, "./imgs", None)._pool_size())


if __name__ == "__main__":
    unittest.main()
