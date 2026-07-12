"""Regression tests: fsck/validate must handle cloud-backed (remote) assets.

An asset synced against a cloud target records its object URI as ``path``
(e.g. ``s3://bucket/objects/sha256/...``); such assets have no local file to
open. ``run_fsck`` used to crash on the first remote asset and
``validate_project`` flooded the report with false ``image.unreadable`` errors.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.models import Asset
from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.fsck import run_fsck
from visionpack.validation import validate_project


def _seed_with_remote_asset(root: Path) -> Project:
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("cat\n", encoding="utf-8")
    Project.init(root, name="remote-checks")
    Image.new("RGB", (30, 30), (10, 20, 30)).save(raw / "images" / "local.png", format="PNG")
    (raw / "labels" / "local.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    YoloImporter(Project.open(root), raw).run()

    project = Project.open(root)
    digest = "b" * 64
    project.index.upsert_asset(
        Asset(
            id=f"asset_{digest[:16]}",
            sha256=digest,
            media_type="image",
            path=f"s3://bucket/objects/sha256/{digest[:2]}/{digest[2:4]}/{digest}",
            original_path="s3://bucket/imgs/remote.jpg",
            width=64,
            height=64,
            channels=3,
            format="jpeg",
            size_bytes=1234,
            phash="00000000000000ff",
        )
    )
    project.index.save()
    return Project.open(root)


class RemoteAssetChecksTest(unittest.TestCase):
    def test_fsck_skips_remote_assets_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_with_remote_asset(Path(tmp))
            report = run_fsck(project)
            self.assertTrue(report.ok)
            self.assertEqual(report.checked_assets, 2)  # remote asset still counted

    def test_fsck_deep_skips_remote_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_with_remote_asset(Path(tmp))
            report = run_fsck(project, deep=True)
            self.assertTrue(report.ok)

    def test_validate_does_not_flag_remote_assets_as_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = _seed_with_remote_asset(Path(tmp))
            report = validate_project(project)
            unreadable = [issue for issue in report.issues if issue.code == "image.unreadable"]
            self.assertEqual(unreadable, [])


if __name__ == "__main__":
    unittest.main()
