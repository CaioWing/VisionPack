from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.models import Split
from visionpack.core.project import Project
from visionpack.duplicates import cross_split_leakage, near_duplicate_pairs, phash_map
from visionpack.formats.yolo import YoloImporter
from visionpack.validation import validate_project


def _gradient(seed: int) -> Image.Image:
    """A non-flat image so its perceptual hash is meaningful (not all zeros)."""
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = ((x * 4 + seed * 9) % 256, (y * 4) % 256, ((x + y) * 2 + seed * 5) % 256)
    return img


def _seed(root: Path) -> Project:
    """Two near-duplicates (same content, PNG vs JPEG -> different sha) plus a
    visually distinct image."""
    project = Project.init(root, name="dedup-demo")
    images = root / "raw" / "images"
    images.mkdir(parents=True)
    _gradient(0).save(images / "a.png", format="PNG")
    _gradient(0).save(images / "b.jpg", format="JPEG", quality=95)
    _gradient(40).save(images / "c.png", format="PNG")
    YoloImporter(project, root / "raw").run()
    return Project.open(root)


class DuplicatesTest(unittest.TestCase):
    def test_near_duplicates_have_distinct_assets_but_close_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            # PNG and JPEG of the same content must be separate assets (different sha).
            self.assertEqual(len(project.index.assets()), 3)
            phashes = phash_map(project)
            self.assertEqual(len(phashes), 3)
            pairs = near_duplicate_pairs(phashes, threshold=5)
            # Exactly one near-duplicate pair: a ~ b, and c stays out of it.
            self.assertEqual(len(pairs), 1)

    def test_validate_warns_on_near_duplicate_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            report = validate_project(project)
            codes = [issue.code for issue in report.issues]
            self.assertIn("asset.near_duplicate", codes)

    def test_validate_off_disables_perceptual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            project.manifest.validation["duplicates"]["perceptual"] = "off"
            report = validate_project(project)
            codes = [issue.code for issue in report.issues]
            self.assertNotIn("asset.near_duplicate", codes)

    def test_cross_split_leakage_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            ids = {Path(asset.original_path).name: asset.id for asset in project.index.assets()}
            # Put the two near-duplicates on opposite sides of the split.
            split = Split(
                id="default",
                strategy="manual",
                sets={"train": [ids["a.png"]], "val": [ids["c.png"]], "test": [ids["b.jpg"]]},
            )
            project.index.upsert_split(split)
            project.index.save()

            leaks = cross_split_leakage(Project.open(root), "default")
            self.assertEqual(len(leaks), 1)
            self.assertEqual({leaks[0].set_a, leaks[0].set_b}, {"train", "test"})

            report = validate_project(Project.open(root))
            self.assertIn("split.near_duplicate_leakage", [issue.code for issue in report.errors])

    def test_phash_is_persisted_on_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed(root)
            # Re-open from disk: phash should already be stored, not recomputed.
            for asset in Project.open(root).index.assets():
                self.assertTrue(asset.phash)


if __name__ == "__main__":
    unittest.main()
