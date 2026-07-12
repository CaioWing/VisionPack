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

    def test_identical_hashes_expand_to_all_distance_zero_pairs(self) -> None:
        # Re-encoded copies share the exact hash; the collapsed LSH path must
        # still report every pair, at distance 0, and leave distinct images out.
        phashes = {"a": "00000000000000ff", "b": "00000000000000ff", "c": "00000000000000ff", "d": "ffffffffffffff00"}
        pairs = near_duplicate_pairs(phashes, threshold=5)
        self.assertEqual({(p.asset_a, p.asset_b, p.distance) for p in pairs}, {("a", "b", 0), ("a", "c", 0), ("b", "c", 0)})

    def test_close_but_not_identical_hashes_still_pair_across_groups(self) -> None:
        # 2 bits apart (<= threshold): the pair must survive the identical-hash
        # collapse and carry the true distance.
        phashes = {"a": "00000000000000ff", "b": "00000000000000fc"}
        pairs = near_duplicate_pairs(phashes, threshold=5)
        self.assertEqual([(pairs[0].asset_a, pairs[0].asset_b, pairs[0].distance)], [("a", "b", 2)])

    def test_huge_identical_group_expands_linearly(self) -> None:
        # 20k frames with the same perceptual hash (idle camera) must not
        # materialize the ~200M-pair cross product: star expansion keeps the
        # pair list linear while the cluster still covers every member.
        from visionpack.duplicates import cluster_pairs

        phashes = {f"asset_{i:05d}": "00000000000000ff" for i in range(20_000)}
        pairs = near_duplicate_pairs(phashes, threshold=5)
        self.assertEqual(len(pairs), 19_999)
        clusters = cluster_pairs(pairs)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].asset_ids), 20_000)

    def test_huge_near_identical_groups_pair_linearly_across_values(self) -> None:
        # Two big groups 2 bits apart: every asset must appear in at least one
        # cross pair (so leakage stays visible) without the full cross product.
        group_a = {f"a_{i:05d}": "00000000000000ff" for i in range(200)}
        group_b = {f"b_{i:05d}": "00000000000000fc" for i in range(200)}
        pairs = near_duplicate_pairs({**group_a, **group_b}, threshold=5)
        cross = [p for p in pairs if p.distance == 2]
        self.assertLess(len(cross), 200 * 200)
        covered = {p.asset_a for p in cross} | {p.asset_b for p in cross}
        self.assertTrue(set(group_a) <= covered)
        self.assertTrue(set(group_b) <= covered)

    def test_phash_is_persisted_on_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed(root)
            # Re-open from disk: phash should already be stored, not recomputed.
            for asset in Project.open(root).index.assets():
                self.assertTrue(asset.phash)


if __name__ == "__main__":
    unittest.main()
