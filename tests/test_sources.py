from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.sources import plan_sources, sync_sources


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _label(path: Path, class_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")


def _declare(root: Path, sources: list[dict]) -> Project:
    project = Project.open(root)
    project.manifest.sources = sources
    project.save_manifest()
    return Project.open(root)  # reopen so it round-trips through visionpack.yaml


def _uri(path: Path) -> str:
    return path.as_posix()


class SourceSyncTest(unittest.TestCase):
    def test_relpath_join_across_separate_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            imgs, lbls = root / "imgs", root / "lbls"
            for i, name in enumerate(("a", "b", "c")):
                _png(imgs / f"{name}.png", i)
                _label(lbls / f"{name}.txt", i % 2)
            (lbls / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

            project = _declare(root, [{"name": "s1", "format": "yolo", "images": _uri(imgs), "labels": _uri(lbls), "match": "relpath"}])
            summaries = sync_sources(project)

            self.assertEqual(summaries[0].assets_added, 3)
            self.assertEqual(summaries[0].annotations, 3)
            opened = Project.open(root)
            self.assertEqual({c.name for c in opened.manifest.classes}, {"cat", "dog"})
            self.assertTrue(all(asset.source == "s1" for asset in opened.index.assets()))

    def test_stem_join_when_structures_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "train" / "a.png", 1)
            _label(lbls / "anything" / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")

            # relpath would NOT match (train/a vs anything/a); stem does.
            project = _declare(root, [{"name": "s1", "images": _uri(imgs), "labels": _uri(lbls), "match": "stem"}])
            summaries = sync_sources(project)
            self.assertEqual(summaries[0].assets_added, 1)
            self.assertEqual(summaries[0].annotations, 1)

    def test_root_shorthand_expands_to_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            base = root / "dataset"
            _png(base / "images" / "a.png", 1)
            _label(base / "labels" / "a.txt", 0)
            (base / "labels" / "classes.txt").write_text("cat\n", encoding="utf-8")

            project = _declare(root, [{"name": "s1", "root": _uri(base)}])
            summaries = sync_sources(project)
            self.assertEqual(summaries[0].assets_added, 1)
            self.assertEqual(summaries[0].annotations, 1)

    def test_dry_run_reports_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _png(imgs / "b.png", 2)  # no label for b
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")

            project = _declare(root, [{"name": "s1", "images": _uri(imgs), "labels": _uri(lbls)}])
            plans = plan_sources(project)
            self.assertEqual(plans[0].images_found, 2)
            self.assertEqual(plans[0].matched, 1)
            self.assertEqual(plans[0].images_without_label, 1)
            self.assertEqual(plans[0].class_names, ["cat"])
            # Dry-run must not have ingested anything.
            self.assertEqual(len(Project.open(root).index.assets()), 0)

    def test_sync_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")
            sources = [{"name": "s1", "images": _uri(imgs), "labels": _uri(lbls)}]

            sync_sources(_declare(root, sources))
            second = sync_sources(Project.open(root))
            self.assertEqual(second[0].assets_added, 0)
            self.assertEqual(second[0].assets_existing, 1)
            self.assertEqual(len(Project.open(root).index.assets()), 1)

    def test_class_map_remaps_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            imgs, lbls = root / "imgs", root / "lbls"
            _png(imgs / "a.png", 1)
            _label(lbls / "a.txt", 0)
            (lbls / "classes.txt").write_text("person\n", encoding="utf-8")

            project = _declare(root, [{"name": "s1", "images": _uri(imgs), "labels": _uri(lbls), "class_map": {"person": "people"}}])
            sync_sources(project)
            opened = Project.open(root)
            self.assertEqual({c.name for c in opened.manifest.classes}, {"people"})

    def test_two_sources_merge_classes_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Project.init(root, name="multi")
            # Source 1: classes [cat, dog]; Source 2: classes [dog, cat] (reordered).
            a_img, a_lbl = root / "a/imgs", root / "a/lbls"
            b_img, b_lbl = root / "b/imgs", root / "b/lbls"
            _png(a_img / "x.png", 1)
            _label(a_lbl / "x.txt", 1)  # index 1 -> dog in source 1
            (a_lbl / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
            _png(b_img / "y.png", 2)
            _label(b_lbl / "y.txt", 1)  # index 1 -> cat in source 2
            (b_lbl / "classes.txt").write_text("dog\ncat\n", encoding="utf-8")

            project = _declare(root, [
                {"name": "a", "images": _uri(a_img), "labels": _uri(a_lbl)},
                {"name": "b", "images": _uri(b_img), "labels": _uri(b_lbl)},
            ])
            sync_sources(project)

            opened = Project.open(root)
            self.assertEqual({c.name for c in opened.manifest.classes}, {"cat", "dog"})
            by_source = {asset.source: asset for asset in opened.index.assets()}
            self.assertEqual(set(by_source), {"a", "b"})
            # source a's index-1 must resolve to "dog"; source b's index-1 to "cat".
            dog_id = opened.manifest.class_id_for_name("dog")
            cat_id = opened.manifest.class_id_for_name("cat")
            self.assertEqual(opened.index.annotation_for_asset(by_source["a"].id).objects[0].class_id, dog_id)
            self.assertEqual(opened.index.annotation_for_asset(by_source["b"].id).objects[0].class_id, cat_id)


if __name__ == "__main__":
    unittest.main()
