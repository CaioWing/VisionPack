from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.errors import VisionPackError
from visionpack.core.project import Project
from visionpack.formats.coco import export_coco
from visionpack.formats.yolo import YoloImporter, export_yolo
from visionpack.split import create_split, lock_split
from visionpack.stats import split_breakdown


def _seed_dataset(root: Path, classes: list[str], images: list[tuple[str, int]]) -> Project:
    """images: list of (name, class_index). Each image gets a unique color so
    its content hash (and thus asset id) is distinct."""
    project = Project.init(root, name="split-demo")
    source = root / "raw"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir(parents=True)
    (source / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    for index, (name, class_index) in enumerate(images):
        color = (index * 7 % 256, index * 13 % 256, index * 29 % 256)
        Image.new("RGB", (50, 50), color=color).save(source / "images" / f"{name}.png", format="PNG")
        (source / "labels" / f"{name}.txt").write_text(f"{class_index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    YoloImporter(project, source).run()
    return Project.open(root)


def _twenty_balanced(root: Path) -> Project:
    images = [(f"a{i}", 0) for i in range(10)] + [(f"b{i}", 1) for i in range(10)]
    return _seed_dataset(root, ["cat", "dog"], images)


class SplitTest(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            first = create_split(Project.open(root), strategy="random", seed=42)
            second = create_split(Project.open(root), strategy="random", seed=42)
            self.assertEqual(first.sets, second.sets)

    def test_random_hits_exact_global_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            split = create_split(Project.open(root), train=0.8, val=0.1, test=0.1, strategy="random")
            self.assertEqual(len(split.sets["train"]), 16)
            self.assertEqual(len(split.sets["val"]), 2)
            self.assertEqual(len(split.sets["test"]), 2)

    def test_sets_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            split = create_split(Project.open(root), strategy="stratified")
            all_ids = [asset_id for ids in split.sets.values() for asset_id in ids]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(len(all_ids), 20)

    def test_stratified_balances_each_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            create_split(Project.open(root), train=0.8, val=0.1, test=0.1, strategy="stratified")
            breakdown = split_breakdown(Project.open(root))
            assert breakdown is not None
            # Each class (10 images) should split 8/1/1 across train/val/test.
            self.assertEqual(breakdown["sets"]["train"]["class_distribution"], {"cat": 8, "dog": 8})
            self.assertEqual(breakdown["sets"]["val"]["class_distribution"], {"cat": 1, "dog": 1})
            self.assertEqual(breakdown["sets"]["test"]["class_distribution"], {"cat": 1, "dog": 1})

    def test_hash_strategy_is_stable_as_data_grows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            before = create_split(Project.open(root), strategy="hash", seed=1)
            assignment_before = {asset_id: name for name, ids in before.sets.items() for asset_id in ids}

            # Add 10 more images and re-create the hash split.
            project = Project.open(root)
            source = root / "raw"
            for i in range(10, 20):
                color = (i * 3 % 256, i * 17 % 256, i * 5 % 256)
                Image.new("RGB", (50, 50), color=color).save(source / "images" / f"c{i}.png", format="PNG")
                (source / "labels" / f"c{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            YoloImporter(project, source).run()

            after = create_split(Project.open(root), strategy="hash", seed=1)
            assignment_after = {asset_id: name for name, ids in after.sets.items() for asset_id in ids}

            # Every originally-assigned asset keeps its set membership.
            for asset_id, set_name in assignment_before.items():
                self.assertEqual(assignment_after[asset_id], set_name)

    def test_yolo_export_is_split_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            create_split(Project.open(root), strategy="stratified")
            output = root / "exports" / "yolo"
            summary = export_yolo(Project.open(root), output, split_id="default")

            self.assertEqual(summary["sets"], {"train": 16, "val": 2, "test": 2})
            for set_name, count in (("train", 16), ("val", 2), ("test", 2)):
                self.assertEqual(len(list((output / "images" / set_name).glob("*.png"))), count)
                self.assertEqual(len(list((output / "labels" / set_name).glob("*.txt"))), count)
            data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("train: images/train", data_yaml)
            self.assertIn("val: images/val", data_yaml)
            self.assertIn("test: images/test", data_yaml)

    def test_coco_export_writes_per_split_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            create_split(Project.open(root), strategy="stratified")
            output = root / "exports" / "coco"
            summary = export_coco(Project.open(root), output, split_id="default")

            self.assertEqual(summary["sets"], {"train": 16, "val": 2, "test": 2})
            for set_name in ("train", "val", "test"):
                self.assertTrue((output / "annotations" / f"instances_{set_name}.json").exists())
                self.assertTrue((output / "images" / set_name).is_dir())

    def test_export_with_unknown_split_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            with self.assertRaises(VisionPackError):
                export_yolo(Project.open(root), root / "out", split_id="nope")

    def test_lock_prevents_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _twenty_balanced(root)
            create_split(Project.open(root), strategy="random")
            lock_split(Project.open(root))
            with self.assertRaises(VisionPackError):
                create_split(Project.open(root), strategy="random")
            # --force overrides.
            create_split(Project.open(root), strategy="random", force=True)


if __name__ == "__main__":
    unittest.main()
