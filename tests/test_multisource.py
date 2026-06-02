from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter


def _make_source(root: Path, name: str, classes: list[str], color: tuple[int, int, int], class_index: int) -> Path:
    source = root / name
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir(parents=True)
    (source / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    Image.new("RGB", (80, 60), color=color).save(source / "images" / f"{name}.png", format="PNG")
    (source / "labels" / f"{name}.txt").write_text(f"{class_index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return source


class MultiSourceTest(unittest.TestCase):
    def test_second_source_with_reordered_classes_is_not_mislabeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="multi")

            # Source A: index 0 == scratch.
            source_a = _make_source(root, "a", ["scratch", "dent"], (10, 20, 30), class_index=0)
            YoloImporter(project, source_a).run()

            # Source B: classes declared in the opposite order, index 0 == dent.
            project = Project.open(root)
            source_b = _make_source(root, "b", ["dent", "scratch"], (200, 100, 50), class_index=0)
            YoloImporter(project, source_b).run()

            reopened = Project.open(root)
            # Classes merged by name, no duplicates, A's order preserved first.
            self.assertEqual([item.name for item in reopened.manifest.classes], ["scratch", "dent"])

            by_original = {
                Path(asset.original_path).stem: reopened.index.annotation_for_asset(asset.id)
                for asset in reopened.index.assets()
            }
            # A's box -> scratch, B's box -> dent (mapped by the source's own names).
            self.assertEqual(by_original["a"].objects[0].class_id, "scratch")
            self.assertEqual(by_original["b"].objects[0].class_id, "dent")

    def test_new_class_from_second_source_is_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Project.init(root, name="multi")
            YoloImporter(project, _make_source(root, "a", ["scratch"], (10, 20, 30), 0)).run()
            project = Project.open(root)
            YoloImporter(project, _make_source(root, "b", ["rust"], (200, 100, 50), 0)).run()

            reopened = Project.open(root)
            self.assertEqual([item.name for item in reopened.manifest.classes], ["scratch", "rust"])


if __name__ == "__main__":
    unittest.main()
