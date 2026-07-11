from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.autolabel import apply_predictions
from visionpack.core.project import Project
from visionpack.curation import rank_for_annotation
from visionpack.formats.yolo import YoloImporter
from visionpack.predictions import load_predictions


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed(root: Path) -> Project:
    """Four images: a and b labeled, c and d unlabeled."""
    data = root / "raw"
    for name, seed in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
        _png(data / f"{name}.png", seed)
    for name in ("a", "b"):
        (data / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (data / "classes.txt").write_text("obj\n", encoding="utf-8")
    project = Project.init(root, name="loop", task="detection")
    YoloImporter(project, data).run()
    return Project.open(root)


def _asset_by_stem(project: Project) -> dict[str, str]:
    return {Path(asset.original_path).stem: asset.id for asset in project.index.assets()}


def _write(root: Path, document: dict) -> Path:
    path = root / "preds.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class AutolabelTest(unittest.TestCase):
    def test_labels_only_unlabeled_and_confident(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            document = {
                "predictions": [
                    {"image": "a", "objects": [{"class": "obj", "confidence": 0.99, "bbox": [1, 1, 5, 5]}]},  # already labeled
                    {"image": "c", "objects": [{"class": "obj", "confidence": 0.9, "bbox": [10, 10, 20, 20]}]},  # confident
                    {"image": "d", "objects": [{"class": "obj", "confidence": 0.2, "bbox": [10, 10, 20, 20]}]},  # low confidence
                ]
            }
            predictions = load_predictions(project, _write(root, document))
            summary = apply_predictions(project, predictions, min_confidence=0.5)

            self.assertEqual(summary["labeled"], 1)
            self.assertEqual(summary["skipped_existing"], 1)
            self.assertEqual(summary["skipped_low_confidence"], 1)

            project = Project.open(root)
            stems = _asset_by_stem(project)
            annotation = project.index.annotation_for_asset(stems["c"])
            self.assertEqual(annotation.source["type"], "model")
            self.assertEqual(annotation.objects[0].confidence, 0.9)
            # The human-labeled asset was not touched.
            self.assertEqual(project.index.annotation_for_asset(stems["a"]).source["type"], "import")
            self.assertIsNone(project.index.annotation_for_asset(stems["d"]))

    def test_replace_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            document = {"predictions": [{"image": "a", "objects": [{"class": "obj", "confidence": 0.9, "bbox": [1, 1, 5, 5]}]}]}
            predictions = load_predictions(project, _write(root, document))
            summary = apply_predictions(project, predictions, min_confidence=0.5, replace=True)
            self.assertEqual(summary["labeled"], 1)
            project = Project.open(root)
            stems = _asset_by_stem(project)
            self.assertEqual(project.index.annotation_for_asset(stems["a"]).source["type"], "model")


class QueueTest(unittest.TestCase):
    def test_unlabeled_without_predictions_rank_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            # Confident predictions for c, nothing for d.
            document = {"predictions": [{"image": "c", "objects": [{"class": "obj", "confidence": 0.9, "bbox": [10, 10, 20, 20]}]}]}
            predictions = load_predictions(project, _write(root, document))
            ranked = rank_for_annotation(project, predictions)
            stems = _asset_by_stem(project)

            self.assertEqual([item["asset_id"] for item in ranked][0], stems["d"])
            self.assertEqual(ranked[0]["score"], 1.0)
            by_id = {item["asset_id"]: item for item in ranked}
            # c has a confident prediction -> low uncertainty score, still queued.
            self.assertAlmostEqual(by_id[stems["c"]]["score"], 0.1, places=4)
            # Labeled assets stay out of the queue without --include-labeled.
            self.assertNotIn(stems["a"], by_id)

    def test_without_predictions_queue_is_the_unlabeled_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            ranked = rank_for_annotation(project, None)
            self.assertEqual(len(ranked), 2)
            self.assertTrue(all(item["score"] == 1.0 for item in ranked))

    def test_disagreement_flags_possible_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            stems = _asset_by_stem(project)
            document = {
                "predictions": [
                    {
                        "image": "a",
                        "objects": [
                            # Matches the ground truth box (10,10,20,20)...
                            {"class": "obj", "confidence": 0.9, "bbox": [10, 10, 20, 20]},
                            # ...plus a confident detection nowhere in the labels.
                            {"class": "obj", "confidence": 0.8, "bbox": [0, 0, 6, 6]},
                        ],
                    },
                    # b: the model perfectly agrees -> should not be queued.
                    {"image": "b", "objects": [{"class": "obj", "confidence": 0.9, "bbox": [10, 10, 20, 20]}]},
                ]
            }
            predictions = load_predictions(project, _write(root, document))
            ranked = rank_for_annotation(project, predictions, include_labeled=True)
            by_id = {item["asset_id"]: item for item in ranked}

            self.assertIn(stems["a"], by_id)
            self.assertNotIn(stems["b"], by_id)
            self.assertTrue(any("missing labels" in reason for reason in by_id[stems["a"]]["reasons"]))


if __name__ == "__main__":
    unittest.main()
