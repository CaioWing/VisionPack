from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.project import Project
from visionpack.eval import bbox_iou, evaluate
from visionpack.formats.classification import ImageFolderImporter
from visionpack.formats.yolo import YoloImporter
from visionpack.predictions import load_predictions
from visionpack.split import create_split


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


def _seed_detection(root: Path, images: int = 8) -> Project:
    """A one-class YOLO dataset where every image carries the same centered box
    (absolute x=10, y=10, w=20, h=20 in a 40x40 image)."""
    data = root / "raw"
    for i in range(images):
        _png(data / f"img{i}.png", i)
        (data / f"img{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (data / "classes.txt").write_text("obj\n", encoding="utf-8")
    project = Project.init(root, name="det", task="detection")
    YoloImporter(project, data).run()
    return Project.open(root)


def _echo_gt_predictions(project: Project, asset_ids: list[str], confidence: float = 0.9) -> dict:
    predictions = []
    for asset_id in asset_ids:
        annotation = project.index.annotation_for_asset(asset_id)
        objects = []
        for obj in annotation.objects:
            box = obj.bbox
            objects.append({"class": "obj", "confidence": confidence, "bbox": [box.x, box.y, box.width, box.height]})
        predictions.append({"image": asset_id, "objects": objects})
    return {"predictions": predictions}


class DetectionEvalTest(unittest.TestCase):
    def test_perfect_predictions_score_map_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_detection(root)
            create_split(project, train=0.5, val=0.25, test=0.25, strategy="stratified")
            project = Project.open(root)
            test_ids = next(s for s in project.index.splits() if s.id == "default").sets["test"]

            preds_path = root / "preds.json"
            preds_path.write_text(json.dumps(_echo_gt_predictions(project, test_ids)), encoding="utf-8")
            result = evaluate(project, load_predictions(project, preds_path))

            self.assertEqual(result["images"], len(test_ids))
            self.assertEqual(result["images_with_predictions"], len(test_ids))
            self.assertEqual(result["metrics"]["mAP50"], 1.0)
            self.assertEqual(result["metrics"]["mAP50_95"], 1.0)
            self.assertEqual(result["metrics"]["precision"], 1.0)
            self.assertEqual(result["metrics"]["recall"], 1.0)

    def test_false_positive_lowers_precision_not_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_detection(root)
            create_split(project, train=0.5, val=0.25, test=0.25, strategy="stratified")
            project = Project.open(root)
            test_ids = next(s for s in project.index.splits() if s.id == "default").sets["test"]

            document = _echo_gt_predictions(project, test_ids)
            # An extra confident box in a corner that overlaps no ground truth.
            document["predictions"][0]["objects"].append({"class": "obj", "confidence": 0.8, "bbox": [0, 0, 5, 5]})
            preds_path = root / "preds.json"
            preds_path.write_text(json.dumps(document), encoding="utf-8")
            result = evaluate(project, load_predictions(project, preds_path))

            self.assertEqual(result["metrics"]["recall"], 1.0)
            self.assertLess(result["metrics"]["precision"], 1.0)

    def test_predictions_outside_the_set_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_detection(root)
            create_split(project, train=0.5, val=0.25, test=0.25, strategy="stratified")
            project = Project.open(root)
            split = next(s for s in project.index.splits() if s.id == "default")

            # Predict (correctly) on train images only: the test set sees nothing.
            preds_path = root / "preds.json"
            preds_path.write_text(json.dumps(_echo_gt_predictions(project, split.sets["train"])), encoding="utf-8")
            result = evaluate(project, load_predictions(project, preds_path))

            self.assertEqual(result["images_with_predictions"], 0)
            self.assertEqual(result["metrics"]["mAP50"], 0.0)

    def test_coco_results_list_and_yolo_txt_predictions_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_detection(root, images=2)
            asset = project.index.assets()[0]

            coco_path = root / "coco_preds.json"
            coco_path.write_text(
                json.dumps([{"file_name": f"{asset.id}.png", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.7}]),
                encoding="utf-8",
            )
            loaded = load_predictions(project, coco_path)
            self.assertEqual(list(loaded.by_asset), [asset.id])
            self.assertEqual(loaded.by_asset[asset.id][0].class_id, "obj")
            self.assertEqual(loaded.by_asset[asset.id][0].confidence, 0.7)

            yolo_dir = root / "yolo_preds"
            yolo_dir.mkdir()
            (yolo_dir / f"{asset.id}.txt").write_text("0 0.5 0.5 0.5 0.5 0.65\n", encoding="utf-8")
            loaded = load_predictions(project, yolo_dir)
            box = loaded.by_asset[asset.id][0].bbox
            self.assertEqual((box.x, box.y, box.width, box.height), (10.0, 10.0, 20.0, 20.0))
            self.assertEqual(loaded.by_asset[asset.id][0].confidence, 0.65)

    def test_unmatched_references_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed_detection(root, images=2)
            preds_path = root / "preds.json"
            preds_path.write_text(json.dumps({"predictions": [{"image": "nope.png", "objects": []}]}), encoding="utf-8")
            loaded = load_predictions(project, preds_path)
            self.assertEqual(loaded.unmatched, ["nope.png"])

    def test_iou(self) -> None:
        from visionpack.core.models import BBox

        self.assertEqual(bbox_iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)), 1.0)
        self.assertEqual(bbox_iou(BBox(0, 0, 10, 10), BBox(20, 20, 10, 10)), 0.0)
        self.assertAlmostEqual(bbox_iou(BBox(0, 0, 10, 10), BBox(5, 0, 10, 10)), 50 / 150)


class ClassificationEvalTest(unittest.TestCase):
    def _seed(self, root: Path) -> Project:
        data = root / "data"
        for i in range(4):
            _png(data / "cat" / f"c{i}.png", i)
        for i in range(4):
            _png(data / "dog" / f"d{i}.png", i + 50)
        project = Project.init(root, name="cls", task="classification")
        ImageFolderImporter(project, data).run()
        return Project.open(root)

    def test_accuracy_and_confusion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._seed(root)
            create_split(project, train=0.5, val=0.25, test=0.25, strategy="stratified")
            project = Project.open(root)
            test_ids = next(s for s in project.index.splits() if s.id == "default").sets["test"]

            # Predict every test image as "cat": cats are right, dogs are wrong.
            document = {"predictions": [{"image": aid, "objects": [{"class": "cat", "confidence": 0.9}]} for aid in test_ids]}
            preds_path = root / "preds.json"
            preds_path.write_text(json.dumps(document), encoding="utf-8")
            result = evaluate(project, load_predictions(project, preds_path))

            truths = [project.index.annotation_for_asset(aid).objects[0].class_id for aid in test_ids]
            expected_accuracy = round(truths.count("cat") / len(truths), 4)
            self.assertEqual(result["metrics"]["accuracy"], expected_accuracy)
            self.assertEqual(sum(result["confusion_matrix"].get("dog", {}).values()), truths.count("dog"))
            self.assertEqual(result["confusion_matrix"].get("dog", {}).get("cat", 0), truths.count("dog"))


if __name__ == "__main__":
    unittest.main()
