from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd
from PIL import Image

from visionpack.core.project import Project
from visionpack.formats.yolo import YoloImporter
from visionpack.packing import pack_training
from visionpack.split import create_split


def _seed(root: Path) -> Project:
    project = Project.init(root, name="wds-demo")
    source = root / "raw"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir(parents=True)
    (source / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    images = [(f"a{i}", 0) for i in range(10)] + [(f"b{i}", 1) for i in range(10)]
    for index, (name, class_index) in enumerate(images):
        color = (index * 7 % 256, index * 13 % 256, index * 29 % 256)
        Image.new("RGB", (50, 50), color=color).save(source / "images" / f"{name}.png", format="PNG")
        (source / "labels" / f"{name}.txt").write_text(f"{class_index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    YoloImporter(project, source).run()
    return Project.open(root)


def _set_training_profile(project: Project, **overrides) -> None:
    project.manifest.pack_profiles["training"] = {"format": "webdataset", "shard_size": 1024, "compression": "none", **overrides}
    project.save_manifest()


def _read_members(shard: Path) -> list[str]:
    if shard.suffix == ".zst":
        with shard.open("rb") as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                return [member.name for member in tar]
    with tarfile.open(shard, mode="r") as tar:
        return tar.getnames()


class TrainingPackTest(unittest.TestCase):
    def test_split_aware_pack_writes_per_set_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            _set_training_profile(project)
            create_split(Project.open(root), strategy="stratified")

            output = root / "exports" / "wds"
            summary = pack_training(Project.open(root), output=output, split_id="default")

            self.assertEqual(summary.sets, {"train": 16, "val": 2, "test": 2})
            self.assertEqual(summary.samples, 20)
            doc = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["split"], "default")
            self.assertEqual({s["split"] for s in doc["shards"]}, {"train", "val", "test"})
            # One shard per set at the default shard size.
            self.assertTrue((output / "train-000000.tar").exists())
            self.assertTrue((output / "val-000000.tar").exists())
            self.assertTrue((output / "test-000000.tar").exists())

    def test_each_sample_pairs_image_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            _set_training_profile(project)
            create_split(Project.open(root), strategy="stratified")
            output = root / "exports" / "wds"
            pack_training(Project.open(root), output=output, split_id="default")

            members = _read_members(output / "train-000000.tar")
            # 16 train samples -> 16 images + 16 json, image directly before its json.
            self.assertEqual(len(members), 32)
            for i in range(0, len(members), 2):
                key_image = members[i].rsplit(".", 1)[0]
                key_json = members[i + 1].rsplit(".", 1)[0]
                self.assertEqual(key_image, key_json)
                self.assertTrue(members[i + 1].endswith(".json"))

    def test_label_payload_has_normalized_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            _set_training_profile(project)
            output = root / "exports" / "wds"
            pack_training(Project.open(root), output=output)  # flat

            shard = output / "data-000000.tar"
            self.assertTrue(shard.exists())
            with tarfile.open(shard, mode="r") as tar:
                json_member = next(m for m in tar.getmembers() if m.name.endswith(".json"))
                payload = json.loads(tar.extractfile(json_member).read().decode("utf-8"))
            self.assertEqual(len(payload["objects"]), 1)
            obj = payload["objects"][0]
            self.assertIn(obj["class_index"], (0, 1))
            cx, cy, w, h = obj["bbox_normalized"]
            self.assertAlmostEqual(cx, 0.5, places=5)
            self.assertAlmostEqual(w, 0.4, places=5)

    def test_shard_size_chunks_into_multiple_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            _set_training_profile(project, shard_size=8)
            create_split(Project.open(root), strategy="stratified")
            output = root / "exports" / "wds"
            summary = pack_training(Project.open(root), output=output, split_id="default")

            # train has 16 samples -> two shards of 8.
            train_shards = sorted(p.name for p in output.glob("train-*.tar"))
            self.assertEqual(train_shards, ["train-000000.tar", "train-000001.tar"])
            self.assertEqual(summary.shards, len(list(output.glob("*.tar"))))

    def test_zstd_compression_produces_readable_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _seed(root)
            _set_training_profile(project, compression="zstd")
            output = root / "exports" / "wds"
            pack_training(Project.open(root), output=output)

            shard = output / "data-000000.tar.zst"
            self.assertTrue(shard.exists())
            members = _read_members(shard)
            self.assertEqual(len(members), 40)  # 20 images + 20 json


if __name__ == "__main__":
    unittest.main()
