from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import orjson

from visionpack.core.models import Annotation, Asset, BBox, ObjectAnnotation, Split
from visionpack.index.sqlite_index import SqliteIndex


def _asset(i: int) -> Asset:
    h = f"{i:016x}" + "0" * 48
    return Asset(
        id=f"asset_{h[:16]}",
        sha256=h,
        media_type="image",
        path=f".vp/objects/sha256/{h[:2]}/{h[2:4]}/{h}",
        original_path=f"raw/img{i}.jpg",
        width=100,
        height=80,
        channels=3,
        format="jpeg",
        size_bytes=1000,
        phash=f"{i:016x}",
        source="cam",
    )


def _ann(asset_id: str) -> Annotation:
    return Annotation(
        id=f"ann_{asset_id}",
        asset_id=asset_id,
        task="detection",
        format="internal",
        objects=[ObjectAnnotation(class_id="cat", geometry=BBox(1, 2, 3, 4))],
    )


class SqliteIndexTest(unittest.TestCase):
    def test_write_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idx = SqliteIndex(root)
            a = _asset(1)
            idx.upsert_asset(a)
            idx.upsert_annotation(_ann(a.id))
            idx.upsert_split(Split(id="default", strategy="random", sets={"train": [a.id]}))
            idx.set_orphan_labels(["x.txt"])
            idx.add_import_record({"format": "yolo"})
            idx.save()

            reopened = SqliteIndex(root)
            self.assertEqual([x.id for x in reopened.assets()], [a.id])
            self.assertEqual(reopened.annotation_for_asset(a.id).objects[0].class_id, "cat")
            self.assertEqual([s.id for s in reopened.splits()], ["default"])
            self.assertEqual(reopened.orphan_labels(), ["x.txt"])

    def test_reads_see_unsaved_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idx = SqliteIndex(Path(tmp))
            a = _asset(2)
            idx.upsert_asset(a)  # not saved yet
            self.assertEqual([x.id for x in idx.assets()], [a.id])

    def test_save_is_incremental(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idx = SqliteIndex(root)
            for i in range(5):
                idx.upsert_asset(_asset(i))
            idx.save()
            # A second save touching one row must not duplicate or drop the rest.
            idx.upsert_asset(_asset(99))
            idx.save()
            with closing(sqlite3.connect(root / ".vp" / "db" / "index.db")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            self.assertEqual(count, 6)

    def test_streaming_pairs_assets_with_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idx = SqliteIndex(root)
            a1, a2 = _asset(1), _asset(2)
            idx.upsert_asset(a1)
            idx.upsert_asset(a2)
            idx.upsert_annotation(_ann(a1.id))  # a2 has no annotation
            idx.save()

            by_id = {asset.id: ann for asset, ann in SqliteIndex(root).iter_assets_with_annotations()}
            self.assertEqual(set(by_id), {a1.id, a2.id})
            self.assertIsNotNone(by_id[a1.id])
            self.assertIsNone(by_id[a2.id])

    def test_streaming_reflects_unsaved_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idx = SqliteIndex(Path(tmp))
            a = _asset(3)
            idx.upsert_asset(a)  # not saved -> must use the in-memory fallback
            streamed = [asset.id for asset in idx.iter_assets()]
            self.assertEqual(streamed, [a.id])
            paired = [(asset.id, ann) for asset, ann in idx.iter_assets_with_annotations()]
            self.assertEqual(paired, [(a.id, None)])

    def test_migrates_legacy_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_dir = root / ".vp" / "db"
            db_dir.mkdir(parents=True)
            legacy = {
                "assets": {"asset_x": _asset(7).to_dict() | {"id": "asset_x"}},
                "annotations": {"ann_asset_x": _ann("asset_x").to_dict() | {"id": "ann_asset_x", "asset_id": "asset_x"}},
                "splits": {},
                "imports": [],
                "metadata": {"orphan_labels": ["o.txt"]},
            }
            (db_dir / "index.json").write_bytes(orjson.dumps(legacy))

            idx = SqliteIndex(root)
            self.assertEqual([a.id for a in idx.assets()], ["asset_x"])
            self.assertEqual(idx.annotation_for_asset("asset_x").objects[0].class_id, "cat")
            self.assertEqual(idx.orphan_labels(), ["o.txt"])
            # Legacy file moved aside, db created.
            self.assertFalse((db_dir / "index.json").exists())
            self.assertTrue((db_dir / "index.json.migrated").exists())
            self.assertTrue((db_dir / "index.db").exists())

    def test_does_not_remigrate_when_db_has_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idx = SqliteIndex(root)
            idx.upsert_asset(_asset(1))
            idx.save()
            # A stray legacy file should be ignored once the db is populated.
            (root / ".vp" / "db" / "index.json").write_bytes(orjson.dumps({"assets": {"asset_zzz": _asset(2).to_dict()}}))
            reopened = SqliteIndex(root)
            self.assertEqual([a.id for a in reopened.assets()], [_asset(1).id])


if __name__ == "__main__":
    unittest.main()
