from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import orjson

from visionpack.core.models import Annotation, Asset, Split

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS annotations (id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, data BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_annotations_asset ON annotations(asset_id);
CREATE TABLE IF NOT EXISTS splits (id TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS imports (seq INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, data BLOB NOT NULL);
"""


class SqliteIndex:
    """SQLite-backed local index.

    Mirrors :class:`JsonIndex`'s public API so callers don't change, but scales
    to hundreds of thousands of records:

    - **opening is instant** — nothing is loaded until a read actually needs it;
    - **saving is incremental and atomic** — only the rows touched since the last
      save are written, inside one transaction, instead of rewriting the whole
      index (which was the dominant cost and a corruption risk with JSON);
    - connections are short-lived (opened per load/save), so no file handle
      lingers — important on Windows where an open handle blocks directory cleanup.

    A legacy ``index.json`` is migrated transparently on first open.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".vp" / "db" / "index.db"
        self._legacy = root / ".vp" / "db" / "index.json"
        # Lazily-populated read caches; None means "not loaded from the DB yet".
        self._assets: dict[str, Asset] | None = None
        self._annotations: dict[str, Annotation] | None = None
        self._annotation_by_asset: dict[str, Annotation] | None = None
        self._splits: dict[str, Split] | None = None
        self._metadata: dict[str, Any] | None = None
        # Write buffers flushed on save().
        self._dirty_assets: dict[str, Asset] = {}
        self._dirty_annotations: dict[str, Annotation] = {}
        self._dirty_splits: dict[str, Split] = {}
        self._dirty_metadata: dict[str, Any] = {}
        self._new_imports: list[dict[str, Any]] = []
        self._ensure_schema()
        self._migrate_legacy_if_needed()

    # -- connection helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> None:
        """Drop caches so the next read re-queries the DB (parity with JsonIndex)."""
        self._assets = None
        self._annotations = None
        self._annotation_by_asset = None
        self._splits = None
        self._metadata = None

    def save(self) -> None:
        if not (
            self._dirty_assets
            or self._dirty_annotations
            or self._dirty_splits
            or self._dirty_metadata
            or self._new_imports
        ):
            return
        with closing(self._connect()) as conn:
            if self._dirty_assets:
                conn.executemany(
                    "INSERT OR REPLACE INTO assets(id, data) VALUES (?, ?)",
                    [(a.id, orjson.dumps(a.to_dict())) for a in self._dirty_assets.values()],
                )
            if self._dirty_annotations:
                conn.executemany(
                    "INSERT OR REPLACE INTO annotations(id, asset_id, data) VALUES (?, ?, ?)",
                    [(n.id, n.asset_id, orjson.dumps(n.to_dict())) for n in self._dirty_annotations.values()],
                )
            if self._dirty_splits:
                conn.executemany(
                    "INSERT OR REPLACE INTO splits(id, data) VALUES (?, ?)",
                    [(s.id, orjson.dumps(s.to_dict())) for s in self._dirty_splits.values()],
                )
            if self._new_imports:
                conn.executemany(
                    "INSERT INTO imports(data) VALUES (?)",
                    [(orjson.dumps(record),) for record in self._new_imports],
                )
            if self._dirty_metadata:
                conn.executemany(
                    "INSERT OR REPLACE INTO metadata(key, data) VALUES (?, ?)",
                    [(key, orjson.dumps(value)) for key, value in self._dirty_metadata.items()],
                )
            conn.commit()
        self._dirty_assets.clear()
        self._dirty_annotations.clear()
        self._dirty_splits.clear()
        self._dirty_metadata.clear()
        self._new_imports.clear()

    # -- writes ---------------------------------------------------------------

    def upsert_asset(self, asset: Asset) -> None:
        self._dirty_assets[asset.id] = asset
        if self._assets is not None:
            self._assets[asset.id] = asset

    def upsert_annotation(self, annotation: Annotation) -> None:
        self._dirty_annotations[annotation.id] = annotation
        if self._annotations is not None:
            self._annotations[annotation.id] = annotation
        if self._annotation_by_asset is not None:
            self._annotation_by_asset[annotation.asset_id] = annotation

    def upsert_split(self, split: Split) -> None:
        self._dirty_splits[split.id] = split
        if self._splits is not None:
            self._splits[split.id] = split

    def add_import_record(self, record: dict[str, Any]) -> None:
        self._new_imports.append(record)

    def set_orphan_labels(self, paths: list[str]) -> None:
        value = [str(item) for item in paths]
        self._dirty_metadata["orphan_labels"] = value
        if self._metadata is not None:
            self._metadata["orphan_labels"] = value

    # -- reads ----------------------------------------------------------------

    def assets(self) -> list[Asset]:
        if self._assets is None:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT data FROM assets").fetchall()
            self._assets = {}
            for (blob,) in rows:
                asset = Asset.from_dict(orjson.loads(blob))
                self._assets[asset.id] = asset
            for asset_id, asset in self._dirty_assets.items():  # overlay unsaved writes
                self._assets[asset_id] = asset
        return list(self._assets.values())

    def annotations(self) -> list[Annotation]:
        if self._annotations is None:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT data FROM annotations").fetchall()
            self._annotations = {}
            for (blob,) in rows:
                annotation = Annotation.from_dict(orjson.loads(blob))
                self._annotations[annotation.id] = annotation
            for ann_id, annotation in self._dirty_annotations.items():
                self._annotations[ann_id] = annotation
        return list(self._annotations.values())

    def splits(self) -> list[Split]:
        if self._splits is None:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT data FROM splits").fetchall()
            self._splits = {}
            for (blob,) in rows:
                split = Split.from_dict(orjson.loads(blob))
                self._splits[split.id] = split
            for split_id, split in self._dirty_splits.items():
                self._splits[split_id] = split
        return list(self._splits.values())

    def annotation_for_asset(self, asset_id: str) -> Annotation | None:
        if self._annotation_by_asset is None:
            self._annotation_by_asset = {item.asset_id: item for item in self.annotations()}
        return self._annotation_by_asset.get(asset_id)

    def count_assets(self) -> int:
        if self._dirty_assets or self._assets is not None:
            return len(self.assets())
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0])

    # -- streaming reads ------------------------------------------------------
    #
    # These iterate rows straight off a DB cursor so a full-scan command never has
    # to hold every record (and the auxiliary maps) in RAM at once. When there are
    # unsaved writes or an already-materialized cache, they fall back to the
    # in-memory view to stay correct.

    def iter_assets(self) -> Iterator[Asset]:
        if self._dirty_assets or self._assets is not None:
            yield from self.assets()
            return
        conn = self._connect()
        try:
            for (blob,) in conn.execute("SELECT data FROM assets"):
                yield Asset.from_dict(orjson.loads(blob))
        finally:
            conn.close()

    def iter_annotations(self) -> Iterator[Annotation]:
        if self._dirty_annotations or self._annotations is not None:
            yield from self.annotations()
            return
        conn = self._connect()
        try:
            for (blob,) in conn.execute("SELECT data FROM annotations"):
                yield Annotation.from_dict(orjson.loads(blob))
        finally:
            conn.close()

    def iter_assets_with_annotations(self) -> Iterator[tuple[Asset, Annotation | None]]:
        """Stream each asset paired with its annotation (or None) via a LEFT JOIN.

        This is what export/stats want: one pass, no full asset list and no
        asset->annotation map materialized.
        """
        if self._dirty_assets or self._dirty_annotations or self._assets is not None or self._annotations is not None:
            for asset in self.assets():
                yield asset, self.annotation_for_asset(asset.id)
            return
        conn = self._connect()
        try:
            query = "SELECT a.data, n.data FROM assets a LEFT JOIN annotations n ON n.asset_id = a.id"
            for asset_blob, ann_blob in conn.execute(query):
                asset = Asset.from_dict(orjson.loads(asset_blob))
                annotation = Annotation.from_dict(orjson.loads(ann_blob)) if ann_blob is not None else None
                yield asset, annotation
        finally:
            conn.close()

    def orphan_labels(self) -> list[str]:
        return [str(item) for item in self._meta().get("orphan_labels", [])]

    def _meta(self) -> dict[str, Any]:
        if self._metadata is None:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT key, data FROM metadata").fetchall()
            self._metadata = {key: orjson.loads(blob) for key, blob in rows}
            self._metadata.update(self._dirty_metadata)
        return self._metadata

    # -- migration ------------------------------------------------------------

    def _migrate_legacy_if_needed(self) -> None:
        if not self._legacy.exists() or self._has_rows():
            return
        data = orjson.loads(self._legacy.read_bytes())
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO assets(id, data) VALUES (?, ?)",
                [(aid, orjson.dumps(rec)) for aid, rec in data.get("assets", {}).items()],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO annotations(id, asset_id, data) VALUES (?, ?, ?)",
                [(nid, str(rec.get("asset_id", "")), orjson.dumps(rec)) for nid, rec in data.get("annotations", {}).items()],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO splits(id, data) VALUES (?, ?)",
                [(sid, orjson.dumps(rec)) for sid, rec in data.get("splits", {}).items()],
            )
            conn.executemany(
                "INSERT INTO imports(data) VALUES (?)",
                [(orjson.dumps(record),) for record in data.get("imports", [])],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO metadata(key, data) VALUES (?, ?)",
                [(key, orjson.dumps(value)) for key, value in data.get("metadata", {}).items()],
            )
            conn.commit()
        # Keep the old file but take it out of the way so it isn't migrated twice.
        self._legacy.rename(self._legacy.with_suffix(".json.migrated"))

    def _has_rows(self) -> bool:
        with closing(self._connect()) as conn:
            for table in ("assets", "annotations", "splits"):
                if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                    return True
        return False
