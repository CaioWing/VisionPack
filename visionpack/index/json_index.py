from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson

from visionpack.core.models import Annotation, Asset, Split


class JsonIndex:
    """Small local index used by the MVP.

    The public methods mirror a future DuckDB-backed implementation so callers
    do not need to know how records are persisted. Deserialized records are
    cached so repeated reads (validation, stats, export) don't re-parse the JSON
    on every call, and an asset->annotation map keeps lookups O(1).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".vp" / "db" / "index.json"
        self._data: dict[str, Any] = {
            "schema_version": 1,
            "assets": {},
            "annotations": {},
            "splits": {},
            "imports": [],
            "metadata": {"orphan_labels": []},
        }
        self._asset_cache: list[Asset] | None = None
        self._annotation_cache: list[Annotation] | None = None
        self._annotation_by_asset: dict[str, Annotation] | None = None
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self._data = orjson.loads(self.path.read_bytes())
        self._invalidate()

    def save(self) -> None:
        # orjson (no pretty-print) is ~7x faster and ~30% smaller than json with
        # indent; write to a temp file and os.replace so a crash mid-write can't
        # corrupt the index (the rename is atomic on the same filesystem).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = orjson.dumps(self._data, option=orjson.OPT_SORT_KEYS)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, self.path)

    def _invalidate(self) -> None:
        self._asset_cache = None
        self._annotation_cache = None
        self._annotation_by_asset = None

    def upsert_asset(self, asset: Asset) -> None:
        self._data["assets"][asset.id] = asset.to_dict()
        self._asset_cache = None

    def upsert_annotation(self, annotation: Annotation) -> None:
        self._data["annotations"][annotation.id] = annotation.to_dict()
        self._annotation_cache = None
        self._annotation_by_asset = None

    def upsert_split(self, split: Split) -> None:
        self._data["splits"][split.id] = split.to_dict()

    def assets(self) -> list[Asset]:
        if self._asset_cache is None:
            self._asset_cache = [Asset.from_dict(item) for item in self._data.get("assets", {}).values()]
        return self._asset_cache

    def annotations(self) -> list[Annotation]:
        if self._annotation_cache is None:
            self._annotation_cache = [Annotation.from_dict(item) for item in self._data.get("annotations", {}).values()]
        return self._annotation_cache

    def splits(self) -> list[Split]:
        return [Split.from_dict(item) for item in self._data.get("splits", {}).values()]

    def asset_ids(self) -> set[str]:
        return set(self._data.get("assets", {}))

    def annotation_for_asset(self, asset_id: str) -> Annotation | None:
        if self._annotation_by_asset is None:
            self._annotation_by_asset = {item.asset_id: item for item in self.annotations()}
        return self._annotation_by_asset.get(asset_id)

    def add_import_record(self, record: dict[str, Any]) -> None:
        self._data.setdefault("imports", []).append(record)

    def set_orphan_labels(self, paths: list[str]) -> None:
        self._data.setdefault("metadata", {})["orphan_labels"] = paths

    def orphan_labels(self) -> list[str]:
        return [str(item) for item in self._data.get("metadata", {}).get("orphan_labels", [])]

    def raw(self) -> dict[str, Any]:
        return self._data
