from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionpack.core.models import Annotation, Asset, Split


class JsonIndex:
    """Small local index used by the MVP.

    The public methods mirror a future DuckDB-backed implementation so callers
    do not need to know how records are persisted.
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
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def upsert_asset(self, asset: Asset) -> None:
        self._data["assets"][asset.id] = asset.to_dict()

    def upsert_annotation(self, annotation: Annotation) -> None:
        self._data["annotations"][annotation.id] = annotation.to_dict()

    def upsert_split(self, split: Split) -> None:
        self._data["splits"][split.id] = split.to_dict()

    def assets(self) -> list[Asset]:
        return [Asset.from_dict(item) for item in self._data.get("assets", {}).values()]

    def annotations(self) -> list[Annotation]:
        return [Annotation.from_dict(item) for item in self._data.get("annotations", {}).values()]

    def splits(self) -> list[Split]:
        return [Split.from_dict(item) for item in self._data.get("splits", {}).values()]

    def annotation_for_asset(self, asset_id: str) -> Annotation | None:
        for item in self.annotations():
            if item.asset_id == asset_id:
                return item
        return None

    def add_import_record(self, record: dict[str, Any]) -> None:
        self._data.setdefault("imports", []).append(record)

    def set_orphan_labels(self, paths: list[str]) -> None:
        self._data.setdefault("metadata", {})["orphan_labels"] = paths

    def orphan_labels(self) -> list[str]:
        return [str(item) for item in self._data.get("metadata", {}).get("orphan_labels", [])]

    def raw(self) -> dict[str, Any]:
        return self._data
