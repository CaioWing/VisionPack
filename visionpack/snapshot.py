from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionpack.core.models import utc_now
from visionpack.core.project import Project
from visionpack.stats import collect_stats
from visionpack.storage.hash import sha256_bytes, sha256_file, stable_json_hash


def create_snapshot(project: Project, message: str) -> dict[str, Any]:
    snapshot_dir = project.root / ".vp" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing = list_snapshots(project)
    version = f"v{len(existing) + 1}"
    parent = existing[-1]["version"] if existing else None
    inventory = _inventory(project)
    # The inventory is the bulky part of a snapshot and is often unchanged
    # between versions, so store it as a content-addressed blob and reference it
    # by hash. Identical inventories are written once and shared across versions.
    inventory_hash = _store_inventory(project, inventory)
    payload: dict[str, Any] = {
        "version": version,
        "message": message,
        "created_at": utc_now(),
        "manifest_hash": sha256_file(project.manifest_path),
        "assets_hash": stable_json_hash(inventory["assets"]),
        "annotations_hash": stable_json_hash(inventory["annotations"]),
        "splits_hash": stable_json_hash(inventory["splits"]),
        "parent": parent,
        "stats": collect_stats(project),
        "inventory_hash": inventory_hash,
    }
    (snapshot_dir / f"{version}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return load_snapshot(project, version)


def list_snapshots(project: Project) -> list[dict[str, Any]]:
    snapshot_dir = project.root / ".vp" / "snapshots"
    if not snapshot_dir.exists():
        return []
    snapshots = []
    for path in sorted(snapshot_dir.glob("v*.json"), key=_snapshot_sort_key):
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))
    return snapshots


def load_snapshot(project: Project, version: str) -> dict[str, Any]:
    path = project.root / ".vp" / "snapshots" / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {version}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Rehydrate the inventory from its blob unless an older snapshot still
    # embeds it inline.
    if "inventory" not in payload and "inventory_hash" in payload:
        payload["inventory"] = _load_inventory(project, payload["inventory_hash"])
    return payload


def _blob_path(project: Project, inventory_hash: str) -> Path:
    return project.root / ".vp" / "snapshots" / "blobs" / f"{inventory_hash}.json"


def _store_inventory(project: Project, inventory: dict[str, Any]) -> str:
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = sha256_bytes(payload)
    path = _blob_path(project, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return digest


def _load_inventory(project: Project, inventory_hash: str) -> dict[str, Any]:
    path = _blob_path(project, inventory_hash)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot inventory blob missing: {inventory_hash}")
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory(project: Project) -> dict[str, Any]:
    annotations = project.index.annotations()
    return {
        "assets": {
            asset.id: {
                "sha256": asset.sha256,
                "original_path": asset.original_path,
                "width": asset.width,
                "height": asset.height,
                "size_bytes": asset.size_bytes,
            }
            for asset in project.index.assets()
        },
        "annotations": {
            annotation.asset_id: stable_json_hash([obj.to_dict() for obj in annotation.objects])
            for annotation in annotations
        },
        "classes": [item.to_dict() for item in project.manifest.classes],
        "splits": {split.id: split.to_dict() for split in project.index.splits()},
    }


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("v") and stem[1:].isdigit():
        return int(stem[1:]), stem
    return 0, stem
