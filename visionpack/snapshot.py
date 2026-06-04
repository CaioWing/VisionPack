from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from visionpack.core.errors import VisionPackError
from visionpack.core.models import ClassDef, utc_now
from visionpack.core.project import Project
from visionpack.stats import collect_stats
from visionpack.storage.hash import sha256_bytes, sha256_file, stable_json_hash


def create_snapshot(project: Project, message: str) -> dict[str, Any]:
    snapshot_dir = project.root / ".vp" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing = list_snapshots(project)
    version = f"v{len(existing) + 1}"
    parent = existing[-1]["version"] if existing else None
    project.index.save()  # flush any pending writes so the frozen db is current
    inventory = _inventory(project)
    # The inventory is the bulky part of a snapshot and is often unchanged
    # between versions, so store it as a content-addressed blob and reference it
    # by hash. Identical inventories are written once and shared across versions.
    inventory_hash = _store_inventory(project, inventory)
    # Freeze the index itself (content-addressed) so the snapshot is restorable:
    # `vp export --snapshot vN` opens this frozen db and streams from it. Images
    # are referenced from the shared CAS, never copied.
    index_db_hash = _freeze_index(project)
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
        "index_db_hash": index_db_hash,
        # Stored directly (small) so opening a snapshot doesn't need to load the
        # whole inventory blob just to recover the class list.
        "classes": [item.to_dict() for item in project.manifest.classes],
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


def _read_snapshot_file(project: Project, version: str) -> dict[str, Any]:
    """Read the snapshot record (vN.json) without rehydrating the inventory blob."""
    path = project.root / ".vp" / "snapshots" / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {version}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(project: Project, version: str) -> dict[str, Any]:
    payload = _read_snapshot_file(project, version)
    # Rehydrate the inventory from its blob unless an older snapshot still
    # embeds it inline. (Callers like diff/show need it; open_snapshot does not.)
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


def _dbs_dir(project: Project) -> Path:
    return project.root / ".vp" / "snapshots" / "dbs"


def _freeze_index(project: Project) -> str | None:
    """Copy the live index db into the content-addressed snapshot db store.

    Returns its hash, or ``None`` if there is no index yet. Identical indexes
    (nothing changed between two snapshots) share one frozen file.
    """
    source = project.root / ".vp" / "db" / "index.db"
    if not source.exists():
        return None
    digest = sha256_file(source)
    target = _dbs_dir(project) / f"{digest}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return digest


def open_snapshot(project: Project, version: str) -> Project:
    """Return a read-only view of the project as it was at ``version``.

    The view shares the project's root and object store (so images resolve from
    the shared CAS) but swaps in the frozen index and the snapshot's class list,
    so export/stats stream from that exact state without touching the live one.
    """
    # Light read: don't rehydrate the inventory blob (it can be tens of MB at
    # 100k+); everything open needs is in the small payload.
    payload = _read_snapshot_file(project, version)
    index_db_hash = payload.get("index_db_hash")
    if not index_db_hash:
        raise VisionPackError(
            f"Snapshot {version} predates restorable snapshots (no frozen index). "
            "Re-create it with `vp snapshot create` to export from it."
        )
    db_path = _dbs_dir(project) / f"{index_db_hash}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Frozen snapshot index missing: {index_db_hash}")

    classes_data = payload.get("classes")
    if classes_data is None:  # older snapshot: classes only live in the inventory blob
        inventory = _load_inventory(project, payload["inventory_hash"]) if payload.get("inventory_hash") else {}
        classes_data = inventory.get("classes", [])

    from visionpack.index.sqlite_index import SqliteIndex

    view = copy.copy(project)
    view.index = SqliteIndex(project.root, db_path=db_path)
    view.manifest = replace(project.manifest, classes=[ClassDef.from_dict(c) for c in classes_data])
    return view


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
