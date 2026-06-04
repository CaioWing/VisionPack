from __future__ import annotations

from dataclasses import dataclass

from visionpack.core.project import Project
from visionpack.snapshot import list_snapshots
from visionpack.storage.hash import sha256_file


@dataclass(slots=True)
class FsckIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass(slots=True)
class FsckReport:
    issues: list[FsckIssue]
    checked_assets: int
    checked_objects: int

    @property
    def errors(self) -> list[FsckIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[FsckIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def run_fsck(project: Project, deep: bool = False, check_orphans: bool = True) -> FsckReport:
    """Verify the dataset is internally consistent and backed by real bytes.

    Cheap checks (default): every asset's stored object exists, no annotation or
    split references a missing asset, snapshot inventory blobs are present, and no
    unreferenced objects linger in the store. ``deep`` additionally re-hashes
    every stored object and compares it to the recorded ``sha256`` — catching
    silent corruption / bit-rot, at the cost of reading all bytes.
    """
    issues: list[FsckIssue] = []
    asset_ids: set[str] = set()
    referenced_sha: set[str] = set()
    checked_assets = 0

    for asset in project.index.iter_assets():
        checked_assets += 1
        asset_ids.add(asset.id)
        referenced_sha.add(asset.sha256)
        path = asset.resolved_path(project.root)
        if not path.exists():
            issues.append(FsckIssue("error", "object.missing", f"{asset.id}: stored object not found at {path}"))
            continue
        if deep:
            actual = sha256_file(path)
            if actual != asset.sha256:
                issues.append(
                    FsckIssue(
                        "error",
                        "object.hash_mismatch",
                        f"{asset.id}: content changed (recorded {asset.sha256[:12]}…, found {actual[:12]}…) at {path}",
                    )
                )

    for annotation in project.index.iter_annotations():
        if annotation.asset_id not in asset_ids:
            issues.append(
                FsckIssue("error", "annotation.orphan", f"{annotation.id} references missing asset {annotation.asset_id}")
            )

    for split in project.index.splits():
        for set_name, ids in split.sets.items():
            for asset_id in ids:
                if asset_id not in asset_ids:
                    issues.append(
                        FsckIssue("error", "split.missing_asset", f"split {split.id!r}/{set_name} references missing asset {asset_id}")
                    )

    blobs_dir = project.root / ".vp" / "snapshots" / "blobs"
    for snapshot in list_snapshots(project):
        inventory_hash = snapshot.get("inventory_hash")
        if inventory_hash and not (blobs_dir / f"{inventory_hash}.json").exists():
            issues.append(
                FsckIssue(
                    "error",
                    "snapshot.blob_missing",
                    f"snapshot {snapshot.get('version')} inventory blob {inventory_hash[:12]}… is missing",
                )
            )

    checked_objects = 0
    objects_root = project.root / ".vp" / "objects" / "sha256"
    if check_orphans and objects_root.exists():
        for path in objects_root.rglob("*"):
            if not path.is_file():
                continue
            checked_objects += 1
            if path.name not in referenced_sha:
                issues.append(
                    FsckIssue("warning", "object.orphan", f"unreferenced object in store: {path.name[:12]}… ({path})")
                )

    return FsckReport(issues=issues, checked_assets=checked_assets, checked_objects=checked_objects)
