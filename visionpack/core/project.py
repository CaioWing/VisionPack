from __future__ import annotations

import shutil
from pathlib import Path

from visionpack.core.errors import ProjectNotFoundError
from visionpack.core.manifest import Manifest, read_manifest, write_manifest
from visionpack.index import JsonIndex
from visionpack.storage.object_store import ObjectStore


class Project:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = self.root / "visionpack.yaml"
        if not self.manifest_path.exists():
            raise ProjectNotFoundError(f"No visionpack.yaml found at {self.root}")
        self.manifest = read_manifest(self.manifest_path)
        self.index = JsonIndex(self.root)
        self.object_store = ObjectStore(self.root)

    @classmethod
    def open(cls, root: str | Path = ".") -> "Project":
        root_path = Path(root).resolve()
        manifest = _find_manifest(root_path)
        if manifest is None:
            raise ProjectNotFoundError("No VisionPack project found. Run `vp init` first.")
        return cls(manifest.parent)

    @classmethod
    def init(cls, root: str | Path = ".", name: str | None = None, task: str = "detection") -> "Project":
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        dataset_name = name or root_path.name
        manifest_path = root_path / "visionpack.yaml"
        if not manifest_path.exists():
            write_manifest(manifest_path, Manifest.default(dataset_name, task))
        for directory in [
            root_path / ".vp" / "db",
            root_path / ".vp" / "objects",
            root_path / ".vp" / "snapshots",
            root_path / ".vp" / "cache",
            root_path / ".vp" / "logs",
            root_path / "assets",
            root_path / "annotations",
            root_path / "exports",
            root_path / "reports",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        _ensure_readme(root_path / "assets" / "README.md", "Raw or materialized dataset assets.")
        _ensure_readme(root_path / "annotations" / "README.md", "Source annotation files and interchange exports.")
        project = cls(root_path)
        project.index.save()
        return project

    def save_manifest(self) -> None:
        write_manifest(self.manifest_path, self.manifest)

    def materialize_asset(self, asset_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset_path, output_path)


Dataset = Project


def _find_manifest(start: Path) -> Path | None:
    current = start
    while True:
        candidate = current / "visionpack.yaml"
        if candidate.exists():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def _ensure_readme(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content + "\n", encoding="utf-8")
