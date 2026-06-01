from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

CopyMode = Literal["copy", "move", "hardlink", "reference", "ingest"]


class ObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects_root = root / ".vp" / "objects" / "sha256"

    def object_path(self, sha256: str) -> Path:
        return self.objects_root / sha256[:2] / sha256[2:4] / sha256

    def store(self, source: Path, sha256: str, mode: CopyMode = "ingest") -> str:
        if mode == "reference":
            return str(source.resolve())

        destination = self.object_path(sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if mode == "move" and source.exists() and source.resolve() != destination.resolve():
                source.unlink()
            return str(destination.relative_to(self.root))

        if mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        elif mode == "move":
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
        return str(destination.relative_to(self.root))
