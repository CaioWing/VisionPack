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

    def store(self, source: Path, sha256: str, mode: CopyMode = "ingest", data: bytes | None = None) -> str:
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
                self._write(destination, source, data)
        elif mode == "move":
            shutil.move(str(source), str(destination))
        else:  # copy / ingest
            self._write(destination, source, data)
        return str(destination.relative_to(self.root))

    @staticmethod
    def _write(destination: Path, source: Path, data: bytes | None) -> None:
        # Objects in the store are immutable and identified by content hash, so
        # we don't need copy2's metadata preservation; reuse in-memory bytes when
        # the caller already read them (e.g. for hashing) to avoid a second read.
        if data is not None:
            destination.write_bytes(data)
        else:
            shutil.copy2(source, destination)
