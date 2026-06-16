from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from visionpack.core.models import Asset


def link_or_copy(src: Path, dest: Path) -> None:
    """Place ``src`` at ``dest`` without duplicating bytes when possible.

    CAS objects are immutable, so the export can share their inode via a hardlink
    (``cp -l``) instead of copying every image — zero extra bytes. Falls back to a
    copy when a hardlink can't span the two locations (different filesystem, or a
    platform/permission that disallows it).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


class AssetMaterializer:
    """Places asset bytes into an export tree without duplicating them.

    Local assets are hardlinked from the CAS (see :func:`link_or_copy`); remote
    (cloud-backed) assets are **not** downloaded — each is recorded in
    ``manifest.jsonl`` as ``(image, uri, ...)`` so a trainer streams it straight
    from the bucket. One export tree is therefore byte-free for cloud datasets and
    inode-shared for local ones. See docs/SPEC-cloud-sync.md.
    """

    def __init__(self, output: Path, root: Path) -> None:
        self.output = output
        self.root = root
        self._manifest: list[dict[str, Any]] = []

    def place(self, asset: Asset, dest: Path, extra: dict[str, Any] | None = None) -> None:
        """Materialize ``asset`` at ``dest`` (local) or record it (remote)."""
        if asset.is_remote:
            entry = {"image": dest.relative_to(self.output).as_posix(), "uri": asset.path}
            if extra:
                entry.update(extra)
            self._manifest.append(entry)
            return
        link_or_copy(asset.resolved_path(self.root), dest)

    def flush(self) -> int:
        """Write ``manifest.jsonl`` for any recorded remote assets; return the count."""
        if not self._manifest:
            return 0
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for entry in self._manifest:
                handle.write(json.dumps(entry) + "\n")
        return len(self._manifest)
