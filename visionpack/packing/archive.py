from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import zstandard as zstd

from visionpack.core.errors import VisionPackError
from visionpack.core.models import utc_now
from visionpack.core.project import Project
from visionpack.stats import collect_stats


@dataclass(slots=True)
class ArchivePackSummary:
    path: Path
    format: str
    files: int
    size_bytes: int
    assets: int


def pack_archive(project: Project, output: Path | None = None, profile_name: str = "archive") -> ArchivePackSummary:
    profile = project.manifest.pack_profiles.get(profile_name)
    if profile is None:
        raise VisionPackError(f"Pack profile not found in visionpack.yaml: {profile_name}")

    archive_format = str(profile.get("format", "tar.zst"))
    if archive_format not in {"tar", "tar.zst"}:
        raise VisionPackError(
            f"Archive profile {profile_name!r} uses unsupported format {archive_format!r}. "
            "Supported formats: tar, tar.zst"
        )

    output_path = output or _default_output_path(project, profile_name, archive_format)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    packer = _TarZstWriter(output_path, compression_level=int(profile.get("compression_level", 10)))
    files = 0
    assets = project.index.assets()
    include_assets = bool(profile.get("include_assets", profile.get("include_raw", True)))
    include_metadata = bool(profile.get("include_metadata", True))

    with packer.open(archive_format) as tar:
        files += _add_path(tar, project.manifest_path, "visionpack.yaml")

        index_path = project.root / ".vp" / "db" / "index.db"
        if index_path.exists():
            files += _add_path(tar, index_path, ".vp/db/index.db")

        if include_metadata:
            for snapshot_path in sorted((project.root / ".vp" / "snapshots").glob("*.json")):
                files += _add_path(tar, snapshot_path, f".vp/snapshots/{snapshot_path.name}")
            # Snapshot inventories are stored as content-addressed blobs; pack
            # them so archived snapshots stay self-contained.
            for blob_path in sorted((project.root / ".vp" / "snapshots" / "blobs").glob("*.json")):
                files += _add_path(tar, blob_path, f".vp/snapshots/blobs/{blob_path.name}")

        if include_assets:
            seen_assets: set[str] = set()
            for asset in sorted(assets, key=lambda item: item.sha256):
                source = asset.resolved_path(project.root)
                if not source.exists():
                    raise VisionPackError(f"Cannot pack missing asset {asset.id}: {source}")
                if source.is_relative_to(project.root):
                    arcname = source.relative_to(project.root).as_posix()
                else:
                    arcname = f"external_assets/{asset.sha256}{Path(asset.original_path).suffix.lower()}"
                if arcname in seen_assets:
                    continue
                seen_assets.add(arcname)
                files += _add_path(tar, source, arcname)

        metadata = {
            "tool": "visionpack",
            "created_at": utc_now(),
            "profile": profile_name,
            "format": archive_format,
            "dataset": project.manifest.name,
            "stats": collect_stats(project),
            "assets": [
                {
                    "id": asset.id,
                    "sha256": asset.sha256,
                    "path": asset.path,
                    "original_path": asset.original_path,
                    "size_bytes": asset.size_bytes,
                }
                for asset in sorted(assets, key=lambda item: item.id)
            ],
        }
        files += _add_bytes(tar, "pack.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"))

    return ArchivePackSummary(
        path=output_path,
        format=archive_format,
        files=files,
        size_bytes=output_path.stat().st_size,
        assets=len(assets) if include_assets else 0,
    )


class _TarZstWriter:
    def __init__(self, path: Path, compression_level: int) -> None:
        self.path = path
        self.compression_level = compression_level
        self._file: BinaryIO | None = None
        self._zstd: Any | None = None

    def open(self, archive_format: str) -> _TarContext:
        self._file = self.path.open("wb")
        if archive_format == "tar.zst":
            compressor = zstd.ZstdCompressor(level=self.compression_level)
            self._zstd = compressor.stream_writer(self._file, closefd=False)
            tar = tarfile.open(fileobj=self._zstd, mode="w|")
        else:
            tar = tarfile.open(fileobj=self._file, mode="w|")
        return _TarContext(tar, self)

    def close(self) -> None:
        if self._zstd is not None:
            self._zstd.close()
        if self._file is not None:
            self._file.close()


class _TarContext:
    def __init__(self, tar: tarfile.TarFile, writer: _TarZstWriter) -> None:
        self.tar = tar
        self.writer = writer

    def __enter__(self) -> tarfile.TarFile:
        return self.tar

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.tar.close()
        self.writer.close()


def _default_output_path(project: Project, profile_name: str, archive_format: str) -> Path:
    suffix = ".tar.zst" if archive_format == "tar.zst" else ".tar"
    return project.root / "exports" / "archive" / f"{project.manifest.name}-{profile_name}{suffix}"


def _add_path(tar: tarfile.TarFile, source: Path, arcname: str) -> int:
    info = tar.gettarinfo(str(source), arcname)
    info.mtime = 0
    with source.open("rb") as handle:
        tar.addfile(info, handle)
    return 1


def _add_bytes(tar: tarfile.TarFile, arcname: str, payload: bytes) -> int:
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))
    return 1
