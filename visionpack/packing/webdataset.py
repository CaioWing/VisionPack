from __future__ import annotations

import io
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import zstandard as zstd

from visionpack.core.errors import VisionPackError
from visionpack.core.models import Asset, Keypoints, Polygon, utc_now
from visionpack.core.project import Project
from visionpack.split import resolve_export_sets


@dataclass(slots=True)
class TrainingPackSummary:
    path: Path
    format: str
    shards: int
    samples: int
    sets: dict[str, int]
    skipped: int = 0


def pack_training(
    project: Project,
    output: Path | None = None,
    profile_name: str = "training",
    split_id: str | None = None,
) -> TrainingPackSummary:
    """Pack the dataset into WebDataset shards for streaming training.

    Each sample is two consecutive tar members sharing a key (the asset id):
    ``<key>.<imgext>`` (the original image bytes) and ``<key>.json`` (normalized
    detection labels). With ``split_id`` each set gets its own shard series
    (``train-000000.tar`` ...), so a trainer can point each loader at the right
    glob. ``dataset.json`` at the root describes shards, classes and counts.
    """
    profile = project.manifest.pack_profiles.get(profile_name)
    if profile is None:
        raise VisionPackError(f"Pack profile not found in visionpack.yaml: {profile_name}")
    if str(profile.get("format")) != "webdataset":
        raise VisionPackError(
            f"Training pack expects a WebDataset profile, but {profile_name!r} has "
            f"format={profile.get('format')!r}. Set 'format: webdataset' in pack_profiles."
        )

    shard_size = int(profile.get("shard_size", 1024))
    if shard_size < 1:
        raise VisionPackError("shard_size (samples per shard) must be >= 1.")
    compression = str(profile.get("compression", "none")).lower()
    if compression not in {"none", "zstd"}:
        raise VisionPackError(f"Unsupported WebDataset compression {compression!r}. Use 'none' or 'zstd'.")
    level = int(profile.get("compression_level", 10))
    extension = ".tar.zst" if compression == "zstd" else ".tar"

    output_dir = (output or project.root / "exports" / "webdataset").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_for_asset, set_names = resolve_export_sets(project, split_id)
    classes = project.manifest.classes
    class_index = {item.id: idx for idx, item in enumerate(classes)}

    buckets: dict[str, list[Asset]] = defaultdict(list)
    skipped = 0
    for asset in project.index.assets():
        name = set_for_asset(asset.id)
        if name is None:
            skipped += 1
            continue
        buckets[name].append(asset)

    ordered = set_names or sorted(buckets)
    shard_records: list[dict[str, Any]] = []
    set_counts: dict[str, int] = {}
    total_samples = 0

    for set_name in ordered:
        assets = sorted(buckets.get(set_name, []), key=lambda item: item.id)
        set_counts[set_name] = len(assets)
        # Flat (no split) packs read nicer as "data-*.tar" than "all-*.tar".
        prefix = "data" if (split_id is None and set_name == "all") else set_name
        for shard_index, start in enumerate(range(0, len(assets), shard_size)):
            chunk = assets[start : start + shard_size]
            shard_name = f"{prefix}-{shard_index:06d}{extension}"
            count = _write_shard(project, output_dir / shard_name, chunk, class_index, compression, level)
            shard_records.append({"name": shard_name, "split": set_name, "samples": count})
            total_samples += count

    dataset_doc = {
        "tool": "visionpack",
        "format": "webdataset",
        "created_at": utc_now(),
        "dataset": project.manifest.name,
        "task": project.manifest.task,
        "split": split_id,
        "shard_size": shard_size,
        "compression": compression,
        "classes": [{"index": idx, "id": item.id, "name": item.name} for idx, item in enumerate(classes)],
        "sets": set_counts,
        "samples": total_samples,
        "shards": shard_records,
    }
    (output_dir / "dataset.json").write_text(json.dumps(dataset_doc, indent=2), encoding="utf-8")
    (output_dir / "classes.txt").write_text(
        "\n".join(item.name for item in classes) + ("\n" if classes else ""), encoding="utf-8"
    )

    return TrainingPackSummary(
        path=output_dir,
        format="webdataset",
        shards=len(shard_records),
        samples=total_samples,
        sets=set_counts,
        skipped=skipped,
    )


def _write_shard(
    project: Project,
    path: Path,
    assets: list[Asset],
    class_index: dict[str, int],
    compression: str,
    level: int,
) -> int:
    handle: BinaryIO = path.open("wb")
    compressor = None
    try:
        if compression == "zstd":
            compressor = zstd.ZstdCompressor(level=level).stream_writer(handle, closefd=False)
            tar = tarfile.open(fileobj=compressor, mode="w|")
        else:
            tar = tarfile.open(fileobj=handle, mode="w|")
        with tar:
            for asset in assets:
                _write_sample(tar, project, asset, class_index)
    finally:
        if compressor is not None:
            compressor.close()
        handle.close()
    return len(assets)


def _write_sample(tar: tarfile.TarFile, project: Project, asset: Asset, class_index: dict[str, int]) -> None:
    source = asset.resolved_path(project.root)
    if not source.exists():
        raise VisionPackError(f"Cannot pack missing asset {asset.id}: {source}")
    image_bytes = source.read_bytes()
    suffix = (Path(asset.original_path).suffix or f".{asset.format}").lower()

    # The two members must share the same key and be consecutive so WebDataset
    # groups them into one sample.
    _add_bytes(tar, f"{asset.id}{suffix}", image_bytes)
    _add_bytes(tar, f"{asset.id}.json", _sample_label(project, asset, class_index))


def _sample_label(project: Project, asset: Asset, class_index: dict[str, int]) -> bytes:
    annotation = project.index.annotation_for_asset(asset.id)
    objects: list[dict[str, Any]] = []
    if annotation:
        for obj in annotation.objects:
            if obj.class_id not in class_index:
                continue
            entry: dict[str, Any] = {"class_id": obj.class_id, "class_index": class_index[obj.class_id]}
            bbox = obj.bbox
            if bbox is not None:
                entry["bbox"] = [bbox.x, bbox.y, bbox.width, bbox.height]
                entry["bbox_normalized"] = [
                    (bbox.x + bbox.width / 2) / asset.width,
                    (bbox.y + bbox.height / 2) / asset.height,
                    bbox.width / asset.width,
                    bbox.height / asset.height,
                ]
            if isinstance(obj.geometry, Polygon):
                entry["polygon"] = obj.geometry.rings
            elif isinstance(obj.geometry, Keypoints):
                entry["keypoints"] = obj.geometry.points
            objects.append(entry)
    document = {"key": asset.id, "width": asset.width, "height": asset.height, "objects": objects}
    return json.dumps(document).encode("utf-8")


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))
