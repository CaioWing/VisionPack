from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionpack.core.errors import FormatError, VisionPackError
from visionpack.core.models import Annotation, Asset, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import ImportSummary, IngestFailure, safe_path_component
from visionpack.media import image_info_from_bytes, is_image_path
from visionpack.perceptual import dhash_bytes
from visionpack.progress import ProgressCallback
from visionpack.storage.hash import sha256_bytes
from visionpack.storage.object_store import CopyMode


@dataclass(slots=True)
class _ProcessedImage:
    asset: Asset
    annotation: Annotation


class ImageFolderImporter:
    """Import a classification dataset laid out as ``root/<class>/<image>``.

    Each top-level subdirectory is a class; every image under it gets a
    whole-image label (an ``ObjectAnnotation`` with no geometry). This is the
    ubiquitous "ImageFolder" / ImageNet convention.
    """

    def __init__(self, project: Project, source: Path, copy_mode: CopyMode = "ingest") -> None:
        self.project = project
        self.source = source.resolve()
        self.copy_mode : CopyMode = copy_mode

    def run(self, progress: ProgressCallback | None = None) -> ImportSummary:
        if not self.source.exists() or not self.source.is_dir():
            raise FormatError(
                f"ImageFolder source must be a directory of class subfolders: {self.source}"
            )

        class_dirs = sorted(path for path in self.source.iterdir() if path.is_dir())
        if not class_dirs:
            raise FormatError(
                f"No class subdirectories found under {self.source}. "
                "Expected layout: <root>/<class-name>/<image-files>."
            )

        class_names = [path.name for path in class_dirs]
        classes_added = self.project.manifest.merge_classes(class_names)
        name_to_id = {item.name: item.id for item in self.project.manifest.classes}

        tasks: list[tuple[Path, str]] = []
        for class_dir in class_dirs:
            class_id = name_to_id[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and is_image_path(path):
                    tasks.append((path, class_id))

        summary = ImportSummary(classes_added=classes_added)

        def process(item: tuple[Path, str]) -> _ProcessedImage | IngestFailure:
            try:
                return self._process_image(*item)
            except (VisionPackError, OSError) as exc:
                return IngestFailure(path=str(item[0]), error=str(exc))

        total = len(tasks)
        with ThreadPoolExecutor() as pool:
            for done, outcome in enumerate(pool.map(process, tasks), 1):
                if isinstance(outcome, IngestFailure):
                    summary.failures.append(outcome)
                else:
                    processed = outcome
                    self.project.index.upsert_asset(processed.asset)
                    self.project.index.upsert_annotation(processed.annotation)
                    summary.assets += 1
                    summary.annotations += 1
                    summary.objects += 1
                if progress is not None:
                    progress(done, total)

        self.project.index.add_import_record(
            {
                "format": "imagefolder",
                "source": str(self.source),
                "copy_mode": self.copy_mode,
                "created_at": utc_now(),
                "assets": summary.assets,
                "annotations": summary.annotations,
                "objects": summary.objects,
            }
        )
        self.project.index.save()
        if classes_added:
            self.project.save_manifest()
        return summary

    def _process_image(self, image_path: Path, class_id: str) -> _ProcessedImage:
        data = image_path.read_bytes()
        digest = sha256_bytes(data)
        width, height, channels, image_format = image_info_from_bytes(data, image_path)
        asset_id = f"asset_{digest[:16]}"
        stored_path = self.project.object_store.store(image_path, digest, self.copy_mode, data=data)
        asset = Asset(
            id=asset_id,
            sha256=digest,
            media_type="image",
            path=stored_path,
            original_path=str(image_path),
            width=width,
            height=height,
            channels=channels,
            format=image_format,
            size_bytes=len(data),
            phash=dhash_bytes(data),
        )
        annotation = Annotation(
            id=f"ann_{asset_id}",
            asset_id=asset_id,
            task="classification",
            format="internal",
            objects=[ObjectAnnotation(class_id=class_id, geometry=None)],
            source={"type": "import", "format": "imagefolder", "path": str(image_path.parent), "imported_at": utc_now()},
        )
        return _ProcessedImage(asset=asset, annotation=annotation)


def export_imagefolder(
    project: Project, output: Path, split_id: str | None = None, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    """Export a classification dataset to the ImageFolder layout.

    Writes ``<set>/<class>/<image>`` when a split is given, else ``<class>/<image>``.
    Each image is filed under its label's class; images with no label are skipped.
    """
    from visionpack.formats.materialize import AssetMaterializer
    from visionpack.split import resolve_export_sets

    output = output.resolve()
    id_to_name = {item.id: item.name for item in project.manifest.classes}
    set_for_asset, _ = resolve_export_sets(project, split_id)
    materializer = AssetMaterializer(output, project.root)

    exported = 0
    skipped = 0
    per_set: dict[str, int] = {}
    total = project.index.count_assets()
    done = 0
    for asset, annotation in project.index.iter_assets_with_annotations():
        done += 1
        if progress is not None:
            progress(done, total)
        set_name = set_for_asset(asset.id)
        if set_name is None:
            skipped += 1
            continue
        label_obj = next((obj for obj in annotation.objects), None) if annotation else None
        if label_obj is None:
            skipped += 1
            continue
        class_name = safe_path_component(id_to_name.get(label_obj.class_id, label_obj.class_id))
        parts = [output, set_name, class_name] if split_id else [output, class_name]
        target_dir = Path(*[str(part) for part in parts])
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(asset.original_path).suffix.lower() or f".{asset.format}"
        materializer.place(asset, target_dir / f"{asset.id}{suffix}", {"set": set_name, "class": class_name})
        exported += 1
        per_set[set_name] = per_set.get(set_name, 0) + 1

    streamed = materializer.flush()
    result: dict[str, Any] = {"images": exported}
    if streamed:
        result["streamed"] = streamed
    if split_id:
        result["sets"] = per_set
        result["skipped"] = skipped
    return result
