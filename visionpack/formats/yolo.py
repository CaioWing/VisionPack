from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionpack.core.errors import FormatError, VisionPackError
from visionpack.core.models import Annotation, Asset, BBox, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import ImportSummary, IngestFailure
from visionpack.formats.materialize import AssetMaterializer
from visionpack.media import image_info_from_bytes, is_image_path
from visionpack.perceptual import dhash_bytes
from visionpack.progress import ProgressCallback
from visionpack.split import resolve_export_sets
from visionpack.storage.hash import sha256_bytes
from visionpack.storage.object_store import CopyMode


@dataclass(slots=True)
class _ProcessedImage:
    asset: Asset
    annotation: Annotation | None
    object_count: int
    label_path: Path | None


class YoloImporter:
    def __init__(self, project: Project, source: Path, copy_mode: CopyMode = "ingest") -> None:
        self.project = project
        self.source = source.resolve()
        self.copy_mode = copy_mode

    def run(self, progress: ProgressCallback | None = None) -> ImportSummary:
        if not self.source.exists():
            raise FormatError(f"YOLO source does not exist: {self.source}")
        image_root = self.source if self.source.is_dir() else self.source.parent
        images = sorted(path for path in image_root.rglob("*") if path.is_file() and is_image_path(path))
        label_files = sorted(path for path in image_root.rglob("*.txt") if _looks_like_label_file(path))
        source_class_names = _discover_class_names(image_root) or _infer_class_names_from_labels(label_files)
        classes_added = self.project.manifest.merge_classes(source_class_names)
        # Map this source's label indices to manifest class ids *by name*, so a
        # second source whose classes are in a different order is not mislabeled.
        name_to_id = {item.name: item.id for item in self.project.manifest.classes}
        self._index_to_class_id = {index: name_to_id[name] for index, name in enumerate(source_class_names)}
        matched_label_files: set[Path] = set()
        summary = ImportSummary(classes_added=classes_added)

        # Reading bytes, hashing, image probing and storing are per-file and
        # I/O-bound, so fan them out across threads. Index mutation stays on the
        # calling thread, and ThreadPoolExecutor.map preserves input order so the
        # resulting index is deterministic regardless of worker scheduling.
        def process(path: Path) -> _ProcessedImage | IngestFailure:
            try:
                return self._process_image(path, image_root)
            except (VisionPackError, OSError) as exc:  # corrupt/unreadable image, missing file
                return IngestFailure(path=str(path), error=str(exc))

        total = len(images)
        with ThreadPoolExecutor() as pool:
            for done, outcome in enumerate(pool.map(process, images), 1):
                if isinstance(outcome, IngestFailure):
                    summary.failures.append(outcome)
                else:
                    processed = outcome
                    self.project.index.upsert_asset(processed.asset)
                    summary.assets += 1
                    if processed.annotation is not None:
                        matched_label_files.add(processed.label_path)  # type: ignore[arg-type]
                        self.project.index.upsert_annotation(processed.annotation)
                        summary.annotations += 1
                        summary.objects += processed.object_count
                if progress is not None:
                    progress(done, total)

        orphan_labels = [str(path) for path in label_files if path not in matched_label_files]
        self.project.index.set_orphan_labels(orphan_labels)
        self.project.index.add_import_record(
            {
                "format": "yolo",
                "source": str(self.source),
                "copy_mode": self.copy_mode,
                "created_at": utc_now(),
                "assets": summary.assets,
                "annotations": summary.annotations,
                "objects": summary.objects,
                "orphan_labels": len(orphan_labels),
            }
        )
        self.project.index.save()
        if classes_added:
            self.project.save_manifest()
        summary.orphan_labels = len(orphan_labels)
        return summary

    def _process_image(self, image_path: Path, image_root: Path) -> _ProcessedImage:
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

        label_path = _label_path_for_image(image_path, image_root)
        annotation: Annotation | None = None
        object_count = 0
        if label_path is not None:
            objects = _parse_yolo_label(label_path, width, height, self._index_to_class_id)
            object_count = len(objects)
            annotation = Annotation(
                id=f"ann_{asset_id}",
                asset_id=asset_id,
                task=self.project.manifest.task,
                format="internal",
                objects=objects,
                source={"type": "import", "format": "yolo", "path": str(label_path), "imported_at": utc_now()},
            )
        return _ProcessedImage(asset=asset, annotation=annotation, object_count=object_count, label_path=label_path)


def export_yolo(
    project: Project, output: Path, split_id: str | None = None, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    """Export to the YOLO layout.

    Without ``split_id`` everything goes to flat ``images/`` and ``labels/``
    directories. With a split, images and labels are written under
    ``images/<set>/`` and ``labels/<set>/`` and ``data.yaml`` points each of
    train/val/test at its set, which is what trainers (e.g. Ultralytics) expect.
    """
    output = output.resolve()
    classes = project.manifest.classes
    class_index = {item.id: idx for idx, item in enumerate(classes)}

    set_for_asset, set_names = resolve_export_sets(project, split_id)
    materializer = AssetMaterializer(output, project.root)

    exported_images = 0
    exported_labels = 0
    exported_objects = 0
    per_set: dict[str, int] = {name: 0 for name in set_names}
    skipped = 0

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
        images_dir = output / "images" / set_name if split_id else output / "images"
        labels_dir = output / "labels" / set_name if split_id else output / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(asset.original_path).suffix or f".{asset.format}"
        image_dest = images_dir / f"{asset.id}{suffix.lower()}"
        label_dest = labels_dir / f"{asset.id}.txt"
        materializer.place(
            asset, image_dest, {"label": label_dest.relative_to(output).as_posix(), "set": set_name}
        )
        exported_images += 1
        per_set[set_name] = per_set.get(set_name, 0) + 1

        lines: list[str] = []
        if annotation:
            for obj in annotation.objects:
                if obj.class_id not in class_index:
                    continue
                bbox = obj.bbox
                if bbox is None:  # whole-image label (classification): no box to write
                    continue
                x_center = (bbox.x + bbox.width / 2) / asset.width
                y_center = (bbox.y + bbox.height / 2) / asset.height
                width = bbox.width / asset.width
                height = bbox.height / asset.height
                lines.append(
                    f"{class_index[obj.class_id]} {_fmt(x_center)} {_fmt(y_center)} {_fmt(width)} {_fmt(height)}"
                )
                exported_objects += 1
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported_labels += 1

    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(item.name for item in classes) + ("\n" if classes else ""), encoding="utf-8")
    (output / "data.yaml").write_text(_render_data_yaml(classes, split_id, set_names), encoding="utf-8")
    streamed = materializer.flush()

    result: dict[str, Any] = {"images": exported_images, "labels": exported_labels, "objects": exported_objects}
    if streamed:
        result["streamed"] = streamed
    if split_id:
        result["sets"] = per_set
        result["skipped"] = skipped
    return result


def _render_data_yaml(classes: list[Any], split_id: str | None, set_names: list[str]) -> str:
    if split_id:
        lines = ["path: ."]
        for name in ("train", "val", "test"):
            if name in set_names:
                lines.append(f"{name}: images/{name}")
    else:
        lines = ["path: .", "train: images", "val: images"]
    lines.append(f"nc: {len(classes)}")
    lines.append(f"names: {[item.name for item in classes]!r}")
    return "\n".join(lines) + "\n"


def _parse_yolo_label(path: Path, width: int, height: int, index_to_class_id: dict[int, str]) -> list[ObjectAnnotation]:
    return parse_yolo_label_text(path.read_text(encoding="utf-8"), str(path), width, height, index_to_class_id)


def parse_yolo_label_text(
    text: str, origin: str, width: int, height: int, index_to_class_id: dict[int, str]
) -> list[ObjectAnnotation]:
    """Parse YOLO label lines into absolute-coordinate objects.

    ``origin`` is only used in error messages (a path or URI), so this works the
    same whether the text came from disk or a remote object.
    """
    objects: list[ObjectAnnotation] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = _clean_label_line(line)
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise FormatError(f"Invalid YOLO label at {origin}:{line_number}. Expected: class x_center y_center width height")
        try:
            class_index = int(float(parts[0]))
            x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
        except ValueError as exc:
            raise FormatError(f"Invalid numeric YOLO label value at {origin}:{line_number}") from exc
        abs_width = box_width * width
        abs_height = box_height * height
        x = x_center * width - abs_width / 2
        y = y_center * height - abs_height / 2
        objects.append(
            ObjectAnnotation(
                class_id=index_to_class_id.get(class_index, f"class_{class_index}"),
                geometry=BBox(x=x, y=y, width=abs_width, height=abs_height),
            )
        )
    return objects


def _label_path_for_image(image_path: Path, root: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    parts = image_path.relative_to(root).parts
    if "images" in parts:
        replaced = ["labels" if part == "images" else part for part in parts]
        candidates.append((root / Path(*replaced)).with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_class_names(root: Path) -> list[str]:
    for name in ("classes.txt", "obj.names"):
        path = root / name
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        names = _read_names_from_data_yaml(data_yaml)
        if names:
            return names
    return []


def _read_names_from_data_yaml(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            value = stripped.split(":", 1)[1].strip()
            if value.startswith("[") and value.endswith("]"):
                return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "names:":
            names: list[str] = []
            for child in lines[idx + 1 :]:
                stripped = child.strip()
                if not stripped.startswith("- "):
                    break
                names.append(stripped[2:].strip().strip("'\""))
            return names
    return []


def _looks_like_label_file(path: Path) -> bool:
    return path.name not in {"classes.txt"} and path.suffix.lower() == ".txt"


def _infer_class_names_from_labels(label_files: list[Path]) -> list[str]:
    max_class = -1
    for path in label_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = _clean_label_line(line)
            if not stripped:
                continue
            try:
                max_class = max(max_class, int(float(stripped.split()[0])))
            except (ValueError, IndexError):
                continue
    return [f"class_{idx}" for idx in range(max_class + 1)]


def _clean_label_line(line: str) -> str:
    return line.strip().lstrip("\ufeff")


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")
