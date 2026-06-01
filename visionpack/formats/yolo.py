from __future__ import annotations

import shutil
from pathlib import Path

from visionpack.core.errors import FormatError
from visionpack.core.models import Annotation, Asset, BBox, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import ImportSummary
from visionpack.media import is_image_path, image_info
from visionpack.storage.hash import sha256_file
from visionpack.storage.object_store import CopyMode


class YoloImporter:
    def __init__(self, project: Project, source: Path, copy_mode: CopyMode = "ingest") -> None:
        self.project = project
        self.source = source.resolve()
        self.copy_mode = copy_mode

    def run(self) -> ImportSummary:
        if not self.source.exists():
            raise FormatError(f"YOLO source does not exist: {self.source}")
        image_root = self.source if self.source.is_dir() else self.source.parent
        images = sorted(path for path in image_root.rglob("*") if path.is_file() and is_image_path(path))
        label_files = sorted(path for path in image_root.rglob("*.txt") if _looks_like_label_file(path))
        class_names = _discover_class_names(image_root) or _infer_class_names_from_labels(label_files)
        classes_added = int(self.project.manifest.ensure_classes(class_names))
        matched_label_files: set[Path] = set()
        summary = ImportSummary(classes_added=classes_added)

        for image_path in images:
            width, height, channels, image_format = image_info(image_path)
            digest = sha256_file(image_path)
            asset_id = f"asset_{digest[:16]}"
            stored_path = self.project.object_store.store(image_path, digest, self.copy_mode)
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
                size_bytes=image_path.stat().st_size,
            )
            self.project.index.upsert_asset(asset)
            summary.assets += 1

            label_path = _label_path_for_image(image_path, image_root)
            if label_path is not None:
                matched_label_files.add(label_path)
                objects = _parse_yolo_label(label_path, width, height, self.project)
                annotation = Annotation(
                    id=f"ann_{asset_id}",
                    asset_id=asset_id,
                    task=self.project.manifest.task,
                    format="internal",
                    objects=objects,
                    source={"type": "import", "format": "yolo", "path": str(label_path), "imported_at": utc_now()},
                )
                self.project.index.upsert_annotation(annotation)
                summary.annotations += 1
                summary.objects += len(objects)

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


def export_yolo(project: Project, output: Path) -> dict[str, int]:
    output = output.resolve()
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    classes = project.manifest.classes
    class_index = {item.id: idx for idx, item in enumerate(classes)}
    exported_images = 0
    exported_labels = 0
    exported_objects = 0

    for asset in project.index.assets():
        source = asset.resolved_path(project.root)
        suffix = Path(asset.original_path).suffix or f".{asset.format}"
        image_name = f"{asset.id}{suffix.lower()}"
        shutil.copy2(source, images_dir / image_name)
        exported_images += 1

        annotation = project.index.annotation_for_asset(asset.id)
        lines: list[str] = []
        if annotation:
            for obj in annotation.objects:
                if obj.class_id not in class_index:
                    continue
                bbox = obj.bbox
                x_center = (bbox.x + bbox.width / 2) / asset.width
                y_center = (bbox.y + bbox.height / 2) / asset.height
                width = bbox.width / asset.width
                height = bbox.height / asset.height
                lines.append(
                    f"{class_index[obj.class_id]} {_fmt(x_center)} {_fmt(y_center)} {_fmt(width)} {_fmt(height)}"
                )
                exported_objects += 1
        (labels_dir / f"{asset.id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported_labels += 1

    (output / "classes.txt").write_text("\n".join(item.name for item in classes) + ("\n" if classes else ""), encoding="utf-8")
    (output / "data.yaml").write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(classes)}\n"
        f"names: {[item.name for item in classes]!r}\n",
        encoding="utf-8",
    )
    return {"images": exported_images, "labels": exported_labels, "objects": exported_objects}


def _parse_yolo_label(path: Path, width: int, height: int, project: Project) -> list[ObjectAnnotation]:
    objects: list[ObjectAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise FormatError(f"Invalid YOLO label at {path}:{line_number}. Expected: class x_center y_center width height")
        try:
            class_index = int(float(parts[0]))
            x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
        except ValueError as exc:
            raise FormatError(f"Invalid numeric YOLO label value at {path}:{line_number}") from exc
        abs_width = box_width * width
        abs_height = box_height * height
        x = x_center * width - abs_width / 2
        y = y_center * height - abs_height / 2
        objects.append(
            ObjectAnnotation(
                class_id=project.manifest.class_id_for_index(class_index),
                bbox=BBox(x=x, y=y, width=abs_width, height=abs_height),
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
            stripped = line.strip()
            if not stripped:
                continue
            try:
                max_class = max(max_class, int(float(stripped.split()[0])))
            except (ValueError, IndexError):
                continue
    return [f"class_{idx}" for idx in range(max_class + 1)]


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")
