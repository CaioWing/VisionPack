from __future__ import annotations

import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionpack.core.errors import FormatError
from visionpack.core.manifest import class_id_from_name
from visionpack.core.models import Annotation, Asset, BBox, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import ImportSummary
from visionpack.media import image_info_from_bytes
from visionpack.perceptual import dhash_bytes
from visionpack.split import resolve_export_sets
from visionpack.storage.hash import sha256_bytes
from visionpack.storage.object_store import CopyMode


@dataclass(slots=True)
class _ProcessedImage:
    asset: Asset
    annotation: Annotation | None
    object_count: int


class CocoImporter:
    """Import a COCO *instances* JSON (object detection)."""

    def __init__(self, project: Project, source: Path, images_dir: Path, copy_mode: CopyMode = "ingest") -> None:
        self.project = project
        self.source = source.resolve()
        self.images_dir = images_dir.resolve()
        self.copy_mode = copy_mode

    def run(self) -> ImportSummary:
        if not self.source.exists():
            raise FormatError(f"COCO annotation file does not exist: {self.source}")
        if not self.images_dir.exists():
            raise FormatError(f"COCO images directory does not exist: {self.images_dir}")

        document = _load_coco(self.source)
        categories = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in document.get("categories", [])}
        # Preserve category order as declared in the file when seeding classes.
        category_names = [categories[cat_id] for cat_id in categories]
        classes_added = self.project.manifest.merge_classes(category_names)
        name_to_class_id = {item.name: item.id for item in self.project.manifest.classes}
        category_to_class_id = {
            cat_id: name_to_class_id.get(name, class_id_from_name(name)) for cat_id, name in categories.items()
        }

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        images = document.get("images", [])
        summary = ImportSummary(classes_added=classes_added)

        def process(image_record: dict[str, Any]) -> _ProcessedImage:
            return self._process_image(image_record, annotations_by_image, category_to_class_id)

        with ThreadPoolExecutor() as pool:
            for processed in pool.map(process, images):
                self.project.index.upsert_asset(processed.asset)
                summary.assets += 1
                if processed.annotation is not None:
                    self.project.index.upsert_annotation(processed.annotation)
                    summary.annotations += 1
                    summary.objects += processed.object_count

        self.project.index.add_import_record(
            {
                "format": "coco",
                "source": str(self.source),
                "images_dir": str(self.images_dir),
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

    def _process_image(
        self,
        image_record: dict[str, Any],
        annotations_by_image: dict[int, list[dict[str, Any]]],
        category_to_class_id: dict[int, str],
    ) -> _ProcessedImage:
        file_name = str(image_record["file_name"])
        image_path = self.images_dir / file_name
        if not image_path.exists():
            raise FormatError(
                f"COCO image file not found: {image_path}\n"
                f"  declared as file_name={file_name!r} for image id={image_record.get('id')}\n"
                f"  looked under --images {self.images_dir}"
            )
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

        records = annotations_by_image.get(int(image_record["id"]), [])
        objects: list[ObjectAnnotation] = []
        for record in records:
            bbox = record.get("bbox")
            if not bbox or len(bbox) < 4:
                raise FormatError(f"COCO annotation {record.get('id')} for image {file_name} has an invalid bbox: {bbox!r}")
            x, y, box_width, box_height = (float(value) for value in bbox[:4])
            objects.append(
                ObjectAnnotation(
                    class_id=category_to_class_id.get(int(record["category_id"]), str(record["category_id"])),
                    bbox=BBox(x=x, y=y, width=box_width, height=box_height),
                    attributes={"iscrowd": int(record["iscrowd"])} if record.get("iscrowd") else {},
                )
            )

        annotation: Annotation | None = None
        if objects:
            annotation = Annotation(
                id=f"ann_{asset_id}",
                asset_id=asset_id,
                task=self.project.manifest.task,
                format="internal",
                objects=objects,
                source={"type": "import", "format": "coco", "path": str(self.source), "imported_at": utc_now()},
            )
        return _ProcessedImage(asset=asset, annotation=annotation, object_count=len(objects))


def export_coco(project: Project, output: Path, split_id: str | None = None) -> dict[str, Any]:
    """Export to COCO instances JSON.

    Without ``split_id`` a single ``annotations.json`` plus a flat ``images/``
    directory is written. With a split, images go to ``images/<set>/`` and each
    set gets its own ``annotations/instances_<set>.json``.
    """
    output = output.resolve()
    classes = project.manifest.classes
    # COCO category ids are 1-based by convention.
    class_to_category = {item.id: index + 1 for index, item in enumerate(classes)}
    categories = [{"id": index + 1, "name": item.name, "supercategory": ""} for index, item in enumerate(classes)]

    set_for_asset, _ = resolve_export_sets(project, split_id)

    # Accumulate images/annotations per set; flat export uses a single bucket.
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"images": [], "annotations": []})
    next_image_id: dict[str, int] = defaultdict(lambda: 1)
    next_annotation_id: dict[str, int] = defaultdict(lambda: 1)
    exported_objects = 0
    skipped = 0

    for asset in project.index.assets():
        set_name = set_for_asset(asset.id)
        if set_name is None:
            skipped += 1
            continue
        images_dir = output / "images" / set_name if split_id else output / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        source = asset.resolved_path(project.root)
        suffix = Path(asset.original_path).suffix.lower() or f".{asset.format}"
        file_name = f"{asset.id}{suffix}"
        shutil.copy2(source, images_dir / file_name)

        image_id = next_image_id[set_name]
        next_image_id[set_name] += 1
        buckets[set_name]["images"].append(
            {"id": image_id, "file_name": file_name, "width": asset.width, "height": asset.height}
        )

        annotation = project.index.annotation_for_asset(asset.id)
        if not annotation:
            continue
        for obj in annotation.objects:
            if obj.class_id not in class_to_category:
                continue
            bbox = obj.bbox
            buckets[set_name]["annotations"].append(
                {
                    "id": next_annotation_id[set_name],
                    "image_id": image_id,
                    "category_id": class_to_category[obj.class_id],
                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                    "area": bbox.width * bbox.height,
                    "iscrowd": int(obj.attributes.get("iscrowd", 0)),
                }
            )
            next_annotation_id[set_name] += 1
            exported_objects += 1

    total_images = 0
    total_annotations = 0
    per_set: dict[str, int] = {}
    output.mkdir(parents=True, exist_ok=True)
    for set_name, content in buckets.items():
        document = {
            "info": {"description": project.manifest.name, "date_created": utc_now()},
            "images": content["images"],
            "annotations": content["annotations"],
            "categories": categories,
        }
        if split_id:
            annotations_dir = output / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            (annotations_dir / f"instances_{set_name}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        else:
            (output / "annotations.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        total_images += len(content["images"])
        total_annotations += len(content["annotations"])
        per_set[set_name] = len(content["images"])

    result: dict[str, Any] = {"images": total_images, "annotations": total_annotations, "objects": exported_objects}
    if split_id:
        result["sets"] = per_set
        result["skipped"] = skipped
    return result


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormatError(f"COCO annotation file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(document, dict):
        raise FormatError(f"COCO annotation file must contain a JSON object at the top level: {path}")
    return document
