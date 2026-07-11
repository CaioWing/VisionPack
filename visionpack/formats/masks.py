"""Semantic-segmentation export: one class-index PNG mask per image.

``vp export --format masks`` writes ``images/`` plus ``masks/`` (split-aware,
like the other exporters) where each mask is an 8-bit grayscale PNG whose pixel
value is ``class index + 1`` in manifest order — 0 is reserved for background.
``classes.txt`` records the mapping (``0 __background__`` first) so trainers
and reviewers never have to guess it.

Polygons are rasterized ring by ring; box geometries rasterize as filled
rectangles so detection-labeled data can still produce coarse masks. Objects
are drawn in annotation order, so later objects overwrite earlier ones where
they overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from visionpack.core.errors import VisionPackError
from visionpack.core.models import BBox, Polygon
from visionpack.core.project import Project
from visionpack.formats.materialize import AssetMaterializer
from visionpack.progress import ProgressCallback
from visionpack.split import resolve_export_sets

BACKGROUND_NAME = "__background__"


def export_masks(
    project: Project, output: Path, split_id: str | None = None, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    output = output.resolve()
    classes = project.manifest.classes
    if len(classes) > 255:
        raise VisionPackError(f"masks export writes 8-bit PNGs, which fit at most 255 classes (project has {len(classes)}).")
    pixel_for_class = {item.id: index + 1 for index, item in enumerate(classes)}

    set_for_asset, set_names = resolve_export_sets(project, split_id)
    materializer = AssetMaterializer(output, project.root)

    exported_images = 0
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
        masks_dir = output / "masks" / set_name if split_id else output / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(asset.original_path).suffix.lower() or f".{asset.format}"
        mask_dest = masks_dir / f"{asset.id}.png"
        materializer.place(asset, images_dir / f"{asset.id}{suffix}", {"mask": mask_dest.relative_to(output).as_posix(), "set": set_name})

        mask = Image.new("L", (asset.width, asset.height), 0)
        draw = ImageDraw.Draw(mask)
        if annotation:
            for obj in annotation.objects:
                value = pixel_for_class.get(obj.class_id)
                if value is None:
                    continue
                if isinstance(obj.geometry, Polygon):
                    drew = False
                    for ring in obj.geometry.rings:
                        if len(ring) >= 6:
                            draw.polygon(list(zip(ring[0::2], ring[1::2], strict=True)), fill=value)
                            drew = True
                    exported_objects += int(drew)
                elif isinstance(obj.geometry, BBox):
                    box = obj.geometry
                    draw.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), fill=value)
                    exported_objects += 1
        mask.save(mask_dest, format="PNG")
        exported_images += 1
        per_set[set_name] = per_set.get(set_name, 0) + 1

    output.mkdir(parents=True, exist_ok=True)
    class_lines = [f"0 {BACKGROUND_NAME}"] + [f"{index + 1} {item.name}" for index, item in enumerate(classes)]
    (output / "classes.txt").write_text("\n".join(class_lines) + "\n", encoding="utf-8")
    streamed = materializer.flush()

    result: dict[str, Any] = {"images": exported_images, "masks": exported_images, "objects": exported_objects}
    if streamed:
        result["streamed"] = streamed
    if split_id:
        result["sets"] = per_set
        result["skipped"] = skipped
    return result
