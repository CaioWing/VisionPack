"""Load model predictions and resolve them onto project assets.

Predictions are the shared input of ``vp eval`` (benchmarking), ``vp autolabel``
(model-assisted labeling) and ``vp queue`` (active-learning ranking). They reuse
``ObjectAnnotation`` (with ``confidence`` set) rather than a parallel model, so
everything downstream — geometry, class ids, export — keeps working unchanged.

Three input formats are accepted:

- ``vp``: a JSON document ``{"predictions": [{"image": <ref>, "objects":
  [{"class": <name|id|index>, "confidence": 0.9, "bbox": [x, y, w, h],
  "polygon": [[x1, y1, ...], ...]}]}]}`` with absolute pixel coordinates.
- ``coco``: either a COCO *results* list (``[{"image_id"|"file_name",
  "category_id", "bbox", "score", "segmentation"?}]``) or a full COCO document
  with ``images``/``annotations``/``categories``. Category ids follow the
  1-based convention used by ``vp export --format coco``.
- ``yolo``: a directory of ``<image>.txt`` files with normalized
  ``class cx cy w h [conf]`` lines (or ``class x1 y1 x2 y2 ... [conf]``
  segment lines), which is exactly what Ultralytics ``predict`` writes with
  ``save_txt=True, save_conf=True`` when run over a ``vp export`` layout.

Image references resolve by asset id first (exports name files by asset id, so
predictions produced on an exported layout match trivially), then by original
filename / stem. Ambiguous stems and unknown references are reported in
``unmatched`` instead of being silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionpack.core.errors import FormatError
from visionpack.core.models import BBox, ObjectAnnotation, Polygon
from visionpack.core.project import Project

FORMATS = ("auto", "vp", "coco", "yolo")


@dataclass(slots=True)
class PredictionSet:
    """Model predictions keyed by asset id, plus everything that didn't resolve."""

    by_asset: dict[str, list[ObjectAnnotation]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    unknown_classes: list[str] = field(default_factory=list)
    origin: str = ""

    @property
    def objects(self) -> int:
        return sum(len(items) for items in self.by_asset.values())


def load_predictions(project: Project, path: Path, fmt: str = "auto") -> PredictionSet:
    path = path.resolve()
    if not path.exists():
        raise FormatError(f"Predictions not found: {path}")
    if fmt not in FORMATS:
        raise FormatError(f"Unknown predictions format {fmt!r}. Use one of: {', '.join(FORMATS)}.")
    if fmt == "auto":
        fmt = _sniff_format(path)

    refs = _asset_refs(project)
    resolve_class = _class_resolver(project)
    result = PredictionSet(origin=str(path))
    if fmt == "yolo":
        _load_yolo_dir(project, path, refs, resolve_class, result)
    else:
        document = _load_json(path)
        if fmt == "vp":
            _load_vp(document, refs, resolve_class, result)
        else:
            _load_coco(document, refs, resolve_class, result)
    return result


def _sniff_format(path: Path) -> str:
    if path.is_dir():
        return "yolo"
    if path.suffix.lower() != ".json":
        raise FormatError(f"Cannot infer predictions format from {path}; pass --format vp|coco|yolo.")
    document = _load_json(path)
    if isinstance(document, dict) and "predictions" in document:
        return "vp"
    return "coco"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormatError(f"Predictions file is not valid JSON: {path} ({exc})") from exc


def _asset_refs(project: Project) -> dict[str, str | None]:
    """Map image references to asset ids; ambiguous references map to ``None``."""
    refs: dict[str, str | None] = {}

    def put(key: str, asset_id: str) -> None:
        if not key:
            return
        if key in refs and refs[key] != asset_id:
            refs[key] = None
        else:
            refs[key] = asset_id

    for asset in project.index.assets():
        put(asset.id, asset.id)
        original = Path(asset.original_path)
        put(original.name, asset.id)
        put(original.stem, asset.id)
    return refs


def _resolve_ref(refs: dict[str, str | None], ref: str) -> str | None:
    for candidate in (ref, Path(ref).name, Path(ref).stem):
        asset_id = refs.get(candidate)
        if asset_id:
            return asset_id
    return None


def _class_resolver(project: Project):
    classes = project.manifest.classes
    by_name = {item.name: item.id for item in classes}
    ids = {item.id for item in classes}

    def resolve(ref: Any, *, one_based: bool = False) -> str | None:
        if isinstance(ref, str) and (ref in by_name or ref in ids):
            return by_name.get(ref, ref)
        try:
            index = int(ref) - (1 if one_based else 0)
        except (TypeError, ValueError):
            return None
        return classes[index].id if 0 <= index < len(classes) else None

    return resolve


def _record(result: PredictionSet, asset_id: str, obj: ObjectAnnotation) -> None:
    result.by_asset.setdefault(asset_id, []).append(obj)


def _load_vp(document: Any, refs: dict[str, str | None], resolve_class, result: PredictionSet) -> None:
    if not isinstance(document, dict) or not isinstance(document.get("predictions"), list):
        raise FormatError('vp predictions must be a JSON object with a "predictions" list.')
    for item in document["predictions"]:
        ref = str(item.get("image", ""))
        asset_id = _resolve_ref(refs, ref)
        if asset_id is None:
            result.unmatched.append(ref or "(missing image ref)")
            continue
        for obj in item.get("objects", []):
            class_ref = obj.get("class", obj.get("class_id", obj.get("label")))
            class_id = resolve_class(class_ref)
            if class_id is None:
                result.unknown_classes.append(str(class_ref))
                continue
            geometry = None
            if obj.get("polygon") is not None:
                rings = obj["polygon"]
                if rings and not isinstance(rings[0], list):  # accept a single flat ring
                    rings = [rings]
                geometry = Polygon(rings=[[float(v) for v in ring] for ring in rings])
            elif obj.get("bbox") is not None:
                x, y, w, h = (float(v) for v in obj["bbox"][:4])
                geometry = BBox(x=x, y=y, width=w, height=h)
            confidence = float(obj.get("confidence", obj.get("score", 1.0)))
            _record(result, asset_id, ObjectAnnotation(class_id=class_id, geometry=geometry, confidence=confidence))


def _load_coco(document: Any, refs: dict[str, str | None], resolve_class, result: PredictionSet) -> None:
    image_names: dict[int, str] = {}
    category_names: dict[int, str] = {}
    if isinstance(document, dict):
        image_names = {int(img["id"]): str(img["file_name"]) for img in document.get("images", []) if "file_name" in img}
        category_names = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in document.get("categories", [])}
        records = document.get("annotations", [])
    elif isinstance(document, list):
        records = document
    else:
        raise FormatError("COCO predictions must be a results list or a COCO JSON object.")

    for record in records:
        ref = record.get("file_name")
        if ref is None and record.get("image_id") is not None:
            image_id = record["image_id"]
            ref = image_names.get(int(image_id)) if isinstance(image_id, int) and image_names else str(image_id)
        if ref is None:
            result.unmatched.append("(missing image ref)")
            continue
        asset_id = _resolve_ref(refs, str(ref))
        if asset_id is None:
            result.unmatched.append(str(ref))
            continue

        category = record.get("category_id", record.get("category_name"))
        if category_names and isinstance(category, int):
            class_id = resolve_class(category_names.get(category, category))
        else:
            # Bare results lists carry no category table; ids follow the 1-based
            # positional convention `vp export --format coco` writes.
            class_id = resolve_class(category, one_based=isinstance(category, int))
        if class_id is None:
            result.unknown_classes.append(str(category))
            continue

        geometry = None
        segmentation = record.get("segmentation")
        if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
            geometry = Polygon(rings=[[float(v) for v in ring] for ring in segmentation if ring])
        elif record.get("bbox"):
            x, y, w, h = (float(v) for v in record["bbox"][:4])
            geometry = BBox(x=x, y=y, width=w, height=h)
        confidence = float(record.get("score", 1.0))
        _record(result, asset_id, ObjectAnnotation(class_id=class_id, geometry=geometry, confidence=confidence))


def _load_yolo_dir(project: Project, path: Path, refs: dict[str, str | None], resolve_class, result: PredictionSet) -> None:
    if not path.is_dir():
        raise FormatError(f"YOLO predictions must be a directory of .txt files: {path}")
    assets = {asset.id: asset for asset in project.index.assets()}
    for label_path in sorted(path.rglob("*.txt")):
        asset_id = _resolve_ref(refs, label_path.stem)
        if asset_id is None:
            result.unmatched.append(label_path.name)
            continue
        asset = assets[asset_id]
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip().lstrip("\ufeff")
            if not stripped:
                continue
            parts = stripped.split()
            try:
                values = [float(v) for v in parts]
            except ValueError as exc:
                raise FormatError(f"Invalid numeric value in YOLO prediction at {label_path}:{line_number}") from exc
            class_id = resolve_class(int(values[0]))
            if class_id is None:
                result.unknown_classes.append(parts[0])
                continue
            geometry, confidence = _parse_yolo_prediction(values, asset.width, asset.height, label_path, line_number)
            _record(result, asset_id, ObjectAnnotation(class_id=class_id, geometry=geometry, confidence=confidence))


def _parse_yolo_prediction(values: list[float], width: int, height: int, path: Path, line_number: int) -> tuple[BBox | Polygon, float]:
    """Decode one prediction line: bbox (5–6 values) or segment (7+ values).

    An optional trailing confidence disambiguates by parity: bbox lines have 5
    (no conf) or 6 (conf) values; segment lines are ``class + 2n coords`` — an
    odd total — so an even total of 8+ means a trailing confidence.
    """
    count = len(values)
    if count in (5, 6):
        confidence = values[5] if count == 6 else 1.0
        cx, cy, w, h = values[1:5]
        return BBox(x=cx * width - w * width / 2, y=cy * height - h * height / 2, width=w * width, height=h * height), confidence
    if count >= 7:
        coords = values[1:]
        confidence = 1.0
        if count % 2 == 0:  # class + 2n coords + conf
            confidence = coords[-1]
            coords = coords[:-1]
        ring = [coords[i] * (width if i % 2 == 0 else height) for i in range(len(coords))]
        return Polygon(rings=[ring]), confidence
    raise FormatError(f"Invalid YOLO prediction at {path}:{line_number}: expected 5+ values, got {count}")
