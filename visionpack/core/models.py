from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class ClassDef:
    id: str
    name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassDef:
        return cls(id=str(data["id"]), name=str(data.get("name", data["id"])))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BBox:
    x: float
    y: float
    width: float
    height: float
    coordinate_system: str = "xywh_absolute"

    @property
    def kind(self) -> str:
        return "bbox"

    def bounding_box(self) -> BBox:
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BBox:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            coordinate_system=str(data.get("coordinate_system", "xywh_absolute")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "bbox",
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "coordinate_system": self.coordinate_system,
        }


@dataclass(slots=True)
class Polygon:
    """One or more rings of absolute (x, y) vertices (COCO ``segmentation``).

    Multiple rings represent a multipart instance (e.g. an object split by
    occlusion), each ring a flat ``[x1, y1, x2, y2, ...]`` list.
    """

    rings: list[list[float]]

    @property
    def kind(self) -> str:
        return "polygon"

    def bounding_box(self) -> BBox:
        xs = [point for ring in self.rings for point in ring[0::2]]
        ys = [point for ring in self.rings for point in ring[1::2]]
        if not xs or not ys:
            return BBox(0.0, 0.0, 0.0, 0.0)
        min_x, min_y = min(xs), min(ys)
        return BBox(min_x, min_y, max(xs) - min_x, max(ys) - min_y)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Polygon:
        return cls(rings=[[float(value) for value in ring] for ring in data["rings"]])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "polygon", "rings": self.rings}


@dataclass(slots=True)
class Keypoints:
    """COCO-style keypoints: a flat ``[x1, y1, v1, x2, y2, v2, ...]`` list where
    ``v`` is visibility (0 not labeled, 1 labeled-but-hidden, 2 visible)."""

    points: list[float]
    names: list[str] | None = None

    @property
    def kind(self) -> str:
        return "keypoints"

    def bounding_box(self) -> BBox:
        xs = [self.points[i] for i in range(0, len(self.points), 3) if self.points[i + 2] > 0]
        ys = [self.points[i + 1] for i in range(0, len(self.points), 3) if self.points[i + 2] > 0]
        if not xs or not ys:
            return BBox(0.0, 0.0, 0.0, 0.0)
        min_x, min_y = min(xs), min(ys)
        return BBox(min_x, min_y, max(xs) - min_x, max(ys) - min_y)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keypoints:
        return cls(points=[float(value) for value in data["points"]], names=data.get("names"))

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {"kind": "keypoints", "points": self.points}
        if self.names is not None:
            document["names"] = self.names
        return document


Geometry = BBox | Polygon | Keypoints

_GEOMETRY_TYPES: dict[str, Any] = {"bbox": BBox, "polygon": Polygon, "keypoints": Keypoints}


def parse_geometry(data: dict[str, Any]) -> Geometry:
    kind = str(data.get("kind", "bbox"))
    geometry_type = _GEOMETRY_TYPES.get(kind)
    if geometry_type is None:
        raise ValueError(f"Unknown geometry kind: {kind!r}")
    return geometry_type.from_dict(data)


@dataclass(slots=True)
class ObjectAnnotation:
    """A labeled element of an image.

    ``geometry`` is ``None`` for whole-image labels (classification); a ``BBox``
    for detection; a ``Polygon`` for instance segmentation; ``Keypoints`` for
    pose. The ``bbox`` property always yields an enclosing box (derived for
    polygons/keypoints), so detection-oriented code keeps working across tasks.
    """

    class_id: str
    geometry: Geometry | None = None
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> BBox | None:
        return self.geometry.bounding_box() if self.geometry is not None else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectAnnotation:
        geometry: Geometry | None = None
        if data.get("geometry") is not None:
            geometry = parse_geometry(data["geometry"])
        elif data.get("bbox") is not None:  # legacy schema: bare bbox
            geometry = BBox.from_dict(data["bbox"])
        return cls(
            class_id=str(data["class_id"]),
            geometry=geometry,
            confidence=data.get("confidence"),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "geometry": self.geometry.to_dict() if self.geometry is not None else None,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass(slots=True)
class Asset:
    id: str
    sha256: str
    media_type: Literal["image"]
    path: str
    original_path: str
    width: int
    height: int
    channels: int | None
    format: str
    size_bytes: int
    phash: str | None = None
    source: str | None = None
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Asset:
        return cls(
            id=str(data["id"]),
            sha256=str(data["sha256"]),
            media_type="image",
            path=str(data["path"]),
            original_path=str(data["original_path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            channels=data.get("channels"),
            format=str(data["format"]),
            size_bytes=int(data["size_bytes"]),
            phash=data.get("phash"),
            source=data.get("source"),
            created_at=str(data.get("created_at", utc_now())),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_path(self, root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else root / path


@dataclass(slots=True)
class Annotation:
    id: str
    asset_id: str
    task: str
    format: str
    objects: list[ObjectAnnotation]
    source: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        return cls(
            id=str(data["id"]),
            asset_id=str(data["asset_id"]),
            task=str(data.get("task", "detection")),
            format=str(data.get("format", "internal")),
            objects=[ObjectAnnotation.from_dict(obj) for obj in data.get("objects", [])],
            source=dict(data.get("source", {})),
            created_at=str(data.get("created_at", utc_now())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "task": self.task,
            "format": self.format,
            "objects": [obj.to_dict() for obj in self.objects],
            "source": self.source,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Split:
    id: str
    strategy: str
    sets: dict[str, list[str]]
    locked: bool = False
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Split:
        return cls(
            id=str(data["id"]),
            strategy=str(data.get("strategy", "manual")),
            sets={key: [str(v) for v in values] for key, values in data.get("sets", {}).items()},
            locked=bool(data.get("locked", False)),
            created_at=str(data.get("created_at", utc_now())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
