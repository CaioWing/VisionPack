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
    def from_dict(cls, data: dict[str, Any]) -> "ClassDef":
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            coordinate_system=str(data.get("coordinate_system", "xywh_absolute")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObjectAnnotation:
    class_id: str
    bbox: BBox
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectAnnotation":
        return cls(
            class_id=str(data["class_id"]),
            bbox=BBox.from_dict(data["bbox"]),
            confidence=data.get("confidence"),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "bbox": self.bbox.to_dict(),
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
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Asset":
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
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
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
    def from_dict(cls, data: dict[str, Any]) -> "Split":
        return cls(
            id=str(data["id"]),
            strategy=str(data.get("strategy", "manual")),
            sets={key: [str(v) for v in values] for key, values in data.get("sets", {}).items()},
            locked=bool(data.get("locked", False)),
            created_at=str(data.get("created_at", utc_now())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
