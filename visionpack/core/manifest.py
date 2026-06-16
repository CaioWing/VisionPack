from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from visionpack.core.errors import ManifestError
from visionpack.core.models import ClassDef


class _ClassDefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None


class _LocationModel(BaseModel):
    """A place to read bytes from: a local path, bucket prefix, or git subpath.

    Plain strings (``./imgs``, ``s3://bucket/prefix``) are accepted directly; the
    object form adds reproducibility/credentials (``ref`` for git, ``region``,
    ``credentials`` pointing at a profile — never raw secrets).
    """

    model_config = ConfigDict(extra="forbid")

    uri: str
    ref: str | None = None
    path: str | None = None
    region: str | None = None
    credentials: dict[str, Any] = Field(default_factory=dict)


_LocationField = str | _LocationModel


class _SourceModel(BaseModel):
    """One contribution to the dataset: where images live, where labels live, how
    to pair them, and in what format."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    format: str = "yolo"
    root: _LocationField | None = None
    images: _LocationField | None = None
    labels: _LocationField | None = None
    classes: _LocationField | None = None
    match: str | None = None
    class_map: dict[str, str] = Field(default_factory=dict)
    # `copy` shadows BaseModel.copy, so store it as copy_mode and expose the
    # friendly `copy:` key in YAML via the alias.
    copy_mode: str = Field(default="ingest", alias="copy")
    credentials: dict[str, Any] = Field(default_factory=dict)


class _ManifestModel(BaseModel):
    """Validation schema for ``visionpack.yaml``.

    Kept separate from the :class:`Manifest` dataclass so the rest of the
    codebase keeps using the dataclass, while parsing gets pydantic's precise,
    field-level error messages. ``extra="forbid"`` turns typos in top-level keys
    (e.g. ``validaton:``) into actionable errors instead of silently-ignored
    config.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = 1
    task: str = "detection"
    classes: list[_ClassDefModel] = Field(default_factory=list)
    sources: list[_SourceModel] = Field(default_factory=list)
    # Where `copy`-mode sync lands objects, content-addressed and server-side
    # (e.g. s3://my-bucket/datasets/foo). Omit to keep objects in the local CAS.
    target: _LocationField | None = None
    storage: dict[str, Any] = Field(default_factory=lambda: {"mode": "content-addressed", "hash": "sha256"})
    validation: dict[str, Any] = Field(default_factory=dict)
    splits: dict[str, Any] = Field(default_factory=dict)
    exports: dict[str, Any] = Field(default_factory=dict)
    pack_profiles: dict[str, Any] = Field(default_factory=dict)
    annotation: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class Manifest:
    name: str
    version: int = 1
    task: str = "detection"
    classes: list[ClassDef] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    target: str | dict[str, Any] | None = None
    storage: dict[str, Any] = field(default_factory=lambda: {"mode": "content-addressed", "hash": "sha256"})
    validation: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    pack_profiles: dict[str, Any] = field(default_factory=dict)
    annotation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, name: str, task: str = "detection") -> Manifest:
        return cls(
            name=name,
            task=task,
            validation={
                "require_annotations": False,
                "allow_empty_images": True,
                "bbox": {"min_area_px": 4, "allow_out_of_bounds": False},
                "splits": {"prevent_leakage": True},
                "duplicates": {"exact": "warn", "perceptual": "warn", "perceptual_threshold": 5},
            },
            splits={"default": {"strategy": "random", "train": 0.8, "val": 0.1, "test": 0.1}},
            exports={
                "yolo": {"image_format": "original", "normalized_coordinates": True},
                "coco": {"include_empty_images": True},
            },
            pack_profiles={
                "archive": {"format": "tar.zst", "compression_level": 10, "include_metadata": True, "include_assets": True},
                "training": {"format": "webdataset", "shard_size": 1024, "compression": "none"},
            },
            annotation={"preferred_tool": "cvat", "review_required": True},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        # Drop explicit nulls so that, like the previous hand-rolled parser, an
        # empty section (e.g. ``classes:`` with no value) falls back to its
        # default instead of failing type validation.
        cleaned = {key: value for key, value in data.items() if value is not None}
        try:
            model = _ManifestModel.model_validate(cleaned)
        except ValidationError as exc:
            raise ManifestError(_format_validation_error(exc)) from exc
        return cls(
            name=model.name,
            version=model.version,
            task=model.task,
            classes=[ClassDef(id=item.id, name=item.name or item.id) for item in model.classes],
            sources=[source.model_dump(by_alias=True, exclude_none=True) for source in model.sources],
            target=(
                None
                if model.target is None
                else model.target
                if isinstance(model.target, str)
                else model.target.model_dump(exclude_none=True)
            ),
            storage=model.storage,
            validation=model.validation,
            splits=model.splits,
            exports=model.exports,
            pack_profiles=model.pack_profiles,
            annotation=model.annotation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "classes": [item.to_dict() for item in self.classes],
            "sources": self.sources,
            "target": self.target,
            "storage": self.storage,
            "validation": self.validation,
            "splits": self.splits,
            "exports": self.exports,
            "pack_profiles": self.pack_profiles,
            "annotation": self.annotation,
        }

    @property
    def class_ids(self) -> set[str]:
        return {item.id for item in self.classes}

    def merge_classes(self, class_names: list[str]) -> int:
        """Add any classes not already present (matched by name), returning the count added.

        Used by importers so datasets from different sources merge into one class
        set by *name* rather than by positional index. Slug collisions (e.g.
        ``Dog`` vs ``dog``) are disambiguated with a numeric suffix.
        """
        existing_names = {item.name for item in self.classes}
        existing_ids = {item.id for item in self.classes}
        added = 0
        for name in class_names:
            if name in existing_names:
                continue
            base_id = class_id_from_name(name)
            class_id = base_id
            suffix = 2
            while class_id in existing_ids:
                class_id = f"{base_id}_{suffix}"
                suffix += 1
            self.classes.append(ClassDef(id=class_id, name=name))
            existing_names.add(name)
            existing_ids.add(class_id)
            added += 1
        return added

    def class_id_for_name(self, name: str) -> str | None:
        for item in self.classes:
            if item.name == name:
                return item.id
        return None


def read_manifest(path: Path) -> Manifest:
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = _loads_mapping(text)
    return Manifest.from_dict(data)


# Section order + a one-line comment for each, so the generated manifest reads as
# documentation rather than a flat dump. name/version/task are rendered together
# as the header; empty sections are skipped.
_SECTION_COMMENTS: list[tuple[str, str]] = [
    ("classes", "Classes. Merged by name across sources; ids are slugs."),
    ("sources", "Sources: where images & labels come from. `vp sync` reconciles these."),
    ("target", "Content-addressed sink for `copy`-mode sync (e.g. an S3 bucket)."),
    ("splits", "Train / val / test splits (vp split create)."),
    ("validation", "Validation policy (vp validate)."),
    ("exports", "Export options (vp export)."),
    ("pack_profiles", "Pack profiles (vp pack --profile <name>)."),
    ("storage", "Asset storage."),
    ("annotation", "Annotation workflow."),
]


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.write_text(_render_manifest(manifest), encoding="utf-8")


def _render_manifest(manifest: Manifest) -> str:
    data = manifest.to_dict()
    header = [
        "# VisionPack dataset manifest. See ARCHITECTURE.md for the full schema.",
        f"name: {_format_scalar(data['name'])}",
        f"version: {_format_scalar(data['version'])}",
        f"task: {_format_scalar(data['task'])}",
    ]
    blocks = ["\n".join(header)]
    for key, comment in _SECTION_COMMENTS:
        value = data.get(key)
        if not value:
            continue
        blocks.append(f"# {comment}\n{_dump_yaml({key: value}).rstrip()}")
    return "\n\n".join(blocks) + "\n"


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["visionpack.yaml is invalid:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def class_id_from_name(name: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "class"


def _loads_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ManifestError("visionpack.yaml must contain a mapping at the top level")
        return loaded
    except ModuleNotFoundError:
        return _parse_limited_yaml(text)


def _parse_limited_yaml(text: str) -> dict[str, Any]:
    """Parse the YAML subset generated by VisionPack when PyYAML is absent."""
    result: dict[str, Any] = {}
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    current_key: str | None = None
    current_list: list[dict[str, Any]] | None = None
    for line in lines:
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            if value == "":
                if current_key == "classes":
                    current_list = []
                    result[current_key] = current_list
                else:
                    result[current_key] = {}
                    current_list = None
            else:
                result[current_key] = _scalar(value)
                current_list = None
        elif current_key == "classes" and current_list is not None:
            stripped = line.strip()
            if stripped.startswith("- "):
                key, value = stripped[2:].split(":", 1)
                current_list.append({key.strip(): _scalar(value.strip())})
            elif current_list and ":" in stripped:
                key, value = stripped.split(":", 1)
                current_list[-1][key.strip()] = _scalar(value.strip())
    return result


def _scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value)
    return value.strip('"')


def _dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(value, indent + 2).rstrip())
            else:
                lines.append(f"{pad}{key}: {{}}")
        elif isinstance(value, list):
            if value:
                lines.append(f"{pad}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        first = True
                        for child_key, child_value in item.items():
                            prefix = "- " if first else "  "
                            lines.append(f"{pad}  {prefix}{child_key}: {_format_scalar(child_value)}")
                            first = False
                    else:
                        lines.append(f"{pad}  - {_format_scalar(item)}")
            else:
                lines.append(f"{pad}{key}: []")
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}"):
        return json.dumps(text)
    return text
