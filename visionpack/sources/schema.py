from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Location:
    uri: str
    ref: str | None = None
    path: str | None = None
    region: str | None = None
    credentials: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any) -> "Location | None":
        if value is None:
            return None
        if isinstance(value, str):
            return cls(uri=value)
        return cls(
            uri=str(value["uri"]),
            ref=value.get("ref"),
            path=value.get("path"),
            region=value.get("region"),
            credentials=dict(value.get("credentials", {})),
        )

    def child(self, *parts: str) -> "Location":
        """A sub-location under this one (e.g. ``root`` -> ``root/images``)."""
        if self.ref is not None:
            # git: descend within the repo's in-tree path, keep repo uri/ref.
            return Location(uri=self.uri, ref=self.ref, path=_join(self.path, *parts), region=self.region, credentials=dict(self.credentials))
        # plain path / bucket prefix: extend the uri itself.
        return Location(uri=_join(self.uri, *parts), ref=None, path=self.path, region=self.region, credentials=dict(self.credentials))

    def resolved_uri(self) -> str:
        """The concrete URI to hand a resolver (``uri`` joined with ``path``)."""
        return _join(self.uri, self.path) if self.path else self.uri


@dataclass(slots=True)
class Source:
    name: str
    format: str
    images: Location | None
    labels: Location | None
    classes: Location | None
    match: str
    class_map: dict[str, str]
    copy: str
    credentials: dict[str, Any]
    root: Location | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        fmt = str(data.get("format", "yolo"))
        match = data.get("match") or ("embedded" if fmt == "coco" else "relpath")
        return cls(
            name=str(data["name"]),
            format=fmt,
            images=Location.parse(data.get("images")),
            labels=Location.parse(data.get("labels")),
            classes=Location.parse(data.get("classes")),
            match=str(match),
            class_map={str(k): str(v) for k, v in data.get("class_map", {}).items()},
            copy=str(data.get("copy", "ingest")),
            credentials=dict(data.get("credentials", {})),
            root=Location.parse(data.get("root")),
        )


def _join(base: str | None, *parts: str) -> str:
    pieces = [piece for piece in ((base or "").rstrip("/"), *parts) if piece]
    return "/".join(piece.strip("/") if i else piece for i, piece in enumerate(pieces))
