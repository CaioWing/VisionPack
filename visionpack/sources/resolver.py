from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from visionpack.core.errors import VisionPackError


@dataclass(slots=True)
class FileRef:
    """A file discovered under a source location.

    ``relkey`` is the path relative to the listed root with the extension
    stripped (posix-normalized); it is what the ``relpath`` join matches on.
    ``stem`` is just the final component without extension, for ``stem`` joins.
    ``uri`` round-trips back to :meth:`Resolver.read_bytes`.
    """

    uri: str
    relkey: str
    stem: str
    suffix: str


class Resolver(ABC):
    """Reads bytes and lists files behind a single URI scheme.

    Keeping every backend behind this interface means the importer never knows
    whether bytes come from a local disk, an S3 bucket or a git checkout — new
    schemes (fsspec-backed) plug in without touching the import pipeline.
    """

    @abstractmethod
    def exists(self, uri: str) -> bool: ...

    @abstractmethod
    def list_files(self, uri: str, suffixes: set[str] | None = None) -> list[FileRef]: ...

    @abstractmethod
    def read_bytes(self, uri: str) -> bytes: ...

    @abstractmethod
    def local_path(self, uri: str) -> Path | None:
        """Local filesystem path for ``uri`` if one exists, else ``None``.

        Lets ingest hardlink/copy directly and lets format importers that expect
        a path (e.g. COCO) run unchanged on local sources.
        """


class LocalResolver(Resolver):
    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()

    def list_files(self, uri: str, suffixes: set[str] | None = None) -> list[FileRef]:
        root = self._path(uri)
        if not root.exists():
            raise VisionPackError(f"Source location does not exist: {uri}")
        if root.is_file():
            files = [root]
            base = root.parent
        else:
            files = [path for path in root.rglob("*") if path.is_file()]
            base = root
        refs: list[FileRef] = []
        for path in sorted(files):
            suffix = path.suffix.lower()
            if suffixes is not None and suffix not in suffixes:
                continue
            rel = path.relative_to(base)
            refs.append(
                FileRef(
                    uri=str(path),
                    relkey=rel.with_suffix("").as_posix(),
                    stem=path.stem,
                    suffix=suffix,
                )
            )
        return refs

    def read_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def local_path(self, uri: str) -> Path | None:
        return self._path(uri)

    @staticmethod
    def _path(uri: str) -> Path:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            return Path(url2pathname(unquote(parsed.path)))
        return Path(uri)


def scheme_of(uri: str) -> str:
    """The URI scheme, or "" for a local path.

    Guards against Windows drive letters (``C:\\...``) parsing as a one-letter
    scheme.
    """
    parsed = urlparse(uri)
    if len(parsed.scheme) <= 1:  # "" or a drive letter like "c"
        return ""
    return parsed.scheme.lower()


# Schemes we intend to support via fsspec extras in Phase 2. Listed so the error
# message can point users at the right install instead of a generic failure.
_REMOTE_EXTRAS = {"s3": "s3", "gs": "gcs", "gcs": "gcs", "az": "azure", "abfs": "azure", "git": "git"}


def get_resolver(uri: str) -> Resolver:
    scheme = scheme_of(uri)
    if scheme in ("", "file"):
        return LocalResolver()
    extra = _REMOTE_EXTRAS.get(scheme)
    if extra:
        raise VisionPackError(
            f"Source scheme {scheme!r} is not available yet (planned via fsspec). "
            f"Once enabled, install it with: pip install 'visionpack[{extra}]'."
        )
    raise VisionPackError(f"Unsupported source scheme {scheme!r} in URI: {uri}")
