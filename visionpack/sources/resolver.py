from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from visionpack.core.errors import VisionPackError
from visionpack.sources.retry import with_retries


@dataclass(slots=True)
class ObjectStat:
    """Metadata of one object, obtained without reading its body.

    ``etag`` is a *per-key change detector* — same URI + same etag means the
    bytes are unchanged — and is **never** compared across different URIs to
    decide identity (see docs/SPEC-cloud-sync.md). It is ``None`` when the
    backend exposes no usable validator; callers must then treat the object as
    changed (re-read) rather than trust it.
    """

    size: int
    etag: str | None = None


@dataclass(slots=True)
class FileRef:
    """A file discovered under a source location.

    ``relkey`` is the path relative to the listed root with the extension
    stripped (posix-normalized); it is what the ``relpath`` join matches on.
    ``stem`` is just the final component without extension, for ``stem`` joins.
    ``uri`` round-trips back to :meth:`Resolver.read_bytes`. ``stat`` carries the
    size + change-detector captured in the **same listing**, so a sync never
    needs a per-object metadata round-trip (``None`` only for backends that
    can't list it cheaply; callers fall back to :meth:`Resolver.stat`).
    """

    uri: str
    relkey: str
    stem: str
    suffix: str
    stat: ObjectStat | None = None


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
    def stat(self, uri: str) -> ObjectStat:
        """Size and a change-detector for ``uri``, without reading the body.

        Lets ``sync`` decide whether an object is unchanged since last time and
        skip re-reading it entirely.
        """

    @abstractmethod
    def read_bytes(self, uri: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, uri: str, data: bytes) -> None:
        """Write ``data`` to ``uri``, creating parents as the backend requires.

        The upload half of a cross-provider relay: when a server-side copy is
        impossible (source and target live on different providers), sync uploads
        the bytes it already read for hashing — one read, one write, no second
        round-trip.
        """

    @abstractmethod
    def server_copy(self, src_uri: str, dst_uri: str) -> None:
        """Copy ``src_uri`` to ``dst_uri`` without routing the bytes through us.

        Same-provider only: the copy happens inside the backend (S3
        ``CopyObject`` / GCS rewrite / a local filesystem copy), so the client
        never downloads-then-uploads. Used by ``sync`` to land objects in a
        content-addressed target bucket; cross-provider transfers go through
        :meth:`write_bytes` instead (the relay path).
        """

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
            files = [(root, root.stat())]
            base = root.parent
        else:
            files = _scan_tree(root)
            base = root
        refs: list[FileRef] = []
        for path, info in sorted(files, key=lambda item: item[0]):
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
                    stat=ObjectStat(size=info.st_size, etag=str(info.st_mtime_ns)),
                )
            )
        return refs

    def stat(self, uri: str) -> ObjectStat:
        info = self._path(uri).stat()
        # mtime_ns is the local equivalent of an etag: a cheap per-path validator
        # that changes whenever the file is rewritten. Like rsync/git's mtime+size
        # heuristic it can miss a rewrite that preserves both — the sha256 remains
        # the real identity; this only gates whether we bother re-reading.
        return ObjectStat(size=info.st_size, etag=str(info.st_mtime_ns))

    def read_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def write_bytes(self, uri: str, data: bytes) -> None:
        destination = self._path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def server_copy(self, src_uri: str, dst_uri: str) -> None:
        dst = self._path(dst_uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(src_uri), dst)

    def local_path(self, uri: str) -> Path | None:
        return self._path(uri)

    @staticmethod
    def _path(uri: str) -> Path:
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            return Path(url2pathname(unquote(parsed.path)))
        return Path(uri)


def _scan_tree(root: Path) -> list[tuple[Path, os.stat_result]]:
    """Walk ``root`` with ``os.scandir``, capturing each file's stat in the pass.

    ``rglob`` costs two syscalls per entry (``is_file`` + the later ``stat``);
    scandir's dirent caching folds that into one, which matters on large trees
    and network filesystems. Directory symlinks are not followed and file
    symlinks are, matching ``rglob``'s behavior.
    """
    files: list[tuple[Path, os.stat_result]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=True):
                    files.append((Path(entry.path), entry.stat()))
    return files


class FsspecResolver(Resolver):
    """One resolver for every fsspec-backed scheme (s3, gcs, az, ...).

    The provider library (``s3fs``/``gcsfs``/``adlfs``) is an optional extra; it
    is imported lazily by :func:`get_resolver`, never by the core. Filesystem
    instances are cached by fsspec itself, so resolving per call is cheap.

    Every network call goes through :func:`with_retries`: transient provider
    errors (throttling, dropped connections, 5xx) are retried with exponential
    backoff, and exhaustion surfaces as a :class:`VisionPackError` so ingest
    records a per-object failure instead of aborting the sync.
    """

    def __init__(self, scheme: str, storage_options: dict[str, Any] | None = None) -> None:
        self._scheme = scheme
        # Credentials/region from the source declaration (see schema.Location),
        # forwarded to the provider filesystem constructor. Empty for ambient
        # (env/instance-role) auth, which stays the default.
        self._storage_options = storage_options or {}

    def _fs_and_path(self, uri: str):
        import fsspec

        return fsspec.core.url_to_fs(uri, **self._storage_options)

    def _to_uri(self, path: str) -> str:
        # fsspec strips the protocol from listed paths; restore it so the URI
        # round-trips back through url_to_fs in read_bytes/stat.
        return path if "://" in path else f"{self._scheme}://{path}"

    @staticmethod
    def _stat_from_info(info: dict[str, Any]) -> ObjectStat:
        size = int(info.get("size") or info.get("Size") or 0)
        # Providers spell the validator differently; none is required.
        etag = info.get("ETag") or info.get("etag") or info.get("md5Hash") or info.get("crc32c")
        if etag is not None:
            etag = str(etag).strip('"')
        return ObjectStat(size=size, etag=etag)

    def exists(self, uri: str) -> bool:
        fs, path = self._fs_and_path(uri)
        return bool(with_retries(f"exists({uri})", lambda: fs.exists(path)))

    def list_files(self, uri: str, suffixes: set[str] | None = None) -> list[FileRef]:
        fs, root = self._fs_and_path(uri)
        if not with_retries(f"exists({uri})", lambda: fs.exists(root)):
            raise VisionPackError(f"Source location does not exist: {uri}")
        base = root.rstrip("/")
        refs: list[FileRef] = []
        # detail=True returns size + etag in the same paginated LIST, so a sync
        # needs no per-object HEAD (the metadata-only listing the spec promises).
        listing = with_retries(f"list({uri})", lambda: fs.find(root, detail=True))
        for path, info in sorted(listing.items()):
            name = path.rsplit("/", 1)[-1]
            suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if suffixes is not None and suffix not in suffixes:
                continue
            rel = path[len(base) :].lstrip("/") if path.startswith(base) else name
            stem = rel.rsplit("/", 1)[-1]
            stem = stem[: -len(suffix)] if suffix else stem
            relkey = rel[: -len(suffix)] if suffix else rel
            refs.append(
                FileRef(uri=self._to_uri(path), relkey=relkey, stem=stem, suffix=suffix, stat=self._stat_from_info(info))
            )
        return refs

    def stat(self, uri: str) -> ObjectStat:
        fs, path = self._fs_and_path(uri)
        return self._stat_from_info(with_retries(f"stat({uri})", lambda: fs.info(path)))

    def read_bytes(self, uri: str) -> bytes:
        fs, path = self._fs_and_path(uri)
        return with_retries(f"read({uri})", lambda: fs.cat_file(path))

    def write_bytes(self, uri: str, data: bytes) -> None:
        fs, path = self._fs_and_path(uri)
        with_retries(f"write({uri})", lambda: fs.pipe_file(path, data))

    def server_copy(self, src_uri: str, dst_uri: str) -> None:
        src_fs, src_path = self._fs_and_path(src_uri)
        dst_fs, dst_path = self._fs_and_path(dst_uri)
        # A cross-provider copy can't be server-side (it would have to transit
        # the client), so refuse it loudly rather than silently
        # downloading-then-uploading — callers relay via write_bytes instead.
        if type(src_fs) is not type(dst_fs):
            raise VisionPackError(
                f"Server-side copy needs source and target on the same provider "
                f"(got {src_uri!r} -> {dst_uri!r}); use the relay path (write_bytes) for cross-provider transfer."
            )
        with_retries(f"copy({src_uri} -> {dst_uri})", lambda: dst_fs.copy(src_path, dst_path))

    def local_path(self, uri: str) -> Path | None:
        # Remote objects have no local path; ingest works from in-memory bytes.
        return None


def scheme_of(uri: str) -> str:
    """The URI scheme, or "" for a local path.

    Guards against Windows drive letters (``C:\\...``) parsing as a one-letter
    scheme.
    """
    parsed = urlparse(uri)
    if len(parsed.scheme) <= 1:  # "" or a drive letter like "c"
        return ""
    return parsed.scheme.lower()


# Schemes we intend to support via fsspec extras. Listed so the error message can
# point users at the right install instead of a generic failure.
_REMOTE_EXTRAS = {"s3": "s3", "gs": "gcs", "gcs": "gcs", "az": "azure", "abfs": "azure", "git": "git"}

# Schemes fsspec handles in-process with no provider library — `memory` is the
# in-memory filesystem the cloud paths are tested against, so it must resolve
# like any other backend (no extra to install).
_BUILTIN_FSSPEC = {"memory"}


def get_resolver(uri: str, storage_options: dict[str, Any] | None = None) -> Resolver:
    scheme = scheme_of(uri)
    if scheme in ("", "file"):
        return LocalResolver()
    extra = _REMOTE_EXTRAS.get(scheme)
    if extra is None and scheme not in _BUILTIN_FSSPEC:
        raise VisionPackError(f"Unsupported source scheme {scheme!r} in URI: {uri}")
    try:
        import fsspec  # noqa: F401
    except ModuleNotFoundError as exc:
        hint = f"Install it with: pip install 'visionpack[{extra}]'." if extra else "Install fsspec."
        raise VisionPackError(f"Source scheme {scheme!r} needs the optional backend. {hint}") from exc
    return FsspecResolver(scheme, storage_options)
