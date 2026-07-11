from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionpack.core.errors import VisionPackError
from visionpack.core.models import Annotation, Asset, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import IngestFailure
from visionpack.formats.yolo import parse_yolo_label_text
from visionpack.media import IMAGE_EXTENSIONS, image_info_from_bytes
from visionpack.perceptual import dhash_bytes
from visionpack.progress import ProgressCallback
from visionpack.sources.join import join_refs
from visionpack.sources.resolver import FileRef, Resolver, get_resolver, scheme_of
from visionpack.sources.schema import Location, Source
from visionpack.sources.target import CloudTarget
from visionpack.storage.hash import sha256_bytes

_LABEL_SUFFIXES = {".txt"}
_CLASS_FILES = ("classes.txt", "obj.names")
# Fields persisted in the blob cache and replayed on an unchanged re-sync.
_PROBE_KEYS = ("sha256", "width", "height", "channels", "format", "phash")
# (uri, etag, size, probe) to hand to index.put_blob_probe — produced on a miss.
_CacheWrite = tuple[str, "str | None", int, dict]


@dataclass(slots=True)
class SourcePlan:
    """What a `vp sync --dry-run` would do for one source. No writes."""

    name: str
    format: str
    images_uri: str
    labels_uri: str | None
    images_found: int = 0
    labels_found: int = 0
    matched: int = 0
    images_without_label: int = 0
    labels_without_image: int = 0
    class_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceSyncSummary:
    name: str
    assets_added: int = 0
    assets_existing: int = 0
    annotations: int = 0
    objects: int = 0
    classes_added: int = 0
    images_without_label: int = 0
    labels_without_image: int = 0
    failures: list[IngestFailure] = field(default_factory=list)


class SourceSyncer:
    """Reconciles the index with one declared source.

    The YAML is the source of truth: ``plan`` reports what would be ingested and
    ``run`` ingests it. Re-running is idempotent — assets are content-addressed,
    so images already in the store are recognized and not re-added.
    """

    def __init__(self, project: Project, source: Source, max_workers: int | None = None) -> None:
        self.project = project
        # Manifest paths are project-relative; resolve them against the project
        # root (not the cwd) so `vp sync` works from any directory.
        self.source = _rebase_source(source, project.root)
        self.max_workers = max_workers
        # The cloud sink for `copy` mode, built only when one is actually needed
        # (copy + a declared target); otherwise objects stay in the local CAS.
        self._target = self._build_target()
        # Label texts read once during class inference, replayed at ingest so a
        # remote label file never costs two GETs.
        self._label_texts: dict[str, str] = {}

    # -- resolvers ------------------------------------------------------------

    def _resolver(self, loc: Location) -> Resolver:
        """A resolver for ``loc``, wired with this source's credentials/region.

        All resolver access goes through here so cloud auth declared in the YAML
        actually reaches the provider filesystem (the schema promises it).
        """
        return get_resolver(loc.resolved_uri(), _resolver_options(loc, self.source.credentials))

    def _build_target(self) -> CloudTarget | None:
        # A target only matters for `copy`; `reference`/`ingest` never write to it.
        if self.source.copy != "copy" or not self.project.manifest.target:
            return None
        loc = _rebase_location(Location.parse(self.project.manifest.target), self.project.root)
        assert loc is not None  # parse of a non-empty value is never None
        target_uri = loc.resolved_uri()
        # Same provider -> server-side copy (bytes never transit the client).
        # Different providers (local -> s3, s3 -> gcs, ...) -> relay: upload the
        # bytes the sync already read for hashing; still a single read.
        server_side = self._source_scheme() == scheme_of(target_uri)
        return CloudTarget(
            base_uri=target_uri,
            resolver=get_resolver(target_uri, _resolver_options(loc)),
            server_side=server_side,
        )

    def _source_scheme(self) -> str:
        loc = self.source.images or self.source.root
        return scheme_of(loc.resolved_uri()) if loc is not None else ""

    def _pool_size(self) -> int | None:
        """Worker count for the ingest pool (``--jobs`` wins).

        Object-store throughput is latency-bound, not CPU-bound, so the
        CPU-derived executor default undersizes remote syncs on small machines;
        remote sources get a floor of 16 concurrent transfers. Local sources
        keep the executor default (disk parallelism saturates early).
        """
        if self.max_workers is not None:
            return self.max_workers
        if self._source_scheme() not in ("", "file"):
            cpus = os.cpu_count() or 1
            return max(16, min(32, cpus + 4))
        return None

    # -- public ---------------------------------------------------------------

    def plan(self) -> SourcePlan:
        if self.source.format == "imagefolder":
            return self._plan_imagefolder()
        images_loc, labels_loc = self._locations()
        image_res = self._resolver(images_loc)
        images = image_res.list_files(images_loc.resolved_uri(), IMAGE_EXTENSIONS)
        plan = SourcePlan(
            name=self.source.name,
            format=self.source.format,
            images_uri=images_loc.resolved_uri(),
            labels_uri=labels_loc.resolved_uri() if labels_loc else None,
            images_found=len(images),
        )
        if self.source.format == "coco":
            return self._plan_coco(plan, images, image_res, labels_loc)
        return self._plan_yolo(plan, images, labels_loc)

    def run(self, progress: ProgressCallback | None = None) -> SourceSyncSummary:
        if self.source.format == "coco":
            return self._run_coco(progress)
        if self.source.format == "imagefolder":
            return self._run_imagefolder(progress)
        return self._run_yolo(progress)

    # -- YOLO -----------------------------------------------------------------

    def _plan_yolo(self, plan: SourcePlan, images: list[FileRef], labels_loc: Location | None) -> SourcePlan:
        images_loc, _ = self._locations()
        labels, label_res = self._list_labels(labels_loc)
        plan.labels_found = len(labels)
        plan.class_names = self._yolo_class_names(labels, labels_loc, label_res, images_loc)
        result = join_refs(images, labels, self.source.match)
        plan.matched = sum(1 for _, label in result.pairs if label is not None)
        plan.images_without_label = result.images_without_label
        plan.labels_without_image = len(result.labels_without_image)
        return plan

    def _run_yolo(self, progress: ProgressCallback | None = None) -> SourceSyncSummary:
        images_loc, labels_loc = self._locations()
        image_res = self._resolver(images_loc)
        images = image_res.list_files(images_loc.resolved_uri(), IMAGE_EXTENSIONS)
        labels, label_res = self._list_labels(labels_loc)

        class_names = self._yolo_class_names(labels, labels_loc, label_res, images_loc)
        classes_added = self.project.manifest.merge_classes(class_names)
        name_to_id = {item.name: item.id for item in self.project.manifest.classes}
        index_to_class_id = {index: name_to_id[name] for index, name in enumerate(class_names)}

        result = join_refs(images, labels, self.source.match)
        summary = SourceSyncSummary(
            name=self.source.name,
            classes_added=classes_added,
            images_without_label=result.images_without_label,
            labels_without_image=len(result.labels_without_image),
        )
        # Load every cached probe for this run's images in one query, so the
        # per-image lookup in the threads below never opens a connection.
        self.project.index.prime_blob_cache([image.uri for image, _ in result.pairs])

        def process(
            pair: tuple[FileRef, FileRef | None],
        ) -> tuple[Asset, Annotation | None, int, _CacheWrite | None] | IngestFailure:
            image_ref, label_ref = pair
            try:
                label_text = self._label_text(label_ref, label_res) if label_ref else None
                origin = label_ref.uri if label_ref else None
                return self._ingest(image_ref, image_res, label_text, origin, index_to_class_id)
            except (VisionPackError, OSError) as exc:
                return IngestFailure(path=image_ref.uri, error=str(exc))

        self._drain_pool(process, result.pairs, summary, progress)
        self._record(summary, images_loc, labels_loc)
        return summary

    def _label_text(self, label_ref: FileRef, label_res: Resolver | None) -> str:
        # Class inference may already have fetched this label; never GET twice.
        cached = self._label_texts.pop(label_ref.uri, None)
        if cached is not None:
            return cached
        assert label_res is not None  # a label ref implies a label resolver
        return label_res.read_bytes(label_ref.uri).decode("utf-8")

    def _drain_pool(self, process, items, summary: SourceSyncSummary, progress: ProgressCallback | None) -> None:
        """Fan ``process`` out over ``items`` and fold outcomes into ``summary``.

        Reading bytes, hashing, probing, perceptual-hashing and storing are
        per-item and I/O-bound, so they run across threads. Index mutation
        stays on this thread; ``pool.map`` preserves input order, so the result
        is deterministic regardless of scheduling.
        """
        existing_ids = {asset.id for asset in self.project.index.assets()}
        total = len(items)
        with ThreadPoolExecutor(max_workers=self._pool_size()) as pool:
            for done, outcome in enumerate(pool.map(process, items), 1):
                if isinstance(outcome, IngestFailure):
                    summary.failures.append(outcome)
                else:
                    asset, annotation, object_count, cache_write = outcome
                    # Cache writes mutate the index, so they stay on this thread.
                    if cache_write is not None:
                        self.project.index.put_blob_probe(*cache_write)
                    self.project.index.upsert_asset(asset)
                    if asset.id in existing_ids:
                        summary.assets_existing += 1
                    else:
                        summary.assets_added += 1
                        existing_ids.add(asset.id)
                    if annotation is not None:
                        self.project.index.upsert_annotation(annotation)
                        summary.annotations += 1
                        summary.objects += object_count
                if progress is not None:
                    progress(done, total)

    def _ingest(
        self,
        image_ref: FileRef,
        image_res: Resolver,
        label_text: str | None,
        label_origin: str | None,
        index_to_class_id: dict[int, str],
    ) -> tuple[Asset, Annotation | None, int, _CacheWrite | None]:
        asset, cache_write = self._ingest_asset(image_ref, image_res)
        annotation: Annotation | None = None
        object_count = 0
        if label_text is not None:
            objects = parse_yolo_label_text(
                label_text, label_origin or image_ref.uri, asset.width, asset.height, index_to_class_id
            )
            object_count = len(objects)
            annotation = Annotation(
                id=f"ann_{asset.id}",
                asset_id=asset.id,
                task=self.project.manifest.task,
                format="internal",
                objects=objects,
                source={"type": "sync", "format": "yolo", "source": self.source.name, "path": label_origin, "imported_at": utc_now()},
            )
        return asset, annotation, object_count, cache_write

    def _ingest_asset(self, image_ref: FileRef, image_res: Resolver) -> tuple[Asset, _CacheWrite | None]:
        """Read/probe/store one image and build its ``Asset`` — format-agnostic.

        The shared core of every source format: YOLO wraps it with label
        parsing, ImageFolder with a whole-image label, COCO with its record
        conversion. All of them inherit the probe cache and the copy-mode /
        cloud-target routing for free.
        """
        probe, cache_write = self._resolve_blob(image_ref, image_res)
        digest = probe["sha256"]
        asset = Asset(
            id=f"asset_{digest[:16]}",
            sha256=digest,
            media_type="image",
            path=probe["stored_path"],
            original_path=image_ref.uri,
            width=probe["width"],
            height=probe["height"],
            channels=probe["channels"],
            format=probe["format"],
            size_bytes=probe["size_bytes"],
            phash=probe["phash"],
            source=self.source.name,
        )
        return asset, cache_write

    def _resolve_blob(self, image_ref: FileRef, image_res: Resolver) -> tuple[dict, _CacheWrite | None]:
        """Probe an image, skipping the body read when it is provably unchanged.

        Returns the fields needed to build the ``Asset`` plus, on a cache *miss*,
        the tuple to persist so the next sync can skip it. The stat comes from the
        listing (``FileRef.stat``) — no per-object metadata round-trip — falling
        back to a single ``stat`` call only if the backend didn't carry it.
        """
        stat = image_ref.stat or image_res.stat(image_ref.uri)
        cached = self.project.index.cached_blob_probe(image_ref.uri, stat.etag, stat.size)
        if cached is not None:
            stored_path = self._reuse_stored(cached["sha256"], image_ref, image_res)
            if stored_path is not None:
                full = dict(cached, size_bytes=stat.size, stored_path=stored_path)
                return full, None

        data = image_res.read_bytes(image_ref.uri)
        digest = sha256_bytes(data)
        width, height, channels, image_format = image_info_from_bytes(data, Path(image_ref.uri))
        stored_path = self._store_blob(image_ref, image_res, data, digest)
        full = {
            "sha256": digest,
            "width": width,
            "height": height,
            "channels": channels,
            "format": image_format,
            "phash": dhash_bytes(data),
            "size_bytes": len(data),
            "stored_path": stored_path,
        }
        cache_probe = {key: full[key] for key in _PROBE_KEYS}
        return full, (image_ref.uri, stat.etag, stat.size, cache_probe)

    def _store_blob(self, image_ref: FileRef, image_res: Resolver, data: bytes, digest: str) -> str:
        """Materialize the just-read bytes and return the asset ``path``.

        ``reference`` keeps no copy; ``copy`` with a target lands the object in
        the target CAS — server-side when source and target share a provider,
        otherwise by relaying the bytes we already read for the hash (still one
        read total); everything else writes the local CAS.
        """
        if self.source.copy == "reference":
            return self._reference_path(image_ref, image_res)
        if self._target is not None:  # copy + a declared target
            return self._target.ensure_object(image_ref.uri, digest, data=data)
        local = image_res.local_path(image_ref.uri) or Path(image_ref.uri)
        return self.project.object_store.store(local, digest, self.source.copy, data=data)

    def _reference_path(self, image_ref: FileRef, image_res: Resolver) -> str:
        # The index points straight at the source: a local path when we have one,
        # otherwise the remote object URI as-is.
        local = image_res.local_path(image_ref.uri)
        return str(local.resolve()) if local is not None else image_ref.uri

    def _reuse_stored(self, sha256: str, image_ref: FileRef, image_res: Resolver) -> str | None:
        """The asset ``path`` to reuse on a cache hit, or ``None`` if it's gone.

        A cached probe is only usable if the bytes it points at still exist. The
        local CAS is cheaply checked; the cloud target is an immutable CAS we
        already wrote, so a warm cache is trusted without a per-object HEAD
        (keeping re-sync metadata-only).
        """
        if self.source.copy == "reference":
            return self._reference_path(image_ref, image_res)
        if self._target is not None:
            return self._target.object_uri(sha256)
        if self.project.object_store.object_path(sha256).exists():
            return self.project.object_store.relpath(sha256)
        return None

    def _yolo_class_names(
        self,
        labels: list[FileRef],
        labels_loc: Location | None,
        label_res: Resolver | None,
        images_loc: Location | None = None,
    ) -> list[str]:
        names = self._explicit_class_names()
        if not names:
            # classes.txt may sit at the source root, beside the images, or beside
            # the labels (YoloImporter searches the root, so sync must too — else a
            # root-level classes.txt is missed and class_N names get invented).
            for loc in (self.source.root, images_loc, labels_loc):
                if loc is None:
                    continue
                found = self._class_names_from_file(loc, self._resolver(loc))
                if found:
                    names = found
                    break
        if not names:
            names = self._infer_class_names(labels, label_res)
        return [self._remap(index, name) for index, name in enumerate(names)]

    def _infer_class_names(self, labels: list[FileRef], label_res: Resolver | None) -> list[str]:
        """Infer ``class_N`` names from the highest class index used in labels.

        Label bodies are fetched in parallel and cached, so ingest replays them
        instead of issuing a second GET per label (labels are tiny; the cache is
        drained as ingest consumes it).
        """
        if label_res is None or not labels:
            return []
        with ThreadPoolExecutor(max_workers=self._pool_size()) as pool:
            texts = pool.map(lambda ref: label_res.read_bytes(ref.uri).decode("utf-8"), labels)
            self._label_texts = {ref.uri: text for ref, text in zip(labels, texts, strict=True)}
        max_class = -1
        for text in self._label_texts.values():
            for line in text.splitlines():
                stripped = line.strip().lstrip("﻿")
                if not stripped:
                    continue
                try:
                    max_class = max(max_class, int(float(stripped.split()[0])))
                except (ValueError, IndexError):
                    continue
        return [f"class_{index}" for index in range(max_class + 1)]

    def _explicit_class_names(self) -> list[str]:
        if self.source.classes is None:
            return []
        loc = self.source.classes
        res = self._resolver(loc)
        text = res.read_bytes(loc.resolved_uri()).decode("utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _class_names_from_file(self, labels_loc: Location, label_res: Resolver) -> list[str]:
        for filename in _CLASS_FILES:
            candidate = labels_loc.child(filename).resolved_uri()
            if label_res.exists(candidate):
                text = label_res.read_bytes(candidate).decode("utf-8")
                return [line.strip() for line in text.splitlines() if line.strip()]
        return []

    def _remap(self, index: int, name: str) -> str:
        return self.source.class_map.get(name) or self.source.class_map.get(str(index)) or name

    # -- COCO (local sources in Phase 1) --------------------------------------

    def _plan_coco(self, plan: SourcePlan, images: list[FileRef], image_res: Resolver, labels_loc: Location | None) -> SourcePlan:
        import json

        if labels_loc is None:
            raise VisionPackError(f"COCO source {self.source.name!r} must declare 'labels' (the instances JSON).")
        label_res = self._resolver(labels_loc)
        document = json.loads(label_res.read_bytes(labels_loc.resolved_uri()).decode("utf-8"))
        file_names = {str(record.get("file_name")) for record in document.get("images", [])}
        present = {ref.uri.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for ref in images}
        plan.labels_found = len(document.get("annotations", []))
        plan.matched = len(file_names & present)
        plan.images_without_label = max(0, len(present) - plan.matched)
        plan.labels_without_image = len(file_names - present)
        plan.class_names = [str(cat.get("name", cat["id"])) for cat in document.get("categories", [])]
        return plan

    def _run_coco(self, progress: ProgressCallback | None = None) -> SourceSyncSummary:
        from visionpack.formats.coco import CocoImporter

        images_loc, labels_loc = self._locations()
        if labels_loc is None:
            raise VisionPackError(f"COCO source {self.source.name!r} must declare 'labels' (the instances JSON).")
        image_res = self._resolver(images_loc)
        label_res = self._resolver(labels_loc)
        images_path = image_res.local_path(images_loc.resolved_uri())
        labels_path = label_res.local_path(labels_loc.resolved_uri())
        if images_path is None or labels_path is None:
            return self._run_coco_remote(images_loc, labels_loc, image_res, label_res, progress)
        before = {asset.id for asset in self.project.index.assets()}
        result = CocoImporter(self.project, labels_path, images_path, copy_mode=self.source.copy).run(progress)
        added = self._tag_provenance(before)
        return SourceSyncSummary(
            name=self.source.name,
            assets_added=added,
            assets_existing=result.assets - added,
            annotations=result.annotations,
            objects=result.objects,
            classes_added=result.classes_added,
            failures=result.failures,
        )

    def _run_coco_remote(
        self,
        images_loc: Location,
        labels_loc: Location,
        image_res: Resolver,
        label_res: Resolver,
        progress: ProgressCallback | None,
    ) -> SourceSyncSummary:
        """COCO over any resolver: the JSON is one read, images stream through
        the shared blob pipeline (probe cache, copy modes, cloud target)."""
        import json
        from collections import defaultdict

        from visionpack.core.manifest import class_id_from_name
        from visionpack.formats.coco import geometry_from_record

        labels_uri = labels_loc.resolved_uri()
        try:
            document = json.loads(label_res.read_bytes(labels_uri).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise VisionPackError(f"COCO annotation file is not valid JSON: {labels_uri} ({exc})") from exc
        if not isinstance(document, dict):
            raise VisionPackError(f"COCO annotation file must contain a JSON object at the top level: {labels_uri}")

        categories = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in document.get("categories", [])}
        # Merge classes by (remapped) name, exactly like the local importer.
        remapped = {cat_id: self._remap(index, name) for index, (cat_id, name) in enumerate(categories.items())}
        classes_added = self.project.manifest.merge_classes(list(remapped.values()))
        name_to_id = {item.name: item.id for item in self.project.manifest.classes}
        category_to_class_id = {
            cat_id: name_to_id.get(name, class_id_from_name(name)) for cat_id, name in remapped.items()
        }

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for record in document.get("annotations", []):
            annotations_by_image[int(record["image_id"])].append(record)

        refs = image_res.list_files(images_loc.resolved_uri(), IMAGE_EXTENSIONS)
        # file_name is relative to the images root; fall back to the bare
        # basename only when it is unambiguous across the listing.
        by_rel = {f"{ref.relkey}{ref.suffix}": ref for ref in refs}
        by_name: dict[str, FileRef | None] = {}
        for ref in refs:
            name = f"{ref.stem}{ref.suffix}"
            by_name[name] = None if name in by_name else ref

        def resolve_ref(file_name: str) -> FileRef | None:
            rel = file_name.replace("\\", "/").lstrip("./")
            return by_rel.get(rel) or by_name.get(rel.rsplit("/", 1)[-1])

        images = document.get("images", [])
        matched = [resolve_ref(str(record.get("file_name"))) for record in images]
        self.project.index.prime_blob_cache([ref.uri for ref in matched if ref is not None])
        summary = SourceSyncSummary(name=self.source.name, classes_added=classes_added)
        task = self.project.manifest.task

        def process(record: dict) -> tuple[Asset, Annotation | None, int, _CacheWrite | None] | IngestFailure:
            file_name = str(record.get("file_name"))
            try:
                ref = resolve_ref(file_name)
                if ref is None:
                    raise VisionPackError(
                        f"COCO image not found under {images_loc.resolved_uri()}: file_name={file_name!r} "
                        f"(image id={record.get('id')})"
                    )
                asset, cache_write = self._ingest_asset(ref, image_res)
                objects = [
                    ObjectAnnotation(
                        class_id=category_to_class_id.get(int(item["category_id"]), str(item["category_id"])),
                        geometry=geometry_from_record(item, task, file_name),
                        attributes={"iscrowd": int(item["iscrowd"])} if item.get("iscrowd") else {},
                    )
                    for item in annotations_by_image.get(int(record["id"]), [])
                ]
                annotation: Annotation | None = None
                if objects:
                    annotation = Annotation(
                        id=f"ann_{asset.id}",
                        asset_id=asset.id,
                        task=task,
                        format="internal",
                        objects=objects,
                        source={"type": "sync", "format": "coco", "source": self.source.name, "path": labels_uri, "imported_at": utc_now()},
                    )
                return asset, annotation, len(objects), cache_write
            except (VisionPackError, OSError) as exc:
                return IngestFailure(path=file_name, error=str(exc))

        self._drain_pool(process, images, summary, progress)
        self._record(summary, images_loc, labels_loc)
        return summary

    # -- ImageFolder (classification) -----------------------------------------

    def _imagefolder_root(self) -> Location:
        root = self.source.root or self.source.images
        if root is None:
            raise VisionPackError(
                f"ImageFolder source {self.source.name!r} must declare 'root' (the directory of class subfolders)."
            )
        return root

    def _plan_imagefolder(self) -> SourcePlan:
        root = self._imagefolder_root()
        resolver = self._resolver(root)
        images = resolver.list_files(root.resolved_uri(), IMAGE_EXTENSIONS)
        # The class is the first path segment under the root.
        class_names = sorted({ref.relkey.split("/")[0] for ref in images if "/" in ref.relkey})
        return SourcePlan(
            name=self.source.name,
            format="imagefolder",
            images_uri=root.resolved_uri(),
            labels_uri=None,
            images_found=len(images),
            labels_found=len(images),
            matched=len(images),
            class_names=[self._remap(index, name) for index, name in enumerate(class_names)],
        )

    def _run_imagefolder(self, progress: ProgressCallback | None = None) -> SourceSyncSummary:
        from visionpack.formats.classification import ImageFolderImporter

        root = self._imagefolder_root()
        resolver = self._resolver(root)
        root_path = resolver.local_path(root.resolved_uri())
        if root_path is None:
            return self._run_imagefolder_remote(root, resolver, progress)
        before = {asset.id for asset in self.project.index.assets()}
        result = ImageFolderImporter(self.project, root_path, copy_mode=self.source.copy).run(progress)
        added = self._tag_provenance(before)
        return SourceSyncSummary(
            name=self.source.name,
            assets_added=added,
            assets_existing=result.assets - added,
            annotations=result.annotations,
            objects=result.objects,
            classes_added=result.classes_added,
            failures=result.failures,
        )

    def _run_imagefolder_remote(
        self, root: Location, resolver: Resolver, progress: ProgressCallback | None
    ) -> SourceSyncSummary:
        """ImageFolder over any resolver: the class is the first path segment
        under the root, each image gets a whole-image label (no geometry)."""
        refs = resolver.list_files(root.resolved_uri(), IMAGE_EXTENSIONS)
        labeled = [(ref, ref.relkey.split("/")[0]) for ref in refs if "/" in ref.relkey]
        if not labeled:
            raise VisionPackError(
                f"No class subdirectories found under {root.resolved_uri()}. "
                "Expected layout: <root>/<class-name>/<image-files>."
            )
        raw_names = sorted({name for _, name in labeled})
        remapped = {raw: self._remap(index, raw) for index, raw in enumerate(raw_names)}
        classes_added = self.project.manifest.merge_classes(list(remapped.values()))
        name_to_id = {item.name: item.id for item in self.project.manifest.classes}

        self.project.index.prime_blob_cache([ref.uri for ref, _ in labeled])
        summary = SourceSyncSummary(name=self.source.name, classes_added=classes_added)
        task = self.project.manifest.task

        def process(item: tuple[FileRef, str]) -> tuple[Asset, Annotation | None, int, _CacheWrite | None] | IngestFailure:
            ref, raw_name = item
            try:
                asset, cache_write = self._ingest_asset(ref, resolver)
                annotation = Annotation(
                    id=f"ann_{asset.id}",
                    asset_id=asset.id,
                    task=task,
                    format="internal",
                    objects=[ObjectAnnotation(class_id=name_to_id[remapped[raw_name]], geometry=None)],
                    source={"type": "sync", "format": "imagefolder", "source": self.source.name, "path": ref.uri, "imported_at": utc_now()},
                )
                return asset, annotation, 1, cache_write
            except (VisionPackError, OSError) as exc:
                return IngestFailure(path=ref.uri, error=str(exc))

        self._drain_pool(process, labeled, summary, progress)
        self._record(summary, root, None)
        return summary

    # -- shared ---------------------------------------------------------------

    def _tag_provenance(self, before: set[str]) -> int:
        """Stamp newly-added assets with this source's name and persist.

        Importers don't know which declared source drove them, so the sync layer
        records provenance after the fact for the assets this run introduced.
        """
        added = 0
        for asset in self.project.index.assets():
            if asset.id in before:
                continue
            added += 1
            if asset.source != self.source.name:
                asset.source = self.source.name
                self.project.index.upsert_asset(asset)
        if added:
            self.project.index.save()
        return added

    def _locations(self) -> tuple[Location, Location | None]:
        source = self.source
        images = source.images or self._under_root("images")
        if images is None:
            raise VisionPackError(f"Source {source.name!r} must declare 'images' or 'root'.")
        labels = source.labels or self._under_root("labels")
        return images, labels

    def _under_root(self, subdir: str) -> Location | None:
        """Expand the `root` shorthand: use root/<subdir> if it exists, else root."""
        if self.source.root is None:
            return None
        candidate = self.source.root.child(subdir)
        try:
            resolver = self._resolver(candidate)
            if resolver.exists(candidate.resolved_uri()):
                return candidate
        except VisionPackError:
            pass
        return self.source.root

    def _list_labels(self, labels_loc: Location | None) -> tuple[list[FileRef], Resolver | None]:
        if labels_loc is None:
            return [], None
        resolver = self._resolver(labels_loc)
        labels = [
            ref
            for ref in resolver.list_files(labels_loc.resolved_uri(), _LABEL_SUFFIXES)
            if ref.uri.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] not in _CLASS_FILES
        ]
        return labels, resolver

    def _record(self, summary: SourceSyncSummary, images_loc: Location, labels_loc: Location | None) -> None:
        self.project.index.add_import_record(
            {
                "format": self.source.format,
                "source": self.source.name,
                "via": "sync",
                "images": images_loc.resolved_uri(),
                "labels": labels_loc.resolved_uri() if labels_loc else None,
                "copy_mode": self.source.copy,
                "created_at": utc_now(),
                "assets_added": summary.assets_added,
                "annotations": summary.annotations,
                "objects": summary.objects,
            }
        )
        self.project.index.save()
        if summary.classes_added:
            self.project.save_manifest()


def _resolver_options(loc: Location, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """fsspec ``storage_options`` for ``loc``: ``base`` creds overlaid by the
    location's own, with ``region`` mapped to where S3 reads it."""
    options: dict[str, Any] = {**(base or {}), **loc.credentials}
    if loc.region and scheme_of(loc.resolved_uri()) == "s3":
        client_kwargs = dict(options.get("client_kwargs", {}))
        client_kwargs.setdefault("region_name", loc.region)
        options["client_kwargs"] = client_kwargs
    return options


def _rebase_location(loc: Location | None, root: Path) -> Location | None:
    """Make a local, relative location absolute against the project root."""
    if loc is None:
        return None
    if loc.ref is None and scheme_of(loc.uri) in ("", "file"):
        path = Path(loc.uri)
        if not path.is_absolute():
            return Location(
                uri=str((root / path).resolve()),
                ref=loc.ref,
                path=loc.path,
                region=loc.region,
                credentials=loc.credentials,
            )
    return loc


def _rebase_source(source: Source, root: Path) -> Source:
    return Source(
        name=source.name,
        format=source.format,
        images=_rebase_location(source.images, root),
        labels=_rebase_location(source.labels, root),
        classes=_rebase_location(source.classes, root),
        match=source.match,
        class_map=source.class_map,
        copy=source.copy,
        credentials=source.credentials,
        root=_rebase_location(source.root, root),
    )


def sync_sources(
    project: Project,
    source_name: str | None = None,
    progress_factory: Callable[[str], AbstractContextManager[ProgressCallback | None]] | None = None,
    max_workers: int | None = None,
) -> list[SourceSyncSummary]:
    """Sync the declared sources. ``progress_factory`` (e.g. ``cli_progress``)
    yields a fresh progress callback per source so each gets its own bar.
    ``max_workers`` overrides the per-source ingest concurrency (``--jobs``)."""
    summaries: list[SourceSyncSummary] = []
    for source in _select_sources(project, source_name):
        syncer = SourceSyncer(project, source, max_workers=max_workers)
        if progress_factory is None:
            summaries.append(syncer.run())
        else:
            with progress_factory(f"Syncing {source.name}") as callback:
                summaries.append(syncer.run(callback))
    return summaries


def plan_sources(project: Project, source_name: str | None = None) -> list[SourcePlan]:
    return [SourceSyncer(project, source).plan() for source in _select_sources(project, source_name)]


def _select_sources(project: Project, source_name: str | None) -> list[Source]:
    declared = [Source.from_dict(item) for item in project.manifest.sources]
    if not declared:
        raise VisionPackError(
            "No sources declared in visionpack.yaml. Add a 'sources:' block, then run `vp sync`."
        )
    if source_name is None:
        return declared
    selected = [source for source in declared if source.name == source_name]
    if not selected:
        names = ", ".join(source.name for source in declared) or "(none)"
        raise VisionPackError(f"No source named {source_name!r}. Declared sources: {names}.")
    return selected
