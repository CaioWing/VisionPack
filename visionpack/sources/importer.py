from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from visionpack.core.errors import FormatError, VisionPackError
from visionpack.core.models import Annotation, Asset, ObjectAnnotation, utc_now
from visionpack.core.project import Project
from visionpack.formats.base import IngestFailure
from visionpack.formats.yolo import parse_yolo_label_text
from visionpack.media import IMAGE_EXTENSIONS, image_info_from_bytes
from visionpack.perceptual import dhash_bytes
from visionpack.sources.join import join_refs
from visionpack.sources.resolver import FileRef, Resolver, get_resolver, scheme_of
from visionpack.sources.schema import Location, Source
from visionpack.storage.hash import sha256_bytes

_LABEL_SUFFIXES = {".txt"}
_CLASS_FILES = ("classes.txt", "obj.names")


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

    def __init__(self, project: Project, source: Source) -> None:
        self.project = project
        # Manifest paths are project-relative; resolve them against the project
        # root (not the cwd) so `vp sync` works from any directory.
        self.source = _rebase_source(source, project.root)

    # -- public ---------------------------------------------------------------

    def plan(self) -> SourcePlan:
        if self.source.format == "imagefolder":
            return self._plan_imagefolder()
        images_loc, labels_loc = self._locations()
        image_res = get_resolver(images_loc.resolved_uri())
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

    def run(self) -> SourceSyncSummary:
        if self.source.format == "coco":
            return self._run_coco()
        if self.source.format == "imagefolder":
            return self._run_imagefolder()
        return self._run_yolo()

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

    def _run_yolo(self) -> SourceSyncSummary:
        images_loc, labels_loc = self._locations()
        image_res = get_resolver(images_loc.resolved_uri())
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
        existing_ids = {asset.id for asset in self.project.index.assets()}

        # Reading bytes, hashing, probing, perceptual-hashing and storing are
        # per-image and I/O-bound, so fan them out across threads (matching
        # YoloImporter). Index mutation stays on this thread; pool.map preserves
        # input order, so the result is deterministic regardless of scheduling.
        def process(pair: tuple[FileRef, FileRef | None]) -> tuple[Asset, Annotation | None, int] | IngestFailure:
            image_ref, label_ref = pair
            try:
                label_text = label_res.read_bytes(label_ref.uri).decode("utf-8") if label_ref else None
                origin = label_ref.uri if label_ref else None
                return self._ingest(image_ref, image_res, label_text, origin, index_to_class_id)
            except (VisionPackError, OSError) as exc:
                return IngestFailure(path=image_ref.uri, error=str(exc))

        with ThreadPoolExecutor() as pool:
            for outcome in pool.map(process, result.pairs):
                if isinstance(outcome, IngestFailure):
                    summary.failures.append(outcome)
                    continue
                asset, annotation, object_count = outcome
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

        self._record(summary, images_loc, labels_loc)
        return summary

    def _ingest(
        self,
        image_ref: FileRef,
        image_res: Resolver,
        label_text: str | None,
        label_origin: str | None,
        index_to_class_id: dict[int, str],
    ) -> tuple[Asset, Annotation | None, int]:
        data = image_res.read_bytes(image_ref.uri)
        digest = sha256_bytes(data)
        width, height, channels, image_format = image_info_from_bytes(data, Path(image_ref.uri))
        asset_id = f"asset_{digest[:16]}"
        local = image_res.local_path(image_ref.uri) or Path(image_ref.uri)
        stored_path = self.project.object_store.store(local, digest, self.source.copy, data=data)
        asset = Asset(
            id=asset_id,
            sha256=digest,
            media_type="image",
            path=stored_path,
            original_path=image_ref.uri,
            width=width,
            height=height,
            channels=channels,
            format=image_format,
            size_bytes=len(data),
            phash=dhash_bytes(data),
            source=self.source.name,
        )

        annotation: Annotation | None = None
        object_count = 0
        if label_text is not None:
            objects = parse_yolo_label_text(label_text, label_origin or image_ref.uri, width, height, index_to_class_id)
            object_count = len(objects)
            annotation = Annotation(
                id=f"ann_{asset_id}",
                asset_id=asset_id,
                task=self.project.manifest.task,
                format="internal",
                objects=objects,
                source={"type": "sync", "format": "yolo", "source": self.source.name, "path": label_origin, "imported_at": utc_now()},
            )
        return asset, annotation, object_count

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
                found = self._class_names_from_file(loc, get_resolver(loc.resolved_uri()))
                if found:
                    names = found
                    break
        if not names:
            names = _infer_class_names(labels, label_res)
        return [self._remap(index, name) for index, name in enumerate(names)]

    def _explicit_class_names(self) -> list[str]:
        if self.source.classes is None:
            return []
        loc = self.source.classes
        res = get_resolver(loc.resolved_uri())
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
        label_res = get_resolver(labels_loc.resolved_uri())
        document = json.loads(label_res.read_bytes(labels_loc.resolved_uri()).decode("utf-8"))
        file_names = {str(record.get("file_name")) for record in document.get("images", [])}
        present = {ref.uri.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for ref in images}
        plan.labels_found = len(document.get("annotations", []))
        plan.matched = len(file_names & present)
        plan.images_without_label = max(0, len(present) - plan.matched)
        plan.labels_without_image = len(file_names - present)
        plan.class_names = [str(cat.get("name", cat["id"])) for cat in document.get("categories", [])]
        return plan

    def _run_coco(self) -> SourceSyncSummary:
        from visionpack.formats.coco import CocoImporter

        images_loc, labels_loc = self._locations()
        if labels_loc is None:
            raise VisionPackError(f"COCO source {self.source.name!r} must declare 'labels' (the instances JSON).")
        image_res = get_resolver(images_loc.resolved_uri())
        label_res = get_resolver(labels_loc.resolved_uri())
        images_path = image_res.local_path(images_loc.resolved_uri())
        labels_path = label_res.local_path(labels_loc.resolved_uri())
        if images_path is None or labels_path is None:
            raise VisionPackError(
                f"COCO source {self.source.name!r} currently supports local images/labels only "
                "(remote COCO arrives with the fsspec backends in Phase 2)."
            )
        before = {asset.id for asset in self.project.index.assets()}
        result = CocoImporter(self.project, labels_path, images_path, copy_mode=self.source.copy).run()
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
        resolver = get_resolver(root.resolved_uri())
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

    def _run_imagefolder(self) -> SourceSyncSummary:
        from visionpack.formats.classification import ImageFolderImporter

        root = self._imagefolder_root()
        resolver = get_resolver(root.resolved_uri())
        root_path = resolver.local_path(root.resolved_uri())
        if root_path is None:
            raise VisionPackError(
                f"ImageFolder source {self.source.name!r} currently supports a local root only "
                "(remote backends arrive with fsspec in Phase 2)."
            )
        before = {asset.id for asset in self.project.index.assets()}
        result = ImageFolderImporter(self.project, root_path, copy_mode=self.source.copy).run()
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
            resolver = get_resolver(candidate.resolved_uri())
            if resolver.exists(candidate.resolved_uri()):
                return candidate
        except VisionPackError:
            pass
        return self.source.root

    def _list_labels(self, labels_loc: Location | None) -> tuple[list[FileRef], Resolver | None]:
        if labels_loc is None:
            return [], None
        resolver = get_resolver(labels_loc.resolved_uri())
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


def _infer_class_names(labels: list[FileRef], label_res: Resolver | None) -> list[str]:
    if label_res is None:
        return []
    max_class = -1
    for ref in labels:
        for line in label_res.read_bytes(ref.uri).decode("utf-8").splitlines():
            stripped = line.strip().lstrip("﻿")
            if not stripped:
                continue
            try:
                max_class = max(max_class, int(float(stripped.split()[0])))
            except (ValueError, IndexError):
                continue
    return [f"class_{index}" for index in range(max_class + 1)]


def sync_sources(project: Project, source_name: str | None = None) -> list[SourceSyncSummary]:
    return [SourceSyncer(project, source).run() for source in _select_sources(project, source_name)]


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
