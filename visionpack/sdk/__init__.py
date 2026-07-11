"""VisionPack SDK: drive the framework from Python instead of the shell.

The CLI's ``--json`` envelopes are the contract for driving VisionPack from
*other processes*; this SDK is the same contract for Python code — notebooks,
training scripts, labeling services, CI jobs — with none of the subprocess
plumbing. One class, :class:`VisionPackClient`, wraps the whole dataset
lifecycle behind stable, typed methods:

    from visionpack.sdk import VisionPackClient

    ds = VisionPackClient.init("./factory-defects", task="detection")
    ds.import_dir("./raw", format="yolo")
    report = ds.validate()
    ds.create_split(train=0.8, val=0.1, test=0.1, strategy="stratified")
    ds.lock_split()
    ds.snapshot("baseline")
    ds.export("./exports/yolo", format="yolo", split="default")

    # ...train, predict, then close the loop:
    metrics = ds.evaluate("runs/predict/labels", format="yolo")
    ds.autolabel("preds.json", min_confidence=0.6)
    for item in ds.annotation_queue("preds.json")[:20]:
        print(item["path"], item["score"])

Guarantees the SDK adds on top of the internal modules:

- **Concurrency safety**: every mutating method takes the same project lock the
  CLI takes, so an SDK caller and a ``vp`` process can never corrupt each
  other's writes.
- **Stable returns**: methods return plain dicts/dataclasses that mirror the
  ``--json`` CLI output, so a service can switch between shelling out and
  importing the SDK without re-parsing.
- **Snapshot views**: :meth:`VisionPackClient.checkout` returns a read-only
  client pinned to a snapshot, for exporting or evaluating historical versions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from visionpack.audit import AuditReport, AuditThresholds, audit_project
from visionpack.autolabel import apply_predictions
from visionpack.core.errors import VisionPackError
from visionpack.core.lock import project_lock
from visionpack.core.models import Annotation, Asset, ClassDef, Split
from visionpack.core.project import Project
from visionpack.curation import rank_for_annotation
from visionpack.diff import diff_snapshots
from visionpack.drift import drift_between
from visionpack.eval import evaluate as _evaluate
from visionpack.predictions import PredictionSet, load_predictions
from visionpack.snapshot import (
    create_snapshot,
    find_snapshots_by_tag,
    list_snapshots,
    load_snapshot,
    open_snapshot,
    tag_snapshot,
    untag_snapshot,
)
from visionpack.split import create_split as _create_split
from visionpack.split import get_split, lock_split
from visionpack.stats import collect_stats, split_breakdown
from visionpack.validation import ValidationReport, validate_project

__all__ = ["VisionPackClient", "init", "open"]

EXPORT_FORMATS = ("yolo", "coco", "imagefolder", "masks")
IMPORT_FORMATS = ("yolo", "coco", "imagefolder")


class VisionPackClient:
    """A Python handle on one VisionPack dataset (one ``visionpack.yaml``).

    Construct through :meth:`init` (create) or :meth:`open` (existing). The
    client is a thin, stateless facade: every call reads/writes the same
    on-disk project the CLI uses, so the two can be mixed freely.
    """

    def __init__(self, project: Project, *, _readonly: bool = False, _snapshot: str | None = None) -> None:
        self._project = project
        self._readonly = _readonly
        self._snapshot_version = _snapshot

    # -- lifecycle -------------------------------------------------------------

    @classmethod
    def init(cls, root: str | Path = ".", *, name: str | None = None, task: str = "detection") -> VisionPackClient:
        """Create a project at ``root`` (idempotent) and return a client on it."""
        return cls(Project.init(root, name=name, task=task))

    @classmethod
    def open(cls, root: str | Path = ".") -> VisionPackClient:
        """Open the project at (or above) ``root``."""
        return cls(Project.open(root))

    @property
    def project(self) -> Project:
        """The underlying :class:`Project`, for advanced/internal use."""
        return self._project

    @property
    def root(self) -> Path:
        return self._project.root

    @property
    def name(self) -> str:
        return self._project.manifest.name

    @property
    def task(self) -> str:
        return self._project.manifest.task

    @property
    def classes(self) -> list[ClassDef]:
        return list(self._project.manifest.classes)

    @property
    def readonly(self) -> bool:
        """True for snapshot views returned by :meth:`checkout`."""
        return self._readonly

    def __repr__(self) -> str:
        pinned = f", snapshot={self._snapshot_version!r}" if self._snapshot_version else ""
        return f"VisionPackClient({str(self.root)!r}, name={self.name!r}, task={self.task!r}{pinned})"

    # -- data access -----------------------------------------------------------

    def assets(self) -> list[Asset]:
        return self._project.index.assets()

    def annotations(self) -> list[Annotation]:
        return self._project.index.annotations()

    def samples(self) -> Iterator[tuple[Asset, Annotation | None]]:
        """Stream ``(asset, annotation)`` pairs without materializing the index."""
        yield from self._project.index.iter_assets_with_annotations()

    def __len__(self) -> int:
        return self._project.index.count_assets()

    def __iter__(self) -> Iterator[tuple[Asset, Annotation | None]]:
        return self.samples()

    # -- ingest ----------------------------------------------------------------

    def import_dir(
        self,
        source: str | Path,
        *,
        format: str = "auto",
        images: str | Path | None = None,
        copy_mode: str = "ingest",
    ) -> dict[str, Any]:
        """Import a dataset from disk (mirrors ``vp import``).

        ``source`` is the YOLO/ImageFolder root — or, for ``format="coco"``,
        the annotations JSON, with ``images`` pointing at the image directory.
        The default ``format="auto"`` detects the layout from the dataset's
        structure. Returns the import summary as a dict (including per-file
        ``failures``).
        """
        if format == "auto":
            from visionpack.formats.detect import coco_json_in, detect_import_format

            source_path = Path(source)
            format = detect_import_format(source_path)
            if format == "coco" and source_path.is_dir():
                # "instances.json next to the images" layout: the JSON becomes
                # the source and the directory doubles as the images root.
                annotations = coco_json_in(source_path)
                assert annotations is not None  # detection said coco, so the JSON is there
                source = annotations
                images = images or source_path
        if format not in IMPORT_FORMATS:
            raise VisionPackError(f"Unknown import format {format!r}. Use one of: auto, {', '.join(IMPORT_FORMATS)}.")
        from visionpack.formats.classification import ImageFolderImporter
        from visionpack.formats.coco import CocoImporter
        from visionpack.formats.yolo import YoloImporter

        with self._write_lock():
            if format == "coco":
                if images is None:
                    raise VisionPackError("import_dir(format='coco') needs images=<directory holding the image files>.")
                importer = CocoImporter(self._project, Path(source), Path(images), copy_mode=copy_mode)
            elif format == "imagefolder":
                importer = ImageFolderImporter(self._project, Path(source), copy_mode=copy_mode)
            else:
                importer = YoloImporter(self._project, Path(source), copy_mode=copy_mode)
            summary = importer.run()
        return {
            "format": format,
            "assets": summary.assets,
            "annotations": summary.annotations,
            "objects": summary.objects,
            "classes_added": summary.classes_added,
            "orphan_labels": summary.orphan_labels,
            "failures": [asdict(failure) for failure in summary.failures],
        }

    def sync(self, *, source: str | None = None, jobs: int | None = None) -> list[dict[str, Any]]:
        """Pull every source declared in ``visionpack.yaml`` (mirrors ``vp sync``)."""
        from visionpack.sources.importer import sync_sources

        with self._write_lock():
            summaries = sync_sources(self._project, source_name=source, max_workers=jobs)
        return [asdict(summary) for summary in summaries]

    def plan_sync(self, *, source: str | None = None) -> list[dict[str, Any]]:
        """What :meth:`sync` would do, without writing (``vp sync --dry-run``)."""
        from visionpack.sources.importer import plan_sources

        return [asdict(plan) for plan in plan_sources(self._project, source_name=source)]

    # -- quality ---------------------------------------------------------------

    def validate(self, *, strict: bool = False) -> ValidationReport:
        """Correctness checks: corrupt images, bad boxes, duplicates, leakage."""
        return validate_project(self._project, strict=strict)

    def audit(self, **thresholds: Any) -> AuditReport:
        """Label-health audit (mirrors ``vp audit``).

        Keyword thresholds override ``validation.audit`` from the manifest:
        ``min_box_px``, ``duplicate_iou``, ``max_aspect_ratio``,
        ``edge_tolerance_px``, ``covers_image_ratio``, ``imbalance_ratio``,
        ``min_class_count``.
        """
        return audit_project(self._project, AuditThresholds.from_project(self._project, **thresholds))

    def stats(self) -> dict[str, Any]:
        return collect_stats(self._project)

    def split_stats(self, split_id: str = "default") -> dict[str, Any] | None:
        return split_breakdown(self._project, split_id)

    # -- splits ----------------------------------------------------------------

    def create_split(
        self,
        *,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        strategy: str = "stratified",
        seed: int = 0,
        split_id: str = "default",
        force: bool = False,
    ) -> Split:
        with self._write_lock():
            return _create_split(
                self._project, train=train, val=val, test=test, strategy=strategy, seed=seed, split_id=split_id, force=force
            )

    def lock_split(self, split_id: str = "default") -> Split:
        with self._write_lock():
            return lock_split(self._project, split_id)

    def split(self, split_id: str = "default") -> Split | None:
        return get_split(self._project, split_id)

    # -- versions ----------------------------------------------------------------

    def snapshot(self, message: str) -> dict[str, Any]:
        """Freeze the current dataset state as a new version (``vp snapshot create``)."""
        with self._write_lock():
            return create_snapshot(self._project, message)

    def snapshots(self) -> list[dict[str, Any]]:
        return list_snapshots(self._project)

    def get_snapshot(self, version: str) -> dict[str, Any]:
        return load_snapshot(self._project, version)

    def tag_snapshot(self, version: str, tag: str) -> dict[str, Any]:
        """Attach a lineage tag (convention ``key:value``, e.g. ``trained:run-812``)."""
        with self._write_lock():
            return tag_snapshot(self._project, version, tag)

    def untag_snapshot(self, version: str, tag: str) -> dict[str, Any]:
        with self._write_lock():
            return untag_snapshot(self._project, version, tag)

    def snapshots_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Snapshots carrying ``tag`` (a bare ``key:`` prefix matches any value)."""
        return find_snapshots_by_tag(self._project, tag)

    def diff(self, left: str, right: str) -> dict[str, Any]:
        """Structural diff between two snapshots (mirrors ``vp diff``)."""
        return diff_snapshots(self._project, left, right)

    def drift(self, left: str, right: str) -> dict[str, Any]:
        """Class-distribution drift between two snapshots (``vp diff --drift``):
        per-class share deltas plus KL/JS divergence."""
        return drift_between(self._project, left, right)

    def checkout(self, version: str) -> VisionPackClient:
        """A read-only client pinned to snapshot ``version``.

        Reads (``assets``, ``stats``, ``export``, ``evaluate``, ...) reflect
        that exact state; mutating methods raise.
        """
        return VisionPackClient(open_snapshot(self._project, version), _readonly=True, _snapshot=version)

    # -- outputs ---------------------------------------------------------------

    def export(
        self,
        output: str | Path,
        *,
        format: str = "yolo",
        split: str | None = None,
        seg: bool | None = None,
    ) -> dict[str, Any]:
        """Write a ready-to-train layout (mirrors ``vp export``)."""
        from visionpack.formats.classification import export_imagefolder
        from visionpack.formats.coco import export_coco
        from visionpack.formats.masks import export_masks
        from visionpack.formats.yolo import export_yolo

        output_path = Path(output)
        if format == "coco":
            return export_coco(self._project, output_path, split_id=split)
        if format == "imagefolder":
            return export_imagefolder(self._project, output_path, split_id=split)
        if format == "masks":
            return export_masks(self._project, output_path, split_id=split)
        if format == "yolo":
            return export_yolo(self._project, output_path, split_id=split, seg=seg)
        raise VisionPackError(f"Unknown export format {format!r}. Use one of: {', '.join(EXPORT_FORMATS)}.")

    # -- model in the loop -------------------------------------------------------

    def load_predictions(self, predictions: str | Path | PredictionSet, *, format: str = "auto") -> PredictionSet:
        """Load model output (vp JSON / COCO JSON / YOLO txt dir) into a
        :class:`PredictionSet` resolved against this dataset's assets."""
        if isinstance(predictions, PredictionSet):
            return predictions
        return load_predictions(self._project, Path(predictions), fmt=format)

    def evaluate(
        self,
        predictions: str | Path | PredictionSet,
        *,
        format: str = "auto",
        split: str | None = "default",
        set_name: str = "test",
        conf_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Score predictions against a split's labels (mirrors ``vp eval``)."""
        loaded = self.load_predictions(predictions, format=format)
        return _evaluate(self._project, loaded, split_id=split, set_name=set_name, conf_threshold=conf_threshold)

    def autolabel(
        self,
        predictions: str | Path | PredictionSet,
        *,
        format: str = "auto",
        min_confidence: float = 0.5,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Persist confident predictions as annotations (mirrors ``vp autolabel``)."""
        loaded = self.load_predictions(predictions, format=format)
        with self._write_lock():
            return apply_predictions(self._project, loaded, min_confidence=min_confidence, replace=replace)

    def annotation_queue(
        self,
        predictions: str | Path | PredictionSet | None = None,
        *,
        format: str = "auto",
        include_labeled: bool = False,
        confident: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Rank images by annotation value (mirrors ``vp queue``)."""
        loaded = self.load_predictions(predictions, format=format) if predictions is not None else None
        return rank_for_annotation(
            self._project, loaded, include_labeled=include_labeled, confident=confident, iou_threshold=iou_threshold
        )

    # -- internal ----------------------------------------------------------------

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """The project lock every mutating SDK call runs under.

        Snapshot views are frozen history: mutating them would silently write
        into the *live* index (they share the root), so it's refused outright.
        """
        if self._readonly:
            raise VisionPackError(
                f"This client is a read-only view of snapshot {self._snapshot_version!r}; "
                "open the live dataset with VisionPackClient.open() to modify it."
            )
        with project_lock(self._project.root):
            yield


def init(root: str | Path = ".", *, name: str | None = None, task: str = "detection") -> VisionPackClient:
    """Module-level alias: ``visionpack.sdk.init(...)``."""
    return VisionPackClient.init(root, name=name, task=task)


def open(root: str | Path = ".") -> VisionPackClient:  # noqa: A001 - deliberate, mirrors Project.open
    """Module-level alias: ``visionpack.sdk.open(...)``."""
    return VisionPackClient.open(root)
