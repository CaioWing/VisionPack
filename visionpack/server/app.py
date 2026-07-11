from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from visionpack.core.errors import VisionPackError
from visionpack.core.lock import project_lock
from visionpack.core.models import Asset
from visionpack.core.project import Project
from visionpack.server.jobs import JobBusyError, JobManager
from visionpack.snapshot import create_snapshot, list_snapshots
from visionpack.sources import plan_sources, sync_sources
from visionpack.sources.resolver import get_resolver, scheme_of
from visionpack.split import create_split, lock_split
from visionpack.stats import collect_stats, split_breakdown
from visionpack.validation.engine import validate_project

_STATIC = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "tiff": "image/tiff",
}

_MAX_ISSUES = 500


class _ProjectHandle:
    """A cached, shared read view of the project.

    `Project.open` re-reads the manifest and lazily loads the index; doing that
    per thumbnail request would thrash. Reads share one instance; anything that
    mutates the dataset (a sync job, a split, a snapshot) calls
    :meth:`invalidate` so the next read reopens fresh state.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._mutex = threading.Lock()
        self._project: Project | None = None
        self._assets_by_id: dict[str, Asset] | None = None

    def get(self) -> Project:
        with self._mutex:
            if self._project is None:
                self._project = Project.open(self.root)
                self._assets_by_id = None
            return self._project

    def asset(self, asset_id: str) -> Asset | None:
        project = self.get()
        with self._mutex:
            if self._assets_by_id is None:
                self._assets_by_id = {asset.id: asset for asset in project.index.assets()}
            return self._assets_by_id.get(asset_id)

    def invalidate(self) -> None:
        with self._mutex:
            self._project = None
            self._assets_by_id = None


def _sanitize(value: Any) -> Any:
    """Drop credential material at any nesting level before it leaves the API."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if key != "credentials"}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _provider_of(location: Any) -> str:
    if isinstance(location, dict):
        location = location.get("uri", "")
    return scheme_of(str(location or "")) or "local"


class SyncRequest(BaseModel):
    source: str | None = None


class SplitRequest(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    strategy: str = "stratified"
    seed: int = 0
    split_id: str = "default"
    force: bool = False


class SnapshotRequest(BaseModel):
    message: str = Field(min_length=1)


class ExportRequest(BaseModel):
    format: str = "yolo"
    split: str | None = None
    output: str | None = None


def create_app(root: Path) -> FastAPI:
    handle = _ProjectHandle(root.resolve())
    # Any finished job may have written to the index/manifest; reopening on the
    # next read is cheap and keeps the UI consistent without tracking which
    # job kinds mutate.
    jobs = JobManager(on_finish=lambda job: handle.invalidate())
    app = FastAPI(title="VisionPack", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.exception_handler(VisionPackError)
    async def _vp_error(request, exc):  # noqa: ANN001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- UI ---------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    # -- project ------------------------------------------------------------

    @app.get("/api/project")
    def project_info() -> dict[str, Any]:
        project = handle.get()
        manifest = project.manifest
        return {
            "name": manifest.name,
            "task": manifest.task,
            "version": manifest.version,
            "root": str(project.root),
            "classes": [{"id": item.id, "name": item.name} for item in manifest.classes],
            "target": _sanitize(manifest.target),
            "counts": {
                "assets": project.index.count_assets(),
                "sources": len(manifest.sources),
                "splits": len(project.index.splits()),
                "snapshots": len(list_snapshots(project)),
            },
        }

    @app.get("/api/sources")
    def sources() -> list[dict[str, Any]]:
        project = handle.get()
        result = []
        for raw in project.manifest.sources:
            item = _sanitize(dict(raw))
            item["provider"] = _provider_of(raw.get("images") or raw.get("root"))
            result.append(item)
        return result

    # -- pipeline actions -----------------------------------------------------

    @app.post("/api/sync/plan")
    def sync_plan(body: SyncRequest | None = None) -> list[dict[str, Any]]:
        project = handle.get()
        plans = plan_sources(project, body.source if body else None)
        return [asdict(plan) for plan in plans]

    @app.post("/api/sync", status_code=202)
    def sync(body: SyncRequest | None = None) -> dict[str, Any]:
        source = body.source if body else None

        def run(progress):  # noqa: ANN001
            from contextlib import contextmanager

            @contextmanager
            def factory(description: str):
                yield progress

            with project_lock(handle.root):
                summaries = sync_sources(Project.open(handle.root), source, progress_factory=factory)
            return [
                {
                    "name": summary.name,
                    "assets_added": summary.assets_added,
                    "assets_existing": summary.assets_existing,
                    "annotations": summary.annotations,
                    "objects": summary.objects,
                    "classes_added": summary.classes_added,
                    "images_without_label": summary.images_without_label,
                    "labels_without_image": summary.labels_without_image,
                    "failures": [asdict(failure) for failure in summary.failures],
                }
                for summary in summaries
            ]

        return _start(jobs, "sync", run, detail=source or "all sources")

    @app.post("/api/validate", status_code=202)
    def validate() -> dict[str, Any]:
        def run(progress):  # noqa: ANN001
            report = validate_project(Project.open(handle.root))
            issues = [asdict(issue) for issue in report.issues[:_MAX_ISSUES]]
            return {
                "ok": report.ok,
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "truncated": len(report.issues) > _MAX_ISSUES,
                "issues": issues,
            }

        return _start(jobs, "validate", run)

    @app.post("/api/export", status_code=202)
    def export(body: ExportRequest) -> dict[str, Any]:
        if body.format not in ("yolo", "coco", "imagefolder", "masks"):
            raise HTTPException(status_code=400, detail=f"Unknown export format {body.format!r}.")
        output = Path(body.output) if body.output else handle.root / "exports" / body.format
        if not output.is_absolute():
            output = handle.root / output

        def run(progress):  # noqa: ANN001
            from visionpack.formats.classification import export_imagefolder
            from visionpack.formats.coco import export_coco
            from visionpack.formats.masks import export_masks
            from visionpack.formats.yolo import export_yolo

            exporters = {
                "yolo": export_yolo,
                "coco": export_coco,
                "imagefolder": export_imagefolder,
                "masks": export_masks,
            }
            summary = exporters[body.format](Project.open(handle.root), output, split_id=body.split, progress=progress)
            return {"output": str(output), **summary}

        return _start(jobs, "export", run, detail=body.format)

    # -- jobs -----------------------------------------------------------------

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [job.to_dict() for job in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id!r}.")
        return job.to_dict()

    # -- data -----------------------------------------------------------------

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        project = handle.get()
        overall = collect_stats(project)
        class_names = {item.id: item.name for item in project.manifest.classes}
        overall["class_distribution"] = {
            class_names.get(class_id, class_id): count for class_id, count in overall["class_distribution"].items()
        }
        breakdowns = {}
        for split in project.index.splits():
            breakdown = split_breakdown(project, split.id)
            if breakdown is not None:
                breakdowns[split.id] = breakdown
        return {"overall": overall, "splits": breakdowns}

    @app.get("/api/assets")
    def assets(
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=500),
        source: str | None = None,
    ) -> dict[str, Any]:
        project = handle.get()
        class_names = {item.id: item.name for item in project.manifest.classes}
        every = project.index.assets()
        if source:
            every = [asset for asset in every if asset.source == source]
        page = every[offset : offset + limit]
        items = []
        for asset in page:
            annotation = project.index.annotation_for_asset(asset.id)
            labels = sorted({class_names.get(obj.class_id, obj.class_id) for obj in annotation.objects}) if annotation else []
            items.append(
                {
                    "id": asset.id,
                    "width": asset.width,
                    "height": asset.height,
                    "format": asset.format,
                    "size_bytes": asset.size_bytes,
                    "source": asset.source,
                    "provider": scheme_of(asset.path) or "local",
                    "remote": asset.is_remote,
                    "objects": len(annotation.objects) if annotation else 0,
                    "classes": labels,
                }
            )
        return {"total": len(every), "offset": offset, "limit": limit, "items": items}

    @app.get("/api/assets/{asset_id}/file")
    def asset_file(asset_id: str) -> Response:
        project = handle.get()
        asset = handle.asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"No asset {asset_id!r}.")
        try:
            if asset.is_remote:
                data = get_resolver(asset.path).read_bytes(asset.path)
            else:
                data = asset.resolved_path(project.root).read_bytes()
        except (VisionPackError, OSError) as exc:
            raise HTTPException(status_code=502, detail=f"Could not read {asset.path!r}: {exc}") from exc
        media_type = _CONTENT_TYPES.get((asset.format or "").lower(), "application/octet-stream")
        return Response(content=data, media_type=media_type, headers={"Cache-Control": "max-age=3600"})

    # -- versions ---------------------------------------------------------------

    @app.get("/api/splits")
    def splits() -> list[dict[str, Any]]:
        project = handle.get()
        return [
            {
                "id": split.id,
                "strategy": split.strategy,
                "locked": split.locked,
                "sets": {name: len(ids) for name, ids in split.sets.items()},
            }
            for split in project.index.splits()
        ]

    @app.post("/api/splits")
    def split_create(body: SplitRequest) -> dict[str, Any]:
        with project_lock(handle.root):
            split = create_split(
                Project.open(handle.root),
                train=body.train,
                val=body.val,
                test=body.test,
                strategy=body.strategy,
                seed=body.seed,
                split_id=body.split_id,
                force=body.force,
            )
        handle.invalidate()
        return {"id": split.id, "strategy": split.strategy, "sets": {name: len(ids) for name, ids in split.sets.items()}}

    @app.post("/api/splits/{split_id}/lock")
    def split_lock(split_id: str) -> dict[str, Any]:
        with project_lock(handle.root):
            split = lock_split(Project.open(handle.root), split_id)
        handle.invalidate()
        return {"id": split.id, "locked": split.locked}

    @app.get("/api/snapshots")
    def snapshots() -> list[dict[str, Any]]:
        return [_sanitize(record) for record in list_snapshots(handle.get())]

    @app.post("/api/snapshots")
    def snapshot_create(body: SnapshotRequest) -> dict[str, Any]:
        with project_lock(handle.root):
            record = create_snapshot(Project.open(handle.root), body.message)
        handle.invalidate()
        return _sanitize(record)

    return app


def _start(jobs: JobManager, kind: str, body, detail: str = "") -> dict[str, Any]:  # noqa: ANN001
    try:
        return jobs.start(kind, body, detail=detail).to_dict()
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
