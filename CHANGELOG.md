# Changelog

All notable changes to VisionPack are tracked here.

## [Unreleased]

- Distribution drift between snapshots (`vp diff v1 v2 --drift`, SDK
  `ds.drift("v1", "v2")`): per-class object counts and distribution-share
  deltas (biggest movers first) plus smoothed KL and Jensen–Shannon divergence,
  computed from the stats frozen in each snapshot — reproducible forever.
- Dataset → model lineage (`vp snapshot tag v4 trained:run-812`): free-form
  tags on snapshots (add/`--remove`, shown in `snapshot list`/`show`); SDK
  `tag_snapshot`/`untag_snapshot`/`snapshots_by_tag` (a bare `key:` prefix
  matches any value).
- `vp import --format auto` (now the default): the import format is detected
  from the dataset's structure — instances-style JSON means COCO (a directory
  with the JSON next to the images works too), `.txt` labels or
  classes/data.yaml furniture means YOLO, folder-per-class means ImageFolder;
  ambiguous layouts ask for an explicit `--format` instead of guessing. The
  SDK's `import_dir` defaults to `auto` as well.
- Python SDK (`visionpack.sdk`): the whole dataset lifecycle behind one class,
  `VisionPackClient` — init/open, import, sync, validate, audit, stats, splits,
  snapshots (including read-only `checkout(version)` views), export, and the
  model loop (evaluate/autolabel/annotation queue). Mutating methods take the
  same project lock the CLI takes, and summaries come back as the same
  JSON-friendly shapes the `--json` envelopes carry.
- `vp audit`: label-health audit (roadmap Phase B) — duplicate same-class boxes,
  degenerate (tiny) boxes, edge-pinned and whole-image boxes, aspect-ratio
  outliers, rare classes, and class imbalance. Findings are advisory by default
  (`--fail-on-findings` gates CI); thresholds configurable via flags or
  `validation.audit` in `visionpack.yaml`; `--json` supported.
- Security: class names arriving from imported data (folder names, COCO
  categories, `classes.txt`) are sanitized before being used as export path
  components, so a hostile name like `../../x` can no longer write outside the
  export directory.
- Robustness: a decompression-bomb image (header claiming absurd dimensions)
  now records a per-file ingest failure instead of aborting the whole
  import/sync batch.
- Performance: sync/import now read only asset *ids* when checking which assets
  already exist (`SELECT id`), instead of materializing every asset record.
- Model-in-the-loop foundation: a shared predictions loader
  (`visionpack/predictions.py`) reads model output in three formats — vp-native
  JSON, COCO results/instances JSON, and YOLO txt directories (what Ultralytics
  `predict` writes with `save_txt`/`save_conf`) — and resolves images to assets
  by asset id or original filename.
- `vp eval`: score predictions against a split's labels, turning a locked split
  + snapshot into a reproducible benchmark. Detection/segmentation/keypoints get
  COCO-style AP (per-class AP@50, mAP@50, mAP@50-95, precision/recall at a
  confidence threshold); classification gets accuracy, per-class P/R/F1 and a
  confusion matrix. `--json` for machine-readable output.
- `vp autolabel`: persist confident predictions as annotations with
  `source.type = "model"` (auditable, distinguishable from human labels). Only
  unlabeled assets are touched unless `--replace`; `--min-confidence` filters
  objects.
- `vp queue`: active-learning queue that ranks images by annotation value —
  unlabeled images first (by model uncertainty when predictions are given), and
  with `--include-labeled` audits labeled images for ground-truth/prediction
  disagreement (possible missing or stale labels).
- YOLO-seg: YOLO imports now accept polygon label lines
  (`class x1 y1 x2 y2 ...`) as instance-segmentation geometry, and `vp export
  --format yolo` writes YOLO-seg labels for segmentation projects (or with
  `--seg`); plain boxes degrade to four-corner polygons.
- Semantic masks: `vp export --format masks` rasterizes polygon (and box)
  annotations into 8-bit class-index PNGs (0 = background), split-aware, with a
  `classes.txt` documenting the pixel-value mapping.
- `vp --version` now reports the installed package version (was hardcoded and
  out of sync with `pyproject.toml`); removed duplicated README sections.
- Cloud sync foundation (PR 1 of docs/SPEC-cloud-sync.md): `FsspecResolver` for
  remote schemes (`s3://`, `gs://`, `az://`) behind optional extras
  (`visionpack[s3]`, `[gcs]`, `[azure]`); `Resolver.stat` for metadata-only
  probing; and a `blob_cache` so an unchanged re-sync skips re-reading object
  bodies entirely. Listings carry size+etag in one pass (no per-object HEAD) and
  credentials/region from the YAML reach the provider filesystem.
- Cloud-internal sync (PR 2): a `target:` in `visionpack.yaml` is a
  content-addressed sink that `copy`-mode sync lands objects in **server-side**
  (`Resolver.server_copy`), so bytes never round-trip through the client; global
  dedup by content, idempotent re-sync, and non-destructive (no `move`).
  `reference` mode points the index straight at the source object. Same-provider
  only in v1.
- Byte-free export (PR 3): `vp export` (yolo/coco/imagefolder) now **hardlinks**
  local images from the CAS instead of copying them — exports cost zero extra
  bytes. Cloud-backed (remote) assets are not downloaded: each is written to a
  `manifest.jsonl` of `(image, uri, ...)` alongside the generated labels, so a
  trainer streams images straight from the bucket.

## [0.0.1] - 2026-06-15

- No changes documented.

## [0.1.0] - 2026-06-04

- Initial PyPI-ready package metadata, CLI entry points, build workflow, and release publishing workflow.
