# VisionPack — Architecture & Progress

This document describes how VisionPack is built and where it stands. For day-to-day
usage see the [README](README.md) and the [usage guide](docs/usage.md); for the
original product vision see [docs/DESIGN.md](docs/DESIGN.md).

---

## Design principles

1. **CLI-first.** The primary interface is the `vp` command, scriptable in
   notebooks, training servers, and CI. No web app in the core. For Python
   callers the same workflow is exposed as a facade (`visionpack.sdk`) with the
   CLI's locking and result shapes.
2. **The manifest is the source of truth.** `visionpack.yaml` declares the dataset
   (classes, sources, splits, validation policy, pack profiles). Behaviour is
   driven by the manifest + the internal index, never by "a folder with the right
   name".
3. **Immutable data, versioned metadata.** Raw images are content — addressed by
   `sha256` and stored once in a content-addressable store (CAS). Annotations,
   splits, and snapshots are the versioned layer on top.
4. **Interoperability over lock-in.** VisionPack imports and exports the formats
   teams already use (YOLO, COCO, ImageFolder) instead of inventing a closed one.
   It complements CVAT / FiftyOne / DVC / Roboflow rather than replacing them.
5. **Reproducibility.** A locked split + a snapshot + content hashing make it
   possible to answer "which exact dataset produced this artifact?".

---

## Module map

```text
visionpack/
  cli/
    main.py              # argparse wiring; registers every subcommand
    commands/            # one module per command (init, import, sync, validate,
                         #   audit, fsck, stats, split, snapshot, diff, export,
                         #   pack, annotate, eval, autolabel, queue)
  sdk/                   # VisionPackClient: the Python facade over the whole
                         #   workflow (same locking + result shapes as the CLI)
  core/
    project.py           # Project: manifest + index + object store handle
    manifest.py          # Manifest dataclass + pydantic schema (sources, classes…)
    models.py            # Asset, Annotation, ObjectAnnotation + geometry types,
                         #   Split  (the persisted data model)
    errors.py            # the VisionPackError hierarchy (actionable messages)
  storage/
    object_store.py      # content-addressed store; copy/move/hardlink/reference/ingest
    hash.py              # sha256 of bytes / files
  index/
    json_index.py        # local JSON index with cached, O(1) annotation lookups
  formats/
    yolo.py  coco.py  classification.py   # import + export per format (yolo covers YOLO-seg)
    masks.py             # semantic-mask export (class-index PNGs)
  sources/
    resolver.py  schema.py  join.py  importer.py   # declarative multi-source sync
  validation/
    engine.py            # the validate_project check suite
  packing/
    archive.py  webdataset.py             # pack profiles (archive, training)
  perceptual.py          # dHash perceptual hashing
  duplicates.py          # near-duplicate clustering + cross-split leakage
  split.py               # deterministic split creation / resolution
  stats.py               # dataset + per-split statistics
  snapshot.py            # content-addressed snapshots
  predictions.py         # model-prediction loading (vp/COCO/YOLO) resolved to assets
  eval.py                # vp eval: AP/mAP, accuracy, confusion matrix vs a split set
  autolabel.py           # vp autolabel: persist confident predictions as annotations
  curation.py            # vp queue: active-learning ranking + label-quality audit
  audit.py               # vp audit: label-health findings (duplicate/tiny/edge boxes,
                         #   aspect outliers, class imbalance)
```

---

## Data model

- **Asset** — one image. Identified by `sha256`; `id = asset_<sha256[:16]}`. Carries
  width/height/channels/format/size, a perceptual hash (`phash`), and `source`
  provenance (which declared source ingested it). Exact duplicates collapse to one
  asset, so they can never straddle a split.
- **Annotation** — the labels for one asset: a `task`, a list of `ObjectAnnotation`,
  and a `source` record (human/import/sync).
- **ObjectAnnotation** — `class_id` + an optional **tagged geometry**:
  `BBox` (detection) · `Polygon` (instance segmentation) · `Keypoints` (pose) ·
  `None` (whole-image label for classification). A derived `bbox` property yields an
  enclosing box for any geometry, so detection-oriented code (export, packing,
  validation) keeps working across every task. The legacy bare-`bbox` JSON still
  loads, so old datasets upgrade transparently.
- **Split** — `id`, `strategy`, `sets: {train,val,test -> [asset_id]}`, `locked`.
  A first-class versioned object, not just folders.
- **Source** — a declared contribution: an image location, a label location, a join
  rule, a format. Locations are URIs (local now; bucket/git planned).

---

## Storage & index

- **Content-addressable store** at `.vp/objects/sha256/ab/cd/<hash>`. The `ingest`
  copy mode (default) makes the dataset self-contained; `reference` keeps bytes in
  place and stores only the path + hash; `hardlink`/`copy`/`move` are also available.
- **SQLite index** at `.vp/db/index.db` (`index/sqlite_index.py`). Each record is
  stored as an orjson blob keyed by id, with an index on `annotations.asset_id`.
  Chosen for scale: opening is instant (records load lazily, only when a read needs
  them), saving is **incremental and atomic** (only rows touched since the last save
  are written, in one transaction — no full rewrite, no corruption risk), and point
  lookups are indexed queries that don't load everything. Connections are
  short-lived (opened per load/save) so no file handle lingers. A legacy
  `index.json` is migrated transparently on first open and moved aside.
  Measured at 100k assets+annotations: open 6.06s → 0.002s, per-mutation save
  1.06s → 0.009s versus the old JSON index.
- **JSON index** (`index/json_index.py`) is retained for the one-time legacy
  migration and uses orjson + atomic writes.
- **DuckDB — still deferred.** SQLite covers the transactional index well. DuckDB's
  remaining payoff is analytical (SQL group-by for stats/dedup over the SQLite/
  parquet data); it can be added as a query layer when a Phase B feature needs it.
- **Streaming reads.** `SqliteIndex.iter_assets()` / `iter_annotations()` /
  `iter_assets_with_annotations()` iterate straight off a DB cursor (a LEFT JOIN for
  the paired case), so a full-scan command never holds every record in RAM. `stats`
  and the YOLO/COCO/ImageFolder exporters use them — peak RAM for a 100k-record scan
  drops from ~220MB (materialized list + asset→annotation map) to flat. They fall
  back to the in-memory view when there are unsaved writes. Still materializing:
  `validate`, deterministic `split`, dedup, and the WebDataset pack (each needs the
  whole set at once) — candidates for the next pass.

---

## Key subsystems

### Deterministic splits (`split.py`)
Each asset is assigned by hashing its content hash with a seed, so the same dataset
and seed always produce the same split — independent of import order or machine.
Strategies: `stratified` (default, balances each class), `random` (a single
deterministic shuffle cut at exact global ratios via largest-remainder), and `hash`
(threshold on each asset's own hash — the only one **stable as data grows**: new
images never reassign existing ones). Splits can be locked and are captured in
snapshots. `resolve_export_sets()` is the shared entry point that export and packing
both use, so a split flows identically into every output.

### Near-duplicate & cross-split leakage (`perceptual.py`, `duplicates.py`)
A 64-bit dHash is computed at import (reusing the bytes already read for the sha256).
Candidate pairs are generated by **LSH banding** with a pigeonhole guarantee — no
missed pairs and no O(n²) scan as the dataset grows. `vp validate` surfaces
near-duplicate clusters as warnings and **near-duplicate train↔test leakage as
errors**, the case that silently inflates every reported metric. Dependency-free.

### Multi-source sync (`sources/`)
A `sources:` block in the manifest links images and labels that live apart. A
`Resolver` keyed by URI scheme reads bytes and lists files, so remote backends
(fsspec-based s3/gcs/azure/git) drop in without touching the import pipeline.
Images and labels are paired by `relpath` (parallel trees) or `stem` (different
layouts). `vp sync` reconciles the index idempotently (content-addressed, so
re-running skips what's present) and records per-asset provenance; `vp sync
--dry-run` previews found/matched/unmatched/classes per source.

**Multi-provider targets.** With a `target:` and `copy` mode, objects land in a
content-addressed sink. Same-provider transfers are server-side (S3 CopyObject /
GCS rewrite — bytes never transit the client). Cross-provider transfers
(local→S3, S3→GCS, …) *relay* the bytes sync already read to compute the sha256:
one read (needed anyway) + one upload (size-verified against the landed
object), never a second download. Either way an unchanged re-sync stays
metadata-only.

**Request economy & resilience.** Target membership is resolved with one prefix
listing per run (rclone-style fast-list), never a per-object existence check.
Every remote call is wrapped in exponential-backoff retries
(`sources/retry.py`); exhaustion surfaces as a per-object `IngestFailure`, so a
flaky bucket degrades a sync instead of aborting it. Ingest concurrency is
sized for latency-bound object stores (16+ workers for remote sources) and
tunable with `vp sync --jobs`. Labels fetched during class inference are cached
and replayed at ingest, so no remote label is ever read twice.

### Geometry model & task coverage (`core/models.py`, `formats/`)
The tagged geometry above lets one dataset model cover the common CV tasks.
Classification uses the ImageFolder convention; detection uses YOLO/COCO bbox;
instance segmentation (polygons) and keypoints come through COCO, selected by the
project task. Stats, splits, and validation are geometry-agnostic (they key on
`class_id`), so they work across tasks unchanged.

### Snapshots & time-travel (`snapshot.py`)
`vp snapshot create` records an immutable version (hashes + stats + lineage) and
**freezes the index** as a content-addressed SQLite db under
`.vp/snapshots/dbs/<hash>.db` (an unchanged index is frozen once and shared).
`vp export --snapshot vN` opens that frozen index as a read-only view and streams
from it — so you can materialize training sets from any past version without
touching the live state. Images are referenced from the shared CAS, never copied,
so the per-snapshot cost is just the index (MBs), not the images (GBs). `vp diff`
compares two snapshots' inventories. (At scale with many snapshots, per-record
content-addressing of annotations would dedup further — a future step.)

### Packing (`packing/`)
`archive` produces a self-contained `.tar.zst` (manifest + index + snapshots +
assets + metadata). `training` produces split-aware WebDataset shards
(`<key>.<img>` + `<key>.json` per sample, chunked by `shard_size`, optional zstd)
with a self-describing `dataset.json`.

---

## Production robustness

- **Resilient ingest.** Import and sync capture per-image failures (corrupt /
  unreadable / missing) as `IngestFailure` instead of aborting the batch; good
  images still import, the skipped ones are reported, and the command exits
  non-zero so CI can gate on it.
- **Project lock.** Mutating commands take an OS advisory lock on `.vp/lock`
  (`core/lock.py`; fcntl/msvcrt), so a second concurrent writer fails fast instead
  of racing and losing updates. Released automatically if the process dies.
- **`vp fsck`.** Verifies index↔store integrity (missing objects, orphan
  annotations/splits, missing snapshot blobs, orphan objects); `--deep` re-hashes
  every object to catch silent corruption.
- **Progress.** Long operations (import/sync/export) render a rich progress bar
  via a callback (`progress.py`), but only on an interactive terminal — piped/CI
  runs stay quiet. The core stays UI-free; the callback is optional.

## Testing

`unittest` suite under `tests/`. Run with:

```bash
uv run python -m unittest discover -s tests -q
```

Coverage spans the YOLO/COCO/classification flows, the geometry model, media
probing, manifest validation, snapshots, deterministic splits, near-duplicate /
leakage detection, multi-source sync, and training packs.

---

## Progress & roadmap

The roadmap is sequenced so each phase unblocks the next.

### Phase 0 — Foundations ✅
- [x] O(1) index access (cached records + `asset_id -> annotation` map)
- [x] parallel import reading each file once for hash + probe + store
- [x] Pillow-based probing (webp + EXIF-orientation correct)
- [x] pydantic manifest validation with field-level, actionable errors

### Phase A — Scale & format coverage ✅ (DuckDB deferred by decision)
- [x] content-addressed, deduplicated snapshots packed into archives
- [x] COCO import & export
- [x] multi-source class merging by name (no positional mislabeling)
- [x] deterministic, lockable, versionable splits (`vp split create/lock/list/show`)
- [x] per-split stats for comparable metrics (`vp stats --by split`)
- [x] split-aware export (ready-to-train train/val/test layouts)
- [x] `vp pack --profile training` (WebDataset shards)
- [ ] DuckDB index — deferred by decision (see Storage & index)

### Multi-source ingestion ✅
- [x] declarative `sources:` + `vp sync` (+ `--dry-run`), local backends, joins,
      provenance, class reconciliation, idempotent re-sync
- [x] remote YOLO/COCO/ImageFolder sources via fsspec behind extras
      (s3/gcs/azure), metadata-only re-sync, content-addressed cloud target
- [x] cross-provider `copy` targets (local↔S3, S3↔GCS, …): server-side copy when
      providers match, single-pass verified byte relay when they don't
- [x] hardening: retries with backoff on every remote call, fast-list target
      membership (no per-object HEAD), `--jobs` concurrency control,
      single-read remote labels
- [ ] git sources pinned by ref

### Task coverage ✅ (beyond detection)
- [x] tagged geometry model (bbox | polygon | keypoints | none), backward compatible
- [x] classification: ImageFolder import/export
- [x] instance segmentation (polygons) and keypoints via COCO
- [x] semantic segmentation (per-class mask PNGs via `vp export --format masks`)
- [x] YOLO-seg import-export (polygon label lines; `--seg` / segmentation-task default)
- [ ] YOLO-pose import-export; dedicated keypoint importer
- [x] `--format auto` format detection on import (now the default; structural
      detection of YOLO/COCO/ImageFolder, explicit `--format` when ambiguous)

### Phase B — Differentiators
- [x] near-duplicate & cross-split leakage detection (perceptual-hash tier)
- [ ] optional embedding tier (CLIP/DINOv2) for semantic near-duplicates
- [x] label-health audit (`vp audit`): duplicate/degenerate/edge-pinned boxes,
      aspect-ratio outliers, class imbalance (advisory by default;
      `--fail-on-findings` for CI)
- [x] model-in-the-loop quality (`vp queue --include-labeled`: confident
      detections with no matching label, and labels the model never finds)
- [x] distribution-drift diff between snapshots (`vp diff --drift`: per-class
      share deltas, smoothed KL + Jensen–Shannon divergence)
- [x] dataset → model lineage (`vp snapshot tag v4 trained:<run-id>`, free-form
      tags; `snapshots_by_tag` lookup in the SDK)

### Benchmarking
- [x] `vp eval` — score predictions against a split set (per-class AP@50,
      mAP@50, mAP@50-95, precision/recall; accuracy + confusion matrix for
      classification), turning a dataset into a reproducible benchmark.
      Predictions load from vp JSON, COCO JSON, or YOLO txt (`predictions.py`)
- [ ] mask IoU for segmentation eval (bbox IoU today)
- [ ] benchmark objects (snapshot + split + protocol) and a leaderboard
- [ ] dataset/benchmark cards (`vp card`) for publishable, citable artifacts
- [ ] Hugging Face Datasets export

### Phase C — Reporting & polish
- [x] machine-readable output: every pipeline command takes `--json` and prints
      a schema-versioned envelope (`cli/output.py`) — the stable contract for
      driving VisionPack from services/UIs/CI; errors are structured too
- [ ] HTML validation / stats / drift reports
- [ ] richer terminal output with `rich`
- [ ] move CLI plumbing from `argparse` to `typer` once commands stabilize

### Model-in-the-loop ✅ (first slice)
- [x] `vp autolabel` — persist confident predictions as annotations
      (`source.type = "model"`, `--min-confidence`, `--replace` opt-in)
- [x] active-learning queue (`vp queue`) — rank unlabeled images by model
      uncertainty; audit labeled ones for GT/prediction disagreement

### Later
- [ ] `vp annotate prepare` / `ingest`; CVAT and Label Studio packages
- [ ] remote storage integrations (S3/GCS/Azure) for assets
- [ ] optional PyTorch dataset helpers
