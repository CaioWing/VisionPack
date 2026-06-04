# VisionPack — Architecture & Progress

This document describes how VisionPack is built and where it stands. For day-to-day
usage see the [README](README.md) and the [usage guide](docs/usage.md); for the
original product vision see [docs/DESIGN.md](docs/DESIGN.md).

---

## Design principles

1. **CLI-first.** The primary interface is the `vp` command, scriptable in
   notebooks, training servers, and CI. No web app in the core.
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
                         #   stats, split, snapshot, diff, export, pack, annotate)
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
    yolo.py  coco.py  classification.py   # import + export per format
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
- **JSON index** at `.vp/db/index.json`. Deserialized records and an
  `asset_id -> annotation` map are cached so validation/stats/export don't re-parse
  on every call (keeps lookups O(1) at the README's target scale).
- **DuckDB — deferred by decision.** At current scale the cached lists are fine, and
  DuckDB's payoff (SQL aggregations/joins, scale past RAM) is pulled by Phase B. When
  built it will be an in-memory query layer over the portable JSON store (no binary
  index file, no file-lock issues), driven by a real query need.

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

### Geometry model & task coverage (`core/models.py`, `formats/`)
The tagged geometry above lets one dataset model cover the common CV tasks.
Classification uses the ImageFolder convention; detection uses YOLO/COCO bbox;
instance segmentation (polygons) and keypoints come through COCO, selected by the
project task. Stats, splits, and validation are geometry-agnostic (they key on
`class_id`), so they work across tasks unchanged.

### Packing (`packing/`)
`archive` produces a self-contained `.tar.zst` (manifest + index + snapshots +
assets + metadata). `training` produces split-aware WebDataset shards
(`<key>.<img>` + `<key>.json` per sample, chunked by `shard_size`, optional zstd)
with a self-describing `dataset.json`.

---

## Testing

`unittest` suite under `tests/` (55 tests). Run with:

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

### Multi-source ingestion ✅ (remote backends pending)
- [x] declarative `sources:` + `vp sync` (+ `--dry-run`), local backends, joins,
      provenance, class reconciliation, idempotent re-sync
- [ ] remote backends via fsspec behind extras (s3/gcs/azure/git, pinned by ref)
      and COCO-format sources, plugging into the same resolver layer

### Task coverage ✅ (beyond detection)
- [x] tagged geometry model (bbox | polygon | keypoints | none), backward compatible
- [x] classification: ImageFolder import/export
- [x] instance segmentation (polygons) and keypoints via COCO
- [ ] semantic segmentation (per-class mask PNGs)
- [ ] YOLO-seg / YOLO-pose import-export; dedicated keypoint importer
- [ ] `--format auto` task/format detection

### Phase B — Differentiators
- [x] near-duplicate & cross-split leakage detection (perceptual-hash tier)
- [ ] optional embedding tier (CLIP/DINOv2) for semantic near-duplicates
- [ ] label-health audit (`vp audit`): duplicate/degenerate/edge-pinned boxes,
      aspect-ratio outliers, class imbalance
- [ ] model-in-the-loop quality (confident detections with no matching label)
- [ ] distribution-drift diff between snapshots (per-class deltas / KL)
- [ ] dataset → model lineage (`vp snapshot tag v4 trained:<run-id>`)

### Benchmarking (planned)
- [ ] `vp eval` — score predictions against a locked test split (mAP, accuracy,
      confusion matrix), turning a dataset into a reproducible benchmark
- [ ] benchmark objects (snapshot + split + protocol) and a leaderboard
- [ ] dataset/benchmark cards (`vp card`) for publishable, citable artifacts
- [ ] Hugging Face Datasets export

### Phase C — Reporting & polish
- [ ] HTML validation / stats / drift reports
- [ ] JSON report output for stats and diff
- [ ] richer terminal output with `rich`
- [ ] move CLI plumbing from `argparse` to `typer` once commands stabilize

### Later
- [ ] `vp annotate prepare` / `ingest`; CVAT and Label Studio packages
- [ ] active-learning queue (rank unlabeled images by model uncertainty)
- [ ] remote storage integrations (S3/GCS/Azure) for assets
- [ ] optional PyTorch dataset helpers
