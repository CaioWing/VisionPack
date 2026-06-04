# VisionPack

**DatasetOps for Computer Vision** — a Git/Docker-like CLI for the messy part of
training models: turning scattered images and labels into a clean, versioned,
leak-free, ready-to-train dataset.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-active%20development-orange)
![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)

```bash
vp init --name factory-defects --task detection
vp sync                 # pull images + labels from the sources in visionpack.yaml
vp validate             # catch corrupt images, bad boxes, near-duplicate leakage
vp split create         # deterministic, reproducible train/val/test
vp snapshot create -m "baseline"
vp export --format yolo --split          # ready-to-train layout
```

---

## Why VisionPack

Most of the pain in a CV project isn't the model — it's the dataset. VisionPack
targets the failures that quietly cost you accuracy and reproducibility:

- **Train/test leakage that inflates your metrics.** Exact-duplicate detection is
  not enough: a re-encoded or resized copy of a training image landing in the test
  set makes your reported numbers a lie. VisionPack catches **near-duplicate**
  leakage with perceptual hashing.
- **Splits you can't reproduce.** "I shuffled with `random.seed(42)`" breaks the
  moment data is added or reordered. VisionPack splits are a function of image
  *content*, so they're identical across machines and stable as the dataset grows.
- **Data scattered across places.** Images in one bucket, labels in another repo,
  classes in a third. Declare them once and `vp sync` assembles the dataset.
- **"Which dataset trained this model?"** Content-addressed snapshots make that
  answerable instead of a guess.

It's built to **complement, not replace** CVAT, FiftyOne, DVC, Roboflow, and Label
Studio — VisionPack is the DatasetOps layer that imports, validates, versions,
splits, packs, and exports.

---

## Install

VisionPack uses [`uv`](https://github.com/astral-sh/uv). From the repo root:

```bash
uv sync
uv run vp --help
```

Requires Python 3.11+.

---

## Quickstart (60 seconds)

```bash
# 1. create a project (the manifest is visionpack.yaml)
uv run vp init --name factory-defects --task detection

# 2. bring in a YOLO dataset
uv run vp import ./raw --format yolo

# 3. check it for real problems
uv run vp validate

# 4. a deterministic, reproducible split
uv run vp split create --train 0.8 --val 0.1 --test 0.1 --strategy stratified
uv run vp split lock

# 5. freeze a reproducible version
uv run vp snapshot create -m "initial import"

# 6. comparable metrics as the dataset grows
uv run vp stats --by split

# 7. a ready-to-train layout
uv run vp export --format yolo --split
```

---

## Works across the common CV tasks

VisionPack's annotation model carries a tagged geometry, so one tool covers the
tasks you actually use:

| Task | Import | Geometry |
|------|--------|----------|
| **Classification** | ImageFolder (folder-per-class) | whole-image label |
| **Detection** | YOLO, COCO | bounding box |
| **Instance segmentation** | COCO | polygon |
| **Keypoints / pose** | COCO | keypoints |

```bash
# classification from a folder-per-class layout
uv run vp init --name product-grades --task classification
uv run vp import ./train --format imagefolder
uv run vp export --format imagefolder --split        # train/val/test/<class>/…

# detection or instance segmentation from COCO
uv run vp init --name cells --task segmentation
uv run vp import ./instances.json --format coco --images ./images
```

---

## Assemble a dataset from many sources

Images and labels rarely live together. Declare them in `visionpack.yaml`:

```yaml
sources:
  - name: camera-A
    format: yolo
    images: ./repoA/images        # images here…
    labels: ./repoB               # …labels in another repo
    classes: ./repoB/classes.txt
    match: stem                   # pair by filename (or `relpath` for parallel trees)
    copy: ingest
```

Then reconcile the dataset — idempotently, so re-running only pulls what's new:

```bash
uv run vp sync --dry-run     # preview: found / matched / unmatched / classes
uv run vp sync               # ingest; classes merge by name, provenance recorded
```

Classes from different sources merge **by name** (YOLO indices are mapped through
each source's own class order, never positionally), so reordered class lists don't
mislabel your data.

A one-off `vp import` also records what it imported as a source in
`visionpack.yaml`, so the manifest stays the single source of truth and the data
can be re-pulled later with `vp sync` (use `--no-record` for a throwaway import).

---

## What's in the box

- **Deterministic, lockable splits** — `stratified` / `random` / `hash`
  (growth-stable), captured in snapshots.
- **Near-duplicate & cross-split leakage detection** — perceptual-hash tier, no
  extra dependencies, scale-proof via LSH bucketing; surfaced in `vp validate`.
- **Multi-source sync** — declarative `sources:` + `vp sync`, with per-asset
  provenance and a resolver layer ready for remote backends.
- **Content-addressed snapshots & diff** — reproducible versions; compare any two.
- **Strong validation** — unreadable images, missing/orphan labels, unknown
  classes, invalid/out-of-bounds boxes, exact + near duplicates, split leakage.
- **Comparable metrics** — per-split stats so class balance stays auditable as data
  grows.
- **Packing** — `archive` (`.tar.zst`, self-contained) and `training`
  (split-aware WebDataset shards).
- **Interoperable I/O** — YOLO, COCO, ImageFolder in and out.

Full command reference and per-command options live in the
[usage guide](docs/usage.md).

---

## How it works

VisionPack is **manifest-driven** and **content-addressed**: `visionpack.yaml`
declares the dataset, raw images are stored once by `sha256` (immutable), and
annotations / splits / snapshots are the versioned layer on top. The truth is the
manifest + index, never "a folder with the right name".

For the design principles, module map, data model, subsystems, and the full
roadmap, see **[ARCHITECTURE.md](ARCHITECTURE.md)**. For the original product
vision, see **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Status

VisionPack is in **active development** (early but usable). The core workflow —
multi-source ingestion → validation → deterministic splits → snapshots →
ready-to-train export/packing — works end-to-end across classification, detection,
instance segmentation, and keypoints, with 55 passing tests. APIs may still shift;
feedback and contributions are welcome.

```bash
uv run python -m unittest discover -s tests -q
```
