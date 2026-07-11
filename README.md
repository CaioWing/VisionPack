# VisionPack

**DatasetOps for Computer Vision** — a Git/Docker-like CLI for the messy part of
training models: turning scattered images and labels into a clean, versioned,
leak-free, ready-to-train dataset.

[![CI](https://github.com/CaioWing/VisionPack/actions/workflows/ci.yml/badge.svg)](https://github.com/CaioWing/VisionPack/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/visionpack)](https://pypi.org/project/visionpack/)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-active%20development-orange)
![Tests](https://img.shields.io/badge/tests-143%20passing-brightgreen)

[Documentation](https://caiowing.github.io/VisionPack/) ·
[Install](https://caiowing.github.io/VisionPack/installation/) ·
[Quickstart](https://caiowing.github.io/VisionPack/quickstart/) ·
[Cloud Sync](https://caiowing.github.io/VisionPack/cloud-sync/)

```bash
pip install visionpack

vp init --name factory-defects --task detection
vp sync                 # pull images + labels from the sources in visionpack.yaml
vp validate             # catch corrupt images, bad boxes, near-duplicate leakage
vp split create         # deterministic, reproducible train/val/test
vp snapshot create -m "baseline"
vp export --format yolo --split          # ready-to-train layout

# ...train / predict with any framework, then close the loop:
vp eval runs/predict/labels --format yolo    # mAP on the locked test set
vp autolabel preds.json --min-confidence 0.6 # confident predictions -> labels
vp queue --predictions preds.json            # what should a human label next?
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

```bash
pip install visionpack
```

Cloud backends are optional extras: `pip install "visionpack[s3]"` (also `[gcs]`,
`[azure]`). Requires Python 3.11+.

Developing from source uses [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
uv run vp --help
```

📖 **Full documentation: <https://caiowing.github.io/VisionPack/>**

---

## Quickstart (60 seconds)

```bash
# 1. create a project (the manifest is visionpack.yaml)
vp init --name factory-defects --task detection

# 2. bring in a YOLO dataset
vp import ./raw --format yolo

# 3. check it for real problems
vp validate

# 4. a deterministic, reproducible split
vp split create --train 0.8 --val 0.1 --test 0.1 --strategy stratified
vp split lock

# 5. freeze a reproducible version
vp snapshot create -m "initial import"

# 6. comparable metrics as the dataset grows
vp stats --by split

# 7. a ready-to-train layout
vp export --format yolo --split
```

---

## Works across the common CV tasks

VisionPack's annotation model carries a tagged geometry, so one tool covers the
tasks you actually use:

| Task | Import | Geometry |
|------|--------|----------|
| **Classification** | ImageFolder (folder-per-class) | whole-image label |
| **Detection** | YOLO, COCO | bounding box |
| **Instance segmentation** | YOLO-seg, COCO | polygon |
| **Semantic segmentation (export)** | — | class-index mask PNGs (`--format masks`) |
| **Keypoints / pose** | COCO | keypoints |

```bash
# classification from a folder-per-class layout
vp init --name product-grades --task classification
vp import ./train --format imagefolder
vp export --format imagefolder --split        # train/val/test/<class>/…

# detection or instance segmentation from COCO
vp init --name cells --task segmentation
vp import ./instances.json --format coco --images ./images
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
vp sync --dry-run     # preview: found / matched / unmatched / classes
vp sync               # ingest; classes merge by name, provenance recorded
```

Classes from different sources merge **by name** (YOLO indices are mapped through
each source's own class order, never positionally), so reordered class lists don't
mislabel your data.

A one-off `vp import` also records what it imported as a source in
`visionpack.yaml`, so the manifest stays the single source of truth and the data
can be re-pulled later with `vp sync` (use `--no-record` for a throwaway import).

Sources can also live in **S3 / GCS / Azure**. Remote URIs go anywhere a local
path would, and a `target:` lets `copy` mode land objects server-side in a
content-addressed bucket without ever downloading them — re-sync is metadata-only.
See the [Cloud Sync guide](https://caiowing.github.io/VisionPack/cloud-sync/).

---

## What's in the box

- **Deterministic, lockable splits** — `stratified` / `random` / `hash`
  (growth-stable), captured in snapshots.
- **Near-duplicate & cross-split leakage detection** — perceptual-hash tier, no
  extra dependencies, scale-proof via LSH bucketing; surfaced in `vp validate`.
- **Multi-source sync** — declarative `sources:` + `vp sync`, with per-asset
  provenance; idempotent re-sync that only pulls what's new.
- **Cloud-native, multi-provider** — sync YOLO, COCO, and ImageFolder sources
  from S3/GCS/Azure without downloading the whole dataset; server-side `copy`
  into a content-addressed target when source and target share a provider,
  single-pass verified relay across providers (S3→GCS, local→S3, …); one
  fast-list instead of per-object lookups, retries with backoff on every remote
  call, tunable concurrency (`--jobs`); streaming export.
- **Content-addressed snapshots & diff** — reproducible versions; compare any two.
- **Strong validation** — unreadable images, missing/orphan labels, unknown
  classes, invalid/out-of-bounds boxes, exact + near duplicates, split leakage.
- **Comparable metrics** — per-split stats so class balance stays auditable as data
  grows.
- **Benchmarking (`vp eval`)** — score model predictions (vp/COCO JSON or YOLO
  txt) against a split's labels: per-class AP@50, mAP@50, mAP@50-95,
  precision/recall — or accuracy + confusion matrix for classification. A locked
  split + a snapshot = a reproducible benchmark.
- **Model-in-the-loop** — `vp autolabel` persists confident predictions as
  annotations (recorded as `source.type = "model"`, never silently overwriting
  human labels); `vp queue` ranks what a human should label next and audits
  existing labels against the model (missing/stale-label signals).
- **Packing & byte-free export** — `archive` (`.tar.zst`) and `training`
  (WebDataset shards); exports hardlink from the CAS or stream from the cloud.
- **Interoperable I/O** — YOLO (incl. YOLO-seg), COCO, ImageFolder in and out;
  semantic masks out.
- **Machine-readable everything** — every pipeline command takes `--json` and
  prints a stable, schema-versioned envelope on stdout, so services, UIs, and CI
  can drive VisionPack without scraping text. See the
  [JSON Output guide](https://caiowing.github.io/VisionPack/json-output/).

Full command reference and per-command options live in the
[CLI guide](https://caiowing.github.io/VisionPack/usage/).

---

## Release process

Releases are prepared locally, reviewed as a GitHub Release draft, and published
to PyPI only when the GitHub Release is published.

```powershell
.\scripts\prepare-release.ps1 0.1.1
git push origin HEAD
git push origin v0.1.1
gh release create v0.1.1 --draft --title "v0.1.1" --notes-file CHANGELOG.md
```

Review the draft release notes in GitHub. Publishing the release triggers
`.github/workflows/publish.yml`, which builds the package, validates the
artifacts with `twine check`, and publishes to PyPI through Trusted Publishing.

Use `-NoCommit -NoTag` to update files and run the checks without creating the
release commit or tag:

```powershell
.\scripts\prepare-release.ps1 0.1.1 -NoCommit -NoTag
```

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
multi-source ingestion (local and cloud) → validation → deterministic splits →
snapshots → ready-to-train export/packing → evaluation (`vp eval`) and
model-in-the-loop labeling (`vp autolabel` / `vp queue`) — works end-to-end
across classification, detection, instance/semantic segmentation, and keypoints,
with 143 passing tests. APIs may still shift; feedback and contributions are
welcome.

```bash
uv run python -m unittest discover -s tests -q
```
