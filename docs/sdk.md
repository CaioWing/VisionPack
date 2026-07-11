---
title: Python SDK
nav_order: 7
---

# Python SDK

Everything the CLI does is available from Python through the SDK — the same
project on disk, the same locking, the same results the `--json` envelopes
carry, without subprocess plumbing. Use it from notebooks, training scripts,
labeling services, or CI jobs:

```python
from visionpack.sdk import VisionPackClient

ds = VisionPackClient.init("./factory-defects", task="detection")   # or .open(".")
ds.import_dir("./raw", format="yolo")

report = ds.validate()                  # ValidationReport (errors/warnings/ok)
audit = ds.audit()                      # AuditReport (label-health findings)
print(ds.stats()["class_distribution"])

ds.create_split(train=0.8, val=0.1, test=0.1, strategy="stratified")
ds.lock_split()
ds.snapshot("baseline")
ds.export("./exports/yolo", format="yolo", split="default")
```

Close the model-in-the-loop cycle with the same handle:

```python
metrics = ds.evaluate("runs/predict/labels", format="yolo")   # mAP on the test set
ds.autolabel("preds.json", min_confidence=0.6)                # confident preds -> labels
for item in ds.annotation_queue("preds.json")[:20]:           # what to label next
    print(item["score"], item["path"])
```

## The client

| Area | Methods |
|------|---------|
| Lifecycle | `VisionPackClient.init(root, name=..., task=...)`, `VisionPackClient.open(root)` (also `sdk.init` / `sdk.open`) |
| Ingest | `import_dir(source, format="yolo"\|"coco"\|"imagefolder", images=..., copy_mode=...)`, `sync(source=..., jobs=...)`, `plan_sync()` |
| Quality | `validate(strict=...)`, `audit(**thresholds)`, `stats()`, `split_stats()` |
| Splits | `create_split(...)`, `lock_split()`, `split()` |
| Versions | `snapshot(message)`, `snapshots()`, `get_snapshot(version)`, `checkout(version)`, `diff(v1, v2)`, `drift(v1, v2)`, `tag_snapshot(v, tag)`, `untag_snapshot(v, tag)`, `snapshots_by_tag(tag)` |
| Outputs | `export(output, format="yolo"\|"coco"\|"imagefolder"\|"masks", split=..., seg=...)` |
| Model loop | `load_predictions(...)`, `evaluate(...)`, `autolabel(...)`, `annotation_queue(...)` |
| Data access | `assets()`, `annotations()`, `samples()` (streaming iterator), `len(ds)`, `for asset, ann in ds:` |

## Guarantees

- **Safe next to the CLI.** Every mutating method takes the same project lock
  `vp` takes, so an SDK caller and a CLI process can never corrupt each
  other's writes — the second writer fails fast with a clear error.
- **Stable, JSON-friendly returns.** Summaries come back as plain dicts that
  mirror the [`--json` contract]({% link json-output.md %}), so a service can
  switch between shelling out to `vp` and importing the SDK without
  re-parsing anything.
- **Read-only snapshot views.** `ds.checkout("v2")` returns a client pinned to
  that snapshot: exports, stats, and evaluation reflect the frozen state, and
  mutating methods raise instead of silently writing into live history.

## Versions, drift, and lineage

```python
ds.snapshot("after week-30 batch")            # v5
print(ds.drift("v4", "v5")["js_divergence"])  # did the class mix shift?

# after training, link the run to the exact dataset version it consumed:
ds.tag_snapshot("v5", "trained:run-812")
ds.snapshots_by_tag("trained:")               # every version a model trained on
```

## Streaming a dataset

`samples()` iterates straight off the index without materializing it, so a
training-adjacent script can walk 100k+ assets in bounded memory:

```python
for asset, annotation in ds.samples():
    if annotation is None:
        continue
    for obj in annotation.objects:
        box = obj.bbox          # enclosing BBox for any geometry
        ...
```

The lower-level `Project` object stays reachable as `ds.project` for anything
the facade doesn't cover yet.
