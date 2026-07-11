---
title: JSON Output
nav_order: 6
---

# JSON Output — drive VisionPack from other programs
{: .no_toc }

Every pipeline command accepts `--json` and prints **exactly one
machine-readable JSON document to stdout** — no progress bars, no prose. This
is the stable contract for wrapping VisionPack in another program: a backend
service, a UI, CI, a notebook.

1. TOC
{:toc}

## The envelope

Success:

```json
{
  "schema": 1,
  "command": "sync",
  "data": { "...": "command-specific payload" }
}
```

Failure (the process also exits non-zero):

```json
{
  "schema": 1,
  "command": "diff",
  "error": { "type": "VisionPackError", "message": "No snapshot named 'v99'." }
}
```

Rules a consumer can rely on:

- `schema` bumps **only on a breaking change** to the envelope or an existing
  `data` shape. New fields may appear without a bump — parse leniently.
- Success has `data`; failure has `error` (never both). Check `error` +
  the exit code, not stderr.
- Exit codes keep their CLI meaning: `0` success, `1` domain failure (e.g.
  validation errors, ingest failures), `2` command error.

## Commands and payloads

| Command | `data` highlights |
|---|---|
| `vp sync --json` | `summaries[]` (per source: `assets_added`, `assets_existing`, `annotations`, `objects`, `failures[]`), `total_assets_added`, `total_failures` |
| `vp sync --dry-run --json` | `plans[]` (per source: `images_found`, `labels_found`, `matched`, `class_names[]`) |
| `vp import ... --json` | `assets`, `annotations`, `objects`, `classes_added`, `recorded_source`, `failures[]` |
| `vp validate --json` | `ok`, `errors`, `warnings`, `issues[]` (severity, code, message, asset_id, path) |
| `vp stats --json` | `stats` (counts, `class_distribution`, `resolutions`), `splits` (per-split breakdowns) |
| `vp split create/lock/list/show --json` | `id`, `strategy`, `locked`, `sets` (name → count); `show` adds `asset_ids` |
| `vp snapshot create/list/show --json` | snapshot records (`version`, `message`, `created_at`, `stats`) |
| `vp diff v1 v2 --json` | `assets_added/removed`, `annotations_added/removed/modified`, `classes_added/removed`, `splits_changed` |
| `vp export --json` | `format`, `output`, per-format counts (`images`, `objects`, `sets`, `streamed`) |
| `vp pack --json` | `profile`, `format`, shard/archive counts and paths |
| `vp fsck --json` | `ok`, `mode`, `checked_assets`, `checked_objects`, `issues[]` |
| `vp eval ... --json` | full result: `task`, `scope`, `metrics` (mAP@50, mAP@50-95, accuracy…), `per_class` |
| `vp autolabel ... --json` | `labeled`, `objects`, `skipped_existing`, `skipped_low_confidence`, `unmatched`, `unknown_classes[]` |
| `vp queue --json` | `total`, `items[]` (`asset_id`, `score`, `reasons[]`) |

## Example: a pipeline from a script

```bash
set -e
vp sync --json           > sync.json
vp validate --json       > validate.json || echo "validation found errors"
vp split create --json   > split.json
vp snapshot create -m "auto $(date -I)" --json > snapshot.json

jq '.data.total_assets_added' sync.json
jq '.data.errors'             validate.json
jq '.data.version'            snapshot.json
```

Or from Python, without parsing text:

```python
import json, subprocess

def vp(*argv: str) -> dict:
    proc = subprocess.run(["vp", *argv, "--json"], capture_output=True, text=True)
    envelope = json.loads(proc.stdout)
    if "error" in envelope:
        raise RuntimeError(envelope["error"]["message"])
    return envelope["data"]

added = vp("sync")["total_assets_added"]
report = vp("validate")
```

{: .note }
Prefer the JSON contract over importing `visionpack` internals when driving the
tool from another service: the CLI + envelope is the supported integration
surface, and the `schema` field is the compatibility signal.

## See also

- [CLI Guide]({% link usage.md %}) — the same commands, human-readable.
- [Cloud Sync]({% link cloud-sync.md %}) — remote sources and targets.
