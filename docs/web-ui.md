---
title: Web UI & API
nav_order: 6
---

# Web UI & REST API — `vp serve`
{: .no_toc }

A local dashboard that makes the whole pipeline — sources → sync → validate →
split → snapshot → export — point-and-click, backed by a REST API you can also
script against. The manifest and index remain the single source of truth: the
UI is a *view* over the same operations the CLI runs, never a second brain.

1. TOC
{:toc}

## Install & run

The web stack is an optional extra, like the cloud backends:

```bash
pip install "visionpack[server]"

cd my-dataset/     # any directory with a visionpack.yaml
vp serve           # opens http://127.0.0.1:8123
```

Options: `--host` (default `127.0.0.1`, local only), `--port` (default `8123`),
`--no-browser`.

Interactive API docs (OpenAPI) are served at `/api/docs`.

## What the UI gives you

- **Pipeline board** — the six dataset stages as cards with one-click actions,
  so the workflow is legible to people who don't live in the terminal.
- **Sources** — every declared source with its provider (S3 / GCS / Azure /
  local), format, join rule, and copy mode; per-source sync and a **dry-run
  preview** (found / matched / unmatched / classes) before anything is written.
- **Dataset view** — asset/object/class counts, per-class object distribution,
  split composition, and a thumbnail gallery that streams images from wherever
  they live (local CAS or any cloud provider) through one endpoint.
- **Versions** — splits (create / lock) and snapshots (create), the same
  deterministic machinery as the CLI.
- **Jobs** — long operations (sync, validate, export) run as background jobs
  with live progress; the UI polls and reports results and failures.

## Concurrency & safety

- Mutating operations take the same **project lock** as the CLI, so a `vp sync`
  in a terminal and a sync from the UI can't corrupt the index — one of them
  fails fast with a clear message.
- The job runner executes **one job at a time** (mirroring the single-writer
  lock); a second request answers `409` immediately instead of queueing.
- **Credentials never leave the server.** Source and target `credentials`
  blocks are stripped from every API response.
- The server binds to `127.0.0.1` by default. If you expose it on a network,
  put it behind your own auth — it is a single-user tool by design.

## REST API

Everything the UI does is plain JSON over HTTP:

| Method & path | What it does |
|---|---|
| `GET /api/project` | Name, task, classes, counts, target |
| `GET /api/sources` | Declared sources (credentials stripped) |
| `POST /api/sync/plan` | Dry-run: found / matched / unmatched per source |
| `POST /api/sync` | Start a sync job (`{"source": "camera-A"}` for one source) |
| `POST /api/validate` | Start a validation job |
| `POST /api/export` | Start an export job (`format`, optional `split`, `output`) |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Job state, progress, result |
| `GET /api/stats` | Dataset stats + per-split breakdowns |
| `GET /api/assets?offset&limit` | Paginated assets with label summaries |
| `GET /api/assets/{id}/file` | Image bytes — local CAS or streamed from the cloud |
| `GET/POST /api/splits`, `POST /api/splits/{id}/lock` | Deterministic splits |
| `GET/POST /api/snapshots` | Content-addressed versions |

Jobs return `202` with an id; poll `GET /api/jobs/{id}` until `state` is
`done` or `error`. Example:

```bash
job=$(curl -s -X POST localhost:8123/api/sync -H 'content-type: application/json' -d '{}' | jq -r .id)
curl -s localhost:8123/api/jobs/$job | jq '{state, done, total}'
```

## See also

- [Cloud Sync]({% link cloud-sync.md %}) — multi-provider sources and targets.
- [CLI Guide]({% link usage.md %}) — the same operations from the terminal.
