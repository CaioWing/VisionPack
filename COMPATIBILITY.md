# Compatibility policy

VisionPack is built to be driven by other software — CI pipelines, services,
training scripts. This document says exactly what those consumers may rely on,
and how breaking changes are communicated. The stable surfaces below are
guarded by tests (`tests/test_json_contract.py`, `tests/test_compatibility.py`,
`tests/test_manifest.py`), so breaking them accidentally fails CI.

## Versioning

VisionPack follows [semantic versioning](https://semver.org). While the
project is pre-1.0:

- **Patch releases** (`0.x.Y`) never break any stable surface.
- **Minor releases** (`0.X.0`) may break a stable surface, but only with a
  `CHANGELOG.md` entry that names the break and the migration path.
- From 1.0 on, breaking changes to stable surfaces happen only in major
  releases.

Deprecations are announced at least one minor release before removal: the old
form keeps working and warns, the changelog names the replacement.

## Stable surfaces

### 1. The JSON envelope (`--json`)

Every pipeline command accepts `--json` and prints exactly one envelope on
stdout ([docs](https://caiowing.github.io/VisionPack/json-output/)):

```json
{"schema": 1, "command": "validate", "data": { ... }}
{"schema": 1, "command": "validate", "error": {"type": "...", "message": "..."}}
```

- `schema` (`SCHEMA_VERSION` in `visionpack/cli/output.py`) is bumped **only**
  on a breaking change to the envelope or to an existing command's `data`
  shape. Adding new fields or new commands does not bump it.
- Consumers must tolerate unknown fields, and should dispatch on `schema`,
  `command`, and the presence of `error` plus the process exit code — never on
  stderr text.
- Human-facing (non-`--json`) output is **not** a stable surface; do not
  parse it.

### 2. The manifest (`visionpack.yaml`)

- The manifest's `version` field is the manifest **schema** version
  (`MANIFEST_VERSION` in `visionpack/core/manifest.py`).
- A manifest written by any released VisionPack keeps opening in every later
  release: older versions are upgraded in memory through a migration chain.
- A manifest from a *newer* VisionPack is rejected with an explicit
  "upgrade visionpack" error — it is never misparsed silently.

### 3. The CLI

Command names, their documented flags, and their exit-code convention
(`0` success, non-zero failure, `--json` errors still enveloped) are stable.
Flags may gain new optional values; defaults change only in minor releases
with a changelog entry.

### 4. The Python SDK (`visionpack.sdk`)

The public API is what `visionpack.sdk` exports — `VisionPackClient` and its
public methods. Method signatures may gain keyword-only parameters with
defaults; removing or renaming a public method or changing a return shape is a
breaking change. Every other module under `visionpack.*` is internal: import
it if you like, but it can change in any release.

### 5. Project state on disk (`.vp/`)

The `.vp/` directory layout is internal. What *is* guaranteed: a project
created by an older release keeps working with newer releases (open, validate,
export, snapshot history intact). Reading `.vp/` files directly, or writing
into them, is unsupported — use the CLI or SDK.

## What is explicitly not stable

- Human-readable terminal output, progress rendering, log text.
- The order of items in reports and listings, unless documented.
- Internal modules, private helpers, and the index database schema.
- The exact set of validation/audit finding messages (their *codes* — e.g.
  `asset.near_duplicate` — are stable; message wording is not).
