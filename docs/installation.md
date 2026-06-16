---
title: Installation
nav_order: 2
---

# Installation
{: .no_toc }

1. TOC
{:toc}

## Requirements

- **Python 3.11+**
- Works on Linux, macOS, and Windows.

## Install from PyPI

```bash
pip install visionpack
```

This installs the `vp` command (and a `visionpack` alias). Verify it:

```bash
vp --version
vp --help
```

{: .tip }
Prefer an isolated install? [`pipx`](https://pipx.pypa.io) keeps the CLI off your
global environment: `pipx install visionpack`.

## Cloud backends (optional extras)

Reading from or writing to object stores pulls in [`fsspec`](https://filesystem-spec.readthedocs.io)
plus the provider library. They're **optional** — the core never imports them, so
a local install stays lean. Install only what you need:

```bash
pip install "visionpack[s3]"      # Amazon S3   (s3fs)
pip install "visionpack[gcs]"     # Google Cloud Storage (gcsfs)
pip install "visionpack[azure]"   # Azure Blob  (adlfs)
```

You can combine them: `pip install "visionpack[s3,gcs]"`. See {% link cloud-sync.md %}
for declaring remote sources and a cloud target.

## Develop from source

VisionPack uses [`uv`](https://github.com/astral-sh/uv). From the repo root:

```bash
git clone https://github.com/CaioWing/VisionPack
cd VisionPack
uv sync                 # create the venv and install dependencies
uv run vp --help        # run the CLI
```

Run the test suite:

```bash
uv run python -m unittest discover -s tests
```

You can also invoke the module directly without the console script:

```bash
uv run python -m visionpack --help
```

## Next

- [Quickstart]({% link quickstart.md %}) — build a dataset end to end.
- [CLI Guide]({% link usage.md %}) — the full command reference.
