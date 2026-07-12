#!/usr/bin/env python3
"""Scale benchmark for the core VisionPack pipeline.

Generates a synthetic YOLO dataset of N images and times each pipeline stage
(import, validate, split, snapshot, stats, export) through the public SDK —
the same code paths the CLI drives. Use it to spot index or I/O bottlenecks
before they hit real datasets, and to compare runs across versions:

    uv run python scripts/benchmark.py --images 10000
    uv run python scripts/benchmark.py --images 50000 --json > bench.json

The dataset is built in a temporary directory (or --workdir, kept with
--keep). Times are wall-clock; run on an idle machine and compare medians of
a few runs rather than single numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from visionpack.sdk import VisionPackClient  # noqa: E402

CLASSES = ["scratch", "dent", "crack", "stain"]


def generate_dataset(root: Path, count: int, size: int, seed: int = 7) -> Path:
    """A YOLO tree of `count` distinct images with 1-3 boxes each.

    Images get per-image rectangles so their *perceptual* hashes differ too —
    a benchmark of flat images would measure the (linear, but noisy) massive
    near-duplicate-cluster path instead of normal-case throughput.
    """
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    (raw / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    state = seed
    for i in range(count):
        # xorshift-ish deterministic pseudo-randomness; cheap and dependency-free
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        color = (state % 256, (state >> 8) % 256, (state >> 16) % 256)
        image = Image.new("RGB", (size, size), color)
        draw = ImageDraw.Draw(image)
        for r in range(3):
            bits = state >> (3 * r)
            x0, y0 = bits % (size // 2), (bits >> 5) % (size // 2)
            x1, y1 = x0 + 4 + (bits >> 9) % (size // 2), y0 + 4 + (bits >> 13) % (size // 2)
            draw.rectangle((x0, y0, x1, y1), fill=((bits * 37) % 256, (bits * 59) % 256, (bits * 83) % 256))
        image.save(raw / "images" / f"img{i:06d}.png", format="PNG")

        lines = []
        for b in range(1 + state % 3):
            cx = 0.2 + ((state >> (4 + b)) % 60) / 100.0
            cy = 0.2 + ((state >> (7 + b)) % 60) / 100.0
            lines.append(f"{(i + b) % len(CLASSES)} {cx:.3f} {cy:.3f} 0.15 0.15")
        (raw / "labels" / f"img{i:06d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=int, default=2000, help="Number of synthetic images (default 2000)")
    parser.add_argument("--size", type=int, default=64, help="Square image edge in pixels (default 64)")
    parser.add_argument("--workdir", type=Path, default=None, help="Where to build (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="Keep the generated project instead of deleting it")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON on stdout")
    args = parser.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="vp-bench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(stage: str, fn):
        start = time.perf_counter()
        result = fn()
        timings[stage] = time.perf_counter() - start
        if not args.json:
            print(f"  {stage:<18} {timings[stage]:8.2f}s  ({timings[stage] / args.images * 1000:6.2f} ms/img)")
        return result

    if not args.json:
        print(f"VisionPack benchmark: {args.images} images ({args.size}x{args.size}) in {workdir}")

    generate_start = time.perf_counter()
    raw = generate_dataset(workdir, args.images, args.size)
    timings["generate"] = time.perf_counter() - generate_start
    if not args.json:
        print(f"  {'generate':<18} {timings['generate']:8.2f}s  (fixture, not VisionPack)")

    ds = VisionPackClient.init(workdir / "project", name="bench", task="detection")
    timed("import", lambda: ds.import_dir(raw, format="yolo"))
    timed("validate", lambda: ds.validate())
    timed("split", lambda: ds.create_split(train=0.8, val=0.1, test=0.1, strategy="hash"))
    timed("snapshot", lambda: ds.snapshot("bench-baseline"))
    timed("stats", lambda: ds.stats())
    timed("export-yolo", lambda: ds.export(workdir / "export-yolo", format="yolo", split="default"))

    total = sum(seconds for stage, seconds in timings.items() if stage != "generate")
    if args.json:
        print(
            json.dumps(
                {
                    "images": args.images,
                    "image_size": args.size,
                    "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
                    "pipeline_total_seconds": round(total, 3),
                    "ms_per_image": round(total / args.images * 1000, 3),
                },
                indent=2,
            )
        )
    else:
        print(f"  {'pipeline total':<18} {total:8.2f}s  ({total / args.images * 1000:6.2f} ms/img)")

    if args.keep or args.workdir:
        if not args.json:
            print(f"Kept working tree at {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
