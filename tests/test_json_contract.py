from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from visionpack.cli.main import main
from visionpack.cli.output import SCHEMA_VERSION
from visionpack.core.project import Project


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(path, format="PNG")


class JsonContractTest(unittest.TestCase):
    """`--json` prints exactly one schema-versioned envelope on stdout.

    This is the machine contract external programs (a frontend backend, CI)
    build against, so the assertions pin the envelope keys and each command's
    load-bearing data fields.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        Project.init(self.root, name="jsonctl", task="detection")
        for index in range(1, 4):
            _png(self.root / "raw" / "images" / f"img{index}.png", index)
            label = self.root / "raw" / "labels" / f"img{index}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (self.root / "raw" / "labels" / "classes.txt").write_text("cat\n", encoding="utf-8")
        project = Project.open(self.root)
        project.manifest.sources = [
            {"name": "cam-a", "images": "./raw/images", "labels": "./raw/labels", "match": "stem", "copy": "ingest"}
        ]
        project.save_manifest()
        self._cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _run(self, *argv: str) -> tuple[int, dict]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(list(argv))
        output = buffer.getvalue()
        try:
            envelope = json.loads(output)
        except json.JSONDecodeError as exc:  # pragma: no cover - assertion aid
            raise AssertionError(f"stdout is not a single JSON document:\n{output}") from exc
        self.assertEqual(envelope["schema"], SCHEMA_VERSION)
        return code, envelope

    def _sync(self) -> None:
        code, _ = self._run("sync", "--json")
        self.assertEqual(code, 0)

    def test_sync_dry_run(self) -> None:
        code, envelope = self._run("sync", "--dry-run", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["command"], "sync")
        self.assertTrue(envelope["data"]["dry_run"])
        plan = envelope["data"]["plans"][0]
        self.assertEqual(plan["images_found"], 3)
        self.assertEqual(plan["matched"], 3)
        self.assertEqual(plan["class_names"], ["cat"])

    def test_sync(self) -> None:
        code, envelope = self._run("sync", "--json")
        self.assertEqual(code, 0)
        data = envelope["data"]
        self.assertEqual(data["total_assets_added"], 3)
        self.assertEqual(data["total_failures"], 0)
        self.assertEqual(data["summaries"][0]["name"], "cam-a")

    def test_stats(self) -> None:
        self._sync()
        code, envelope = self._run("stats", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["data"]["stats"]["assets"], 3)
        self.assertIn("splits", envelope["data"])

    def test_validate(self) -> None:
        self._sync()
        code, envelope = self._run("validate", "--json")
        data = envelope["data"]
        self.assertEqual(code, 0 if data["ok"] else 1)
        self.assertIn("issues", data)
        self.assertEqual(data["errors"], sum(1 for i in data["issues"] if i["severity"] == "error"))

    def test_split_create_list_show_lock(self) -> None:
        self._sync()
        code, envelope = self._run("split", "create", "--train", "0.5", "--val", "0.25", "--test", "0.25", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["command"], "split.create")
        self.assertEqual(sum(envelope["data"]["sets"].values()), 3)

        code, envelope = self._run("split", "list", "--json")
        self.assertEqual(envelope["data"]["splits"][0]["id"], "default")

        code, envelope = self._run("split", "show", "--json")
        self.assertTrue(envelope["data"]["found"])
        self.assertIn("asset_ids", envelope["data"])

        code, envelope = self._run("split", "lock", "--json")
        self.assertTrue(envelope["data"]["locked"])

    def test_snapshot_create_list_diff(self) -> None:
        self._sync()
        code, envelope = self._run("snapshot", "create", "-m", "baseline", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["command"], "snapshot.create")
        version = envelope["data"]["version"]

        code, envelope = self._run("snapshot", "list", "--json")
        self.assertEqual(len(envelope["data"]["snapshots"]), 1)

        code, envelope = self._run("snapshot", "create", "-m", "second", "--json")
        second = envelope["data"]["version"]
        code, envelope = self._run("diff", version, second, "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["command"], "diff")
        self.assertEqual(envelope["data"]["left"], version)
        self.assertEqual(envelope["data"]["assets_added"], [])

    def test_export(self) -> None:
        self._sync()
        code, envelope = self._run("export", "--format", "yolo", "--output", "exports/yolo", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(envelope["data"]["images"], 3)
        self.assertTrue((self.root / "exports" / "yolo").exists())

    def test_fsck(self) -> None:
        self._sync()
        code, envelope = self._run("fsck", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(envelope["data"]["ok"])
        self.assertEqual(envelope["data"]["checked_assets"], 3)

    def test_error_envelope(self) -> None:
        code, envelope = self._run("diff", "v98", "v99", "--json")
        self.assertEqual(code, 2)
        self.assertEqual(envelope["command"], "diff")
        self.assertNotIn("data", envelope)
        self.assertIn("message", envelope["error"])
        self.assertTrue(envelope["error"]["type"])


if __name__ == "__main__":
    unittest.main()
