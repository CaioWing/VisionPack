"""Guards for the stable surfaces named in COMPATIBILITY.md.

These tests exist to make breaking a stable surface a *decision* (this file
changes in the same commit, reviewers see it) instead of an accident. Adding
new methods/fields is always fine; removing or renaming what is pinned here
is the breaking case.
"""

from __future__ import annotations

import unittest

from visionpack.cli.output import SCHEMA_VERSION
from visionpack.core.manifest import MANIFEST_VERSION
from visionpack.sdk import VisionPackClient

# The SDK's public API per COMPATIBILITY.md §4. Removing or renaming any of
# these is a breaking change: deprecate first, and update COMPATIBILITY.md and
# CHANGELOG.md in the same release.
SDK_PUBLIC_METHODS = {
    # lifecycle
    "init",
    "open",
    "project",
    "root",
    "name",
    "task",
    "classes",
    "readonly",
    # data access
    "assets",
    "annotations",
    "samples",
    # ingest
    "import_dir",
    "sync",
    # quality
    "validate",
    "audit",
    "stats",
    # splits
    "create_split",
    "lock_split",
    "split",
    # versions
    "snapshot",
    "snapshots",
    "snapshots_by_tag",
    "tag_snapshot",
    "untag_snapshot",
    "checkout",
    "diff",
    "drift",
    # output
    "export",
    # model loop
    "evaluate",
    "autolabel",
    "annotation_queue",
}


class CompatibilityTest(unittest.TestCase):
    def test_sdk_public_surface_is_intact(self) -> None:
        available = {name for name in dir(VisionPackClient) if not name.startswith("_")}
        missing = SDK_PUBLIC_METHODS - available
        self.assertEqual(missing, set(), f"SDK public API lost members (breaking change): {sorted(missing)}")

    def test_schema_versions_change_consciously(self) -> None:
        # Bumping either version is sometimes right — but it must be deliberate:
        # update this test, COMPATIBILITY.md, and the CHANGELOG together.
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(MANIFEST_VERSION, 1)

    def test_error_envelope_shape(self) -> None:
        import io
        import json
        from contextlib import redirect_stdout

        from visionpack.cli.output import emit_json_error

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            emit_json_error("validate", ValueError("boom"))
        envelope = json.loads(buffer.getvalue())
        self.assertEqual(set(envelope), {"schema", "command", "error"})
        self.assertEqual(envelope["error"]["type"], "ValueError")
        self.assertEqual(envelope["error"]["message"], "boom")


if __name__ == "__main__":
    unittest.main()
