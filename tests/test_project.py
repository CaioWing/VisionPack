from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visionpack.core.project import Project


class ProjectInitTest(unittest.TestCase):
    def test_init_creates_git_like_layout_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "proj"
            Project.init(root, name="proj")

            entries = {p.name for p in root.iterdir()}
            self.assertEqual(entries, {".vp", "visionpack.yaml"})
            # The dead scaffold dirs must not come back.
            for dead in ("assets", "annotations", "exports", "reports"):
                self.assertFalse((root / dead).exists(), f"{dead}/ should not be created by init")

    def test_vp_control_dir_has_the_real_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "proj"
            Project.init(root, name="proj")
            vp = root / ".vp"
            self.assertTrue((vp / "db" / "index.json").exists())
            self.assertTrue((vp / "objects").is_dir())
            self.assertTrue((vp / "snapshots").is_dir())


if __name__ == "__main__":
    unittest.main()
