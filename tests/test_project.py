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
            self.assertEqual(entries, {".vp", "visionpack.yaml", ".gitignore"})
            # The dead scaffold dirs must not come back.
            for dead in ("assets", "annotations", "exports", "reports"):
                self.assertFalse((root / dead).exists(), f"{dead}/ should not be created by init")

    def test_init_writes_gitignore_for_heavy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "proj"
            Project.init(root, name="proj")
            text = (root / ".gitignore").read_text(encoding="utf-8")
            rules = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
            self.assertIn(".vp/objects/", rules)
            self.assertIn("/exports/", rules)
            self.assertIn("/reports/", rules)
            # The reproducible truth stays tracked (not an ignore rule).
            self.assertNotIn(".vp/db/", rules)
            self.assertNotIn(".vp/snapshots/", rules)

    def test_init_does_not_clobber_existing_gitignore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "proj"
            root.mkdir(parents=True)
            (root / ".gitignore").write_text("custom-rule/\n", encoding="utf-8")
            Project.init(root, name="proj")
            self.assertEqual((root / ".gitignore").read_text(encoding="utf-8"), "custom-rule/\n")

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
