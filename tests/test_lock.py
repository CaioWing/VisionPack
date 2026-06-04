from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visionpack.core.errors import VisionPackError
from visionpack.core.lock import project_lock


class ProjectLockTest(unittest.TestCase):
    def test_second_acquire_fails_while_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".vp").mkdir()
            with project_lock(root):
                with self.assertRaises(VisionPackError):
                    with project_lock(root):
                        pass

    def test_lock_is_reusable_after_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".vp").mkdir()
            with project_lock(root):
                pass
            # Released — acquiring again must succeed.
            with project_lock(root):
                pass


if __name__ == "__main__":
    unittest.main()
