from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401

    _HAS_SERVER = True
except ModuleNotFoundError:
    _HAS_SERVER = False

from PIL import Image

from visionpack.core.project import Project


def _png_bytes(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (seed * 7 % 256, seed * 13 % 256, seed * 29 % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _wait_for_job(client, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@unittest.skipUnless(_HAS_SERVER, "fastapi/httpx not installed (server extra)")
class ServerApiTest(unittest.TestCase):
    """Drives the `vp serve` app in-process against a real temp project."""

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from visionpack.server.app import create_app

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        Project.init(root, name="webtest", task="detection")

        imgs = root / "raw" / "images"
        lbls = root / "raw" / "labels"
        imgs.mkdir(parents=True)
        lbls.mkdir(parents=True)
        for i in range(3):
            (imgs / f"img{i}.png").write_bytes(_png_bytes(i + 1))
            (lbls / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (lbls / "classes.txt").write_text("cat\n", encoding="utf-8")

        project = Project.open(root)
        project.manifest.sources = [
            {
                "name": "cam-a",
                "images": (root / "raw" / "images").as_posix(),
                "labels": (root / "raw" / "labels").as_posix(),
                "match": "stem",
                "copy": "ingest",
                "credentials": {"key": "SECRET"},
            }
        ]
        project.save_manifest()

        self.root = root
        self.client = TestClient(create_app(root))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _sync(self) -> dict:
        response = self.client.post("/api/sync", json={})
        self.assertEqual(response.status_code, 202)
        job = _wait_for_job(self.client, response.json()["id"])
        self.assertEqual(job["state"], "done", job["error"])
        return job

    def test_project_info(self) -> None:
        info = self.client.get("/api/project").json()
        self.assertEqual(info["name"], "webtest")
        self.assertEqual(info["task"], "detection")
        self.assertEqual(info["counts"]["sources"], 1)

    def test_sources_never_leak_credentials(self) -> None:
        body = self.client.get("/api/sources").text
        self.assertNotIn("SECRET", body)
        sources = self.client.get("/api/sources").json()
        self.assertEqual(sources[0]["name"], "cam-a")
        self.assertEqual(sources[0]["provider"], "local")
        self.assertNotIn("credentials", sources[0])

    def test_sync_plan_previews_without_writing(self) -> None:
        plans = self.client.post("/api/sync/plan", json={}).json()
        self.assertEqual(plans[0]["images_found"], 3)
        self.assertEqual(plans[0]["matched"], 3)
        self.assertEqual(self.client.get("/api/project").json()["counts"]["assets"], 0)

    def test_sync_job_ingests_and_reports(self) -> None:
        job = self._sync()
        self.assertEqual(job["result"][0]["assets_added"], 3)
        self.assertEqual(self.client.get("/api/project").json()["counts"]["assets"], 3)

    def test_assets_page_and_file_bytes(self) -> None:
        self._sync()
        page = self.client.get("/api/assets?limit=2").json()
        self.assertEqual(page["total"], 3)
        self.assertEqual(len(page["items"]), 2)
        asset = page["items"][0]
        self.assertEqual(asset["objects"], 1)
        self.assertEqual(asset["classes"], ["cat"])
        image = self.client.get(f"/api/assets/{asset['id']}/file")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertGreater(len(image.content), 0)

    def test_stats_uses_class_names(self) -> None:
        self._sync()
        stats = self.client.get("/api/stats").json()
        self.assertEqual(stats["overall"]["assets"], 3)
        self.assertIn("cat", stats["overall"]["class_distribution"])

    def test_split_snapshot_flow(self) -> None:
        self._sync()
        split = self.client.post(
            "/api/splits", json={"train": 0.5, "val": 0.25, "test": 0.25, "strategy": "random"}
        ).json()
        self.assertEqual(sum(split["sets"].values()), 3)
        locked = self.client.post("/api/splits/default/lock").json()
        self.assertTrue(locked["locked"])

        snap = self.client.post("/api/snapshots", json={"message": "baseline"}).json()
        self.assertEqual(snap["message"], "baseline")
        snapshots = self.client.get("/api/snapshots").json()
        self.assertEqual(len(snapshots), 1)

    def test_validate_job(self) -> None:
        self._sync()
        response = self.client.post("/api/validate")
        self.assertEqual(response.status_code, 202)
        job = _wait_for_job(self.client, response.json()["id"])
        self.assertEqual(job["state"], "done", job["error"])
        self.assertIn("ok", job["result"])

    def test_export_job(self) -> None:
        self._sync()
        response = self.client.post("/api/export", json={"format": "yolo"})
        self.assertEqual(response.status_code, 202)
        job = _wait_for_job(self.client, response.json()["id"])
        self.assertEqual(job["state"], "done", job["error"])
        self.assertTrue((self.root / "exports" / "yolo").exists())

    def test_unknown_job_is_404(self) -> None:
        self.assertEqual(self.client.get("/api/jobs/job_nope").status_code, 404)

    def test_ui_is_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("VisionPack", response.text)


if __name__ == "__main__":
    unittest.main()
