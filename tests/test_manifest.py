from __future__ import annotations

import unittest
from unittest import mock

from visionpack.core import manifest as manifest_module
from visionpack.core.errors import ManifestError
from visionpack.core.manifest import MANIFEST_VERSION, Manifest


class ManifestValidationTest(unittest.TestCase):
    def test_round_trips_a_valid_manifest(self) -> None:
        original = Manifest.default("factory-defects")
        restored = Manifest.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_empty_sections_fall_back_to_defaults(self) -> None:
        manifest = Manifest.from_dict({"name": "demo", "classes": None, "validation": None})
        self.assertEqual(manifest.classes, [])
        self.assertEqual(manifest.validation, {})

    def test_class_name_defaults_to_id(self) -> None:
        manifest = Manifest.from_dict({"name": "demo", "classes": [{"id": "scratch"}]})
        self.assertEqual(manifest.classes[0].name, "scratch")

    def test_missing_name_is_actionable(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict({"task": "detection"})
        message = str(ctx.exception)
        self.assertIn("name", message)
        self.assertIn("visionpack.yaml is invalid", message)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict({"name": "demo", "validaton": {}})
        self.assertIn("validaton", str(ctx.exception))

    def test_wrong_type_reports_field(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict({"name": "demo", "version": "not-an-int"})
        self.assertIn("version", str(ctx.exception))


class ManifestMigrationTest(unittest.TestCase):
    def test_current_version_passes_through(self) -> None:
        manifest = Manifest.from_dict({"name": "demo", "version": MANIFEST_VERSION})
        self.assertEqual(manifest.version, MANIFEST_VERSION)

    def test_newer_version_is_rejected_with_upgrade_hint(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict({"name": "demo", "version": MANIFEST_VERSION + 1})
        message = str(ctx.exception)
        self.assertIn(f"manifest version {MANIFEST_VERSION + 1}", message)
        self.assertIn("Upgrade visionpack", message)

    def test_older_version_without_migration_is_rejected(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict({"name": "demo", "version": 0})
        self.assertIn("no migration", str(ctx.exception))

    def test_migration_chain_upgrades_old_manifests(self) -> None:
        # Simulate a future schema bump: version 1 files get upgraded through
        # the registered chain before validation, and end at the new version.
        def upgrade_1_to_2(data: dict) -> dict:
            data["task"] = data.pop("kind", data.get("task", "detection"))
            return data

        with (
            mock.patch.object(manifest_module, "MANIFEST_VERSION", 2),
            mock.patch.object(manifest_module, "_MIGRATIONS", {1: upgrade_1_to_2}),
        ):
            data = manifest_module.migrate_manifest_data({"name": "demo", "version": 1, "kind": "segmentation"})
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["task"], "segmentation")
        self.assertNotIn("kind", data)


if __name__ == "__main__":
    unittest.main()
