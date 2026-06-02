from __future__ import annotations

import unittest

from visionpack.core.errors import ManifestError
from visionpack.core.manifest import Manifest


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


if __name__ == "__main__":
    unittest.main()
