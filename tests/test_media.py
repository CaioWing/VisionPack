from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionpack.core.errors import FormatError
from visionpack.media import image_info, image_info_from_bytes, is_image_path


class MediaProbeTest(unittest.TestCase):
    def test_webp_is_probed_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.webp"
            self.assertTrue(is_image_path(path))
            Image.new("RGB", (120, 80), color=(10, 20, 30)).save(path, format="WEBP")
            width, height, channels, image_format = image_info(path)
            self.assertEqual((width, height), (120, 80))
            self.assertEqual(channels, 3)
            self.assertEqual(image_format, "webp")

    def test_exif_orientation_swaps_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            img = Image.new("RGB", (100, 50), color=(0, 0, 0))
            exif = img.getexif()
            exif[0x0112] = 6  # rotate 90deg: displayed as 50x100
            img.save(path, exif=exif)
            width, height, _, _ = image_info(path)
            self.assertEqual((width, height), (50, 100))

    def test_bytes_and_path_probes_agree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (64, 32), color=(255, 255, 255)).save(path, format="PNG")
            self.assertEqual(image_info(path), image_info_from_bytes(path.read_bytes(), path))

    def test_unreadable_file_raises_format_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.png"
            path.write_bytes(b"not really a png")
            with self.assertRaises(FormatError):
                image_info(path)


if __name__ == "__main__":
    unittest.main()
