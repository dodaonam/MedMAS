from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_image.image_quality import (
    ahash_bits,
    compute_or_load_image_metrics,
    load_gray,
)


class ImageQualityTests(unittest.TestCase):
    def test_load_gray_from_l_and_rgba(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            l_path = tmp_path / "gray.png"
            rgba_path = tmp_path / "rgba.png"
            gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
            rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
            Image.fromarray(gray, mode="L").save(l_path)
            Image.fromarray(rgba, mode="RGBA").save(rgba_path)
            self.assertEqual(load_gray(l_path).shape, (4, 4))
            self.assertTrue(np.array_equal(load_gray(l_path), load_gray(rgba_path)))

    def test_ahash_deterministic(self) -> None:
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.assertEqual(ahash_bits(arr), ahash_bits(arr))
        self.assertEqual(len(ahash_bits(arr)), 64)

    def test_metrics_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_dir = tmp_path / "images"
            img_dir.mkdir()
            arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
            Image.fromarray(arr, mode="L").save(img_dir / "a.png")
            metrics_path = tmp_path / "metrics.csv"
            meta_path = tmp_path / "metrics.meta.json"
            metrics, status = compute_or_load_image_metrics(["a.png"], img_dir, metrics_path, meta_path)
            self.assertFalse(bool(status.loc[status["metric"] == "cache_hit", "value"].iloc[0]))
            self.assertEqual(len(metrics), 1)
            cached, status_cached = compute_or_load_image_metrics(["a.png"], img_dir, metrics_path, meta_path)
            self.assertTrue(bool(status_cached.loc[status_cached["metric"] == "cache_hit", "value"].iloc[0]))
            self.assertEqual(len(cached), 1)


if __name__ == "__main__":
    unittest.main()

