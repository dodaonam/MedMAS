from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_image.metadata import add_metadata_features, parse_age_to_years
from eda_image.paths import resolve_paths


class MetadataTests(unittest.TestCase):
    def test_parse_age_to_years(self) -> None:
        self.assertEqual(parse_age_to_years("060Y"), (60.0, "Y"))
        self.assertAlmostEqual(parse_age_to_years("013M")[0], 13 / 12)
        self.assertEqual(parse_age_to_years("013M")[1], "M")
        self.assertAlmostEqual(parse_age_to_years("001D")[0], 1 / 365)
        self.assertEqual(parse_age_to_years("001D")[1], "D")
        self.assertTrue(parse_age_to_years("bad")[0] != parse_age_to_years("bad")[0])

    def test_resolve_paths_uses_existing_png_dir(self) -> None:
        paths = resolve_paths(ROOT)
        self.assertTrue(paths.csv_path.exists())
        self.assertTrue(paths.image_dir.exists())
        self.assertGreater(len(list(paths.image_dir.glob("*.png"))), 0)

    def test_add_metadata_features(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Patient Age": ["060Y", "013M"],
                "Follow-up #": ["1", "2"],
                "Finding Labels": ["No Finding", "Mass|Nodule"],
                "OriginalImageWidth": [2500, 2992],
                "OriginalImageHeight": [2048, 2991],
            }
        )
        enriched = add_metadata_features(df)
        self.assertIn("AgeYears", enriched)
        self.assertIn("LabelList", enriched)
        self.assertEqual(enriched.loc[1, "NumLabels"], 2)
        self.assertEqual(enriched.loc[0, "OriginalResolution"], "2500x2048")


if __name__ == "__main__":
    unittest.main()

