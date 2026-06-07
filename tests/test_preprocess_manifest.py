from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.manifest import build_manifest_all, build_manifest_filtered
from preprocess_image.paths import resolve_paths
from preprocess_image.targets import TARGET_LABELS


class PreprocessManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = resolve_paths(ROOT)
        cls.manifest_all, cls.integrity = build_manifest_all(cls.paths)
        cls.manifest_filtered = build_manifest_filtered(cls.manifest_all)

    def test_manifest_all_integrity(self) -> None:
        self.assertEqual(len(self.manifest_all), 5606)
        self.assertEqual(self.integrity["rows_csv"], 5606)
        self.assertEqual(self.integrity["images_on_disk"], 5606)
        self.assertEqual(self.integrity["csv_images_not_in_disk"], 0)
        self.assertTrue(self.manifest_all["image_path_exists"].all())

    def test_manifest_filtered_counts(self) -> None:
        self.assertEqual(len(self.manifest_filtered), 5101)
        self.assertEqual(len(self.manifest_all) - len(self.manifest_filtered), 505)
        expected_counts = {
            "No Finding": 3044,
            "Infiltration": 967,
            "Effusion": 644,
            "Atelectasis": 508,
            "Nodule": 313,
            "Mass": 284,
        }
        self.assertEqual(self.manifest_filtered[TARGET_LABELS].sum().astype(int).to_dict(), expected_counts)

    def test_filter_keeps_target_plus_out_of_scope(self) -> None:
        retained_with_out_scope = self.manifest_filtered["has_out_of_scope_label"].sum()
        self.assertEqual(int(retained_with_out_scope), 542)
        dropped = self.manifest_all.loc[~self.manifest_all["keep_for_mvp"].astype(bool)]
        self.assertEqual(int(dropped[TARGET_LABELS].sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
