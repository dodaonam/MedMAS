from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.targets import TARGET_LABELS, add_target_columns, filter_to_target_scope, parse_labels


class PreprocessTargetTests(unittest.TestCase):
    def test_target_order(self) -> None:
        self.assertEqual(
            TARGET_LABELS,
            ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass"],
        )

    def test_parse_labels(self) -> None:
        self.assertEqual(parse_labels("Mass|Nodule"), ["Mass", "Nodule"])
        self.assertEqual(parse_labels(" No Finding "), ["No Finding"])

    def test_target_filter_rules(self) -> None:
        df = pd.DataFrame(
            {
                "Finding Labels": [
                    "No Finding",
                    "Mass|Pneumothorax",
                    "Pneumothorax",
                ]
            }
        )
        enriched = add_target_columns(df)
        self.assertEqual(enriched.loc[0, TARGET_LABELS].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(enriched.loc[1, "Mass"], 1)
        self.assertTrue(bool(enriched.loc[1, "has_out_of_scope_label"]))
        self.assertFalse(bool(enriched.loc[2, "keep_for_mvp"]))
        filtered = filter_to_target_scope(enriched)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
