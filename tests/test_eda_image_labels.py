from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_image.labels import (
    build_multi_hot,
    compute_cooccurrence,
    compute_label_summary,
    compute_pairwise_associations,
    get_all_labels,
)


class LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "Finding Labels": [
                    "No Finding",
                    "Mass|Nodule",
                    "Mass|Effusion",
                ]
            }
        )
        self.df["LabelList"] = self.df["Finding Labels"].str.split("|")

    def test_multi_hot_and_summary(self) -> None:
        labels = get_all_labels(self.df)
        multi_hot = build_multi_hot(self.df, labels)
        summary = compute_label_summary(multi_hot)
        self.assertEqual(int(summary.loc["Mass", "count"]), 2)
        self.assertEqual(int(summary.loc["No Finding", "count"]), 1)
        self.assertEqual(int(multi_hot.loc[1, "Nodule"]), 1)

    def test_cooccurrence_and_pairwise(self) -> None:
        labels = get_all_labels(self.df)
        multi_hot = build_multi_hot(self.df, labels)
        co = compute_cooccurrence(multi_hot)
        self.assertEqual(int(co.loc["Mass", "Nodule"]), 1)
        pairs = compute_pairwise_associations(multi_hot)
        self.assertIn("lift", pairs.columns)
        self.assertGreaterEqual(len(pairs), 2)


if __name__ == "__main__":
    unittest.main()

