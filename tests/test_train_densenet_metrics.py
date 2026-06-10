from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import DISEASE_LABELS
from train_densenet.artifacts import label_slug
from train_densenet.metrics import (
    checkpoint_candidate_improved,
    compute_label_metrics,
    compute_multilabel_metrics,
    compute_subgroup_metrics_from_frame,
)


class DenseNetMetricsTests(unittest.TestCase):
    def test_label_metrics_include_one_class_guard(self) -> None:
        result = compute_label_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
        self.assertIsNone(result["auroc"])
        self.assertEqual(result["auroc_reason"], "not_defined_one_class_y_true")
        self.assertIsNone(result["average_precision"])
        self.assertEqual(result["average_precision_reason"], "not_defined_one_class_y_true")

    def test_multilabel_metrics_preserve_disease_macro(self) -> None:
        y_true = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        y_prob = np.array(
            [
                [0.2, 0.8, 0.1, 0.2, 0.3],
                [0.9, 0.3, 0.3, 0.1, 0.2],
                [0.8, 0.7, 0.9, 0.2, 0.3],
                [0.2, 0.2, 0.8, 0.1, 0.2],
                [0.1, 0.2, 0.2, 0.1, 0.1],
            ]
        )
        result = compute_multilabel_metrics(
            y_true,
            y_prob,
            {label: 0.5 for label in DISEASE_LABELS},
            DISEASE_LABELS,
            run_id="unit",
        )
        self.assertEqual(result["run_id"], "unit")
        self.assertIn("derived_no_finding_metrics", result)
        self.assertIn("disease_macro_average_precision", result)
        self.assertIn("disease_micro_f1", result)
        self.assertIsNone(result["per_label"]["Nodule"]["auroc"])
        self.assertEqual(result["per_label"]["Nodule"]["auroc_reason"], "not_defined_one_class_y_true")

    def test_multilabel_metrics_reject_direct_no_finding_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "No Finding"):
            compute_multilabel_metrics([[0, 1]], [[0.2, 0.8]], {"No Finding": 0.5, "Mass": 0.5}, ["No Finding", "Mass"])

    def test_subgroup_metrics_include_ranking_metrics_when_reported(self) -> None:
        rows = []
        for group, true_value, prob_value in [
            ("A", 1, 0.9),
            ("A", 0, 0.1),
            ("B", 1, 0.8),
            ("B", 0, 0.2),
        ]:
            row = {"group": group}
            for label in DISEASE_LABELS:
                slug = label_slug(label)
                row[f"true_{slug}"] = 0
                row[f"prob_{slug}"] = 0.1
                row[f"threshold_{slug}"] = 0.5
            slug = label_slug("Infiltration")
            row[f"true_{slug}"] = true_value
            row[f"prob_{slug}"] = prob_value
            rows.append(row)

        result = compute_subgroup_metrics_from_frame(pd.DataFrame(rows), "group", DISEASE_LABELS, min_positives=1)
        item = result["groups"]["A"]["per_label"]["Infiltration"]
        self.assertTrue(item["reported"])
        self.assertEqual(item["average_precision"], 1.0)
        self.assertEqual(item["auroc"], 1.0)

    def test_checkpoint_candidate_requires_min_delta(self) -> None:
        self.assertTrue(
            checkpoint_candidate_improved(
                candidate_ap=0.25,
                best_ap=-np.inf,
                min_delta=0.001,
                has_best=False,
            )
        )
        self.assertFalse(
            checkpoint_candidate_improved(
                candidate_ap=0.2505,
                best_ap=0.25,
                min_delta=0.001,
                has_best=True,
                candidate_auroc=0.9,
                best_auroc=0.5,
                candidate_val_loss=0.1,
                best_val_loss=1.0,
            )
        )
        self.assertTrue(
            checkpoint_candidate_improved(
                candidate_ap=0.252,
                best_ap=0.25,
                min_delta=0.001,
                has_best=True,
                candidate_auroc=0.4,
                best_auroc=0.5,
                candidate_val_loss=1.0,
                best_val_loss=0.9,
            )
        )


if __name__ == "__main__":
    unittest.main()
