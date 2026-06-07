from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import TARGET_LABELS
from train_densenet.metrics import compute_label_metrics, compute_multilabel_metrics


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
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
            ]
        )
        y_prob = np.array(
            [
                [0.9, 0.2, 0.8, 0.1, 0.2, 0.3],
                [0.2, 0.9, 0.3, 0.3, 0.1, 0.2],
                [0.3, 0.8, 0.7, 0.9, 0.2, 0.3],
                [0.8, 0.2, 0.2, 0.8, 0.1, 0.2],
            ]
        )
        result = compute_multilabel_metrics(
            y_true,
            y_prob,
            {label: 0.5 for label in TARGET_LABELS},
            TARGET_LABELS,
            run_id="unit",
        )
        self.assertEqual(result["run_id"], "unit")
        self.assertIn("raw_no_finding_metrics", result)
        self.assertIn("disease_macro_average_precision", result)
        self.assertIsNone(result["per_label"]["Nodule"]["auroc"])
        self.assertEqual(result["per_label"]["Nodule"]["auroc_reason"], "not_defined_one_class_y_true")


if __name__ == "__main__":
    unittest.main()
