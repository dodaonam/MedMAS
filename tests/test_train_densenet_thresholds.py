from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import TARGET_LABELS
from train_densenet.thresholds import select_threshold_for_label, select_validation_thresholds, threshold_grid


class DenseNetThresholdTests(unittest.TestCase):
    def test_threshold_grid_matches_plan(self) -> None:
        grid = threshold_grid()
        self.assertAlmostEqual(float(grid[0]), 0.05)
        self.assertAlmostEqual(float(grid[-1]), 0.95)
        self.assertEqual(len(grid), 91)

    def test_threshold_selection_marks_unstable_low_positive_label(self) -> None:
        result = select_threshold_for_label(
            [1, 1, 0, 0],
            [0.9, 0.8, 0.7, 0.1],
            label="Mass",
            grid=np.arange(0.05, 0.951, 0.01),
        )
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertTrue(result["threshold_unstable"])
        self.assertEqual(result["positive_count"], 2)

    def test_multilabel_threshold_payload_has_required_keys(self) -> None:
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
        payload = select_validation_thresholds(y_true, y_prob, TARGET_LABELS, run_id="unit")
        self.assertEqual(payload["run_id"], "unit")
        self.assertEqual(payload["threshold_grid"], "np.arange(0.05, 0.951, 0.01)")
        self.assertEqual(set(payload["thresholds"]), set(TARGET_LABELS))
        self.assertIn("tp", payload["per_label"]["No Finding"])


if __name__ == "__main__":
    unittest.main()
