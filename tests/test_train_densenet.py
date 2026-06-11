from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet import (
    TARGET_LABELS,
    artifact_paths,
    auroc,
    average_precision,
    compute_metrics,
    create_run_id,
    label_slug,
    load_manifest,
    load_target_labels,
    resolve_run_dir,
    save_json,
)


class DenseNetSimpleTests(unittest.TestCase):
    def test_labels_and_run_paths_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_path = tmp_path / "target_labels.json"
            labels_path.write_text(json.dumps({"target_labels": TARGET_LABELS}), encoding="utf-8")

            self.assertEqual(label_slug("No Finding"), "no_finding")
            self.assertTrue(create_run_id(seed=7).startswith("densenet121_seed7_"))
            self.assertEqual(load_target_labels(labels_path), TARGET_LABELS)

            run_dir = tmp_path / "runs" / "unit"
            paths = artifact_paths(run_dir)
            save_json(paths.config_path, {"run_id": "unit"})
            self.assertEqual(resolve_run_dir(run_dir), run_dir)
            self.assertEqual(resolve_run_dir(tmp_path / "runs"), run_dir)

    def test_load_manifest_validates_splits_and_patient_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "split_manifest.csv"
            rows = []
            for split, patient in [("train", "p1"), ("val", "p2"), ("test", "p3")]:
                row = {
                    "Image Index": f"{split}.png",
                    "Patient ID": patient,
                    "split": split,
                    "image_path": f"images/{split}.png",
                    "Patient Gender": "M",
                    "View Position": "PA",
                    "AgeBin": "41-60",
                    "has_out_of_scope_label": False,
                }
                for label in TARGET_LABELS:
                    row[label] = 1 if label == "No Finding" else 0
                rows.append(row)
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

            frame = load_manifest(manifest_path, TARGET_LABELS)
            self.assertEqual(len(frame), 3)

            rows[1]["Patient ID"] = "p1"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_manifest(manifest_path, TARGET_LABELS)

    def test_metrics_handle_multilabel_scores(self) -> None:
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]])

        result = compute_metrics(y_true, y_prob, ["A", "B"], threshold=0.5, run_id="unit")
        self.assertEqual(result["run_id"], "unit")
        self.assertEqual(result["micro"]["tp"], 4)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["macro_average_precision"], 1.0)
        self.assertEqual(result["macro_auroc"], 1.0)
        self.assertEqual(average_precision([0, 0], [0.1, 0.2]), None)
        self.assertEqual(auroc([1, 1], [0.8, 0.9]), None)


if __name__ == "__main__":
    unittest.main()
