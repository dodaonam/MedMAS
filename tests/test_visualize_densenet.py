from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet import TARGET_LABELS, artifact_paths, label_slug, save_json
from visualize_densenet.data import load_visualization_artifacts
from visualize_densenet.plots import create_training_plots, prediction_threshold


class DenseNetVisualizationTests(unittest.TestCase):
    def test_load_visualization_artifacts_reads_new_run_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "unit"
            run_dir = Path(tmp) / "densenet121" / run_id
            paths = artifact_paths(run_dir)
            paths.run_dir.mkdir(parents=True)
            save_json(paths.config_path, {"run_id": run_id, "target_labels": TARGET_LABELS})
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "epoch": 1,
                        "train_loss": 1.0,
                        "val_loss": 0.8,
                        "learning_rate": 1e-4,
                        "val_macro_average_precision": 0.5,
                        "val_macro_auroc": 0.6,
                        "val_macro_f1": 0.4,
                    }
                ]
            ).to_csv(paths.history_path, index=False)
            prediction = _prediction_frame(run_id)
            prediction.to_csv(paths.predictions_val_path, index=False)
            prediction.to_csv(paths.predictions_test_path, index=False)
            metrics = {
                "run_id": run_id,
                "labels": TARGET_LABELS,
                "threshold": 0.5,
                "per_label": {
                    label: {"tp": 1, "fp": 0, "fn": 0, "tn": 1, "average_precision": 1.0, "auroc": 1.0, "f1": 1.0}
                    for label in TARGET_LABELS
                },
            }
            save_json(paths.metrics_val_path, metrics)
            save_json(paths.metrics_test_path, metrics)
            labels_path = Path(tmp) / "target_labels.json"
            labels_path.write_text(json.dumps({"target_labels": TARGET_LABELS}), encoding="utf-8")

            artifacts = load_visualization_artifacts(run_dir, labels_path)
            self.assertEqual(artifacts.config["run_id"], run_id)
            self.assertEqual(artifacts.target_labels, TARGET_LABELS)
            self.assertEqual(len(artifacts.predictions_test), 2)

            save_json(paths.metrics_test_path, {**metrics, "run_id": "other"})
            with self.assertRaisesRegex(ValueError, "Run ID mismatch"):
                load_visualization_artifacts(run_dir, labels_path)

    def test_create_training_plots_includes_learning_rate_curve_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figures_dir = Path(tmp) / "figures"
            history = pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        "val_loss": 0.9,
                        "learning_rate": 1e-5,
                        "val_macro_average_precision": 0.4,
                        "val_macro_auroc": 0.5,
                        "val_macro_f1": 0.3,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.8,
                        "val_loss": 0.7,
                        "learning_rate": 1e-4,
                        "val_macro_average_precision": 0.6,
                        "val_macro_auroc": 0.7,
                        "val_macro_f1": 0.5,
                    },
                ]
            )

            paths = create_training_plots(history, figures_dir)
            self.assertEqual(
                sorted(path.name for path in paths),
                ["learning_rate_curve.png", "loss_curve.png", "validation_metrics.png"],
            )
            for path in paths:
                self.assertTrue(path.is_file(), str(path))

    def test_prediction_threshold_reads_saved_per_label_value(self) -> None:
        predictions = _prediction_frame("unit")
        predictions["threshold_no_finding"] = 0.7
        predictions["threshold_infiltration"] = 0.35

        self.assertEqual(prediction_threshold(predictions, "No Finding"), 0.7)
        self.assertEqual(prediction_threshold(predictions, "Infiltration"), 0.35)


def _prediction_frame(run_id: str) -> pd.DataFrame:
    rows = []
    for idx in range(2):
        row = {
            "run_id": run_id,
            "Image Index": f"img_{idx}.png",
            "Patient ID": f"p{idx}",
            "split": "test",
            "image_path": f"images/img_{idx}.png",
            "Patient Gender": "F",
            "View Position": "PA",
            "AgeBin": "41-60",
            "has_out_of_scope_label": False,
        }
        for label in TARGET_LABELS:
            slug = label_slug(label)
            value = int(idx == 0)
            row[label] = value
            row[f"true_{slug}"] = value
            row[f"prob_{slug}"] = 0.8 if value else 0.2
            row[f"threshold_{slug}"] = 0.5
            row[f"pred_{slug}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
