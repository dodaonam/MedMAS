from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import (
    DISEASE_LABELS,
    TARGET_LABELS,
    RunConfig,
    label_slug,
    resolve_artifact_paths,
    save_json,
    write_run_config,
)
from train_densenet.metrics import compute_multilabel_metrics
from train_densenet.thresholds import select_validation_thresholds, thresholds_by_label
from visualize_densenet.data import load_visualization_artifacts, resolve_run_dir
from visualize_densenet.plots_subgroups import out_of_scope_error_count_table


def _prediction_frame(run_id: str) -> pd.DataFrame:
    y_true = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    y_prob = np.array(
        [
            [0.2, 0.8, 0.1, 0.2, 0.3],
            [0.9, 0.3, 0.3, 0.1, 0.2],
            [0.8, 0.7, 0.9, 0.2, 0.3],
            [0.2, 0.2, 0.8, 0.1, 0.2],
        ]
    )
    thresholds = thresholds_by_label(select_validation_thresholds(y_true, y_prob, DISEASE_LABELS, run_id=run_id), DISEASE_LABELS)
    rows = []
    for idx in range(y_true.shape[0]):
        row = {
            "run_id": run_id,
            "Image Index": f"img_{idx}.png",
            "Patient ID": str(idx),
            "split": "test",
            "image_path": f"missing/img_{idx}.png",
            "Patient Gender": "M" if idx % 2 else "F",
            "View Position": "PA",
            "AgeBin": "41-60",
            "has_out_of_scope_label": False,
            "true_no_finding_source": int(np.sum(y_true[idx, :]) == 0),
        }
        disease_preds = []
        for label_idx, label in enumerate(DISEASE_LABELS):
            slug = label_slug(label)
            row[label] = int(y_true[idx, label_idx])
            row[f"true_{slug}"] = int(y_true[idx, label_idx])
            row[f"logit_{slug}"] = float(y_prob[idx, label_idx])
            row[f"prob_{slug}"] = float(y_prob[idx, label_idx])
            row[f"threshold_{slug}"] = float(thresholds[label])
            row[f"pred_{slug}"] = int(y_prob[idx, label_idx] >= thresholds[label])
            disease_preds.append(row[f"pred_{slug}"])
        row["true_no_finding_derived"] = int(np.sum(y_true[idx, :]) == 0)
        row["pred_no_finding_derived"] = int(sum(disease_preds) == 0)
        rows.append(row)
    return pd.DataFrame(rows)


class DenseNetVisualizationDataTests(unittest.TestCase):
    def test_resolve_run_dir_accepts_run_dir_or_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "densenet121_cnn_head"
            older = base_dir / "older"
            newer = base_dir / "newer"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "config.json").write_text("{}", encoding="utf-8")
            (newer / "config.json").write_text("{}", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            self.assertEqual(resolve_run_dir(older), older)
            self.assertEqual(resolve_run_dir(base_dir), newer)

    def test_load_visualization_artifacts_validates_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "unit_run"
            run_dir = Path(tmp) / "densenet121_cnn_head"
            paths = resolve_artifact_paths(run_dir)
            paths.output_dir.mkdir(parents=True)
            write_run_config(paths.config_path, RunConfig(run_id=run_id, output_dir=str(run_dir)))
            save_json(
                paths.loss_config_path,
                {
                    "loss": "asymmetric",
                    "gamma_pos": 0,
                    "gamma_neg": 4,
                    "clip": 0.05,
                    "eps": 1e-8,
                    "reduction": "mean",
                },
            )
            history = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "epoch": 1,
                        "stage": "cnn_head_only",
                        "train_loss": 1.0,
                        "val_loss": 0.9,
                        "learning_rate_cnn_head": 3e-4,
                        "learning_rate_denseblock4_norm5": np.nan,
                        "val_disease_macro_auroc": 0.5,
                        "val_disease_macro_average_precision": 0.4,
                    }
                ]
            )
            history.to_csv(paths.training_history_path, index=False)
            frame = _prediction_frame(run_id)
            frame.to_csv(paths.predictions_val_path, index=False)
            frame.to_csv(paths.predictions_test_path, index=False)
            y_true = frame[[f"true_{label_slug(label)}" for label in DISEASE_LABELS]].to_numpy(dtype=int)
            y_prob = frame[[f"prob_{label_slug(label)}" for label in DISEASE_LABELS]].to_numpy(dtype=float)
            thresholds = select_validation_thresholds(y_true, y_prob, DISEASE_LABELS, run_id=run_id)
            save_json(paths.thresholds_path, thresholds)
            metrics = compute_multilabel_metrics(y_true, y_prob, thresholds_by_label(thresholds, DISEASE_LABELS), DISEASE_LABELS, run_id=run_id)
            save_json(paths.metrics_val_path, metrics)
            save_json(paths.metrics_test_path, metrics)

            labels_path = Path(tmp) / "target_labels.json"
            labels_path.write_text(json.dumps({"target_labels": TARGET_LABELS}), encoding="utf-8")
            artifacts = load_visualization_artifacts(run_dir, labels_path)
            self.assertEqual(artifacts.config["run_id"], run_id)
            self.assertEqual(artifacts.target_labels, DISEASE_LABELS)
            self.assertEqual(len(artifacts.predictions_test), 4)

            save_json(paths.loss_config_path, {"loss": "asymmetric"})
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                load_visualization_artifacts(run_dir, labels_path)

    def test_out_of_scope_error_count_table_splits_fp_and_fn(self) -> None:
        rows = []
        for out_of_scope, true_value, pred_value in [
            (False, 0, 1),
            (True, 1, 0),
        ]:
            row = {"has_out_of_scope_label": out_of_scope}
            for label in DISEASE_LABELS:
                slug = label_slug(label)
                row[f"true_{slug}"] = 0
                row[f"pred_{slug}"] = 0
            slug = label_slug("Infiltration")
            row[f"true_{slug}"] = true_value
            row[f"pred_{slug}"] = pred_value
            rows.append(row)

        table = out_of_scope_error_count_table(pd.DataFrame(rows), DISEASE_LABELS)
        item = table.loc[table["label"] == "Infiltration"].iloc[0]
        self.assertEqual(int(item["in_scope_fp"]), 1)
        self.assertEqual(int(item["out_of_scope_fn"]), 1)


if __name__ == "__main__":
    unittest.main()
