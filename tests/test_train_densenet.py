from __future__ import annotations

import json
import importlib.util
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
    TrainConfig,
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
import train_densenet.train as train_module


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

    def test_tune_thresholds_and_metrics_support_per_label_thresholds(self) -> None:
        y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        y_prob = np.array([[0.9, 0.3], [0.8, 0.8], [0.7, 0.4], [0.1, 0.1]])

        thresholds = train_module.tune_thresholds(y_true, y_prob, ["A", "B"], default_threshold=0.5)
        self.assertEqual(thresholds, {"A": 0.8, "B": 0.4})

        metrics = compute_metrics(y_true, y_prob, ["A", "B"], threshold=thresholds, run_id="unit")
        self.assertEqual(metrics["threshold"], None)
        self.assertEqual(metrics["threshold_mode"], "per_label")
        self.assertEqual(metrics["thresholds"], thresholds)
        self.assertEqual(metrics["per_label"]["A"]["threshold"], 0.8)
        self.assertEqual(metrics["per_label"]["B"]["threshold"], 0.4)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_tune_thresholds_falls_back_to_default_when_support_is_too_low(self) -> None:
        y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        y_prob = np.array([[0.9, 0.3], [0.8, 0.8], [0.7, 0.4], [0.1, 0.1]])

        thresholds = train_module.tune_thresholds(
            y_true,
            y_prob,
            ["A", "B"],
            default_threshold=0.5,
            min_positives_for_tuning=3,
        )

        self.assertEqual(thresholds, {"A": 0.5, "B": 0.5})

    def test_attach_slice_metrics_adds_disease_only_and_scope_splits(self) -> None:
        labels = TARGET_LABELS
        y_true = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0],
            ]
        )
        y_prob = np.array(
            [
                [0.9, 0.1, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.8, 0.7, 0.3, 0.9, 0.2],
                [0.1, 0.2, 0.1, 0.8, 0.2, 0.8],
                [0.8, 0.2, 0.1, 0.1, 0.1, 0.1],
            ]
        )
        metadata_rows = [
            {"Image Index": "a", "has_out_of_scope_label": False},
            {"Image Index": "b", "has_out_of_scope_label": True},
            {"Image Index": "c", "has_out_of_scope_label": False},
            {"Image Index": "d", "has_out_of_scope_label": False},
        ]
        thresholds = {label: 0.5 for label in labels}

        frame = train_module.prediction_frame(metadata_rows, y_true, y_prob, labels, threshold=thresholds, run_id="unit")
        metrics = compute_metrics(y_true, y_prob, labels, threshold=thresholds, run_id="unit")
        enriched = train_module.attach_slice_metrics(metrics, frame, labels, threshold=thresholds)

        self.assertEqual(enriched["disease_only"]["labels"], labels[1:])
        self.assertEqual(enriched["subsets"]["in_scope_only"]["row_count"], 3)
        self.assertEqual(enriched["subsets"]["out_of_scope_only"]["row_count"], 1)
        self.assertEqual(enriched["subsets"]["out_of_scope_only"]["metrics"]["per_label"]["No Finding"]["positive_count"], 0)

    def test_checkpoint_load_allows_script_metadata(self) -> None:
        calls = []

        class FakeTorch:
            def load(self, *args, **kwargs):  # noqa: ANN001
                calls.append((args, kwargs))
                return {"model_state_dict": {}}

        original_torch = train_module.torch
        train_module.torch = FakeTorch()  # type: ignore[assignment]
        try:
            checkpoint = train_module.load_checkpoint(Path("checkpoint_best.pt"), map_location="cpu")
        finally:
            train_module.torch = original_torch

        self.assertEqual(checkpoint, {"model_state_dict": {}})
        self.assertEqual(calls[0][1]["weights_only"], False)

    def test_config_to_json_includes_scheduler_fields(self) -> None:
        config = TrainConfig(
            root=Path("/tmp/root"),
            manifest_path=Path("/tmp/manifest.csv"),
            target_labels_path=Path("/tmp/labels.json"),
            output_dir=Path("/tmp/output"),
        )

        payload = train_module._config_to_json(config)
        self.assertEqual(payload["epochs"], 20)
        self.assertEqual(payload["warmup_epochs"], 2)
        self.assertEqual(payload["warmup_start_factor"], 0.1)
        self.assertEqual(payload["min_lr"], 1e-6)
        self.assertEqual(payload["balanced_sampler"], True)

    @unittest.skipIf(train_module.torch is None, "torch not installed")
    def test_balanced_sample_weights_prioritize_rare_positive_labels(self) -> None:
        rows = []
        for index in range(6):
            row = {
                "split": "train",
                "No Finding": 1 if index < 4 else 0,
                "Infiltration": 0,
                "Effusion": 0,
                "Atelectasis": 0,
                "Nodule": 1 if index == 4 else 0,
                "Mass": 1 if index == 5 else 0,
            }
            rows.append(row)
        frame = pd.DataFrame(rows)

        weights = train_module.balanced_sample_weights(frame, TARGET_LABELS).numpy()

        self.assertLess(weights[0], weights[4])
        self.assertLess(weights[0], weights[5])

        all_common = frame.copy()
        all_common["No Finding"] = 1
        all_common[["Nodule", "Mass"]] = 0
        self.assertTrue(np.all(train_module.balanced_sample_weights(all_common, TARGET_LABELS).numpy() > 0))

    @unittest.skipIf(
        train_module.torch is None or importlib.util.find_spec("torchvision") is None,
        "torch or torchvision not installed",
    )
    def test_build_dataloaders_uses_balanced_sampler_when_enabled(self) -> None:
        rows = []
        for split in ["train", "train", "val", "test"]:
            row = {
                "split": split,
                "image_path": "unused.png",
                "No Finding": 1,
                "Infiltration": 0,
                "Effusion": 0,
                "Atelectasis": 0,
                "Nodule": 0,
                "Mass": 0,
            }
            rows.append(row)
        frame = pd.DataFrame(rows)
        config = TrainConfig(
            root=Path("/tmp/root"),
            manifest_path=Path("/tmp/manifest.csv"),
            target_labels_path=Path("/tmp/labels.json"),
            output_dir=Path("/tmp/output"),
            batch_size=2,
            num_workers=0,
        )

        train_loader, val_loader, test_loader = train_module.build_dataloaders(config, frame, TARGET_LABELS)

        self.assertEqual(train_loader.sampler.__class__.__name__, "WeightedRandomSampler")
        self.assertNotEqual(val_loader.sampler.__class__.__name__, "WeightedRandomSampler")
        self.assertNotEqual(test_loader.sampler.__class__.__name__, "WeightedRandomSampler")

    @unittest.skipIf(train_module.torch is None, "torch not installed")
    def test_build_lr_scheduler_warmup_then_cosine_uses_min_lr_in_last_epoch(self) -> None:
        parameter = train_module.torch.nn.Parameter(train_module.torch.tensor(1.0))
        optimizer = train_module.torch.optim.AdamW([parameter], lr=1e-4)
        config = TrainConfig(
            root=Path("/tmp/root"),
            manifest_path=Path("/tmp/manifest.csv"),
            target_labels_path=Path("/tmp/labels.json"),
            output_dir=Path("/tmp/output"),
            epochs=20,
            warmup_epochs=2,
            warmup_start_factor=0.1,
            min_lr=1e-6,
        )

        scheduler = train_module.build_lr_scheduler(optimizer, config)
        used_lrs = []
        for _ in range(config.epochs):
            used_lrs.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(used_lrs[0], 1e-5, places=12)
        self.assertAlmostEqual(used_lrs[1], 5.5e-5, places=12)
        self.assertAlmostEqual(used_lrs[2], 1e-4, places=12)
        self.assertAlmostEqual(used_lrs[-1], 1e-6, places=12)


if __name__ == "__main__":
    unittest.main()
