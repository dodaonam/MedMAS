from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
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
from train_densenet import _backend as backend_module
from train_densenet import config as config_module
from train_densenet import data as data_module
from train_densenet import evaluation as evaluation_module
from train_densenet import modeling as modeling_module
from train_densenet import pipeline as pipeline_module

train_module = types.SimpleNamespace()
for module in (
    backend_module,
    config_module,
    data_module,
    evaluation_module,
    modeling_module,
    pipeline_module,
):
    for name in dir(module):
        if not name.startswith("__"):
            setattr(train_module, name, getattr(module, name))

train_module._config_to_json = config_module.config_to_json
train_module._score_for_checkpoint = evaluation_module.checkpoint_score
train_module._techniques_payload = pipeline_module.techniques_payload


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

    def test_tune_thresholds_support_low_uses_label_specific_fallbacks(self) -> None:
        y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        y_prob = np.array([[0.9, 0.3], [0.8, 0.8], [0.7, 0.4], [0.1, 0.1]])

        thresholds = train_module.tune_thresholds(
            y_true,
            y_prob,
            ["A", "B"],
            default_threshold=0.5,
            low_support_threshold={"A": 0.65, "B": 0.55},
            min_positives_for_tuning=3,
        )

        self.assertEqual(thresholds, {"A": 0.65, "B": 0.55})

    def test_tune_binary_threshold_stays_within_radius_of_prior(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.95, 0.9, 0.25, 0.24, 0.23, 0.22])

        threshold = train_module.tune_binary_threshold(
            y_true,
            y_prob,
            default_threshold=0.5,
            prior_threshold=0.65,
            fallback_threshold=0.65,
        )

        self.assertGreaterEqual(threshold, 0.55)
        self.assertLessEqual(threshold, 0.75)

    def test_threshold_priors_use_label_priors_with_dynamic_infiltration(self) -> None:
        y_true = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 0],
            ]
        )
        y_prob = np.array(
            [
                [0.9, 0.8],
                [0.7, 0.6],
                [0.4, 0.3],
                [0.2, 0.1],
            ]
        )

        thresholds = train_module.threshold_priors(
            y_true,
            y_prob,
            ["Infiltration", "Nodule"],
            default_threshold=0.5,
        )

        self.assertEqual(thresholds["Infiltration"], 0.7)
        self.assertEqual(thresholds["Nodule"], train_module.LABEL_THRESHOLD_PRIORS["Nodule"])

    def test_score_for_checkpoint_prefers_disease_only_macro_average_precision(self) -> None:
        metrics = {
            "macro_average_precision": 0.6,
            "disease_only": {"macro_average_precision": 0.3},
        }

        score = train_module._score_for_checkpoint(metrics, val_loss=1.0)

        self.assertEqual(score, 0.3)

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

    def test_validate_train_config_rejects_invalid_freeze_schedule(self) -> None:
        config = TrainConfig(
            root=ROOT,
            manifest_path=ROOT / "artifacts" / "preprocess" / "split_manifest.csv",
            target_labels_path=ROOT / "artifacts" / "preprocess" / "target_labels.json",
            output_dir=ROOT / "artifacts" / "training" / "densenet121",
            epochs=2,
            freeze_backbone_epochs=2,
        )

        with self.assertRaisesRegex(ValueError, "freeze_backbone_epochs"):
            train_module.validate_train_config(config)

    def test_validate_train_config_rejects_head_only_with_freeze_schedule(self) -> None:
        config = TrainConfig(
            root=ROOT,
            manifest_path=ROOT / "artifacts" / "preprocess" / "split_manifest.csv",
            target_labels_path=ROOT / "artifacts" / "preprocess" / "target_labels.json",
            output_dir=ROOT / "artifacts" / "training" / "densenet121",
            head_only=True,
            freeze_backbone_epochs=1,
        )

        with self.assertRaisesRegex(ValueError, "head_only"):
            train_module.validate_train_config(config)

    @unittest.skipIf(train_module.torch is None, "PyTorch is not installed")
    def test_set_backbone_trainable_freezes_features_only(self) -> None:
        torch = train_module.torch
        assert torch is not None
        model = torch.nn.Module()
        model.features = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        model.classifier = torch.nn.Linear(4, 2)

        train_module.set_backbone_trainable(model, trainable=False)

        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

        train_module.set_backbone_trainable(model, trainable=True)

        self.assertTrue(all(parameter.requires_grad for parameter in model.features.parameters()))

    @unittest.skipIf(train_module.torch is None, "PyTorch is not installed")
    def test_train_one_epoch_keeps_frozen_backbone_in_eval_mode(self) -> None:
        torch = train_module.torch
        assert torch is not None

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 4))
                self.classifier = torch.nn.Linear(4, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                return self.classifier(x)

        model = TinyModel()
        train_module.set_backbone_trainable(model, trainable=False)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = [
            (
                torch.randn(3, 4),
                torch.randint(0, 2, (3, 1), dtype=torch.float32),
                {},
            )
        ]

        train_module.train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            torch.device("cpu"),
            backbone_trainable=False,
            desc="unit",
        )

        self.assertFalse(model.features.training)
        self.assertTrue(model.classifier.training)

    def test_checkpoint_load_allows_script_metadata(self) -> None:
        calls = []

        class FakeTorch:
            def load(self, *args, **kwargs):  # noqa: ANN001
                calls.append((args, kwargs))
                return {"model_state_dict": {}}

        original_torch = config_module.torch
        config_module.torch = FakeTorch()  # type: ignore[assignment]
        try:
            checkpoint = config_module.load_checkpoint(Path("checkpoint_best.pt"), map_location="cpu")
        finally:
            config_module.torch = original_torch

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

    def test_techniques_payload_records_training_choices(self) -> None:
        config = TrainConfig(
            root=Path("/tmp/root"),
            manifest_path=Path("/tmp/manifest.csv"),
            target_labels_path=Path("/tmp/labels.json"),
            output_dir=Path("/tmp/output"),
            classifier_dropout=0.25,
            loss_name="asl",
            ema_decay=0.99,
            freeze_backbone_epochs=2,
            image_size=320,
        )

        payload = train_module._techniques_payload(config, TARGET_LABELS)

        self.assertIn("imagenet_pretrained_backbone", payload["applied_techniques"])
        self.assertIn("classifier_only_warm_start", payload["applied_techniques"])
        self.assertEqual(payload["model"]["classifier_head"]["type"], "dropout_linear")
        self.assertEqual(payload["model"]["classifier_head"]["num_outputs"], len(TARGET_LABELS))
        self.assertEqual(payload["data"]["sampling_strategy"], "weighted_random_sampler")
        self.assertEqual(payload["data"]["transforms"]["train"]["random_resized_crop"]["size"], 320)
        self.assertEqual(payload["optimization"]["backbone_training_strategy"]["name"], "classifier_only_then_full_finetune")
        self.assertEqual(payload["optimization"]["loss"]["asymmetric_loss"]["gamma_neg"], 4.0)
        self.assertEqual(
            payload["evaluation"]["checkpoint_selection_metric"],
            "disease_only_macro_average_precision_then_macro_average_precision_then_negative_val_loss",
        )

    def test_techniques_payload_records_head_only_strategy(self) -> None:
        config = TrainConfig(
            root=Path("/tmp/root"),
            manifest_path=Path("/tmp/manifest.csv"),
            target_labels_path=Path("/tmp/labels.json"),
            output_dir=Path("/tmp/output"),
            head_only=True,
        )

        payload = train_module._techniques_payload(config, TARGET_LABELS)

        self.assertIn("head_only_training", payload["applied_techniques"])
        self.assertEqual(
            payload["optimization"]["backbone_training_strategy"]["name"],
            "classifier_only_frozen_backbone",
        )
        self.assertTrue(payload["optimization"]["backbone_training_strategy"]["head_only"])

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

    @unittest.skipIf(train_module.torch is None, "torch not installed")
    def test_asymmetric_loss_prefers_better_logits(self) -> None:
        criterion = train_module.AsymmetricLoss()
        targets = train_module.torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        better = train_module.torch.tensor([[3.0, -3.0], [-3.0, 3.0]])
        worse = train_module.torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        self.assertLess(float(criterion(better, targets)), float(criterion(worse, targets)))

    @unittest.skipIf(train_module.torch is None, "torch not installed")
    def test_exponential_moving_average_updates_parameters(self) -> None:
        model = train_module.torch.nn.Linear(1, 1, bias=False)
        with train_module.torch.no_grad():
            model.weight.fill_(1.0)
        ema = train_module.ExponentialMovingAverage(model, decay=0.5)
        with train_module.torch.no_grad():
            model.weight.fill_(3.0)

        ema.update(model)

        self.assertAlmostEqual(float(ema.model.weight.item()), 2.0, places=6)

    @unittest.skipIf(train_module.torch is None, "torch not installed")
    def test_exponential_moving_average_copies_non_floating_buffers(self) -> None:
        model = train_module.torch.nn.BatchNorm1d(2)
        ema = train_module.ExponentialMovingAverage(model, decay=0.5)
        with train_module.torch.no_grad():
            model.weight.fill_(3.0)
            model.num_batches_tracked.fill_(7)

        ema.update(model)

        self.assertAlmostEqual(float(ema.model.weight.mean().item()), 2.0, places=6)
        self.assertEqual(int(ema.model.num_batches_tracked.item()), 7)

    @unittest.skipIf(
        train_module.torch is None or importlib.util.find_spec("torchvision") is None,
        "torch or torchvision not installed",
    )
    def test_build_model_uses_dropout_head_when_enabled(self) -> None:
        model = train_module.build_model(2, pretrained=False, classifier_dropout=0.2)

        self.assertEqual(model.classifier.__class__.__name__, "Sequential")
        self.assertEqual(model.classifier[0].__class__.__name__, "Dropout")
        self.assertEqual(model.classifier[1].__class__.__name__, "Linear")


if __name__ == "__main__":
    unittest.main()
