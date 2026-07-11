from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_expected import TRAINING_RECIPE, TrainConfig, default_config
from train_expected.train import TARGET_LABELS, build_config_payload, compute_metrics, tune_thresholds


class TrainExpectedTests(unittest.TestCase):
    def test_default_config_is_fixed_training_recipe(self) -> None:
        config = default_config(ROOT)

        for key in [
            "seed",
            "epochs",
            "batch_size",
            "num_workers",
            "image_size",
            "lr",
            "weight_decay",
            "warmup_epochs",
            "warmup_start_factor",
            "min_lr",
            "threshold",
            "pretrained",
            "device",
        ]:
            self.assertEqual(getattr(config, key), TRAINING_RECIPE[key])

        with self.assertRaises(TypeError):
            TrainConfig(
                root=ROOT,
                manifest_path=ROOT / "artifacts" / "preprocess" / "split_manifest.csv",
                target_labels_path=ROOT / "artifacts" / "preprocess" / "target_labels.json",
                output_dir=ROOT / "artifacts" / "training" / "densenet121",
                epochs=20,
            )

    def test_config_payload_describes_new_training_run(self) -> None:
        config = default_config(ROOT)
        payload = build_config_payload(
            config,
            TARGET_LABELS,
            "densenet121_seed0_unit",
            Path("artifacts/training/densenet121/densenet121_seed0_unit/checkpoint_best.pt"),
            "cuda",
        )

        self.assertEqual(payload["run_id"], "densenet121_seed0_unit")
        self.assertEqual(payload["model_name"], "densenet121")
        self.assertEqual(payload["lr_scheduler"], "linear_warmup_cosine_annealing")
        self.assertEqual(payload["cosine_t_max"], 8)
        self.assertEqual(payload["threshold_strategy"], "per_label_f1_from_val_bounded_by_label_priors_min_positives_50")
        self.assertEqual(payload["recipe"]["loss"], "bce_with_logits_pos_weight")
        self.assertEqual(payload["recipe"]["classifier_head"], "linear")

    def test_metrics_and_threshold_tuning_are_local(self) -> None:
        y_true = [[1, 0], [1, 1], [0, 1], [0, 0]]
        y_prob = [[0.9, 0.3], [0.8, 0.8], [0.7, 0.4], [0.1, 0.1]]

        thresholds = tune_thresholds(
            y_true,
            y_prob,
            ["A", "B"],
            default_threshold=0.5,
            low_support_threshold={"A": 0.5, "B": 0.5},
            min_positives_for_tuning=0,
        )
        metrics = compute_metrics(y_true, y_prob, ["A", "B"], threshold=thresholds, run_id="unit")

        self.assertEqual(metrics["run_id"], "unit")
        self.assertEqual(metrics["threshold_mode"], "per_label")
        self.assertEqual(metrics["macro_average_precision"], 1.0)

    def test_module_has_no_train_densenet_dependency(self) -> None:
        source = (ROOT / "src" / "train_expected" / "train.py").read_text(encoding="utf-8")

        self.assertNotIn("train_densenet", source)


if __name__ == "__main__":
    unittest.main()
