from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import (
    MODEL_NAME,
    TARGET_LABELS,
    RunConfig,
    build_class_weights_payload,
    build_class_weights_payload_from_manifest,
    create_run_id,
    ensure_artifact_tree,
    label_slug,
    load_target_labels,
    resolve_artifact_paths,
    selected_pos_weights_from_payload,
    write_run_config,
)


class DenseNetArtifactTests(unittest.TestCase):
    def test_label_slug_is_stable(self) -> None:
        self.assertEqual(label_slug("No Finding"), "no_finding")
        self.assertEqual(label_slug("Atelectasis"), "atelectasis")

    def test_run_id_contains_model_and_seed(self) -> None:
        run_id = create_run_id(seed=0)
        self.assertTrue(run_id.startswith(f"{MODEL_NAME}_seed0_"))

    def test_artifact_tree_and_target_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            paths = resolve_artifact_paths(run_dir)
            ensure_artifact_tree(paths)
            self.assertTrue((paths.figures_dir / "06_gradcam").is_dir())

            labels_path = Path(tmp) / "target_labels.json"
            labels_path.write_text(json.dumps({"target_labels": TARGET_LABELS}), encoding="utf-8")
            self.assertEqual(load_target_labels(labels_path), TARGET_LABELS)

            config = RunConfig(run_id="unit", output_dir=str(run_dir))
            write_run_config(paths.config_path, config)
            loaded = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["stage2_name"], "denseblock4_norm5_finetune")

    def test_class_weights_are_derived_and_stabilized_from_train_counts(self) -> None:
        labels = ["No Finding", "Mass"]
        payload = build_class_weights_payload(
            target_labels=labels,
            train_positive_counts={"No Finding": 3, "Mass": 1},
            train_negative_counts={"No Finding": 1, "Mass": 19},
        )

        self.assertAlmostEqual(payload["raw_train_only_pos_weight"]["No Finding"], 1 / 3)
        self.assertEqual(payload["selected_clipped_pos_weight"]["No Finding"], 1.0)
        self.assertEqual(payload["raw_train_only_pos_weight"]["Mass"], 19.0)
        self.assertEqual(payload["selected_clipped_pos_weight"]["Mass"], 10.0)
        self.assertEqual(selected_pos_weights_from_payload(payload, labels), {"No Finding": 1.0, "Mass": 10.0})

    def test_class_weights_require_positive_train_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive samples"):
            build_class_weights_payload(
                target_labels=["Mass"],
                train_positive_counts={"Mass": 0},
                train_negative_counts={"Mass": 10},
            )

    def test_class_weights_are_computed_from_train_split_only(self) -> None:
        manifest = pd.DataFrame(
            [
                {"split": "train", "No Finding": 1, "Mass": 1},
                {"split": "train", "No Finding": 1, "Mass": 0},
                {"split": "train", "No Finding": 0, "Mass": 0},
                {"split": "val", "No Finding": 0, "Mass": 1},
                {"split": "test", "No Finding": 0, "Mass": 1},
            ]
        )

        payload = build_class_weights_payload_from_manifest(manifest, ["No Finding", "Mass"])

        self.assertEqual(payload["train_positive_counts"], {"No Finding": 2, "Mass": 1})
        self.assertEqual(payload["train_negative_counts"], {"No Finding": 1, "Mass": 2})
        self.assertAlmostEqual(payload["raw_train_only_pos_weight"]["No Finding"], 0.5)
        self.assertAlmostEqual(payload["raw_train_only_pos_weight"]["Mass"], 2.0)


if __name__ == "__main__":
    unittest.main()
