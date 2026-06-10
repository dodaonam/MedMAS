from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import (
    MODEL_NAME,
    TARGET_LABELS,
    RunConfig,
    create_run_id,
    ensure_artifact_tree,
    label_slug,
    labels_for_target_mode,
    load_target_labels,
    resolve_artifact_paths,
    save_json,
    write_run_config,
)


class DenseNetArtifactTests(unittest.TestCase):
    def test_label_slug_is_stable(self) -> None:
        self.assertEqual(label_slug("No Finding"), "no_finding")
        self.assertEqual(label_slug("Atelectasis"), "atelectasis")

    def test_run_id_contains_model_and_seed(self) -> None:
        run_id = create_run_id(seed=0)
        self.assertTrue(run_id.startswith(f"{MODEL_NAME}_disease5_320_asl_seed0_"))

    def test_artifact_tree_and_target_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            paths = resolve_artifact_paths(run_dir)
            ensure_artifact_tree(paths)
            self.assertTrue((paths.figures_dir / "06_gradcam").is_dir())
            save_json(paths.loss_config_path, {"loss": "asymmetric"})
            self.assertTrue(paths.loss_config_path.is_file())

            labels_path = Path(tmp) / "target_labels.json"
            labels_path.write_text(json.dumps({"target_labels": TARGET_LABELS}), encoding="utf-8")
            self.assertEqual(load_target_labels(labels_path), TARGET_LABELS)

            config = RunConfig(run_id="unit", output_dir=str(run_dir))
            write_run_config(paths.config_path, config)
            loaded = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["stage2_name"], "denseblock4_norm5_finetune")
            self.assertEqual(loaded["target_mode"], "disease_only")
            self.assertEqual(loaded["target_labels"], labels_for_target_mode("disease_only"))
            self.assertEqual(loaded["target_label_order"], labels_for_target_mode("disease_only"))
            self.assertEqual(loaded["checkpoint_metric"], "validation_disease_macro_average_precision")
            self.assertEqual(loaded["stage1_config"]["epochs"], 5)
            self.assertEqual(loaded["stage2_config"]["trainable"][-1], "backbone.norm5")
            self.assertIn("cnn_head_config", loaded)
            self.assertIn("batchnorm_policy", loaded)
            self.assertIn("amp_enabled", loaded)

    def test_target_mode_is_disease_only(self) -> None:
        self.assertEqual(labels_for_target_mode("disease_only"), TARGET_LABELS[1:])
        with self.assertRaisesRegex(ValueError, "Unsupported target_mode"):
            labels_for_target_mode("all_labels")


if __name__ == "__main__":
    unittest.main()
