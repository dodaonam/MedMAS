from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_preprocess_artifacts.py"


def load_entrypoint_module():
    spec = importlib.util.spec_from_file_location("generate_preprocess_artifacts", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PreprocessEntrypointTests(unittest.TestCase):
    def test_entrypoint_generates_required_artifacts(self) -> None:
        module = load_entrypoint_module()
        with tempfile.TemporaryDirectory() as tmp:
            summary = module.generate_artifacts(artifacts_dir=Path(tmp), train_frac=0.60, val_frac=0.20)

            self.assertEqual(summary["rows_all"], 5606)
            self.assertEqual(summary["rows_filtered"], 5101)

            required_files = {
                "preprocess_config.json",
                "target_labels.json",
                "manifest_all.csv",
                "manifest_filtered.csv",
                "split_manifest.csv",
                "split_audit.json",
                "figures/split_label_counts.png",
                "figures/split_label_prevalence.png",
                "figures/split_view_distribution.png",
                "figures/split_gender_distribution.png",
                "figures/split_age_distribution.png",
            }
            for relative_path in required_files:
                artifact_path = Path(tmp) / relative_path
                self.assertTrue(artifact_path.exists(), artifact_path)
                self.assertGreater(artifact_path.stat().st_size, 0)

            config = json.loads((Path(tmp) / "preprocess_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["split_ratio"], {"train": 0.60, "val": 0.20, "test": 0.20})


if __name__ == "__main__":
    unittest.main()
