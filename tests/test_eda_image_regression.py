from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_image.image_quality import compute_or_load_image_metrics, summarize_ahash_collisions
from eda_image.labels import build_multi_hot, compute_label_summary, compute_multilabel_summary, get_all_labels
from eda_image.metadata import add_metadata_features, compute_integrity_summary, load_metadata
from eda_image.paths import resolve_paths
from eda_image.splits import compute_patient_dynamics, simulate_split_leakage


class EdaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = resolve_paths(ROOT)
        cls.baseline = json.loads(cls.paths.baseline_path.read_text())
        cls.df_raw = load_metadata(cls.paths.csv_path)
        cls.df = add_metadata_features(cls.df_raw)
        cls.labels = get_all_labels(cls.df)
        cls.multi_hot = build_multi_hot(cls.df, cls.labels)

    def test_integrity_and_labels_match_baseline(self) -> None:
        integrity = compute_integrity_summary(self.df_raw, self.paths.image_dir).set_index("metric")["value"]
        label_summary = compute_label_summary(self.multi_hot)
        multilabel = compute_multilabel_summary(self.df, len(self.labels)).set_index("metric")["value"]
        self.assertEqual(int(integrity["rows_csv"]), self.baseline["csv_rows"])
        self.assertEqual(int(integrity["images_on_disk"]), self.baseline["images_on_disk"])
        self.assertEqual(len(self.labels), self.baseline["unique_labels"])
        self.assertEqual(int(label_summary.loc["No Finding", "count"]), self.baseline["no_finding_count"])
        self.assertEqual(int(label_summary.loc["Hernia", "count"]), self.baseline["hernia_count"])
        self.assertAlmostEqual(float(multilabel["label_cardinality"]), self.baseline["label_cardinality"], places=6)

    def test_patient_and_split_metrics_match_baseline(self) -> None:
        dynamics = compute_patient_dynamics(self.df).set_index("metric")["value"]
        sim = simulate_split_leakage(self.df, self.multi_hot, repeats=80, seed=123)
        sim_summary = sim.groupby("split_type").agg(["mean", "std", "min", "max"])
        self.assertEqual(int(dynamics["unique_patients"]), self.baseline["unique_patients"])
        self.assertEqual(int(dynamics["patients_with_multivisit"]), self.baseline["patients_with_multivisit"])
        self.assertAlmostEqual(
            float(sim_summary.loc["image_random", ("patient_overlap", "mean")]),
            self.baseline["image_random_patient_overlap_mean"],
            places=4,
        )
        self.assertEqual(
            float(sim_summary.loc["patient_wise", ("patient_overlap", "mean")]),
            self.baseline["patient_wise_patient_overlap_mean"],
        )

    def test_image_cache_metrics_match_baseline(self) -> None:
        image_metrics, status = compute_or_load_image_metrics(
            self.df["Image Index"].tolist(),
            self.paths.image_dir,
            self.paths.image_metrics_path,
            self.paths.image_metrics_meta_path,
            force=False,
        )
        self.assertTrue(bool(status.loc[status["metric"] == "cache_hit", "value"].iloc[0]))
        self.assertEqual(len(image_metrics), self.baseline["image_metrics_cache_rows"])
        self.assertEqual(int((image_metrics["mode"] == "L").sum()), self.baseline["image_mode_l_count"])
        self.assertEqual(int((image_metrics["mode"] == "RGBA").sum()), self.baseline["image_mode_rgba_count"])
        df_img = self.df.merge(image_metrics, on="Image Index", how="left")
        ahash_summary, _ = summarize_ahash_collisions(image_metrics, df_img)
        ahash = ahash_summary.set_index("metric")["value"]
        self.assertEqual(int(ahash["collision_groups"]), self.baseline["ahash_collision_groups"])
        self.assertEqual(int(ahash["images_in_collision_groups"]), self.baseline["ahash_images_in_collision_groups"])


if __name__ == "__main__":
    unittest.main()

