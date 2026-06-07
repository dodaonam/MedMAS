from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.manifest import build_manifest_all, build_manifest_filtered
from preprocess_image.paths import resolve_paths
from preprocess_image.split import assign_patient_split, patient_overlap_matrix, search_patient_split, target_counts_by_split, validate_split
from preprocess_image.targets import TARGET_LABELS


class PreprocessSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths = resolve_paths(ROOT)
        manifest_all, _ = build_manifest_all(paths)
        manifest_filtered = build_manifest_filtered(manifest_all)
        cls.result = search_patient_split(manifest_filtered)
        cls.split_manifest = cls.result.manifest

    def test_patient_split_passes_gates(self) -> None:
        passes, diagnostics = validate_split(self.split_manifest)
        self.assertTrue(passes, diagnostics)
        self.assertLessEqual(self.result.selected_seed, 100)
        self.assertEqual(set(self.split_manifest["split"].unique()), {"train", "val", "test"})

    def test_patient_overlap_is_zero(self) -> None:
        overlap = patient_overlap_matrix(self.split_manifest)
        off_diagonal = int(overlap.values.sum() - overlap.values.diagonal().sum())
        self.assertEqual(off_diagonal, 0)

    def test_all_targets_exist_in_each_split(self) -> None:
        counts = target_counts_by_split(self.split_manifest, TARGET_LABELS)
        self.assertTrue((counts[TARGET_LABELS] > 0).all().all())

    def test_patient_split_accepts_custom_ratios(self) -> None:
        frame = self.split_manifest.drop(columns=["split"]).drop_duplicates("Patient ID").head(10).copy()
        split = assign_patient_split(frame, seed=0, train_frac=0.60, val_frac=0.20)
        patient_counts = split.groupby("split")["Patient ID"].nunique().to_dict()

        self.assertEqual(patient_counts, {"test": 2, "train": 6, "val": 2})

    def test_patient_split_rejects_invalid_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than 1"):
            assign_patient_split(self.split_manifest, seed=0, train_frac=0.80, val_frac=0.20)


if __name__ == "__main__":
    unittest.main()
