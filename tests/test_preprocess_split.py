from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.manifest import build_manifest_all, build_manifest_filtered
from preprocess_image.paths import resolve_paths
from preprocess_image.split import patient_overlap_matrix, search_patient_split, target_counts_by_split, validate_split
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


if __name__ == "__main__":
    unittest.main()
