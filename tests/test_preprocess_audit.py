from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.audit import create_all_plots, create_split_audit
from preprocess_image.manifest import build_manifest_all, build_manifest_filtered
from preprocess_image.paths import resolve_paths
from preprocess_image.split import search_patient_split


class PreprocessAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths = resolve_paths(ROOT)
        cls.manifest_all, cls.integrity = build_manifest_all(paths)
        cls.manifest_filtered = build_manifest_filtered(cls.manifest_all)
        cls.result = search_patient_split(cls.manifest_filtered)
        cls.audit = create_split_audit(
            cls.manifest_all,
            cls.manifest_filtered,
            cls.result.manifest,
            cls.integrity,
            cls.result.selected_seed,
        )

    def test_audit_contains_required_summaries(self) -> None:
        required = {
            "selected_seed",
            "integrity_summary",
            "rows_per_split",
            "patients_per_split",
            "patient_overlap_matrix",
            "positive_count_per_target_per_split",
            "prevalence_per_target_per_split",
            "dropped_row_summary",
        }
        self.assertTrue(required.issubset(self.audit))
        self.assertEqual(self.audit["dropped_row_summary"]["retained_rows"], 5101)
        self.assertEqual(self.audit["dropped_row_summary"]["dropped_out_of_scope_only_rows"], 505)

    def test_audit_plots_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plot_paths = create_all_plots(self.result.manifest, Path(tmp))
            self.assertEqual(len(plot_paths), 5)
            for plot_path in plot_paths:
                self.assertTrue(plot_path.exists(), plot_path)
                self.assertGreater(plot_path.stat().st_size, 0)

    def test_preprocess_notebook_has_required_outputs(self) -> None:
        notebook_candidates = [
            ROOT / "notebook" / "preprocess_ready_for_training.ipynb",
            ROOT / "notebook" / "images_preprocessing.ipynb",
        ]
        notebook_path = next((path for path in notebook_candidates if path.exists()), None)
        self.assertIsNotNone(notebook_path, f"Missing preprocess notebook: {notebook_candidates}")
        assert notebook_path is not None
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        required_tags = {
            "integrity_summary",
            "filtering_funnel",
            "retained_examples",
            "dropped_examples",
            "target_counts",
            "split_counts",
            "patient_overlap",
            "target_split_plots",
            "view_distribution",
            "gender_distribution",
            "age_distribution",
            "labels_per_image",
            "images_per_patient",
            "readiness_checklist",
        }
        outputs_by_tag = {}
        for cell in notebook.get("cells", []):
            tags = set(cell.get("metadata", {}).get("tags", []))
            if tags & required_tags:
                outputs_by_tag.update({tag: len(cell.get("outputs", [])) for tag in tags & required_tags})
        missing_tags = required_tags - set(outputs_by_tag)
        empty_tags = {tag for tag, output_count in outputs_by_tag.items() if output_count == 0}
        self.assertFalse(missing_tags, f"Missing notebook tags: {sorted(missing_tags)}")
        self.assertFalse(empty_tags, f"Tagged notebook cells without outputs: {sorted(empty_tags)}")


if __name__ == "__main__":
    unittest.main()
