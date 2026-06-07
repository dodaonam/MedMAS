from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_image.audit import create_all_plots, create_split_audit, write_split_audit
from preprocess_image.manifest import build_manifest_all, build_manifest_filtered, write_preprocess_config
from preprocess_image.paths import ensure_workspace, resolve_paths
from preprocess_image.split import search_patient_split, validate_split
from preprocess_image.targets import TARGET_LABELS, write_target_labels


def generate_artifacts(
    *,
    start: Path | None = None,
    artifacts_dir: Path | None = None,
    initial_start: int = 0,
    initial_end: int = 100,
    fallback_end: int = 1000,
) -> dict[str, object]:
    paths = resolve_paths(start or ROOT, artifacts_dir=artifacts_dir)
    ensure_workspace(paths)

    manifest_all, integrity_summary = build_manifest_all(paths)
    manifest_filtered = build_manifest_filtered(manifest_all)
    split_result = search_patient_split(
        manifest_filtered,
        target_labels=TARGET_LABELS,
        initial_start=initial_start,
        initial_end=initial_end,
        fallback_end=fallback_end,
    )
    split_ok, split_diagnostics = validate_split(split_result.manifest, TARGET_LABELS)
    if not split_ok:
        raise RuntimeError(f"Selected split failed validation: {split_diagnostics}")

    audit = create_split_audit(
        manifest_all,
        manifest_filtered,
        split_result.manifest,
        integrity_summary,
        split_result.selected_seed,
        TARGET_LABELS,
    )

    write_preprocess_config(paths.config_path)
    write_target_labels(paths.target_labels_path, TARGET_LABELS)
    manifest_all.to_csv(paths.manifest_all_path, index=False)
    manifest_filtered.to_csv(paths.manifest_filtered_path, index=False)
    split_result.manifest.to_csv(paths.split_manifest_path, index=False)
    write_split_audit(paths.split_audit_path, audit)
    plot_paths = create_all_plots(split_result.manifest, paths.figures_dir)

    return {
        "artifacts_dir": str(paths.artifacts_dir),
        "selected_seed": split_result.selected_seed,
        "rows_all": len(manifest_all),
        "rows_filtered": len(manifest_filtered),
        "rows_per_split": audit["rows_per_split"],
        "patients_per_split": audit["patients_per_split"],
        "written_files": [
            str(paths.config_path),
            str(paths.target_labels_path),
            str(paths.manifest_all_path),
            str(paths.manifest_filtered_path),
            str(paths.split_manifest_path),
            str(paths.split_audit_path),
            *[str(path) for path in plot_paths],
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate filter/split preprocess artifacts from the local chest X-ray dataset."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/preprocess under the project root.",
    )
    parser.add_argument("--initial-start", type=int, default=0, help="First seed to try.")
    parser.add_argument("--initial-end", type=int, default=100, help="Initial seed range end, inclusive.")
    parser.add_argument("--fallback-end", type=int, default=1000, help="Fallback seed range end, inclusive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_artifacts(
        artifacts_dir=args.artifacts_dir,
        initial_start=args.initial_start,
        initial_end=args.initial_end,
        fallback_end=args.fallback_end,
    )

    print("Generated preprocess artifacts")
    print(f"  artifacts_dir: {summary['artifacts_dir']}")
    print(f"  selected_seed: {summary['selected_seed']}")
    print(f"  rows_all: {summary['rows_all']}")
    print(f"  rows_filtered: {summary['rows_filtered']}")
    print(f"  rows_per_split: {summary['rows_per_split']}")
    print(f"  patients_per_split: {summary['patients_per_split']}")
    print("  written_files:")
    for path in summary["written_files"]:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
