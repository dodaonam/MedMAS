from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plots from DenseNet121 training artifacts.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "training" / "densenet121")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "split_manifest.csv")
    parser.add_argument("--target-labels-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "target_labels.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from train_densenet import resolve_run_dir
    from visualize_densenet import generate_visualization_report

    run_dir = resolve_run_dir(args.run_dir)
    paths = generate_visualization_report(
        run_dir=run_dir,
        manifest_path=args.manifest_path,
        target_labels_path=args.target_labels_path,
        root=ROOT,
    )
    print("Generated DenseNet121 visualization files")
    print(f"  run_dir: {run_dir}")
    print(f"  figure_count: {len(paths)}")
    for path in paths:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
