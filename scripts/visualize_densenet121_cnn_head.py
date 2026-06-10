from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate DenseNet121 CNN-head training visualizations from saved artifacts."
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "training" / "densenet121_cnn_head")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "split_manifest.csv")
    parser.add_argument("--target-labels-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "target_labels.json")
    parser.add_argument("--include-gradcam", action="store_true", help="Also generate Grad-CAM figures from checkpoint_best.pt.")
    parser.add_argument("--device", default=None, help="Device for Grad-CAM when --include-gradcam is used.")
    parser.add_argument(
        "--gradcam-target-layer",
        default="cnn_head_final_conv",
        choices=["cnn_head_final_conv", "backbone.denseblock4"],
        help="Target layer for Grad-CAM when --include-gradcam is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from visualize_densenet.data import resolve_run_dir
    from visualize_densenet.report import generate_visualization_report

    run_dir = resolve_run_dir(args.run_dir)
    paths = generate_visualization_report(
        run_dir=run_dir,
        manifest_path=args.manifest_path,
        target_labels_path=args.target_labels_path,
        root=ROOT,
        include_gradcam=args.include_gradcam,
        device=args.device,
        gradcam_target_layer=args.gradcam_target_layer,
    )
    print("Generated DenseNet visualization files")
    print(f"  run_dir: {run_dir}")
    print(f"  figure_count: {len(paths)}")
    for path in paths:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
