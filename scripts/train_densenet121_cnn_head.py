from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune torchvision DenseNet121 on preprocess artifacts.")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "split_manifest.csv")
    parser.add_argument("--target-labels-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "target_labels.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "training" / "densenet121")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to CUDA when available.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights.")
    parser.add_argument("--dry-run-smoke", action="store_true", help="Build data/model and run one forward pass.")
    parser.add_argument("--finalize-run-dir", type=Path, default=None, help="Only run final val/test export for an existing run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from train_densenet import TrainConfig, finalize_run, smoke_check, train_model
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency. Install torch, torchvision, pillow, pandas, and numpy first.") from exc

    config = TrainConfig(
        root=ROOT,
        manifest_path=args.manifest_path,
        target_labels_path=args.target_labels_path,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        pretrained=not args.no_pretrained,
        device=args.device,
    )
    if args.dry_run_smoke:
        try:
            summary = smoke_check(config)
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency. Install torch and torchvision before running DenseNet training.") from exc
        print("DenseNet121 smoke check passed")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0

    if args.finalize_run_dir is not None:
        try:
            summary = finalize_run(config, args.finalize_run_dir)
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency. Install torch and torchvision before running DenseNet training.") from exc
        print("DenseNet121 final evaluation export complete")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0

    try:
        summary = train_model(config)
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency. Install torch and torchvision before running DenseNet training.") from exc
    print("DenseNet121 fine-tuning complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
