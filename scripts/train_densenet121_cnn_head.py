from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_densenet.artifacts import RARE_SAMPLER_WEAKCROP_VARIANT, V2_LOCKED_VARIANT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 with the compact CNN head on saved CXR preprocess artifacts."
    )
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "split_manifest.csv")
    parser.add_argument("--target-labels-path", type=Path, default=ROOT / "artifacts" / "preprocess" / "target_labels.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "training" / "densenet121_cnn_head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--recipe-variant",
        default=V2_LOCKED_VARIANT,
        choices=[V2_LOCKED_VARIANT, RARE_SAMPLER_WEAKCROP_VARIANT],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--resize-size", type=int, default=352)
    parser.add_argument("--target-mode", default="disease_only", choices=["disease_only"])
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage2-min-epochs-before-early-stop", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--head-lr-stage1", type=float, default=3e-4)
    parser.add_argument("--head-lr-stage2", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-stage2", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="warmup_cosine", choices=["warmup_cosine"])
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--min-lr-factor", type=float, default=0.01)
    parser.add_argument("--loss", default="asymmetric", choices=["asymmetric"])
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-eps", type=float, default=1e-8)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to CUDA when available.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument(
        "--dry-run-smoke",
        action="store_true",
        help="Build data/model and run only the Stage 0 smoke check. Does not train epochs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from train_densenet.artifacts import (
            labels_for_target_mode,
            load_target_labels,
        )
        from train_densenet.model import DenseNet121CNNHead, configure_stage1
        from train_densenet.train import (
            TrainingConfig,
            build_dataloaders,
            create_criterion,
            resolve_device,
            run_stage0_smoke,
            set_seed,
            train_model,
            validate_recipe_config,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing training dependency. Install project dependencies, including torch and torchvision, before running training."
        ) from exc

    config = TrainingConfig(
        root=ROOT,
        manifest_path=args.manifest_path,
        target_labels_path=args.target_labels_path,
        output_dir=args.output_dir,
        seed=args.seed,
        recipe_variant=args.recipe_variant,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        input_size=args.input_size,
        resize_size=args.resize_size,
        target_mode=args.target_mode,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_min_epochs_before_early_stop=args.stage2_min_epochs_before_early_stop,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        head_lr_stage1=args.head_lr_stage1,
        head_lr_stage2=args.head_lr_stage2,
        backbone_lr_stage2=args.backbone_lr_stage2,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        min_lr_factor=args.min_lr_factor,
        loss=args.loss,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
        asl_eps=args.asl_eps,
        use_amp=not args.no_amp,
        device=args.device,
    )
    if args.dry_run_smoke:
        load_target_labels(config.target_labels_path)
        labels = labels_for_target_mode(config.target_mode)
        validate_recipe_config(config, labels)
        set_seed(config.seed)
        device = resolve_device(config.device)
        dataloaders = build_dataloaders(config, labels)
        model = DenseNet121CNNHead(num_classes=len(labels)).to(device)
        configure_stage1(model)
        criterion = create_criterion(config)
        run_stage0_smoke(model, dataloaders["train"], criterion, device, labels, input_size=config.input_size)
        print("Stage 0 smoke check passed. No training epochs were run.")
        return 0

    summary = train_model(config)
    print("DenseNet121 CNN-head training complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
