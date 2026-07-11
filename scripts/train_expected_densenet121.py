from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed DenseNet121 chest X-ray classifier.")
    parser.add_argument("--dry-run-smoke", action="store_true", help="Build data/model and run one forward pass.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from train_expected import default_config, smoke_check, train_model
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency. Install torch, torchvision, pillow, pandas, and numpy first.") from exc

    config = default_config(ROOT)
    if args.dry_run_smoke:
        summary = smoke_check(config)
        print("DenseNet121 smoke check passed")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0

    summary = train_model(config)
    print("DenseNet121 training complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
