#!/usr/bin/env python
"""Run KOA ResNet50-layer3 + MECS VersionB with CE or fixed DAST.

The runner reuses the existing KOA training scripts without editing them.  It
injects ``MECS_VersionB`` before their model is instantiated and assigns a
dedicated run tag so VersionA checkpoints are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for search_path in (PROJECT_ROOT, THIS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from MECS_VersionB import MECS_VersionB  # noqa: E402


TRAINING_SCRIPTS = {
    "ce": THIS_DIR / "ResNet_layer3+MECS+CE.py",
    "dast": THIS_DIR / "ResNet_layer3+MECS+Loss4.py",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KOA ResNet50 layer3 + MECS VersionB with CE or DAST."
    )
    parser.add_argument("--loss-mode", required=True, choices=("ce", "dast"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("epochs, patience, and batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative.")
    if args.dast_tau <= 0 or args.dast_gamma < 0:
        parser.error("dast-tau must be > 0 and dast-gamma must be >= 0.")

    args.data_root = args.data_root.expanduser().resolve()
    if args.run_tag is None:
        args.run_tag = f"mecsb_{args.loss_mode}_seed{args.seed}"
    args.run_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_tag).strip("._-")
    if not args.run_tag:
        parser.error("run-tag becomes empty after sanitization.")
    return args


def configure_environment(args: argparse.Namespace) -> None:
    values = {
        "KNEE_DATA_ROOT": args.data_root,
        "KNEE_SEED": args.seed,
        "KNEE_EPOCHS": args.epochs,
        "KNEE_PATIENCE": args.patience,
        "KNEE_BATCH_SIZE": args.batch_size,
        "KNEE_NUM_WORKERS": args.num_workers,
        "KNEE_IMAGE_SIZE": args.image_size,
        "KNEE_LR_BACKBONE": args.lr_backbone,
        "KNEE_LR_HEAD": args.lr_head,
        "KNEE_WEIGHT_DECAY": args.weight_decay,
        "KNEE_EARLY_DELTA": args.early_stop_delta,
        "KNEE_DAST_TAU": args.dast_tau,
        "KNEE_DAST_GAMMA": args.dast_gamma,
        "KNEE_RUN_TAG": args.run_tag,
    }
    for key, value in values.items():
        os.environ[key] = str(value)


def expected_checkpoint(args: argparse.Namespace) -> Path:
    base_name = (
        "best_resnet50_mecs_layer3_knee_oa"
        if args.loss_mode == "ce"
        else "best_resnet50_mecs_layer3_dast_knee_oa_new"
    )
    return THIS_DIR / "checkpoints" / f"{base_name}_{args.run_tag}.pt"


def load_training_module(args: argparse.Namespace) -> ModuleType:
    path = TRAINING_SCRIPTS[args.loss_mode]
    if not path.is_file():
        raise FileNotFoundError(f"Training script not found: {path}")
    module_name = f"koa_mecsb_{args.loss_mode}_{args.seed}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for split in ("train", "val", "test"):
        split_path = args.data_root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Required KOA split not found: {split_path}")

    checkpoint = expected_checkpoint(args).resolve()
    if checkpoint.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing checkpoint: {checkpoint}\n"
            "Choose a different --run-tag."
        )

    configure_environment(args)
    training = load_training_module(args)

    # CustomResNet50MECS resolves this module global when __init__ is called.
    # Replacing it here changes only this process and leaves source files intact.
    training.MECS_VersionA = MECS_VersionB
    injected = training.CustomResNet50MECS.__init__.__globals__["MECS_VersionA"]
    if injected is not MECS_VersionB:
        raise RuntimeError("MECS VersionB injection failed.")

    print("=" * 72)
    print("KOA MECS VersionB experiment")
    print(f"Loss mode : {args.loss_mode.upper()}")
    print(f"Seed      : {args.seed}")
    print(f"Epochs    : {args.epochs}")
    print(f"Patience  : {args.patience}")
    print(f"Data root : {args.data_root}")
    print(f"Run tag   : {args.run_tag}")
    print(f"Checkpoint: {checkpoint}")
    if args.loss_mode == "dast":
        print(f"DAST      : tau={args.dast_tau}, gamma={args.dast_gamma}")
    print("Module    : MECS_VersionB (median-anchored dynamic router)")
    print("=" * 72)

    training.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
