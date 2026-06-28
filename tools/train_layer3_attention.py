#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a layer3 attention baseline using the existing dataset runners."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_modules import build_attention_module  # noqa: E402


def set_common_env(prefix: str, args: argparse.Namespace) -> None:
    os.environ[f"{prefix}_LOSSES"] = args.loss
    os.environ[f"{prefix}_RUN_TAG"] = args.run_tag
    if args.seed is not None:
        os.environ["GLOBAL_EXPERIMENT_SEED"] = str(args.seed)
        os.environ[f"{prefix}_SEED"] = str(args.seed)
    if args.epochs is not None:
        os.environ[f"{prefix}_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)


def run_ham10000(args: argparse.Namespace) -> None:
    from HAM10000_Loss.ham10000_loss_experiment_common import ResNet50WithInsertedModule, run_ham10000_medical_losses_experiments

    set_common_env("HAM10000", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_ham10000_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_adni(args: argparse.Namespace) -> None:
    from Alzheimer_MRI_Loss.alzheimer_mri_loss_experiment_common import ResNet50WithInsertedModule, run_alzheimer_mri_medical_losses_experiments

    set_common_env("ALZHEIMER", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_alzheimer_mri_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_brain(args: argparse.Namespace) -> None:
    from Brain_Tumor_MRI_Loss.brain_tumor_mri_loss_experiment_common import ResNet50WithInsertedModule, run_brain_tumor_mri_medical_losses_experiments

    set_common_env("BRAIN_MRI", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_brain_tumor_mri_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_chest(args: argparse.Namespace) -> None:
    from chest_xray_loss_import import run_chest_attention

    run_chest_attention(args, build_attention_module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ham10000", "adni", "brain", "chest"), required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--loss", choices=("ce", "dast"), default="ce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    if not args.run_tag:
        args.run_tag = f"layer3_{args.attention}_{args.loss}_seed{args.seed}"

    runners = {
        "ham10000": run_ham10000,
        "adni": run_adni,
        "brain": run_brain,
        "chest": run_chest,
    }
    runners[args.dataset](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
