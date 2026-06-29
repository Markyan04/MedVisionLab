#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Import helper for the chest-x-ray-image_Loss folder whose name contains hyphens."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEST_DIR = PROJECT_ROOT / "chest-x-ray-image_Loss"
if str(CHEST_DIR) not in sys.path:
    sys.path.insert(0, str(CHEST_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_chest_attention(args, module_factory) -> None:
    os.environ["CHESTXRAY_LOSSES"] = args.loss
    os.environ["CHESTXRAY_RUN_TAG"] = args.run_tag
    if args.seed is not None:
        os.environ["GLOBAL_EXPERIMENT_SEED"] = str(args.seed)
        os.environ["CHESTXRAY_SEED"] = str(args.seed)
    if args.epochs is not None:
        os.environ["CHESTXRAY_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ["CHESTXRAY_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        os.environ["CHESTXRAY_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        os.environ["CHESTXRAY_NUM_WORKERS"] = str(args.num_workers)
    if args.patience is not None:
        os.environ["CHESTXRAY_PATIENCE"] = str(args.patience)
    if args.early_delta is not None:
        os.environ["CHESTXRAY_EARLY_DELTA"] = str(args.early_delta)

    from chest_xray_loss_experiment_common import ResNet50WithInsertedModule, run_chestxray_medical_losses_experiments

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, module_factory(args.attention, 1024), "layer3")

    run_chestxray_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )
