#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from alzheimer_mri_loss_experiment_common import (  # noqa: E402
    Bottleneck,
    run_alzheimer_mri_medical_losses_experiments,
)


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 3, 64, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 4, 128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 6, 256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 3, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        downsample = (self.in_channels != block.expansion * channels) or (stride != 1)
        layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = block.expansion * channels
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, stride=1, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.size(0), -1)
        logits = self.fc(h)
        return logits, h


def build_model(num_classes: int):
    return ResNet50Baseline(num_classes=num_classes)


LEGACY_OPTIMIZER_GROUP_DIVISORS = (
    ("conv1", 10),
    ("bn1", 10),
    ("layer1", 10),
    ("layer2", 10),
    ("layer3", 10),
    ("layer4", 10),
    ("fc", 1),
)

MATCHED_MESC_OPTIMIZER_GROUP_DIVISORS = (
    ("conv1", 10),
    ("bn1", 10),
    ("layer1", 8),
    ("layer2", 6),
    ("layer3", 4),
    ("layer4", 2),
    ("fc", 1),
)


def resolve_optimizer_profile():
    """Return the requested LR profile and its layer-wise divisors.

    ``matched_mesc`` is the default so future ADNI baseline runs use the same
    pretrained-backbone schedule as the MESC runs.  ``legacy`` remains
    available to reproduce historical results.
    """

    raw_profile = os.getenv("ALZHEIMER_BASELINE_LR_PROFILE", "matched_mesc")
    profile = raw_profile.strip().lower().replace("-", "_")
    aliases = {
        "matched": "matched_mesc",
        "mecsb": "matched_mesc",
        "progressive": "matched_mesc",
        "old": "legacy",
        "uniform": "legacy",
    }
    profile = aliases.get(profile, profile)
    if profile == "matched_mesc":
        return profile, MATCHED_MESC_OPTIMIZER_GROUP_DIVISORS
    if profile == "legacy":
        return profile, LEGACY_OPTIMIZER_GROUP_DIVISORS
    raise ValueError(
        "ALZHEIMER_BASELINE_LR_PROFILE must be 'matched_mesc' or 'legacy', "
        f"got {raw_profile!r}."
    )


if __name__ == "__main__":
    optimizer_profile, optimizer_group_divisors = resolve_optimizer_profile()
    print(f"BASELINE_LR_PROFILE: {optimizer_profile}", flush=True)
    run_alzheimer_mri_medical_losses_experiments(
        script_stem="ResNet_baseline",
        model_builder=build_model,
        optimizer_group_divisors=optimizer_group_divisors,
        module_name=f"Baseline ({optimizer_profile} LR profile)",
        insert_after="none",
    )
