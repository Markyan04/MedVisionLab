#!/usr/bin/env python
"""Execute a supported training script after injecting a MECS routing variant.

This adapter keeps the original KOA and ADNI training sources unchanged.  It
supports the median-anchored VersionB router and its strict raw-logit control.
For insertion-position ablations it can also move the injected module to any
of the four ResNet stages while retaining the original training protocol.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torchvision.models as models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MECS_VersionB import MECS_VersionB  # noqa: E402
from MECS_RawRouting import MECS_RawRouting  # noqa: E402


SUPPORTED_SCRIPTS = {
    (PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py").resolve(): "koa",
    (PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py").resolve(): "koa",
    (PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py").resolve(): "adni",
}
VARIANTS = {
    "median": (MECS_VersionB, "MECS_VersionB", "median-anchored dynamic router"),
    "raw": (MECS_RawRouting, "MECS_RawRouting", "raw-logit dynamic router"),
}
LAYER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="median",
        help="Routing representation to inject. Default keeps existing VersionB behavior.",
    )
    parser.add_argument(
        "--insert-after",
        choices=tuple(LAYER_CHANNELS),
        default="layer3",
        help="ResNet stage after which to insert MECS. Default: layer3.",
    )
    args = parser.parse_args(argv)
    args.script = args.script.expanduser().resolve()
    if args.script not in SUPPORTED_SCRIPTS:
        supported = "\n".join(f"  - {path}" for path in SUPPORTED_SCRIPTS)
        parser.error(f"Unsupported training script: {args.script}\nSupported scripts:\n{supported}")
    return args


def load_module(script: Path) -> ModuleType:
    module_name = f"mecsb_injected_{script.parent.name}_{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def inject_attention(module: ModuleType, attention_class) -> None:
    if not hasattr(module, "MECS_VersionA"):
        raise RuntimeError("Target script does not expose MECS_VersionA for injection.")
    module.MECS_VersionA = attention_class

    if hasattr(module, "CustomResNet50MECS"):
        resolved = module.CustomResNet50MECS.__init__.__globals__.get("MECS_VersionA")
    elif hasattr(module, "build_model"):
        resolved = module.build_model.__globals__.get("MECS_VersionA")
    else:
        raise RuntimeError("Cannot locate the target script's model construction function.")
    if resolved is not attention_class:
        raise RuntimeError(f"MECS injection verification failed for {attention_class.__name__}.")


def inject_version_b(module: ModuleType) -> None:
    """Backward-compatible helper used by existing VersionB tooling."""
    inject_attention(module, MECS_VersionB)


def inject_raw_routing(module: ModuleType) -> None:
    inject_attention(module, MECS_RawRouting)


def make_koa_position_model(
    attention_class: Type[nn.Module],
    insert_after: str,
) -> Type[nn.Module]:
    """Build the KOA script-compatible model for a non-default insertion."""
    channels = LAYER_CHANNELS[insert_after]

    class CustomResNet50MECSPosition(nn.Module):
        def __init__(self, num_classes: int = 5) -> None:
            super().__init__()
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            # Keep the attribute name expected by the KOA optimizer grouping.
            self.mecs = attention_class(in_channels=channels, out_channels=channels)
            self.avgpool = base_model.avgpool
            self.fc = nn.Linear(base_model.fc.in_features, num_classes)
            self.insert_after = insert_after

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            for layer_name in LAYER_CHANNELS:
                x = getattr(self, layer_name)(x)
                if layer_name == self.insert_after:
                    x = self.mecs(x)
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))

    CustomResNet50MECSPosition.__name__ = "CustomResNet50MECS"
    return CustomResNet50MECSPosition


def configure_koa_position(
    module: ModuleType,
    attention_class: Type[nn.Module],
    insert_after: str,
) -> None:
    """Replace only the model and checkpoint stage label in a KOA trainer."""
    if insert_after == "layer3":
        return
    module.CustomResNet50MECS = make_koa_position_model(attention_class, insert_after)
    original_resolver = module.resolve_checkpoint_path

    def resolve_position_checkpoint(filename: str) -> str:
        return original_resolver(filename.replace("layer3", insert_after))

    module.resolve_checkpoint_path = resolve_position_checkpoint


def run_adni(
    module: ModuleType,
    attention_class: Type[nn.Module],
    module_name: str,
    insert_after: str,
) -> None:
    channels = LAYER_CHANNELS[insert_after]

    def build_position_model(num_classes: int) -> nn.Module:
        attention = attention_class(in_channels=channels, out_channels=channels)
        return module.ResNet50WithInsertedModule(
            num_classes=num_classes,
            inserted_module=attention,
            insert_after=insert_after,
        )

    module.run_alzheimer_mri_medical_losses_experiments(
        script_stem=f"ResNet_{insert_after}+MECS",
        model_builder=build_position_model,
        optimizer_group_divisors=[
            ("conv1", 10),
            ("bn1", 10),
            ("layer1", 8),
            ("layer2", 6),
            ("layer3", 4),
            ("inserted_module", 3),
            ("layer4", 2),
            ("fc", 1),
        ],
        module_name=module_name,
        insert_after=insert_after,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset = SUPPORTED_SCRIPTS[args.script]
    module = load_module(args.script)
    attention_class, module_name, description = VARIANTS[args.variant]
    inject_attention(module, attention_class)
    if dataset == "koa":
        configure_koa_position(module, attention_class, args.insert_after)

    print("=" * 90)
    print(f"Injected attention module: {module_name}")
    print(f"Routing representation    : {description}")
    print(f"Original training script : {args.script}")
    print(f"Dataset                  : {dataset.upper()}")
    print(f"Insert after             : {args.insert_after}")
    print(f"MESC channels            : {LAYER_CHANNELS[args.insert_after]}")
    print("Original source modified : no")
    print("=" * 90)

    if dataset == "koa":
        module.main()
    else:
        run_adni(
            module,
            attention_class=attention_class,
            module_name=module_name,
            insert_after=args.insert_after,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
