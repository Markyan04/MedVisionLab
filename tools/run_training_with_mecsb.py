#!/usr/bin/env python
"""Execute a supported training script after injecting MECS VersionB.

This adapter keeps the original KOA and ADNI training sources unchanged.  It
is intended to be launched by ``run_mecsb_controlled_ablation.py`` only for
the two MESC arms of the controlled ablation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MECS_VersionB import MECS_VersionB  # noqa: E402


SUPPORTED_SCRIPTS = {
    (PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py").resolve(): "koa",
    (PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+Loss4.py").resolve(): "koa",
    (PROJECT_ROOT / "Alzheimer_MRI_Loss" / "ResNet_layer3+MECS.py").resolve(): "adni",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
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


def inject_version_b(module: ModuleType) -> None:
    if not hasattr(module, "MECS_VersionA"):
        raise RuntimeError("Target script does not expose MECS_VersionA for injection.")
    module.MECS_VersionA = MECS_VersionB

    if hasattr(module, "CustomResNet50MECS"):
        resolved = module.CustomResNet50MECS.__init__.__globals__.get("MECS_VersionA")
    elif hasattr(module, "build_model"):
        resolved = module.build_model.__globals__.get("MECS_VersionA")
    else:
        raise RuntimeError("Cannot locate the target script's model construction function.")
    if resolved is not MECS_VersionB:
        raise RuntimeError("MECS VersionB injection verification failed.")


def run_adni(module: ModuleType) -> None:
    module.run_alzheimer_mri_medical_losses_experiments(
        script_stem="ResNet_layer3+MECS",
        model_builder=module.build_model,
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
        module_name="MECS VersionB",
        insert_after="layer3",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset = SUPPORTED_SCRIPTS[args.script]
    module = load_module(args.script)
    inject_version_b(module)

    print("=" * 90)
    print("Injected attention module: MECS_VersionB")
    print(f"Original training script : {args.script}")
    print(f"Dataset                  : {dataset.upper()}")
    print("Original source modified : no")
    print("=" * 90)

    if dataset == "koa":
        module.main()
    else:
        run_adni(module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
