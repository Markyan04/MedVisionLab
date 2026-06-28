#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure parameters, rough FLOPs, and inference latency for ResNet50 variants."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MECS_old import MECS_VersionA  # noqa: E402
from HAM10000_Loss.ham10000_loss_experiment_common import ResNet50Baseline, ResNet50WithInsertedModule  # noqa: E402


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def conv_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_c, out_h, out_w = output.shape
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    return int(batch * out_c * out_h * out_w * kernel_ops)


def linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    batch = output.shape[0] if output.ndim > 1 else 1
    return int(batch * module.in_features * module.out_features)


def rough_flops(model: nn.Module, x: torch.Tensor) -> int:
    total = 0
    hooks = []

    def hook(module, _inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += conv_flops(module, output)
        elif isinstance(module, nn.Linear):
            total += linear_flops(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total


def latency_ms(model: nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def build_models(num_classes: int) -> Dict[str, nn.Module]:
    mesc = MECS_VersionA(in_channels=1024, out_channels=1024)
    return {
        "ResNet50": ResNet50Baseline(num_classes=num_classes),
        "ResNet50 + MESC (layer3)": ResNet50WithInsertedModule(num_classes=num_classes, inserted_module=mesc, insert_after="layer3"),
        "ResNet50 + MESC (layer3) + DAST": ResNet50WithInsertedModule(num_classes=num_classes, inserted_module=MECS_VersionA(1024, 1024), insert_after="layer3"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "analysis_tables" / "model_complexity.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in build_models(args.num_classes).items():
        model = model.to(device).eval()
        flops = rough_flops(model, x)
        ms = latency_ms(model, x, args.warmup, args.repeats)
        rows.append({
            "method": name,
            "parameters": count_params(model),
            "flops": flops,
            "inference_time_ms": ms,
            "input_size": f"{args.input_size}x{args.input_size}",
            "device": str(device),
        })
        print(rows[-1])
    with args.out.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
