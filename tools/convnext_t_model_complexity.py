#!/usr/bin/env python
"""Measure parameters, rough FLOPs, and latency for the ConvNeXt-T table.

MESC-B is inserted after ConvNeXt-T stage3 (``features[5]``, 384 channels).
DAST changes only the training loss and therefore does not need a separate
row in this architecture-complexity table.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_convnext_t_experiment import (  # noqa: E402
    ConvNeXtTinyWithStage3Attention,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "analysis_tables" / "convnext_t_model_complexity.csv",
    )
    args = parser.parse_args(argv)
    if min(args.num_classes, args.input_size, args.batch_size, args.repeats) <= 0:
        parser.error("class count, input size, batch size, and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup cannot be negative")
    return args


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def conv_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_channels, out_height, out_width = output.shape
    kernel_ops = (
        module.kernel_size[0]
        * module.kernel_size[1]
        * (module.in_channels // module.groups)
    )
    return int(batch * out_channels * out_height * out_width * kernel_ops)


def linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    positions = output.numel() // module.out_features
    return int(positions * module.in_features * module.out_features)


def rough_flops(model: nn.Module, inputs: torch.Tensor) -> int:
    """Count Conv2d/Linear multiplications for one forward pass."""

    total = 0
    hooks = []

    def hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += conv_flops(module, output)
        elif isinstance(module, nn.Linear):
            total += linear_flops(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        for registered_hook in hooks:
            registered_hook.remove()
    return total


def latency_ms(
    model: nn.Module,
    inputs: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def build_model(num_classes: int, attention: str) -> nn.Module:
    return ConvNeXtTinyWithStage3Attention(
        num_classes=num_classes,
        attention=attention,
        loss_name="ce",
        pretrained=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    inputs = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=device,
    )
    configurations = (
        ("ConvNeXt-T", "none", "CE"),
        ("ConvNeXt-T + MESC-B (after stage3)", "mesc", "CE"),
    )
    rows: list[dict[str, object]] = []
    for method, attention, loss in configurations:
        model = build_model(args.num_classes, attention).to(device).eval()
        total_params, trainable_params = count_params(model)
        flops = rough_flops(model, inputs)
        latency = latency_ms(model, inputs, args.warmup, args.repeats)
        row = {
            "method": method,
            "backbone": "ConvNeXt-T",
            "attention": attention,
            "training_loss": loss,
            "insertion_point": "after_stage3_features_5" if attention == "mesc" else "none",
            "stage3_channels": 384 if attention == "mesc" else "",
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "rough_flops": flops,
            "rough_gflops": flops / 1e9,
            "inference_time_ms": latency,
            "batch_size": args.batch_size,
            "input_size": f"{args.input_size}x{args.input_size}",
            "device": str(device),
        }
        rows.append(row)
        print(row)
        del model

    output = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
