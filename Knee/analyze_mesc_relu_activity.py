#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure per-image nonzero ratios after MESC shared-MLP ReLU on KOA."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data

from analyze_mesc_branch_disagreement import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEST_DIR,
    CustomResNet50MECS,
    build_test_dataset,
    load_checkpoint,
)
from MECS_old import global_median_pooling


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            THIS_DIR
            / "analysis_results"
            / f"mesc_relu_activity_koa_test_seed1234_{stamp}"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    return parser.parse_args()


def set_reproducibility() -> None:
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forward_to_mesc_input(
    model: CustomResNet50MECS, images: torch.Tensor
) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    return model.mecs.act(model.mecs.pre_conv(x))


def branch_relu_nonzero_ratios(
    model: CustomResNet50MECS, mecs_input: torch.Tensor
):
    descriptors = {
        "avg": F.adaptive_avg_pool2d(mecs_input, output_size=(1, 1)),
        "max": F.adaptive_max_pool2d(mecs_input, output_size=(1, 1)),
        "median": global_median_pooling(mecs_input),
    }
    output = {}
    for branch, descriptor in descriptors.items():
        relu = F.relu(model.mecs.channel_attention.fc1(descriptor), inplace=False)
        nonzero_count = (relu > 0).flatten(1).sum(dim=1)
        output[branch] = {
            "count": nonzero_count,
            "fraction": nonzero_count.float() / relu[0].numel(),
            "hidden_elements": relu[0].numel(),
        }
    return output


@torch.inference_mode()
def run(
    model: CustomResNet50MECS,
    loader: data.DataLoader,
    test_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for batch_index, (images, labels, paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        ratios = branch_relu_nonzero_ratios(
            model, forward_to_mesc_input(model, images)
        )
        labels_list = labels.tolist()
        branch_counts = {
            branch: values["count"].cpu().tolist()
            for branch, values in ratios.items()
        }
        branch_fractions = {
            branch: values["fraction"].cpu().tolist()
            for branch, values in ratios.items()
        }

        for index, path_text in enumerate(paths):
            path = Path(path_text).resolve()
            rows.append(
                {
                    "sample_id": path.relative_to(test_dir).as_posix(),
                    "image_path": str(path),
                    "true_label": int(labels_list[index]),
                    "relu_hidden_elements": int(ratios["avg"]["hidden_elements"]),
                    "avg_relu_nonzero_count": int(branch_counts["avg"][index]),
                    "avg_relu_nonzero_fraction": float(branch_fractions["avg"][index]),
                    "max_relu_nonzero_count": int(branch_counts["max"][index]),
                    "max_relu_nonzero_fraction": float(branch_fractions["max"][index]),
                    "median_relu_nonzero_count": int(branch_counts["median"][index]),
                    "median_relu_nonzero_fraction": float(
                        branch_fractions["median"][index]
                    ),
                }
            )

        if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
            print(
                f"Processed {batch_index + 1}/{len(loader)} batches "
                f"({len(rows)}/{len(loader.dataset)} images)",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch in ("avg", "max", "median"):
        fraction = samples[f"{branch}_relu_nonzero_fraction"]
        count = samples[f"{branch}_relu_nonzero_count"]
        rows.append(
            {
                "branch": branch,
                "n_images": int(len(samples)),
                "mean_nonzero_fraction": float(fraction.mean()),
                "std_nonzero_fraction": float(fraction.std(ddof=0)),
                "min_nonzero_fraction": float(fraction.min()),
                "max_nonzero_fraction": float(fraction.max()),
                "mean_nonzero_count": float(count.mean()),
                "max_nonzero_count": int(count.max()),
                "images_with_zero_nonzero_elements": int((count == 0).sum()),
                "images_with_any_nonzero_element": int((count > 0).sum()),
                "fraction_images_with_any_nonzero_element": float(
                    (count > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    checkpoint: Path,
    best_val_qwk: float | None,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    lines = [
        "# KOA MESC ReLU 逐图非零比例",
        "",
        f"- checkpoint：`{checkpoint}`",
        f"- 最佳验证 QWK：{best_val_qwk}",
        f"- test 样本数：{len(samples)}",
        "- 每个分支的 ReLU hidden shape：`[256,1,1]`，每张图共 256 个元素。",
        "- 非零定义：`ReLU output > 0`。",
        "",
        "| branch | 平均非零比例 | 标准差 | 最小值 | 最大值 | 全零图片数 | 存在非零元素图片数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.mean_nonzero_fraction:.10f} | "
            f"{row.std_nonzero_fraction:.10f} | {row.min_nonzero_fraction:.10f} | "
            f"{row.max_nonzero_fraction:.10f} | "
            f"{row.images_with_zero_nonzero_elements} | "
            f"{row.images_with_any_nonzero_element} |"
        )
    lines.extend(
        [
            "",
            "> 这里统计的是 shared MLP 第一层后 ReLU 的激活比例，不是最终 attention 值。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split not found: {test_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    set_reproducibility()
    device = torch.device(args.device)
    dataset = build_test_dataset(test_dir)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = CustomResNet50MECS().to(device).eval()
    best_val_qwk = load_checkpoint(model, checkpoint, device)

    samples = run(model, loader, test_dir, device)
    summary = summarize(samples)
    sample_csv = output_dir / "koa_mesc_relu_nonzero_per_image.csv"
    summary_csv = output_dir / "koa_mesc_relu_nonzero_summary.csv"
    json_path = output_dir / "koa_mesc_relu_nonzero_summary.json"
    report_path = output_dir / "koa_mesc_relu_nonzero_report.md"
    samples.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_best_val_qwk": best_val_qwk,
                "test_dir": str(test_dir),
                "n_images": len(samples),
                "relu_hidden_elements_per_branch_per_image": 256,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        render_report(checkpoint, best_val_qwk, samples, summary),
        encoding="utf-8",
    )
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"Per-image CSV: {sample_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
