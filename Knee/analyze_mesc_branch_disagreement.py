#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze MESC channel-branch disagreement on the fixed KOA test split.

This script is inference-only. It loads an existing checkpoint, uses the same
deterministic test preprocessing as the training script, and writes results to
a newly created output directory so existing artifacts cannot be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.stats import fisher_exact, spearmanr


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MECS_old import MECS_VersionA  # noqa: E402


DEFAULT_CHECKPOINT = (
    THIS_DIR
    / "checkpoints"
    / "best_resnet50_mecs_layer3_knee_oa_controlled_koa_resnet50_plus_mesc_seed1234.pt"
)
DEFAULT_TEST_DIR = PROJECT_ROOT / "Knee_Osteoarthritis" / "test"
EXPECTED_CLASS_TO_IDX = {str(index): index for index in range(5)}
IMAGE_SIZE = 224
NUM_CLASSES = 5
CHANNELS = 1024


class ImageFolderWithPath(datasets.ImageFolder):
    """ImageFolder variant that also returns the source path."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, label, self.samples[index][0]


class CustomResNet50MECS(nn.Module):
    """Architecture used by ``Knee/ResNet_layer3+MECS+CE.py``."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # The checkpoint replaces every parameter, so no download is needed.
        base_model = models.resnet50(weights=None)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.mecs = MECS_VersionA(in_channels=CHANNELS, out_channels=CHANNELS)
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor, return_branch_attentions: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if return_branch_attentions:
            x, branch_attentions = self.mecs(
                x, return_branch_attentions=True
            )
        else:
            x = self.mecs(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        logits = self.fc(torch.flatten(x, 1))
        if return_branch_attentions:
            return logits, branch_attentions
        return logits


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        THIS_DIR
        / "analysis_results"
        / f"mesc_branch_disagreement_koa_test_seed1234_{stamp}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    return parser.parse_args()


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> float | None:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        best_score = checkpoint.get("best_score")
    else:
        state_dict = checkpoint
        best_score = None

    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return None if best_score is None else float(best_score)


def build_test_dataset(test_dir: Path) -> ImageFolderWithPath:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = ImageFolderWithPath(str(test_dir), transform=transform)
    if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
        raise RuntimeError(
            "KOA class mapping differs from the trained model: "
            f"expected {EXPECTED_CLASS_TO_IDX}, got {dataset.class_to_idx}"
        )
    return dataset


def branch_distances(
    branch_attentions: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    avg_att = branch_attentions["avg_att"]
    max_att = branch_attentions["max_att"]
    med_att = branch_attentions["med_att"]

    expected_shape = (avg_att.shape[0], CHANNELS, 1, 1)
    for name, tensor in branch_attentions.items():
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}"
            )

    denominator = 2.0 * CHANNELS
    avg_max = (avg_att - max_att).abs().flatten(1).sum(dim=1) / denominator
    avg_median = (avg_att - med_att).abs().flatten(1).sum(dim=1) / denominator
    max_median = (max_att - med_att).abs().flatten(1).sum(dim=1) / denominator
    total = avg_max + avg_median + max_median
    return avg_max, avg_median, max_median, total


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    loader: data.DataLoader,
    test_dir: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, float, Dict[str, float | int]]:
    model.eval()
    rows = []
    default_forward_max_abs_diff = 0.0
    attention_counts = {
        "attention_elements_per_branch": 0,
        "avg_att_exact_zero": 0,
        "avg_att_exact_one": 0,
        "max_att_exact_zero": 0,
        "max_att_exact_one": 0,
        "med_att_exact_zero": 0,
        "med_att_exact_one": 0,
        "avg_max_exact_equal": 0,
        "avg_median_exact_equal": 0,
        "max_median_exact_equal": 0,
    }

    for batch_index, (images, labels, paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, branches = model(images, return_branch_attentions=True)

        # On the first batch, explicitly verify that the optional return path
        # does not alter the original/default logits.
        if batch_index == 0:
            default_logits = model(images)
            default_forward_max_abs_diff = float(
                (default_logits - logits).abs().max().item()
            )
            if default_forward_max_abs_diff > 1e-7:
                raise RuntimeError(
                    "Optional attention return changed model logits: "
                    f"max_abs_diff={default_forward_max_abs_diff}"
                )

        predictions = logits.argmax(dim=1)
        avg_max, avg_median, max_median, total = branch_distances(branches)

        avg_att = branches["avg_att"]
        max_att = branches["max_att"]
        med_att = branches["med_att"]
        attention_counts["attention_elements_per_branch"] += avg_att.numel()
        attention_counts["avg_att_exact_zero"] += int((avg_att == 0).sum().item())
        attention_counts["avg_att_exact_one"] += int((avg_att == 1).sum().item())
        attention_counts["max_att_exact_zero"] += int((max_att == 0).sum().item())
        attention_counts["max_att_exact_one"] += int((max_att == 1).sum().item())
        attention_counts["med_att_exact_zero"] += int((med_att == 0).sum().item())
        attention_counts["med_att_exact_one"] += int((med_att == 1).sum().item())
        attention_counts["avg_max_exact_equal"] += int(
            (avg_att == max_att).sum().item()
        )
        attention_counts["avg_median_exact_equal"] += int(
            (avg_att == med_att).sum().item()
        )
        attention_counts["max_median_exact_equal"] += int(
            (max_att == med_att).sum().item()
        )

        labels_cpu = labels.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()
        avg_max_cpu = avg_max.cpu().numpy()
        avg_median_cpu = avg_median.cpu().numpy()
        max_median_cpu = max_median.cpu().numpy()
        total_cpu = total.cpu().numpy()

        for index, image_path_text in enumerate(paths):
            image_path = Path(image_path_text)
            label = int(labels_cpu[index])
            prediction = int(predictions_cpu[index])
            rows.append(
                {
                    "sample_id": image_path.relative_to(test_dir).as_posix(),
                    "image_path": str(image_path.resolve()),
                    "true_label": label,
                    "pred_label": prediction,
                    "is_correct": prediction == label,
                    "abs_pred_minus_label": abs(prediction - label),
                    "avg_max_distance": float(avg_max_cpu[index]),
                    "avg_median_distance": float(avg_median_cpu[index]),
                    "max_median_distance": float(max_median_cpu[index]),
                    "total_disagreement": float(total_cpu[index]),
                }
            )

        if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
            print(
                f"Processed {batch_index + 1}/{len(loader)} batches "
                f"({len(rows)}/{len(loader.dataset)} samples)",
                flush=True,
            )

    denominator = attention_counts["attention_elements_per_branch"]
    attention_diagnostics: Dict[str, float | int] = dict(attention_counts)
    for key, value in attention_counts.items():
        if key != "attention_elements_per_branch":
            attention_diagnostics[f"{key}_fraction"] = value / denominator
    return pd.DataFrame(rows), default_forward_max_abs_diff, attention_diagnostics


def assign_equal_count_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign four size-balanced groups with deterministic tie breaking.

    ImageFolder orders samples by class and path. Hashing sample_id within equal
    disagreement values prevents that unrelated ordering from biasing groups.
    """

    tie_breaker = frame["sample_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    sorted_indices = (
        frame.assign(_tie_breaker=tie_breaker)
        .sort_values(["total_disagreement", "_tie_breaker"], kind="mergesort")
        .index.to_numpy()
    )
    labels = ("Q1_low", "Q2", "Q3", "Q4_high")
    quartile = pd.Series(index=frame.index, dtype="object")
    for label, indices in zip(labels, np.array_split(sorted_indices, 4)):
        quartile.loc[indices] = label
    result = frame.copy()
    result["disagreement_quartile"] = quartile
    return result


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def calculate_statistics(
    samples: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float | int | None]]:
    quartile_order = ("Q1_low", "Q2", "Q3", "Q4_high")
    quartile_rows = []
    for quartile in quartile_order:
        group = samples[samples["disagreement_quartile"] == quartile]
        accuracy = float(group["is_correct"].mean())
        quartile_rows.append(
            {
                "quartile": quartile,
                "n": int(len(group)),
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
                "mae": float(group["abs_pred_minus_label"].mean()),
                "mean_disagreement": float(group["total_disagreement"].mean()),
                "min_disagreement": float(group["total_disagreement"].min()),
                "max_disagreement": float(group["total_disagreement"].max()),
            }
        )
    quartiles = pd.DataFrame(quartile_rows)

    spearman = spearmanr(
        samples["total_disagreement"],
        samples["abs_pred_minus_label"],
    )
    correct_mean = float(
        samples.loc[samples["is_correct"], "total_disagreement"].mean()
    )
    incorrect_mean = float(
        samples.loc[~samples["is_correct"], "total_disagreement"].mean()
    )

    q1 = samples[samples["disagreement_quartile"] == "Q1_low"]
    q4 = samples[samples["disagreement_quartile"] == "Q4_high"]
    q1_errors = int((~q1["is_correct"]).sum())
    q4_errors = int((~q4["is_correct"]).sum())
    q1_error_rate = q1_errors / len(q1)
    q4_error_rate = q4_errors / len(q4)
    fisher_result = fisher_exact(
        [
            [q4_errors, len(q4) - q4_errors],
            [q1_errors, len(q1) - q1_errors],
        ],
        alternative="two-sided",
    )

    statistics: Dict[str, float | int | None] = {
        "n": int(len(samples)),
        "accuracy": float(samples["is_correct"].mean()),
        "error_rate": float((~samples["is_correct"]).mean()),
        "mae": float(samples["abs_pred_minus_label"].mean()),
        "mean_disagreement": float(samples["total_disagreement"].mean()),
        "unique_disagreement_values": int(
            samples["total_disagreement"].nunique(dropna=False)
        ),
        "zero_disagreement_n": int((samples["total_disagreement"] == 0).sum()),
        "zero_disagreement_fraction": float(
            (samples["total_disagreement"] == 0).mean()
        ),
        "spearman_rho_disagreement_vs_abs_error": finite_or_none(
            float(spearman.statistic)
        ),
        "spearman_p_value": finite_or_none(float(spearman.pvalue)),
        "correct_mean_disagreement": correct_mean,
        "incorrect_mean_disagreement": incorrect_mean,
        "incorrect_minus_correct_mean_disagreement": incorrect_mean - correct_mean,
        "q1_error_rate": q1_error_rate,
        "q4_error_rate": q4_error_rate,
        "q4_minus_q1_error_rate": q4_error_rate - q1_error_rate,
        "q4_vs_q1_error_risk_ratio": (
            q4_error_rate / q1_error_rate if q1_error_rate > 0 else None
        ),
        "q4_vs_q1_fisher_odds_ratio": finite_or_none(
            float(fisher_result.statistic)
        ),
        "q4_vs_q1_fisher_p_value": finite_or_none(float(fisher_result.pvalue)),
    }
    return quartiles, statistics


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(
    checkpoint: Path,
    test_dir: Path,
    best_val_qwk: float | None,
    default_forward_max_abs_diff: float,
    quartiles: pd.DataFrame,
    statistics: Dict[str, float | int | None],
    attention_diagnostics: Dict[str, float | int],
) -> str:
    rho = statistics["spearman_rho_disagreement_vs_abs_error"]
    spearman_p = statistics["spearman_p_value"]
    q_gap = statistics["q4_minus_q1_error_rate"]
    fisher_p = statistics["q4_vs_q1_fisher_p_value"]
    mean_gap = statistics["incorrect_minus_correct_mean_disagreement"]
    quartiles_degenerate = bool(
        statistics["unique_disagreement_values"] < 4
        or statistics["zero_disagreement_fraction"] >= 0.75
    )
    positive_ordinal = bool(
        not quartiles_degenerate
        and rho is not None
        and rho > 0
        and spearman_p is not None
        and spearman_p < 0.05
    )
    higher_error = bool(
        not quartiles_degenerate and q_gap is not None and q_gap > 0 and mean_gap > 0
    )
    clearly_higher = bool(
        not quartiles_degenerate
        and q_gap is not None
        and q_gap > 0
        and fisher_p is not None
        and fisher_p < 0.05
    )
    worth_adaptive_dast = higher_error and positive_ordinal

    table_lines = [
        "| 四分位组 | n | Accuracy | 错误率 | MAE | 平均 disagreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in quartiles.itertuples(index=False):
        table_lines.append(
            f"| {row.quartile} | {row.n} | {percent(row.accuracy)} | "
            f"{percent(row.error_rate)} | {row.mae:.6f} | "
            f"{row.mean_disagreement:.8f} |"
        )

    best_score_text = "N/A" if best_val_qwk is None else f"{best_val_qwk:.4f}"
    rho_text = "N/A" if rho is None else f"{rho:.6f}"
    spearman_p_text = "N/A" if spearman_p is None else f"{spearman_p:.6g}"
    fisher_p_text = "N/A" if fisher_p is None else f"{fisher_p:.6g}"
    risk_ratio = statistics["q4_vs_q1_error_risk_ratio"]
    risk_ratio_text = "N/A" if risk_ratio is None else f"{risk_ratio:.3f}"

    return "\n".join(
        [
            "# KOA MESC 分支差异验证报告",
            "",
            "## 设置",
            "",
            f"- checkpoint：`{checkpoint}`",
            f"- checkpoint 最佳验证 QWK：{best_score_text}",
            f"- 固定测试集：`{test_dir}`（n={statistics['n']}）",
            "- 模型：ResNet50 + layer3 MESC + CE（未重新训练、未修改 loss）",
            "- 距离定义：三个 pairwise L1 距离均除以 `2*C`；三者之和为 `total_disagreement`。",
            "- 四个等人数分组按 disagreement 升序；并列值使用 sample_id 的 SHA-256 确定性打散，避免 ImageFolder 的类别/路径顺序污染分组。",
            f"- 默认 forward 与可选返回路径的最大 logit 绝对差：{default_forward_max_abs_diff:.3g}",
            "",
            "## 四分位结果",
            "",
            *table_lines,
            "",
            "## 总体统计",
            "",
            f"- Accuracy：{percent(statistics['accuracy'])}",
            f"- MAE：{statistics['mae']:.6f}",
            f"- disagreement 唯一值数量：{statistics['unique_disagreement_values']}",
            f"- disagreement 精确为 0：{statistics['zero_disagreement_n']}/{statistics['n']}（{percent(statistics['zero_disagreement_fraction'])}）",
            f"- disagreement 与 `abs(pred-label)` 的 Spearman rho：{rho_text}（p={spearman_p_text}）",
            f"- 正确样本平均 disagreement：{statistics['correct_mean_disagreement']:.8f}",
            f"- 错误样本平均 disagreement：{statistics['incorrect_mean_disagreement']:.8f}",
            f"- Q4-Q1 错误率差：{q_gap * 100:.2f} 个百分点；风险比：{risk_ratio_text}",
            f"- Q4 vs Q1 错误率 Fisher 精确检验：p={fisher_p_text}",
            f"- avg-max / avg-median / max-median 注意力元素精确相等率：{percent(attention_diagnostics['avg_max_exact_equal_fraction'])} / {percent(attention_diagnostics['avg_median_exact_equal_fraction'])} / {percent(attention_diagnostics['max_median_exact_equal_fraction'])}",
            "",
            "## 结论",
            "",
            f"- disagreement 高的样本是否更容易预测错误：{'是' if higher_error else '当前结果不支持；disagreement 大量并列为 0，无法形成可辨识的高低梯度'}。",
            f"- disagreement 是否与序数误差正相关：{'是' if positive_ordinal else '否；观测相关不显著且指标退化'}。",
            f"- 最高四分位错误率是否明显高于最低四分位：{'是' if clearly_higher else '否；四分位主要由 0 值并列打散产生'}。",
            f"- 是否值得继续尝试 adaptive DAST：{'值得作为后续受控实验继续验证' if worth_adaptive_dast else '不建议仅依据本次 post-sigmoid disagreement 结果推进；应先解决或重新定义可辨识的分支差异指标'}。",
            "",
            "> 这里的 disagreement 仅表示 MESC 三个通道注意力分支的内部差异，不解释为临床不确定性。",
            "",
        ]
    )


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
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    # Match the seed and cuDNN reproducibility settings used by the selected
    # controlled run. Evaluation transforms themselves contain no randomness.
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = build_test_dataset(test_dir)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = CustomResNet50MECS().to(device)
    best_val_qwk = load_checkpoint(model, checkpoint, device)

    print(f"Checkpoint: {checkpoint}")
    print(f"Best validation QWK: {best_val_qwk}")
    print(f"Test split: {test_dir} (n={len(dataset)})")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    samples, default_diff, attention_diagnostics = run_inference(
        model, loader, test_dir, device
    )
    samples = assign_equal_count_quartiles(samples)
    quartiles, statistics = calculate_statistics(samples)

    sample_csv = output_dir / "koa_mesc_branch_disagreement_samples.csv"
    quartile_csv = output_dir / "koa_mesc_branch_disagreement_quartiles.csv"
    statistics_json = output_dir / "koa_mesc_branch_disagreement_statistics.json"
    report_path = output_dir / "koa_mesc_branch_disagreement_report.md"

    samples.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    quartiles.to_csv(quartile_csv, index=False, encoding="utf-8-sig")
    statistics_payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_best_val_qwk": best_val_qwk,
        "test_dir": str(test_dir),
        "class_to_idx": dataset.class_to_idx,
        "default_forward_max_abs_logit_diff": default_diff,
        "attention_diagnostics": attention_diagnostics,
        **statistics,
    }
    statistics_json.write_text(
        json.dumps(statistics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        render_report(
            checkpoint,
            test_dir,
            best_val_qwk,
            default_diff,
            quartiles,
            statistics,
            attention_diagnostics,
        ),
        encoding="utf-8",
    )

    print("\nQuartile summary:")
    print(quartiles.to_string(index=False))
    print("\nOverall statistics:")
    print(json.dumps(statistics_payload, indent=2, ensure_ascii=False))
    print(f"\nSample CSV: {sample_csv}")
    print(f"Quartile CSV: {quartile_csv}")
    print(f"Statistics JSON: {statistics_json}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
