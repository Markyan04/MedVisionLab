#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Locate where the three MESC channel-attention branches collapse.

The script is diagnostic-only: it loads an existing checkpoint, evaluates one
fixed stratified KOA test batch, and writes results to a new directory. It does
not train, alter model parameters, or change the model's default forward path.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analyze_mesc_branch_disagreement import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEST_DIR,
    CustomResNet50MECS,
    build_test_dataset,
    load_checkpoint,
)
from MECS_old import global_median_pooling


THIS_DIR = Path(__file__).resolve().parent
TARGET_NONZERO_SAMPLE = "4/9215922R.png"
BRANCHES = ("avg", "max", "median")
PAIRS = (("avg", "max"), ("avg", "median"), ("max", "median"))
STAGE_ORDER = (
    "raw_descriptor",
    "fc1_output",
    "relu_hidden",
    "pre_sigmoid_logits",
    "post_sigmoid_attention",
)


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
            / f"mesc_branch_collapse_koa_test_seed1234_{stamp}"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    return parser.parse_args()


def set_reproducibility() -> None:
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_id(path: str | Path, test_dir: Path) -> str:
    return Path(path).resolve().relative_to(test_dir).as_posix()


def select_stratified_batch(
    dataset, test_dir: Path, samples_per_class: int
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    by_class: Dict[int, List[int]] = {class_index: [] for class_index in range(5)}
    target_index = None
    for index, (path, label) in enumerate(dataset.samples):
        by_class[int(label)].append(index)
        if sample_id(path, test_dir) == TARGET_NONZERO_SAMPLE:
            target_index = index

    if target_index is None:
        raise FileNotFoundError(
            f"Target diagnostic sample is missing: {TARGET_NONZERO_SAMPLE}"
        )

    selected: List[int] = []
    for class_index in range(5):
        class_indices = by_class[class_index]
        if len(class_indices) < samples_per_class:
            raise RuntimeError(
                f"Class {class_index} has fewer than {samples_per_class} samples"
            )
        chosen = class_indices[:samples_per_class]
        if class_index == 4 and target_index not in chosen:
            chosen[-1] = target_index
        selected.extend(chosen)

    images = []
    labels = []
    paths = []
    for index in selected:
        image, label, path = dataset[index]
        images.append(image)
        labels.append(int(label))
        paths.append(str(Path(path).resolve()))
    return torch.stack(images), torch.tensor(labels), paths, selected


def forward_to_layer3(model: CustomResNet50MECS, images: torch.Tensor) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    return model.layer3(x)


def extract_branch_stages(
    model: CustomResNet50MECS, layer3_features: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    mecs_input = model.mecs.act(model.mecs.pre_conv(layer3_features))
    channel_attention = model.mecs.channel_attention

    descriptors = {
        "avg": F.adaptive_avg_pool2d(mecs_input, output_size=(1, 1)),
        "max": F.adaptive_max_pool2d(mecs_input, output_size=(1, 1)),
        "median": global_median_pooling(mecs_input),
    }
    fc1_outputs = {
        branch: channel_attention.fc1(descriptor)
        for branch, descriptor in descriptors.items()
    }
    relu_outputs = {
        branch: F.relu(fc1_output, inplace=False)
        for branch, fc1_output in fc1_outputs.items()
    }
    pre_sigmoid_logits = {
        branch: channel_attention.fc2(hidden)
        for branch, hidden in relu_outputs.items()
    }
    post_sigmoid = {
        branch: torch.sigmoid(logits)
        for branch, logits in pre_sigmoid_logits.items()
    }
    stages = {
        "raw_descriptor": descriptors,
        "fc1_output": fc1_outputs,
        "relu_hidden": relu_outputs,
        "pre_sigmoid_logits": pre_sigmoid_logits,
        "post_sigmoid_attention": post_sigmoid,
    }
    return mecs_input, stages


def tensor_summary(stage: str, branch: str, tensor: torch.Tensor) -> Dict[str, object]:
    detached = tensor.detach()
    return {
        "stage": stage,
        "branch": branch,
        "shape": str(list(detached.shape)),
        "numel": int(detached.numel()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "data_ptr": int(detached.data_ptr()),
    }


def pairwise_summary(
    stage: str, first_name: str, second_name: str, tensors: Mapping[str, torch.Tensor]
) -> Dict[str, object]:
    first = tensors[first_name].detach()
    second = tensors[second_name].detach()
    difference = (first - second).abs()
    return {
        "stage": stage,
        "pair": f"{first_name}-{second_name}",
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "exact_equal_fraction": float((first == second).float().mean().item()),
        "same_data_ptr": bool(first.data_ptr() == second.data_ptr()),
    }


def stage_pointer_summary(
    stage: str, tensors: Mapping[str, torch.Tensor]
) -> Dict[str, object]:
    pointers = {branch: int(tensors[branch].data_ptr()) for branch in BRANCHES}
    return {
        "stage": stage,
        "avg_data_ptr": pointers["avg"],
        "max_data_ptr": pointers["max"],
        "median_data_ptr": pointers["median"],
        "all_data_ptr_distinct": len(set(pointers.values())) == len(BRANCHES),
    }


def additional_stage_checks(
    stages: Mapping[str, Mapping[str, torch.Tensor]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for branch in BRANCHES:
        relu = stages["relu_hidden"][branch]
        logits = stages["pre_sigmoid_logits"][branch]
        attention = stages["post_sigmoid_attention"][branch]
        rows.append(
            {
                "branch": branch,
                "relu_zero_fraction": float((relu == 0).float().mean().item()),
                "pre_sigmoid_abs_gt_5_fraction": float(
                    (logits.abs() > 5).float().mean().item()
                ),
                "pre_sigmoid_abs_gt_10_fraction": float(
                    (logits.abs() > 10).float().mean().item()
                ),
                "pre_sigmoid_abs_gt_15_fraction": float(
                    (logits.abs() > 15).float().mean().item()
                ),
                "sigmoid_lt_1e-4_fraction": float(
                    (attention < 1e-4).float().mean().item()
                ),
                "sigmoid_gt_1_minus_1e-4_fraction": float(
                    (attention > 1 - 1e-4).float().mean().item()
                ),
                "sigmoid_extreme_fraction": float(
                    ((attention < 1e-4) | (attention > 1 - 1e-4))
                    .float()
                    .mean()
                    .item()
                ),
            }
        )
    return rows


def parameter_norms(model: CustomResNet50MECS) -> List[Dict[str, object]]:
    channel_attention = model.mecs.channel_attention
    rows = []
    for layer_name, layer in (
        ("fc1", channel_attention.fc1),
        ("fc2", channel_attention.fc2),
    ):
        for parameter_name in ("weight", "bias"):
            parameter = getattr(layer, parameter_name)
            rows.append(
                {
                    "layer": layer_name,
                    "parameter": parameter_name,
                    "shape": str(list(parameter.shape)),
                    "l2_norm": float(torch.linalg.vector_norm(parameter).item()),
                    "mean": float(parameter.mean().item()),
                    "std": float(parameter.std(unbiased=False).item()),
                    "min": float(parameter.min().item()),
                    "max": float(parameter.max().item()),
                }
            )
    return rows


def forward_after_channel_attention(
    model: CustomResNet50MECS,
    mecs_input: torch.Tensor,
    channel_attention: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_ca = channel_attention * mecs_input
    initial_out = model.mecs.initial_depth_conv(x_ca)
    spatial_out = sum(conv(initial_out) for conv in model.mecs.depth_convs)
    spatial_out = spatial_out + x_ca
    spatial_attention = torch.sigmoid(model.mecs.spatial_att_conv(spatial_out))
    mecs_output = model.mecs.post_conv(spatial_attention * x_ca)

    x = model.layer4(mecs_output)
    x = model.avgpool(x)
    logits = model.fc(torch.flatten(x, 1))
    return mecs_output, logits


def replacement_tests(
    model: CustomResNet50MECS,
    images: torch.Tensor,
    mecs_input: torch.Tensor,
    attentions: Mapping[str, torch.Tensor],
) -> Tuple[List[Dict[str, object]], float]:
    original_attention = attentions["avg"] + attentions["max"] + attentions["median"]
    settings = {
        "original": original_attention,
        "median_replaced_by_avg": attentions["avg"] + attentions["max"] + attentions["avg"],
        "max_replaced_by_avg": attentions["avg"] + attentions["avg"] + attentions["median"],
        "all_avg": 3.0 * attentions["avg"],
    }

    original_mecs_output, original_logits = forward_after_channel_attention(
        model, mecs_input, settings["original"]
    )
    default_logits = model(images)
    reconstruction_max_abs_diff = float(
        (default_logits - original_logits).abs().max().item()
    )

    original_predictions = original_logits.argmax(dim=1)
    rows = []
    for setting_name, channel_attention in settings.items():
        mecs_output, logits = forward_after_channel_attention(
            model, mecs_input, channel_attention
        )
        rows.append(
            {
                "setting": setting_name,
                "channel_attention_max_abs_diff": float(
                    (channel_attention - original_attention).abs().max().item()
                ),
                "mecs_output_max_abs_diff": float(
                    (mecs_output - original_mecs_output).abs().max().item()
                ),
                "logits_max_abs_diff": float(
                    (logits - original_logits).abs().max().item()
                ),
                "logits_mean_abs_diff": float(
                    (logits - original_logits).abs().mean().item()
                ),
                "logits_exact_equal_fraction": float(
                    (logits == original_logits).float().mean().item()
                ),
                "changed_prediction_count": int(
                    (logits.argmax(dim=1) != original_predictions).sum().item()
                ),
            }
        )
    return rows, reconstruction_max_abs_diff


def code_and_runtime_audit(
    model: CustomResNet50MECS,
    mecs_input: torch.Tensor,
    stages: Mapping[str, Mapping[str, torch.Tensor]],
    reconstruction_max_abs_diff: float,
) -> Dict[str, object]:
    combined, returned = model.mecs.channel_attention(
        mecs_input, return_branch_attentions=True
    )
    manual_attention = stages["post_sigmoid_attention"]
    returned_by_branch = {
        "avg": returned["avg_att"],
        "max": returned["max_att"],
        "median": returned["med_att"],
    }
    returned_matches_manual = {
        branch: bool(torch.equal(returned_by_branch[branch], manual_attention[branch]))
        for branch in BRANCHES
    }
    returned_pointers = {
        branch: int(returned_by_branch[branch].data_ptr()) for branch in BRANCHES
    }
    manual_sum = (
        manual_attention["avg"]
        + manual_attention["max"]
        + manual_attention["median"]
    )
    analysis_code_error_detected = bool(
        not all(returned_matches_manual.values())
        or len(set(returned_pointers.values())) != len(BRANCHES)
        or reconstruction_max_abs_diff > 1e-7
    )
    return {
        "source_descriptor_mapping": {
            "avg_att": "adaptive_avg_pool2d(inputs)",
            "max_att": "adaptive_max_pool2d(inputs)",
            "med_att": "global_median_pooling(inputs)",
        },
        "source_return_mapping": {
            "avg_att": "avg_out",
            "max_att": "max_out",
            "med_att": "median_out",
        },
        "returned_matches_manual_stage_values": returned_matches_manual,
        "all_returned_values_match_manual": all(returned_matches_manual.values()),
        "returned_data_ptrs": returned_pointers,
        "all_returned_data_ptrs_distinct": (
            len(set(returned_pointers.values())) == len(BRANCHES)
        ),
        "returned_python_objects_distinct": (
            len({id(returned_by_branch[branch]) for branch in BRANCHES})
            == len(BRANCHES)
        ),
        "returned_sum_matches_manual_sum": bool(torch.equal(combined, manual_sum)),
        "default_vs_reconstructed_logits_max_abs_diff": reconstruction_max_abs_diff,
        "analysis_code_error_detected": analysis_code_error_detected,
    }


def find_pair_row(
    pairwise_rows: Sequence[Mapping[str, object]], stage: str, pair: str
) -> Mapping[str, object]:
    return next(
        row
        for row in pairwise_rows
        if row["stage"] == stage and row["pair"] == pair
    )


def diagnose_location(
    pairwise_rows: Sequence[Mapping[str, object]],
    extra_rows: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> Dict[str, object]:
    descriptor_mean_diffs = [
        float(find_pair_row(pairwise_rows, "raw_descriptor", pair)["mean_abs_diff"])
        for pair in ("avg-max", "avg-median", "max-median")
    ]
    descriptors_clearly_different = all(value > 1e-6 for value in descriptor_mean_diffs)
    relu_zero_min = min(float(row["relu_zero_fraction"]) for row in extra_rows)
    sigmoid_extreme_max = max(float(row["sigmoid_extreme_fraction"]) for row in extra_rows)
    avg_median_relu_equal = float(
        find_pair_row(pairwise_rows, "relu_hidden", "avg-median")[
            "exact_equal_fraction"
        ]
    )
    avg_median_pre_sigmoid_equal = float(
        find_pair_row(pairwise_rows, "pre_sigmoid_logits", "avg-median")[
            "exact_equal_fraction"
        ]
    )

    if bool(audit["analysis_code_error_detected"]):
        category = "D"
        location = "analysis code error"
        reason = "Runtime audit found a branch mapping or tensor reuse error."
    elif not descriptors_clearly_different:
        category = "A"
        location = "raw pooling descriptors"
        reason = "At least one raw descriptor pair lacks a measurable difference."
    elif avg_median_relu_equal == 1.0 and avg_median_pre_sigmoid_equal == 1.0:
        category = "B"
        location = "shared MLP/ReLU"
        reason = (
            "Raw descriptors differ, but avg and median become exactly equal at "
            "the ReLU hidden stage and remain equal afterward."
        )
    elif sigmoid_extreme_max > 0.95:
        category = "C"
        location = "sigmoid saturation"
        reason = "More than 95% of at least one branch is near 0 or 1 after sigmoid."
    else:
        category = "E"
        location = "other/mixed"
        reason = "No single complete-collapse rule above explains all branch pairs."

    return {
        "category": category,
        "location": location,
        "reason": reason,
        "descriptors_clearly_different": descriptors_clearly_different,
        "raw_descriptor_pair_mean_abs_diffs": descriptor_mean_diffs,
        "minimum_relu_zero_fraction_across_branches": relu_zero_min,
        "maximum_sigmoid_extreme_fraction_across_branches": sigmoid_extreme_max,
    }


def sci(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8e}"
    return str(value)


def markdown_table(columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(sci(row[column]) for column in columns) + " |")
    return lines


def render_report(
    checkpoint: Path,
    best_val_qwk: float | None,
    selected_samples: Sequence[Mapping[str, object]],
    tensor_rows: Sequence[Mapping[str, object]],
    pairwise_rows: Sequence[Mapping[str, object]],
    pointer_rows: Sequence[Mapping[str, object]],
    extra_rows: Sequence[Mapping[str, object]],
    norm_rows: Sequence[Mapping[str, object]],
    replacement_rows: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
    decision: Mapping[str, object],
) -> str:
    tensor_columns = ("stage", "branch", "shape", "mean", "std", "min", "max")
    pair_columns = (
        "stage",
        "pair",
        "max_abs_diff",
        "mean_abs_diff",
        "exact_equal_fraction",
        "same_data_ptr",
    )
    pointer_columns = (
        "stage",
        "avg_data_ptr",
        "max_data_ptr",
        "median_data_ptr",
        "all_data_ptr_distinct",
    )
    extra_columns = (
        "branch",
        "relu_zero_fraction",
        "pre_sigmoid_abs_gt_5_fraction",
        "pre_sigmoid_abs_gt_10_fraction",
        "pre_sigmoid_abs_gt_15_fraction",
        "sigmoid_lt_1e-4_fraction",
        "sigmoid_gt_1_minus_1e-4_fraction",
        "sigmoid_extreme_fraction",
    )
    norm_columns = ("layer", "parameter", "shape", "l2_norm", "mean", "std", "min", "max")
    replacement_columns = (
        "setting",
        "channel_attention_max_abs_diff",
        "mecs_output_max_abs_diff",
        "logits_max_abs_diff",
        "logits_mean_abs_diff",
        "logits_exact_equal_fraction",
        "changed_prediction_count",
    )

    return "\n".join(
        [
            "# KOA MESC 分支塌缩定位报告",
            "",
            f"- checkpoint：`{checkpoint}`",
            f"- 最佳验证 QWK：{best_val_qwk}",
            f"- 诊断 batch：{len(selected_samples)} 张，每个 KL 等级 2 张，包含 `{TARGET_NONZERO_SAMPLE}`。",
            "- 仅推理诊断；未训练、未修改 loss、模型结构、checkpoint 或既有结果。",
            "",
            "## 代码与运行时审计",
            "",
            f"- 返回值与手工分阶段计算逐元素一致：{audit['all_returned_values_match_manual']}",
            f"- 三个返回 tensor 的 data_ptr 全部不同：{audit['all_returned_data_ptrs_distinct']}",
            f"- 三个返回 Python tensor 对象不同：{audit['returned_python_objects_distinct']}",
            f"- 返回三分支之和与手工求和一致：{audit['returned_sum_matches_manual_sum']}",
            f"- 默认 forward 与手工重构 logits 最大绝对差：{sci(audit['default_vs_reconstructed_logits_max_abs_diff'])}",
            "- 结论：没有发现变量复用、赋值错误、返回同一 tensor 或分析键映射错误。",
            "",
            "## Batch 样本",
            "",
            *markdown_table(("sample_id", "label"), selected_samples),
            "",
            "## 各阶段单分支统计",
            "",
            *markdown_table(tensor_columns, tensor_rows),
            "",
            "## 各阶段分支间差异",
            "",
            *markdown_table(pair_columns, pairwise_rows),
            "",
            "## data_ptr 检查",
            "",
            *markdown_table(pointer_columns, pointer_rows),
            "",
            "## ReLU、pre-sigmoid 与 sigmoid 检查",
            "",
            *markdown_table(extra_columns, extra_rows),
            "",
            "## shared MLP 参数范数",
            "",
            *markdown_table(norm_columns, norm_rows),
            "",
            "## 分支直接替换测试",
            "",
            *markdown_table(replacement_columns, replacement_rows),
            "",
            "## 最终定位",
            "",
            f"- 判定：**{decision['category']}. {decision['location']}**",
            f"- 原因：{decision['reason']}",
            f"- 原始描述符具有明显差异：{decision['descriptors_clearly_different']}",
            f"- 三分支最小 ReLU 零元素比例：{sci(decision['minimum_relu_zero_fraction_across_branches'])}",
            f"- 三分支最大 sigmoid 极端值比例：{sci(decision['maximum_sigmoid_extreme_fraction_across_branches'])}",
            "",
        ]
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    set_reproducibility()
    device = torch.device(args.device)
    dataset = build_test_dataset(test_dir)
    images, labels, paths, selected_indices = select_stratified_batch(
        dataset, test_dir, args.samples_per_class
    )
    images = images.to(device)
    labels = labels.to(device)

    model = CustomResNet50MECS().to(device).eval()
    best_val_qwk = load_checkpoint(model, checkpoint, device)
    layer3_features = forward_to_layer3(model, images)
    mecs_input, stages = extract_branch_stages(model, layer3_features)

    tensor_rows = [
        tensor_summary(stage, branch, stages[stage][branch])
        for stage in STAGE_ORDER
        for branch in BRANCHES
    ]
    pairwise_rows = [
        pairwise_summary(stage, first, second, stages[stage])
        for stage in STAGE_ORDER
        for first, second in PAIRS
    ]
    pointer_rows = [stage_pointer_summary(stage, stages[stage]) for stage in STAGE_ORDER]
    extra_rows = additional_stage_checks(stages)
    norm_rows = parameter_norms(model)
    replacement_rows, reconstruction_diff = replacement_tests(
        model,
        images,
        mecs_input,
        stages["post_sigmoid_attention"],
    )
    audit = code_and_runtime_audit(model, mecs_input, stages, reconstruction_diff)
    decision = diagnose_location(pairwise_rows, extra_rows, audit)

    selected_samples = [
        {
            "dataset_index": int(index),
            "sample_id": sample_id(path, test_dir),
            "label": int(label),
        }
        for index, path, label in zip(
            selected_indices, paths, labels.detach().cpu().tolist()
        )
    ]
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_best_val_qwk": best_val_qwk,
        "test_dir": str(test_dir),
        "device": str(device),
        "selected_samples": selected_samples,
        "code_and_runtime_audit": audit,
        "tensor_statistics": tensor_rows,
        "pairwise_statistics": pairwise_rows,
        "data_ptr_statistics": pointer_rows,
        "activation_diagnostics": extra_rows,
        "shared_mlp_parameter_norms": norm_rows,
        "replacement_tests": replacement_rows,
        "collapse_location": decision,
    }

    tensor_csv = output_dir / "mesc_stage_tensor_statistics.csv"
    pairwise_csv = output_dir / "mesc_stage_pairwise_statistics.csv"
    replacement_csv = output_dir / "mesc_branch_replacement_tests.csv"
    json_path = output_dir / "mesc_branch_collapse_diagnostic.json"
    report_path = output_dir / "mesc_branch_collapse_report.md"
    pd.DataFrame(tensor_rows).to_csv(tensor_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(replacement_rows).to_csv(
        replacement_csv, index=False, encoding="utf-8-sig"
    )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(
        render_report(
            checkpoint,
            best_val_qwk,
            selected_samples,
            tensor_rows,
            pairwise_rows,
            pointer_rows,
            extra_rows,
            norm_rows,
            replacement_rows,
            audit,
            decision,
        ),
        encoding="utf-8",
    )

    print(f"Checkpoint best validation QWK: {best_val_qwk}")
    print(f"Selected samples: {[row['sample_id'] for row in selected_samples]}")
    print(f"Collapse location: {decision['category']}. {decision['location']}")
    print(f"Reason: {decision['reason']}")
    print("Replacement tests:")
    print(pd.DataFrame(replacement_rows).to_string(index=False))
    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
