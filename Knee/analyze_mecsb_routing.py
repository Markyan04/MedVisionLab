#!/usr/bin/env python
"""Analyze learned MECS VersionB routing without training or changing weights.

For each checkpoint this script runs one deterministic KOA split, records
per-sample and per-channel avg/max/median routing statistics, and measures how
far the router has moved from its equal-weight initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Subset


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for search_path in (PROJECT_ROOT, THIS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from MECS_VersionB import MECS_VersionB  # noqa: E402


BRANCH_NAMES = ("avg", "max", "med")
NUM_CLASSES = 5
CHANNELS = 1024
EPS = 1e-12


class RoutingInspectableResNet50MECS(nn.Module):
    """State-dict-compatible ResNet50 layer3 + MECS VersionB model."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        base_model = models.resnet50(weights=None)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.mecs = MECS_VersionB(in_channels=CHANNELS, out_channels=CHANNELS)
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if return_routing:
            x, routing_info = self.mecs(x, return_branch_attentions=True)
        else:
            x = self.mecs(x)
            routing_info = None
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        if return_routing:
            return logits, routing_info
        return logits


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether trained MECS VersionB routing remains uniform."
    )
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "auto_test"), default="test")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic prefix for a smoke test; omit for the full split.",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.num_workers < 0 or args.image_size <= 0:
        parser.error("batch-size/image-size must be positive and num-workers non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive when provided.")

    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    if len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error("Duplicate checkpoint paths were provided.")
    if len({path.stem for path in args.checkpoints}) != len(args.checkpoints):
        parser.error("Checkpoint filenames must have unique stems for output folders.")
    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = THIS_DIR / "analysis_results" / f"mecsb_routing_{args.split}_{stamp}"
    args.output_root = args.output_root.expanduser().resolve()
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(args: argparse.Namespace) -> Tuple[DataLoader, List[str], Mapping[str, int]]:
    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    base_dataset = datasets.ImageFolder(split_dir, transform=transform)
    sample_paths = [str(path) for path, _ in base_dataset.samples]
    dataset: Any = base_dataset
    if args.max_samples is not None:
        limit = min(args.max_samples, len(base_dataset))
        dataset = Subset(base_dataset, range(limit))
        sample_paths = sample_paths[:limit]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sample_paths, base_dataset.class_to_idx


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint object in {path}")
    return checkpoint


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a model state dict.")
    state = dict(state)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    required = (
        "mecs.channel_attention.router.weight",
        "mecs.channel_attention.router.bias",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            "Checkpoint is not MECS VersionB; missing router parameters: "
            + ", ".join(missing)
        )
    return state


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, ensure_ascii=False, indent=2)


def infer_identity(path: Path) -> Tuple[str, Optional[int]]:
    match = re.search(r"mecsb_(ce|dast)_seed(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return "unknown", None
    return match.group(1).lower(), int(match.group(2))


def describe(values: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q1": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_q3": float(np.quantile(values, 0.75)),
        f"{prefix}_max": float(np.max(values)),
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        rho = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(rho) or not math.isfinite(p_value):
            return None, None
        return rho, p_value
    except Exception:
        return None, None


def routing_verdict(summary: Mapping[str, Any]) -> str:
    parameter_max = max(
        float(summary["router_weight_max_abs"]),
        float(summary["router_bias_max_abs"]),
    )
    deviation = float(summary["route_mean_abs_deviation_from_uniform"])
    entropy = float(summary["route_normalized_entropy_mean"])
    if parameter_max <= 1e-8:
        return "Router parameters remain at zero initialization; routing is exactly uniform."
    if deviation < 0.005 and entropy > 0.999:
        return "Router parameters changed, but effective routing remains almost exactly uniform."
    if deviation < 0.020 and entropy > 0.990:
        return "Routing learned only a mild departure from equal weighting."
    return "Routing shows a material departure from equal weighting; inspect sample/channel variation for usefulness."


@torch.no_grad()
def analyze_checkpoint(
    checkpoint_path: Path,
    loader: Iterable[Any],
    sample_paths: Sequence[str],
    device: torch.device,
    output_dir: Path,
    split_name: str,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    state = extract_state_dict(checkpoint)
    model = RoutingInspectableResNet50MECS().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    router_weight = model.mecs.channel_attention.router.weight.detach().cpu().double()
    router_bias = model.mecs.channel_attention.router.bias.detach().cpu().double()

    route_chunks: List[torch.Tensor] = []
    entropy_chunks: List[torch.Tensor] = []
    max_weight_chunks: List[torch.Tensor] = []
    max_deviation_chunks: List[torch.Tensor] = []
    dominance_margin_chunks: List[torch.Tensor] = []
    sample_rows: List[Dict[str, Any]] = []
    all_labels: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    channel_sum = torch.zeros(CHANNELS, 3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(CHANNELS, 3, dtype=torch.float64)
    channel_argmax = torch.zeros(CHANNELS, 3, dtype=torch.float64)
    cursor = 0
    max_weight_sum_error = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, info = model(images, return_routing=True)
        weights = info["routing_weights"].detach().float()
        if weights.ndim != 3 or weights.shape[1:] != (CHANNELS, 3):
            raise RuntimeError(f"Unexpected routing shape: {tuple(weights.shape)}")

        max_weight_sum_error = max(
            max_weight_sum_error,
            float((weights.sum(dim=-1) - 1.0).abs().max().item()),
        )
        predictions = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)
        entropy = -(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(3.0)
        deviations = (weights - (1.0 / 3.0)).abs()
        sorted_weights = torch.sort(weights, dim=-1, descending=True).values
        dominance_margin = sorted_weights[..., 0] - sorted_weights[..., 1]
        branch_argmax = weights.argmax(dim=-1)

        weights_cpu = weights.cpu()
        route_chunks.append(weights_cpu.reshape(-1, 3))
        entropy_chunks.append(normalized_entropy.cpu().reshape(-1))
        max_weight_chunks.append(sorted_weights[..., 0].cpu().reshape(-1))
        max_deviation_chunks.append(deviations.max(dim=-1).values.cpu().reshape(-1))
        dominance_margin_chunks.append(dominance_margin.cpu().reshape(-1))
        channel_sum += weights_cpu.double().sum(dim=0)
        channel_sum_sq += weights_cpu.double().square().sum(dim=0)
        channel_argmax += torch.nn.functional.one_hot(
            branch_argmax.cpu(), num_classes=3
        ).double().sum(dim=0)

        sample_route_mean = weights.mean(dim=1).cpu().numpy()
        sample_route_std = weights.std(dim=1, unbiased=False).cpu().numpy()
        sample_entropy_mean = normalized_entropy.mean(dim=1).cpu().numpy()
        sample_entropy_std = normalized_entropy.std(dim=1, unbiased=False).cpu().numpy()
        sample_uniform_mad = deviations.mean(dim=(1, 2)).cpu().numpy()
        sample_max_weight = sorted_weights[..., 0].mean(dim=1).cpu().numpy()
        sample_margin = dominance_margin.mean(dim=1).cpu().numpy()
        sample_argmax_fraction = torch.nn.functional.one_hot(
            branch_argmax, num_classes=3
        ).float().mean(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()

        for batch_index in range(len(labels_np)):
            sample_index = cursor + batch_index
            row: Dict[str, Any] = {
                "sample_index": sample_index,
                "image_path": sample_paths[sample_index],
                "label": int(labels_np[batch_index]),
                "prediction": int(predictions_np[batch_index]),
                "correct": int(labels_np[batch_index] == predictions_np[batch_index]),
                "absolute_error": int(abs(labels_np[batch_index] - predictions_np[batch_index])),
                "predicted_probability": float(probabilities_np[batch_index].max()),
                "normalized_entropy_mean": float(sample_entropy_mean[batch_index]),
                "normalized_entropy_std": float(sample_entropy_std[batch_index]),
                "mean_abs_deviation_from_uniform": float(sample_uniform_mad[batch_index]),
                "max_route_weight_mean": float(sample_max_weight[batch_index]),
                "dominance_margin_mean": float(sample_margin[batch_index]),
            }
            for branch_index, branch_name in enumerate(BRANCH_NAMES):
                row[f"{branch_name}_route_mean"] = float(
                    sample_route_mean[batch_index, branch_index]
                )
                row[f"{branch_name}_route_std"] = float(
                    sample_route_std[batch_index, branch_index]
                )
                row[f"{branch_name}_argmax_fraction"] = float(
                    sample_argmax_fraction[batch_index, branch_index]
                )
            sample_rows.append(row)

        all_labels.append(labels_np)
        all_predictions.append(predictions_np)
        cursor += len(labels_np)

    if cursor != len(sample_paths):
        raise RuntimeError(f"Processed {cursor} samples but expected {len(sample_paths)}")

    route_values = torch.cat(route_chunks, dim=0).numpy()
    entropy_values = torch.cat(entropy_chunks).numpy()
    max_weight_values = torch.cat(max_weight_chunks).numpy()
    max_deviation_values = torch.cat(max_deviation_chunks).numpy()
    dominance_margin_values = torch.cat(dominance_margin_chunks).numpy()
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    absolute_errors = np.abs(predictions - labels)
    sample_entropy_values = np.array(
        [row["normalized_entropy_mean"] for row in sample_rows], dtype=np.float64
    )
    sample_deviation_values = np.array(
        [row["mean_abs_deviation_from_uniform"] for row in sample_rows], dtype=np.float64
    )
    sample_branch_means = np.array(
        [
            [row[f"{branch}_route_mean"] for branch in BRANCH_NAMES]
            for row in sample_rows
        ],
        dtype=np.float64,
    )

    n_samples = len(sample_rows)
    channel_mean = (channel_sum / n_samples).numpy()
    channel_var = (channel_sum_sq / n_samples).numpy() - np.square(channel_mean)
    channel_std = np.sqrt(np.maximum(channel_var, 0.0))
    channel_argmax_fraction = (channel_argmax / n_samples).numpy()
    channel_rows: List[Dict[str, Any]] = []
    for channel in range(CHANNELS):
        row = {"channel": channel}
        for branch_index, branch_name in enumerate(BRANCH_NAMES):
            row[f"{branch_name}_route_mean"] = float(channel_mean[channel, branch_index])
            row[f"{branch_name}_route_std"] = float(channel_std[channel, branch_index])
            row[f"{branch_name}_argmax_fraction"] = float(
                channel_argmax_fraction[channel, branch_index]
            )
        channel_rows.append(row)

    mode, seed = infer_identity(checkpoint_path)
    entropy_rho, entropy_p = safe_spearman(sample_entropy_values, absolute_errors)
    deviation_rho, deviation_p = safe_spearman(sample_deviation_values, absolute_errors)
    correct_mask = absolute_errors == 0
    incorrect_mask = ~correct_mask

    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "seed": seed,
        "split": split_name,
        "n": n_samples,
        "channels": CHANNELS,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "mae": float(absolute_errors.mean()),
        "router_weight_l1": float(router_weight.abs().sum().item()),
        "router_weight_l2": float(router_weight.norm().item()),
        "router_weight_max_abs": float(router_weight.abs().max().item()),
        "router_weight_exact_zero_fraction": float((router_weight == 0).double().mean().item()),
        "router_bias_l1": float(router_bias.abs().sum().item()),
        "router_bias_l2": float(router_bias.norm().item()),
        "router_bias_max_abs": float(router_bias.abs().max().item()),
        "router_bias_exact_zero_fraction": float((router_bias == 0).double().mean().item()),
        "routing_weight_sum_max_abs_error": max_weight_sum_error,
        "route_normalized_entropy_mean": float(entropy_values.mean()),
        "route_normalized_entropy_std": float(entropy_values.std()),
        "route_effective_branch_count_mean": float(
            np.exp(entropy_values * math.log(3.0)).mean()
        ),
        "route_mean_abs_deviation_from_uniform": float(
            np.abs(route_values - (1.0 / 3.0)).mean()
        ),
        "route_max_abs_deviation_from_uniform": float(max_deviation_values.max()),
        "near_uniform_1e-3_fraction": float((max_deviation_values < 1e-3).mean()),
        "near_uniform_1e-2_fraction": float((max_deviation_values < 1e-2).mean()),
        "near_uniform_5e-2_fraction": float((max_deviation_values < 5e-2).mean()),
        "dominance_tie_1e-6_fraction": float((dominance_margin_values < 1e-6).mean()),
        "dominance_margin_mean": float(dominance_margin_values.mean()),
        "sample_branch_mean_variation": float(sample_branch_means.std(axis=0).mean()),
        "channel_branch_mean_variation": float(channel_mean.std(axis=0).mean()),
        "entropy_vs_absolute_error_spearman_rho": entropy_rho,
        "entropy_vs_absolute_error_spearman_p": entropy_p,
        "uniform_deviation_vs_absolute_error_spearman_rho": deviation_rho,
        "uniform_deviation_vs_absolute_error_spearman_p": deviation_p,
        "correct_normalized_entropy_mean": (
            float(sample_entropy_values[correct_mask].mean()) if correct_mask.any() else None
        ),
        "incorrect_normalized_entropy_mean": (
            float(sample_entropy_values[incorrect_mask].mean()) if incorrect_mask.any() else None
        ),
        "correct_uniform_deviation_mean": (
            float(sample_deviation_values[correct_mask].mean()) if correct_mask.any() else None
        ),
        "incorrect_uniform_deviation_mean": (
            float(sample_deviation_values[incorrect_mask].mean()) if incorrect_mask.any() else None
        ),
    }
    summary.update(describe(max_weight_values, "max_route_weight"))
    summary.update(describe(dominance_margin_values, "dominance_margin"))
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        values = route_values[:, branch_index]
        summary.update(describe(values, f"{branch_name}_route"))
        summary[f"{branch_name}_argmax_fraction"] = float(
            (route_values.argmax(axis=1) == branch_index).mean()
        )
        summary[f"{branch_name}_sample_mean_std"] = float(
            sample_branch_means[:, branch_index].std()
        )
        summary[f"{branch_name}_channel_mean_std"] = float(
            channel_mean[:, branch_index].std()
        )
    summary["verdict"] = routing_verdict(summary)

    write_csv(output_dir / "sample_routing.csv", sample_rows)
    write_csv(output_dir / "channel_routing.csv", channel_rows)
    write_json(output_dir / "routing_summary.json", summary)
    report = routing_report(summary)
    with (output_dir / "routing_report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    return summary


def routing_report(summary: Mapping[str, Any]) -> str:
    branch_rows = []
    for branch in BRANCH_NAMES:
        branch_rows.append(
            f"| {branch} | {summary[f'{branch}_route_mean']:.6f} | "
            f"{summary[f'{branch}_route_std']:.6f} | "
            f"{summary[f'{branch}_argmax_fraction']:.6f} | "
            f"{summary[f'{branch}_sample_mean_std']:.6f} | "
            f"{summary[f'{branch}_channel_mean_std']:.6f} |"
        )
    return "\n".join(
        [
            f"# MECS VersionB routing: {Path(str(summary['checkpoint'])).name}",
            "",
            f"- Mode/seed: {summary['mode']} / {summary['seed']}",
            f"- Split/sample count: {summary['split']} / {summary['n']}",
            f"- ACC / Macro-F1 / QWK / MAE: {summary['accuracy']:.6f} / "
            f"{summary['macro_f1']:.6f} / {summary['qwk']:.6f} / {summary['mae']:.6f}",
            "",
            "| Branch | Mean weight | Global std | Argmax fraction | Sample-mean std | Channel-mean std |",
            "|---|---:|---:|---:|---:|---:|",
            *branch_rows,
            "",
            f"- Normalized route entropy: {summary['route_normalized_entropy_mean']:.6f} ± "
            f"{summary['route_normalized_entropy_std']:.6f} (1.0 is uniform)",
            f"- Effective branch count: {summary['route_effective_branch_count_mean']:.6f} (maximum 3)",
            f"- Mean absolute deviation from 1/3: {summary['route_mean_abs_deviation_from_uniform']:.6f}",
            f"- Mean maximum route weight: {summary['max_route_weight_mean']:.6f}",
            f"- Mean top1-top2 route margin: {summary['dominance_margin_mean']:.6f}",
            f"- Router weight/bias L2: {summary['router_weight_l2']:.6f} / {summary['router_bias_l2']:.6f}",
            "",
            f"Verdict: {summary['verdict']}",
            "",
            "The verdict thresholds are descriptive diagnostics, not a statistical hypothesis test.",
            "",
        ]
    )


def combined_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# MECS VersionB routing across checkpoints",
        "",
        "| Mode | Seed | ACC | QWK | avg route | max route | med route | entropy | |w-1/3| | max weight | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['seed']} | {row['accuracy']:.4f} | "
            f"{row['qwk']:.4f} | {row['avg_route_mean']:.4f} | "
            f"{row['max_route_mean']:.4f} | {row['med_route_mean']:.4f} | "
            f"{row['route_normalized_entropy_mean']:.4f} | "
            f"{row['route_mean_abs_deviation_from_uniform']:.4f} | "
            f"{row['max_route_weight_mean']:.4f} | {row['verdict']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    missing_checkpoints = [path for path in args.checkpoints if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(str(path) for path in missing_checkpoints)
        )
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = select_device(args.device)
    loader, sample_paths, class_to_idx = build_loader(args)

    print(f"Device: {device}")
    print(f"Split: {args.data_root / args.split} (n={len(sample_paths)})")
    print(f"Class mapping: {dict(class_to_idx)}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Output root: {args.output_root}")

    for index, checkpoint_path in enumerate(args.checkpoints, start=1):
        checkpoint_dir = args.output_root / checkpoint_path.stem
        checkpoint_dir.mkdir(parents=False, exist_ok=False)
        print(f"[{index}/{len(args.checkpoints)}] {checkpoint_path}")
        summary = analyze_checkpoint(
            checkpoint_path,
            loader,
            sample_paths,
            device,
            checkpoint_dir,
            args.split,
        )
        print(
            f"  routes=({summary['avg_route_mean']:.6f}, "
            f"{summary['max_route_mean']:.6f}, {summary['med_route_mean']:.6f}) | "
            f"entropy={summary['route_normalized_entropy_mean']:.6f} | "
            f"MAD={summary['route_mean_abs_deviation_from_uniform']:.6f}"
        )
        print(f"  {summary['verdict']}")
        del summary
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Reload summaries from disk so large checkpoint-local tensors are never retained.
    combined: List[Dict[str, Any]] = []
    for checkpoint_path in args.checkpoints:
        summary_path = args.output_root / checkpoint_path.stem / "routing_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            combined.append(json.load(handle))
    write_csv(args.output_root / "checkpoint_routing_summary.csv", combined)
    with (args.output_root / "checkpoint_routing_report.md").open("x", encoding="utf-8") as handle:
        handle.write(combined_report(combined))

    print(f"Combined CSV: {args.output_root / 'checkpoint_routing_summary.csv'}")
    print(f"Combined report: {args.output_root / 'checkpoint_routing_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
