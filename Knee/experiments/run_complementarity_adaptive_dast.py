#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Independent Fixed-DAST vs complementarity-adaptive DAST experiment.

This file intentionally contains all experiment-specific model wrapping, loss,
EMA control, training, evaluation, checkpointing, and reporting logic.  It does
not modify the production MESC module, the original DAST implementation, or the
original KOA training scripts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


SCRIPT_PATH = Path(__file__).resolve()
KNEE_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = KNEE_DIR.parent
BASE_TRAINING_SCRIPT = KNEE_DIR / "ResNet_layer3+MECS+CE.py"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Knee_Osteoarthritis"
DEFAULT_RESULT_ROOT = KNEE_DIR / "experiment_results"
NUM_CLASSES = 5
EPS = 1e-6


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed DAST with reverse-tau DAST controlled by "
            "multi-statistic branch complementarity."
        )
    )
    parser.add_argument(
        "--loss-mode",
        required=True,
        choices=("fixed", "complementarity_adaptive"),
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    # These defaults reproduce Knee/ResNet_layer3+MECS+CE.py.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)

    parser.add_argument("--tau-min", type=float, default=0.5)
    parser.add_argument("--tau-max", type=float, default=1.5)
    parser.add_argument("--tau-base", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--ema-momentum", type=float, default=0.95)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ramp-epochs", type=int, default=5)
    args = parser.parse_args(argv)

    if args.seed != 314:
        parser.error("This initial experiment is restricted to --seed 314.")
    if not 0.0 <= args.ema_momentum < 1.0:
        parser.error("--ema-momentum must be in [0, 1).")
    if not 0.0 < args.tau_min <= args.tau_base <= args.tau_max:
        parser.error("Require 0 < tau_min <= tau_base <= tau_max.")
    if args.ramp_epochs <= 0:
        parser.error("--ramp-epochs must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")

    args.data_root = args.data_root.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    if args.output_dir is None:
        mode_name = (
            "fixed_dast" if args.loss_mode == "fixed"
            else "complementarity_adaptive_dast"
        )
        args.output_dir = args.result_root / f"{mode_name}_seed{args.seed}"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def configure_original_script_environment(args: argparse.Namespace) -> None:
    """Set the original script's import-time configuration without editing it."""
    values = {
        "KNEE_SEED": args.seed,
        "KNEE_DATA_ROOT": args.data_root,
        "KNEE_IMAGE_SIZE": args.image_size,
        "KNEE_BATCH_SIZE": args.batch_size,
        "KNEE_EPOCHS": args.epochs,
        "KNEE_NUM_WORKERS": args.num_workers,
        "KNEE_LR_BACKBONE": args.lr_backbone,
        "KNEE_LR_HEAD": args.lr_head,
        "KNEE_WEIGHT_DECAY": args.weight_decay,
        "KNEE_PATIENCE": args.patience,
        "KNEE_EARLY_DELTA": args.early_stop_delta,
    }
    for key, value in values.items():
        os.environ[key] = str(value)


def load_original_training_module(args: argparse.Namespace) -> ModuleType:
    if not BASE_TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(f"Original training script not found: {BASE_TRAINING_SCRIPT}")
    configure_original_script_environment(args)
    module_name = "koa_resnet50_layer3_mecs_ce_for_complementarity_experiment"
    spec = importlib.util.spec_from_file_location(module_name, BASE_TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {BASE_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_experimental_model(base: ModuleType) -> nn.Module:
    """Subclass the original top-level model only inside this experiment file."""

    class ExperimentalResNet50MECS(base.CustomResNet50MECS):
        def forward(
            self,
            x: torch.Tensor,
            return_branch_logits: bool = False,
        ) -> Any:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            if return_branch_logits:
                # MECS itself still applies the current fused attention.  The
                # three separate logits are observation-only side outputs.
                x, branch_info = self.mecs(x, return_branch_attentions=True)
            else:
                x = self.mecs(x)
                branch_info = None

            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.fc(x)

            if not return_branch_logits:
                return logits
            required = ("avg_logit", "max_logit", "med_logit")
            missing = [key for key in required if key not in branch_info]
            if missing:
                raise RuntimeError(f"Current MESC did not return branch logits: {missing}")
            return logits, {key: branch_info[key] for key in required}

    model = ExperimentalResNet50MECS(num_classes=NUM_CLASSES)
    attention = model.mecs.channel_attention
    if not isinstance(attention.branch_act, nn.LeakyReLU):
        raise RuntimeError("This experiment requires the fixed MESC with LeakyReLU.")
    return model


def compute_branch_complementarity(
    branch_logits: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return detached sample-level multi-statistic branch complementarity."""
    avg_att = torch.sigmoid(branch_logits["avg_logit"])
    max_att = torch.sigmoid(branch_logits["max_logit"])
    med_att = torch.sigmoid(branch_logits["med_logit"])
    if avg_att.ndim != 4 or avg_att.shape[-2:] != (1, 1):
        raise RuntimeError(f"Expected branch tensors [B,C,1,1], got {avg_att.shape}")
    if max_att.shape != avg_att.shape or med_att.shape != avg_att.shape:
        raise RuntimeError("The three MESC branch tensors must have identical shapes.")
    pairwise_diff = (
        torch.abs(avg_att - max_att)
        + torch.abs(avg_att - med_att)
        + torch.abs(max_att - med_att)
    )
    channels = avg_att.shape[1]
    disagreement = pairwise_diff.flatten(1).sum(dim=1) / (2.0 * channels)
    return disagreement.detach()


class ComplementarityAdaptiveDASTLoss(nn.Module):
    """Per-sample DAST plus training-only EMA control for reverse tau mapping."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        loss_mode: str = "complementarity_adaptive",
        tau_min: float = 0.5,
        tau_max: float = 1.5,
        tau_base: float = 1.0,
        gamma: float = 1.5,
        ema_momentum: float = 0.95,
        warmup_epochs: int = 5,
        ramp_epochs: int = 5,
    ) -> None:
        super().__init__()
        if loss_mode not in {"fixed", "complementarity_adaptive"}:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        self.num_classes = int(num_classes)
        self.loss_mode = loss_mode
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_base = float(tau_base)
        self.gamma = float(gamma)
        self.ema_momentum = float(ema_momentum)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.register_buffer("class_ids", torch.arange(num_classes, dtype=torch.float32))
        self.register_buffer("ema_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ema_var", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update_ema(self, disagreement: torch.Tensor) -> None:
        values = disagreement.detach().float()
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        if not bool(self.ema_initialized.item()):
            self.ema_mean.copy_(batch_mean)
            self.ema_var.copy_(batch_var.clamp_min(EPS))
            self.ema_initialized.fill_(True)
            return

        momentum = self.ema_momentum
        old_mean = self.ema_mean.clone()
        delta = batch_mean - old_mean
        new_mean = momentum * old_mean + (1.0 - momentum) * batch_mean
        # The delta term preserves between-batch variance when means move.
        new_var = (
            momentum * self.ema_var
            + (1.0 - momentum) * batch_var
            + momentum * (1.0 - momentum) * delta.square()
        )
        self.ema_mean.copy_(new_mean)
        self.ema_var.copy_(new_var.clamp_min(EPS))

    def control_values(
        self,
        disagreement: torch.Tensor,
        epoch: int,
        update_ema: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sigmoid-normalized complementarity and final per-sample tau."""
        disagreement = disagreement.detach()
        if update_ema:
            self.update_ema(disagreement)
        if not bool(self.ema_initialized.item()):
            raise RuntimeError("EMA must be initialized from training before evaluation.")

        z = (disagreement - self.ema_mean) / torch.sqrt(self.ema_var + EPS)
        complementarity = torch.sigmoid(z)
        if self.loss_mode == "fixed":
            tau = torch.full_like(complementarity, self.tau_base)
            return complementarity.detach(), tau.detach()

        adaptive_tau = self.tau_max - (
            self.tau_max - self.tau_min
        ) * complementarity
        ramp = (float(epoch) - self.warmup_epochs) / float(self.ramp_epochs)
        lambda_t = min(1.0, max(0.0, ramp))
        tau = (1.0 - lambda_t) * self.tau_base + lambda_t * adaptive_tau
        return complementarity.detach(), tau.detach()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected logits [B,{self.num_classes}], got {logits.shape}")
        if targets.ndim != 1 or tau.ndim != 1 or targets.shape != tau.shape:
            raise ValueError("targets and tau must both be [B].")

        distances = torch.abs(
            self.class_ids.to(dtype=logits.dtype).unsqueeze(0)
            - targets.to(dtype=logits.dtype).unsqueeze(1)
        )
        soft_targets = torch.softmax(-distances / tau.unsqueeze(1), dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        soft_ce = -(soft_targets * log_probs).sum(dim=1)
        pt_soft = (probs * soft_targets).sum(dim=1)
        loss = (1.0 - pt_soft).pow(self.gamma) * soft_ce
        return loss.mean()


@dataclass
class TensorStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None:
        x = values.detach().double().flatten()
        if x.numel() == 0:
            return
        self.count += int(x.numel())
        self.total += float(x.sum().item())
        self.total_sq += float(x.square().sum().item())
        self.minimum = min(self.minimum, float(x.min().item()))
        self.maximum = max(self.maximum, float(x.max().item()))

    def as_dict(self, prefix: str) -> Dict[str, float]:
        if self.count == 0:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }


def scalar_metrics(base: ModuleType, y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    values = base.compute_eval_metrics(y, pred, prob)
    return {
        "acc": float(values["acc"]),
        "macro_f1": float(values["macro_f1"]),
        "qwk": float(values["qwk"]),
        "mae": float(values["mae"]),
    }


def run_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    controller: ComplementarityAdaptiveDASTLoss,
    device: torch.device,
    epoch: int,
    base: ModuleType,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    collect_samples: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    disagreement_stats = TensorStats()
    complementarity_stats = TensorStats()
    tau_stats = TensorStats()
    loss_total = 0.0
    sample_count = 0
    all_y: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_disagreement: List[np.ndarray] = []
    all_complementarity: List[np.ndarray] = []
    all_tau: List[np.ndarray] = []

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, branch_logits = model(images, return_branch_logits=True)
            disagreement = compute_branch_complementarity(branch_logits)
            complementarity, tau = controller.control_values(
                disagreement,
                epoch=epoch,
                update_ema=training,
            )
            loss = controller(logits, targets, tau)

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch_size = int(targets.shape[0])
            loss_total += float(loss.item()) * batch_size
            sample_count += batch_size
            disagreement_stats.update(disagreement)
            complementarity_stats.update(complementarity)
            tau_stats.update(tau)

            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = probabilities.argmax(dim=1)
            all_y.append(targets.detach().cpu().numpy())
            all_pred.append(predictions.cpu().numpy())
            all_prob.append(probabilities.cpu().numpy())
            if collect_samples:
                all_disagreement.append(disagreement.cpu().numpy())
                all_complementarity.append(complementarity.cpu().numpy())
                all_tau.append(tau.cpu().numpy())

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)
    output: Dict[str, Any] = {
        "loss": loss_total / sample_count,
        **scalar_metrics(base, y, pred, prob),
        **disagreement_stats.as_dict("disagreement"),
        **complementarity_stats.as_dict("complementarity"),
        **tau_stats.as_dict("tau"),
        "y": y,
        "pred": pred,
        "prob": prob,
    }
    if collect_samples:
        output["disagreement_values"] = np.concatenate(all_disagreement)
        output["complementarity_values"] = np.concatenate(all_complementarity)
        output["tau_values"] = np.concatenate(all_tau)
    return output


def configure_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> Tuple[optim.Optimizer, Any]:
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "fc" in name or "mecs" in name:
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head],
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "base_training_script": str(BASE_TRAINING_SCRIPT),
            "experiment_script": str(SCRIPT_PATH),
            "num_classes": NUM_CLASSES,
            "selection_metric": "validation_qwk",
            "test_used_for_checkpoint_selection": False,
            "complementarity_gradient_detached": True,
        }
    )
    return config


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_epoch_record(
    epoch: int,
    seconds: float,
    train: Mapping[str, Any],
    valid: Mapping[str, Any],
    controller: ComplementarityAdaptiveDASTLoss,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "epoch": epoch,
        "elapsed_seconds": seconds,
        "ema_mean": float(controller.ema_mean.item()),
        "ema_var": float(controller.ema_var.item()),
    }
    fields = (
        "loss", "acc", "macro_f1", "qwk", "mae",
        "disagreement_mean", "disagreement_std",
        "complementarity_mean", "complementarity_std",
        "tau_mean", "tau_std", "tau_min", "tau_max",
    )
    for split_name, values in (("train", train), ("val", valid)):
        for field in fields:
            record[f"{split_name}_{field}"] = values[field]
    return record


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_best_checkpoint(
    path: Path,
    model: nn.Module,
    controller: ComplementarityAdaptiveDASTLoss,
    epoch: int,
    best_val_qwk: float,
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "controller_state_dict": controller.state_dict(),
            "ema_mean": float(controller.ema_mean.item()),
            "ema_var": float(controller.ema_var.item()),
            "epoch": int(epoch),
            "best_val_qwk": float(best_val_qwk),
            "config": dict(config),
        },
        path,
    )


def final_test_metrics(test: Mapping[str, Any], best_epoch: int, best_qwk: float) -> Dict[str, Any]:
    errors = np.abs(test["pred"] - test["y"])
    tau = np.asarray(test["tau_values"], dtype=np.float64)
    disagreement = np.asarray(test["disagreement_values"], dtype=np.float64)
    complementarity = np.asarray(test["complementarity_values"], dtype=np.float64)
    return {
        "best_epoch": int(best_epoch),
        "best_validation_qwk": float(best_qwk),
        "test_n": int(len(errors)),
        "acc": float(test["acc"]),
        "macro_f1": float(test["macro_f1"]),
        "qwk": float(test["qwk"]),
        "mae": float(test["mae"]),
        "adjacent_error_rate": float(np.mean(errors == 1)),
        "severe_error_rate": float(np.mean(errors >= 2)),
        "disagreement_mean": float(disagreement.mean()),
        "disagreement_std": float(disagreement.std()),
        "complementarity_mean": float(complementarity.mean()),
        "complementarity_std": float(complementarity.std()),
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std()),
        "tau_min": float(tau.min()),
        "tau_q1": float(np.quantile(tau, 0.25)),
        "tau_median": float(np.quantile(tau, 0.50)),
        "tau_q3": float(np.quantile(tau, 0.75)),
        "tau_max": float(tau.max()),
        "tau_range": float(tau.max() - tau.min()),
        "tau_iqr": float(np.quantile(tau, 0.75) - np.quantile(tau, 0.25)),
    }


def write_test_predictions(
    path: Path,
    dataset: Any,
    test: Mapping[str, Any],
) -> None:
    samples = getattr(dataset, "samples", None)
    n = len(test["y"])
    if samples is not None and len(samples) != n:
        raise RuntimeError("Test dataset sample order/length does not match predictions.")
    rows = []
    for index in range(n):
        label = int(test["y"][index])
        pred = int(test["pred"][index])
        rows.append(
            {
                "sample_index": index,
                "image_path": str(samples[index][0]) if samples is not None else "",
                "label": label,
                "prediction": pred,
                "correct": int(label == pred),
                "absolute_error": abs(pred - label),
                "disagreement": float(test["disagreement_values"][index]),
                "complementarity": float(test["complementarity_values"][index]),
                "tau": float(test["tau_values"][index]),
            }
        )
    write_csv(path, rows)


def test_report_markdown(
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    checkpoint_path: Path,
) -> str:
    return f"""# {args.loss_mode} (seed {args.seed})

Checkpoint selection used validation QWK only. The test split was evaluated once after loading the best validation checkpoint.

| Metric | Value |
|---|---:|
| Best epoch | {metrics['best_epoch']} |
| Best validation QWK | {metrics['best_validation_qwk']:.6f} |
| Test ACC | {metrics['acc']:.6f} |
| Test Macro-F1 | {metrics['macro_f1']:.6f} |
| Test QWK | {metrics['qwk']:.6f} |
| Test MAE | {metrics['mae']:.6f} |
| Adjacent error rate (`abs(pred-y)==1`) | {metrics['adjacent_error_rate']:.6f} |
| Severe error rate (`abs(pred-y)>=2`) | {metrics['severe_error_rate']:.6f} |
| Tau mean | {metrics['tau_mean']:.6f} |
| Tau std | {metrics['tau_std']:.6f} |
| Tau Q1 / median / Q3 | {metrics['tau_q1']:.6f} / {metrics['tau_median']:.6f} / {metrics['tau_q3']:.6f} |
| Tau min / max | {metrics['tau_min']:.6f} / {metrics['tau_max']:.6f} |

The control signal is **multi-statistic branch complementarity**. Its gradient is detached before tau is constructed. Validation/test normalization uses the EMA saved from training and never re-estimates EMA from evaluation batches.

Checkpoint: `{checkpoint_path}`
"""


def unique_comparison_stem(root: Path, seed: int) -> Path:
    base = root / f"comparison_seed{seed}"
    if not base.with_suffix(".csv").exists() and not base.with_suffix(".md").exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"comparison_seed{seed}_{stamp}"


def maybe_write_comparison(args: argparse.Namespace) -> Optional[Tuple[Path, Path, str]]:
    fixed_path = args.result_root / f"fixed_dast_seed{args.seed}" / "final_metrics.json"
    adaptive_path = (
        args.result_root
        / f"complementarity_adaptive_dast_seed{args.seed}"
        / "final_metrics.json"
    )
    if not fixed_path.is_file() or not adaptive_path.is_file():
        return None
    with fixed_path.open("r", encoding="utf-8") as handle:
        fixed = json.load(handle)
    with adaptive_path.open("r", encoding="utf-8") as handle:
        adaptive = json.load(handle)

    metric_names = (
        "acc", "macro_f1", "qwk", "mae", "adjacent_error_rate",
        "severe_error_rate", "tau_mean", "tau_std", "tau_q1",
        "tau_median", "tau_q3", "tau_min", "tau_max", "tau_range", "tau_iqr",
    )
    rows = [
        {
            "metric": name,
            "fixed_dast": fixed[name],
            "complementarity_adaptive_dast": adaptive[name],
            "adaptive_minus_fixed": adaptive[name] - fixed[name],
        }
        for name in metric_names
    ]

    qwk_up = adaptive["qwk"] > fixed["qwk"]
    mae_down = adaptive["mae"] < fixed["mae"]
    severe_down = adaptive["severe_error_rate"] < fixed["severe_error_rate"]
    acc_loss = fixed["acc"] - adaptive["acc"]
    acceptable_acc = acc_loss <= 0.01
    dynamic_tau = adaptive["tau_range"] >= 0.10 and adaptive["tau_std"] >= 0.02
    positive_count = sum((qwk_up, mae_down, severe_down))
    worth_continuing = positive_count >= 2 and acceptable_acc and dynamic_tau
    verdict = (
        "值得继续到多 seed 验证，但 seed=314 仍只是单次证据。"
        if worth_continuing
        else "seed=314 未显示足够一致的收益，暂不建议直接扩大投入；应先检查控制信号和 tau 动态范围。"
    )

    stem = unique_comparison_stem(args.result_root, args.seed)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    write_csv(csv_path, rows)
    table_lines = [
        "| Metric | Fixed DAST | Complementarity-Adaptive DAST | Adaptive - Fixed |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['metric']} | {row['fixed_dast']:.6f} | "
            f"{row['complementarity_adaptive_dast']:.6f} | "
            f"{row['adaptive_minus_fixed']:+.6f} |"
        )
    md = "\n".join(
        [
            f"# Fixed vs Complementarity-Adaptive DAST (seed {args.seed})",
            "",
            *table_lines,
            "",
            f"- QWK improved: {'yes' if qwk_up else 'no'}",
            f"- MAE decreased: {'yes' if mae_down else 'no'}",
            f"- Severe error rate decreased: {'yes' if severe_down else 'no'}",
            f"- ACC loss: {acc_loss:+.6f} ({'acceptable' if acceptable_acc else 'greater than 1 percentage point'})",
            f"- Adaptive tau has sufficient dynamic range: {'yes' if dynamic_tau else 'no'}",
            "",
            f"Verdict: {verdict}",
            "",
            "This verdict is descriptive for seed=314 and is not a multi-seed significance claim.",
        ]
    )
    with md_path.open("x", encoding="utf-8") as handle:
        handle.write(md + "\n")
    return csv_path, md_path, verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {args.output_dir}"
        )
    for split in ("train", "val", "test"):
        split_path = args.data_root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Required split not found: {split_path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = config_dict(args)
    write_json(args.output_dir / "config.json", config)
    seed_everything(args.seed)
    base = load_original_training_module(args)
    # Reset once more after import-time setup so both loss modes start identically.
    seed_everything(args.seed)
    device = select_device(args.device)

    print(f"Experiment: {args.loss_mode}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print("Checkpoint selection: validation QWK only; test is held until the end.")

    train_loader, val_loader, test_loader, _, _ = base.make_dataloaders()
    model = build_experimental_model(base).to(device)
    controller = ComplementarityAdaptiveDASTLoss(
        num_classes=NUM_CLASSES,
        loss_mode=args.loss_mode,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_base=args.tau_base,
        gamma=args.gamma,
        ema_momentum=args.ema_momentum,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
    ).to(device)
    optimizer, scheduler = configure_optimizer(model, args, len(train_loader))

    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    epoch_records: List[Dict[str, Any]] = []
    best_val_qwk = -math.inf
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_result = run_epoch(
            model, train_loader, controller, device, epoch, base,
            optimizer=optimizer, scheduler=scheduler,
        )
        # update_ema=False is the safeguard that prevents validation leakage.
        val_result = run_epoch(
            model, val_loader, controller, device, epoch, base,
            optimizer=None, scheduler=None,
        )
        elapsed = time.time() - started
        record = compact_epoch_record(
            epoch, elapsed, train_result, val_result, controller
        )
        epoch_records.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | {elapsed:.1f}s | "
            f"train loss={train_result['loss']:.4f} qwk={train_result['qwk']:.4f} | "
            f"val acc={val_result['acc']:.4f} f1={val_result['macro_f1']:.4f} "
            f"qwk={val_result['qwk']:.4f} mae={val_result['mae']:.4f} | "
            f"tau={train_result['tau_mean']:.4f}+/-{train_result['tau_std']:.4f} "
            f"[{train_result['tau_min']:.4f},{train_result['tau_max']:.4f}] | "
            f"EMA=({controller.ema_mean.item():.6f},{controller.ema_var.item():.6f})"
        )

        val_qwk = float(val_result["qwk"])
        if val_qwk > best_val_qwk + args.early_stop_delta:
            best_val_qwk = val_qwk
            bad_epochs = 0
            save_best_checkpoint(
                checkpoint_path,
                model,
                controller,
                epoch,
                best_val_qwk,
                config,
            )
            print(f"  Saved best validation checkpoint (QWK={best_val_qwk:.6f}).")
        else:
            bad_epochs += 1
            print(f"  No validation-QWK improvement: {bad_epochs}/{args.patience}")
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    write_csv(args.output_dir / "epoch_metrics.csv", epoch_records)
    if not checkpoint_path.is_file():
        raise RuntimeError("No checkpoint was saved; validation QWK may be non-finite.")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    controller.load_state_dict(checkpoint["controller_state_dict"], strict=True)
    best_epoch = int(checkpoint["epoch"])
    best_val_qwk = float(checkpoint["best_val_qwk"])
    print(
        f"Loaded best epoch {best_epoch}, validation QWK={best_val_qwk:.6f}; "
        "now evaluating the test split once."
    )
    test_result = run_epoch(
        model,
        test_loader,
        controller,
        device,
        best_epoch,
        base,
        optimizer=None,
        scheduler=None,
        collect_samples=True,
    )
    metrics = final_test_metrics(test_result, best_epoch, best_val_qwk)
    write_json(args.output_dir / "final_metrics.json", metrics)
    write_test_predictions(
        args.output_dir / "test_predictions.csv",
        test_loader.dataset,
        test_result,
    )
    report_path = args.output_dir / "test_report.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(test_report_markdown(args, metrics, checkpoint_path))

    print("\nFinal test metrics:")
    print(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch log: {args.output_dir / 'epoch_metrics.csv'}")
    print(f"Test predictions: {args.output_dir / 'test_predictions.csv'}")
    print(f"Report: {report_path}")

    comparison = maybe_write_comparison(args)
    if comparison is None:
        print("Comparison pending: run the other --loss-mode with the same seed.")
    else:
        csv_path, md_path, verdict = comparison
        print(f"Comparison CSV: {csv_path}")
        print(f"Comparison report: {md_path}")
        print(f"Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
