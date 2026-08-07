#!/usr/bin/env python
"""Train one ConvNeXt-T experiment on KOA or Alzheimer MRI.

For attention experiments the module is inserted after ConvNeXt stage3
(``features[5]``, 384 channels) and before the final downsampling layer.
The data splits and early-stopping monitors match the existing ResNet50
protocols: fixed train/val/test folders and QWK for KOA; seeded stratified
splits and Macro-F1 for Alzheimer MRI.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
ADNI_DIR = PROJECT_ROOT / "Alzheimer_MRI_Loss"
CHEST_DIR = PROJECT_ROOT / "chest-x-ray-image_Loss"
for path in (PROJECT_ROOT, TOOLS_DIR, ADNI_DIR, CHEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attention_modules import (  # noqa: E402
    CBAMBlock,
    ECABlock,
    MSCABlock,
    SEBlock,
)
from chest_xray_loss_experiment_common import compute_eval_metrics  # noqa: E402
from medical_losses import (  # noqa: E402
    CoralOrdinalLoss,
    DistanceAwareSoftTargetLoss,
    LabelSmoothingCrossEntropyLoss,
    OrdinalSoftCrossEntropyLoss,
)
from MECS_old import MECS_VersionA  # noqa: E402
from MECS_RawRouting import MECS_RawRouting  # noqa: E402
from MECS_VersionB import MECS_VersionB  # noqa: E402
from train_ordinal_baseline import (  # noqa: E402
    CoralHead,
    build_adni_bundle,
    build_koa_bundle,
    class_probabilities,
)


ATTENTION_CHOICES = (
    "none",
    "se",
    "cbam",
    "eca",
    "msca",
    "mesc_equal",
    "mesc_direct",
    "mesc",
)
LOSS_CHOICES = ("ce", "dast", "sord_ce", "label_smoothing_ce", "coral")
STAGE3_INDEX = 5
STAGE3_CHANNELS = 384


def sanitize_tag(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("._-")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("koa", "adni"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default="none")
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="ce")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=1e-4)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--koa-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--koa-head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--dast-tau", type=float, default=1.0)
    parser.add_argument("--dast-gamma", type=float, default=1.5)
    parser.add_argument("--sord-tau", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)

    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.image_size,
        args.base_lr,
        args.koa_backbone_lr,
        args.koa_head_lr,
        args.dast_tau,
        args.sord_tau,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs, patience, sizes, learning rates, and tau values must be positive")
    if args.early_delta < 0 or args.weight_decay < 0 or args.dast_gamma < 0:
        parser.error("early delta, weight decay, and DAST gamma cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must satisfy 0 <= value < 1")
    if not 0.0 < args.test_ratio < 1.0 or not 0.0 < args.val_ratio < 1.0:
        parser.error("test and validation ratios must be between 0 and 1")
    args.data_root = args.data_root.expanduser().resolve()
    args.num_workers = args.num_workers if args.num_workers is not None else (4 if args.dataset == "koa" else 2)
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    args.run_tag = sanitize_tag(args.run_tag)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_stage3_attention(name: str, channels: int = STAGE3_CHANNELS) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(channels)
    if name == "cbam":
        return CBAMBlock(channels)
    if name == "eca":
        return ECABlock(channels)
    if name == "msca":
        return MSCABlock(channels)
    if name == "mesc_equal":
        return MECS_VersionA(channels, channels)
    if name == "mesc_direct":
        return MECS_RawRouting(channels, channels)
    if name == "mesc":
        return MECS_VersionB(channels, channels)
    raise ValueError(f"Unknown stage3 attention: {name}")


class ConvNeXtTinyWithStage3Attention(nn.Module):
    """ConvNeXt-T with an optional NCHW attention block after stage3."""

    def __init__(
        self,
        num_classes: int,
        attention: str = "none",
        loss_name: str = "ce",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]
        self.attention_name = attention
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.stage3_attention = build_stage3_attention(attention, STAGE3_CHANNELS)
        feature_dim = backbone.classifier[2].in_features
        self.head = (
            CoralHead(feature_dim, num_classes)
            if loss_name == "coral"
            else nn.Linear(feature_dim, num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == STAGE3_INDEX:
                x = self.stage3_attention(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        return self.head(features), features


def build_criterion(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    if args.loss == "dast":
        return DistanceAwareSoftTargetLoss(
            num_classes=num_classes,
            tau=args.dast_tau,
            gamma=args.dast_gamma,
        )
    if args.loss == "sord_ce":
        return OrdinalSoftCrossEntropyLoss(num_classes=num_classes, tau=args.sord_tau)
    if args.loss == "label_smoothing_ce":
        return LabelSmoothingCrossEntropyLoss(smoothing=args.label_smoothing)
    if args.loss == "coral":
        return CoralOrdinalLoss(num_classes=num_classes)
    raise ValueError(f"Unknown loss: {args.loss}")


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def build_optimizer_and_scheduler(
    model: ConvNeXtTinyWithStage3Attention,
    args: argparse.Namespace,
    steps_per_epoch: int,
):
    total_steps = args.epochs * steps_per_epoch
    if args.dataset == "koa":
        backbone_params = trainable_parameters(model.features) + trainable_parameters(model.norm)
        head_params = trainable_parameters(model.stage3_attention) + trainable_parameters(model.head)
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": args.koa_backbone_lr},
                {"params": head_params, "lr": args.koa_head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        max_lrs = [args.koa_backbone_lr, args.koa_head_lr]
        description = (
            f"AdamW | backbone={args.koa_backbone_lr:.8g}, "
            f"attention+head={args.koa_head_lr:.8g}, weight_decay={args.weight_decay:.8g}"
        )
    else:
        grouped_modules = (
            ("stem", model.features[0], 10.0),
            ("stage1", model.features[1], 8.0),
            ("downsample1", model.features[2], 6.0),
            ("stage2", model.features[3], 6.0),
            ("downsample2", model.features[4], 4.0),
            ("stage3", model.features[5], 4.0),
            ("attention", model.stage3_attention, 3.0),
            ("downsample3", model.features[6], 2.0),
            ("stage4", model.features[7], 2.0),
            ("norm", model.norm, 1.0),
            ("head", model.head, 1.0),
        )
        param_groups = []
        max_lrs = []
        descriptions = []
        for name, module, divisor in grouped_modules:
            params = trainable_parameters(module)
            if not params:
                continue
            learning_rate = args.base_lr / divisor
            param_groups.append({"params": params, "lr": learning_rate})
            max_lrs.append(learning_rate)
            descriptions.append(f"{name}={learning_rate:.8g}")
        optimizer = optim.Adam(param_groups, lr=args.base_lr)
        description = "matched progressive Adam | " + ", ".join(descriptions)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    return optimizer, scheduler, description


def output_probabilities(
    logits: torch.Tensor,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_name == "coral":
        probabilities, cumulative = class_probabilities(logits, "coral")
        predictions = (cumulative > 0.5).sum(dim=1)
        return probabilities, predictions
    probabilities = torch.softmax(logits, dim=1)
    return probabilities, probabilities.argmax(dim=1)


def topk_correct(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[int, int]:
    correct = {1: int(predictions.eq(targets).sum().item())}
    for k in (2, 3):
        indices = probabilities.topk(min(k, num_classes), dim=1).indices
        correct[k] = int(indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
    return correct


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = {1: 0, 2: 0, 3: 0}
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        probabilities, predictions = output_probabilities(logits.detach(), model.loss_name)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_correct = topk_correct(
            probabilities,
            predictions,
            targets,
            model.num_classes,
        )
        for k in correct:
            correct[k] += batch_correct[k]
    top = {f"top{k}": correct[k] / total_samples for k in correct}
    return total_loss / total_samples, top


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = {1: 0, 2: 0, 3: 0}
    all_targets, all_predictions, all_probabilities = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, _ = model(images)
        loss = criterion(logits, targets)
        probabilities, predictions = output_probabilities(logits, model.loss_name)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_correct = topk_correct(
            probabilities,
            predictions,
            targets,
            model.num_classes,
        )
        for k in correct:
            correct[k] += batch_correct[k]
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)
    metrics = compute_eval_metrics(
        y_true,
        y_pred,
        y_prob,
        model.num_classes,
        class_names,
    )
    top = {f"top{k}": correct[k] / total_samples for k in correct}
    return total_loss / total_samples, top, metrics


def metric_line(prefix: str, loss: float, top: dict[str, float], metrics: dict[str, object]) -> str:
    return (
        f"{prefix} | loss={loss:.4f} | top1={top['top1'] * 100:.2f}% "
        f"| top2={top['top2'] * 100:.2f}% | top3={top['top3'] * 100:.2f}% "
        f"| acc={float(metrics['acc']) * 100:.2f}% "
        f"| bal_acc={float(metrics['balanced_acc']) * 100:.2f}% "
        f"| macro_f1={float(metrics['macro_f1']):.4f} "
        f"| qwk={float(metrics['qwk']):.4f} | mae={float(metrics['mae']):.4f} "
        f"| weighted_f1={float(metrics['weighted_f1']):.4f} "
        f"| precision_macro={float(metrics['precision_macro']):.4f} "
        f"| recall_macro={float(metrics['recall_macro']):.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_koa_bundle(args) if args.dataset == "koa" else build_adni_bundle(args)
    monitor_name = "qwk" if args.dataset == "koa" else "macro_f1"

    print(f"Starting ConvNeXt-T experiment on {args.dataset.upper()}...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"Config | seed={args.seed}, attention={args.attention}, loss={args.loss}, "
        f"epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}, "
        f"image_size={args.image_size}, pretrained={not args.no_pretrained}"
    )
    print("Backbone: ConvNeXt-T | stage3=features[5] | stage3_channels=384")
    print(f"Data root: {args.data_root}")
    print(f"Class order: {bundle.class_names}")
    print(
        f"Split sizes | train={len(bundle.train_loader.dataset)}, "
        f"valid={len(bundle.valid_loader.dataset)}, test={len(bundle.test_loader.dataset)}"
    )
    class_counts = np.bincount(bundle.train_targets, minlength=bundle.num_classes)
    print(f"Train class counts: {class_counts.tolist()}")

    model = ConvNeXtTinyWithStage3Attention(
        num_classes=bundle.num_classes,
        attention=args.attention,
        loss_name=args.loss,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = build_criterion(args, bundle.num_classes).to(device)
    optimizer, scheduler, optimizer_description = build_optimizer_and_scheduler(
        model,
        args,
        len(bundle.train_loader),
    )
    print(f"Optimizer: {optimizer_description}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    run_tag = args.run_tag or f"convnext_t_{args.dataset}_{args.attention}_{args.loss}_seed{args.seed}"
    checkpoint_dir = PROJECT_ROOT / ("Knee" if args.dataset == "koa" else "Alzheimer_MRI_Loss") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_convnext_t_{args.attention}_{args.loss}_{run_tag}.pt"
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_top = train_one_epoch(
            model,
            bundle.train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
        )
        valid_loss, valid_top, valid_metrics = evaluate(
            model,
            bundle.valid_loader,
            criterion,
            device,
            bundle.class_names,
        )
        elapsed = int(time.time() - start_time)
        print(f"\nEpoch {epoch + 1:02d}/{args.epochs} | Time {elapsed // 60}m {elapsed % 60}s")
        print(
            f"  Train | loss={train_loss:.4f} | top1={train_top['top1'] * 100:.2f}% "
            f"| top2={train_top['top2'] * 100:.2f}% | top3={train_top['top3'] * 100:.2f}%"
        )
        print("  " + metric_line("Valid", valid_loss, valid_top, valid_metrics))

        score = float(valid_metrics[monitor_name])
        if score > best_score + args.early_delta:
            best_score = score
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "monitor": monitor_name,
                    "dataset": args.dataset,
                    "backbone": "convnext_t",
                    "stage3_index": STAGE3_INDEX,
                    "stage3_channels": STAGE3_CHANNELS,
                    "attention": args.attention,
                    "loss": args.loss,
                    "num_classes": bundle.num_classes,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            print(f" Validation improved. Saved best model ({monitor_name}={best_score:.4f})")
        else:
            bad_epochs += 1
            print(f" No improvement. Bad epochs: {bad_epochs}/{args.patience}")
            if bad_epochs >= args.patience:
                print(f" Early stopping triggered at epoch {epoch + 1}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"\nLoading best model (epoch={checkpoint['best_epoch']}, "
        f"{checkpoint['monitor']}={checkpoint['best_score']:.4f})"
    )
    test_loss, test_top, test_metrics = evaluate(
        model,
        bundle.test_loader,
        criterion,
        device,
        bundle.class_names,
    )
    print("\n" + metric_line("Test", test_loss, test_top, test_metrics))
    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={float(test_metrics['ovr_roc_auc_macro']):.4f} "
            f"| ovr_pr_auc_macro={float(test_metrics['ovr_pr_auc_macro']):.4f}"
        )
    print(f"Checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
