#!/usr/bin/env python
"""Train a ResNet50 CORAL or CORN ordinal baseline on KOA or ADNI."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADNI_DIR = PROJECT_ROOT / "Alzheimer_MRI_Loss"
for path in (PROJECT_ROOT, ADNI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medical_losses import CoralOrdinalLoss, CornOrdinalLoss  # noqa: E402


KOA_CLASS_NAMES = ("0_Normal", "1_Doubtful", "2_Mild", "3_Moderate", "4_Severe")
ADNI_CLASS_NAMES = (
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
)


@dataclass
class DataBundle:
    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    test_loader: data.DataLoader
    class_names: list[str]
    train_targets: np.ndarray

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("koa", "adni"), required=True)
    parser.add_argument("--method", choices=("coral", "corn"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument(
        "--optimizer-profile",
        choices=("legacy", "matched_mesc"),
        default="legacy",
        help=(
            "legacy uses one backbone LR with AdamW; matched_mesc uses Adam "
            "and the same layer-wise LR divisors as the ADNI MESC pipeline"
        ),
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-4,
        help="Base LR used only by --optimizer-profile matched_mesc.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("--epochs, --patience, and --batch-size must be positive.")
    if (
        args.image_size <= 0
        or args.early_delta < 0
        or args.weight_decay < 0
        or args.base_lr <= 0
    ):
        parser.error("invalid image-size, early-delta, weight-decay, or base-lr.")
    if not 0 < args.test_ratio < 1 or not 0 < args.val_ratio < 1:
        parser.error("--test-ratio and --val-ratio must be between 0 and 1.")

    args.data_root = args.data_root.expanduser().resolve()
    args.num_workers = args.num_workers if args.num_workers is not None else (4 if args.dataset == "koa" else 2)
    args.lr_backbone = args.lr_backbone if args.lr_backbone is not None else (1e-4 if args.dataset == "koa" else 1e-5)
    args.lr_head = args.lr_head if args.lr_head is not None else (1e-3 if args.dataset == "koa" else 1e-4)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_koa_bundle(args: argparse.Namespace) -> DataBundle:
    missing = [name for name in ("train", "val", "test") if not (args.data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"KOA data root {args.data_root} is missing: {missing}")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(args.data_root / "train", transform=train_transform)
    valid_set = datasets.ImageFolder(args.data_root / "val", transform=eval_transform)
    test_set = datasets.ImageFolder(args.data_root / "test", transform=eval_transform)
    if len(train_set.classes) != len(KOA_CLASS_NAMES):
        raise RuntimeError(f"Expected 5 ordered KOA classes, found {train_set.classes}.")
    if valid_set.class_to_idx != train_set.class_to_idx or test_set.class_to_idx != train_set.class_to_idx:
        raise RuntimeError("KOA train/val/test class mappings do not match.")
    return DataBundle(
        train_loader=_loader(train_set, args.batch_size, True, args.num_workers),
        valid_loader=_loader(valid_set, args.batch_size, False, args.num_workers),
        test_loader=_loader(test_set, args.batch_size, False, args.num_workers),
        class_names=list(KOA_CLASS_NAMES),
        train_targets=np.asarray(train_set.targets, dtype=np.int64),
    )


def build_adni_bundle(args: argparse.Namespace) -> DataBundle:
    from alzheimer_mri_loss_experiment_common import build_alzheimer_mri_dataloaders

    source = build_alzheimer_mri_dataloaders(
        data_root=args.data_root,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        class_order=ADNI_CLASS_NAMES,
    )
    return DataBundle(
        train_loader=source.train_loader,
        valid_loader=source.valid_loader,
        test_loader=source.test_loader,
        class_names=list(source.class_names),
        train_targets=source.train_targets,
    )


class CoralHead(nn.Module):
    """Shared score plus ordered-initialized threshold biases used by CORAL."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.score = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(0.75, -0.75, num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) + self.bias.unsqueeze(0)


class OrdinalResNet50(nn.Module):
    def __init__(self, method: str, num_classes: int):
        super().__init__()
        self.method = method
        self.num_classes = num_classes
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = (
            CoralHead(feature_dim, num_classes)
            if method == "coral"
            else nn.Linear(feature_dim, num_classes - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def cumulative_probabilities(logits: torch.Tensor, method: str) -> torch.Tensor:
    conditional = torch.sigmoid(logits)
    if method == "corn":
        return torch.cumprod(conditional, dim=1)
    # CORAL's shared score encourages rank consistency. cummin also guards
    # against tiny threshold-bias crossings when converting to class masses.
    return torch.cummin(conditional, dim=1).values


def class_probabilities(logits: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    cumulative = cumulative_probabilities(logits, method)
    first = 1.0 - cumulative[:, :1]
    middle = cumulative[:, :-1] - cumulative[:, 1:]
    last = cumulative[:, -1:]
    probabilities = torch.cat((first, middle, last), dim=1).clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probabilities, cumulative


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "ovr_roc_auc_macro": None,
        "ovr_pr_auc_macro": None,
    }
    try:
        one_hot = np.eye(y_prob.shape[1])[y_true]
        metrics["ovr_roc_auc_macro"] = roc_auc_score(
            one_hot, y_prob, average="macro", multi_class="ovr"
        )
        metrics["ovr_pr_auc_macro"] = average_precision_score(one_hot, y_prob, average="macro")
    except ValueError:
        pass
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = {1: 0, 2: 0, 3: 0}
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        probabilities, cumulative = class_probabilities(logits.detach(), model.method)
        ordinal_pred = (cumulative > 0.5).sum(dim=1)
        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct[1] += int((ordinal_pred == target).sum().item())
        for k in (2, 3):
            top = probabilities.topk(min(k, model.num_classes), dim=1).indices
            correct[k] += int(top.eq(target.unsqueeze(1)).any(dim=1).sum().item())
    return total_loss / total_samples, {f"top{k}": correct[k] / total_samples for k in (1, 2, 3)}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = {1: 0, 2: 0, 3: 0}
    all_target, all_pred, all_prob = [], [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        probabilities, cumulative = class_probabilities(logits, model.method)
        ordinal_pred = (cumulative > 0.5).sum(dim=1)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct[1] += int((ordinal_pred == target).sum().item())
        for k in (2, 3):
            top = probabilities.topk(min(k, model.num_classes), dim=1).indices
            correct[k] += int(top.eq(target.unsqueeze(1)).any(dim=1).sum().item())
        all_target.append(target.cpu().numpy())
        all_pred.append(ordinal_pred.cpu().numpy())
        all_prob.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_target)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    top = {f"top{k}": correct[k] / total_samples for k in (1, 2, 3)}
    return total_loss / total_samples, top, compute_metrics(y_true, y_pred, y_prob)


def metric_line(prefix: str, loss: float, top: dict[str, float], metrics: dict[str, float | None]) -> str:
    return (
        f"{prefix} | loss={loss:.4f} | top1={top['top1'] * 100:.2f}% "
        f"| top2={top['top2'] * 100:.2f}% | top3={top['top3'] * 100:.2f}% "
        f"| acc={metrics['acc'] * 100:.2f}% | bal_acc={metrics['balanced_acc'] * 100:.2f}% "
        f"| macro_f1={metrics['macro_f1']:.4f} | qwk={metrics['qwk']:.4f} "
        f"| mae={metrics['mae']:.4f} | weighted_f1={metrics['weighted_f1']:.4f} "
        f"| precision_macro={metrics['precision_macro']:.4f} "
        f"| recall_macro={metrics['recall_macro']:.4f}"
    )


MATCHED_MESC_LR_DIVISORS = (
    ("conv1", 10.0),
    ("bn1", 10.0),
    ("layer1", 8.0),
    ("layer2", 6.0),
    ("layer3", 4.0),
    ("layer4", 2.0),
)


def build_optimizer_and_scheduler(model, args, steps_per_epoch: int):
    total_steps = args.epochs * steps_per_epoch
    if args.optimizer_profile == "legacy":
        optimizer = optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.head.parameters(), "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )
        max_lrs = [args.lr_backbone, args.lr_head]
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        description = (
            f"legacy AdamW | backbone={args.lr_backbone:.8g}, "
            f"head={args.lr_head:.8g}, weight_decay={args.weight_decay:.8g}"
        )
        return optimizer, scheduler, description

    param_groups = []
    max_lrs = []
    descriptions = []
    for layer_name, divisor in MATCHED_MESC_LR_DIVISORS:
        layer = getattr(model.backbone, layer_name)
        lr = args.base_lr / divisor
        param_groups.append({"params": layer.parameters(), "lr": lr})
        max_lrs.append(lr)
        descriptions.append(f"{layer_name}={lr:.8g}")
    param_groups.append({"params": model.head.parameters(), "lr": args.base_lr})
    max_lrs.append(args.base_lr)
    descriptions.append(f"head={args.base_lr:.8g}")

    optimizer = optim.Adam(param_groups, lr=args.base_lr)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
    )
    description = "matched_mesc Adam | " + ", ".join(descriptions)
    return optimizer, scheduler, description


def sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("._-")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_koa_bundle(args) if args.dataset == "koa" else build_adni_bundle(args)

    print(f"Starting ResNet50 + {args.method.upper()} ordinal baseline on {args.dataset.upper()}...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"Config | seed={args.seed}, epochs={args.epochs}, patience={args.patience}, "
        f"batch_size={args.batch_size}, image_size={args.image_size}"
    )
    print(f"Data root: {args.data_root}")
    print(f"Class order: {bundle.class_names}")
    print(
        f"Split sizes | train={len(bundle.train_loader.dataset)}, "
        f"valid={len(bundle.valid_loader.dataset)}, test={len(bundle.test_loader.dataset)}"
    )

    model = OrdinalResNet50(args.method, bundle.num_classes).to(device)
    criterion = (
        CoralOrdinalLoss(bundle.num_classes)
        if args.method == "coral"
        else CornOrdinalLoss(bundle.num_classes)
    ).to(device)
    optimizer, scheduler, optimizer_description = build_optimizer_and_scheduler(
        model,
        args,
        len(bundle.train_loader),
    )
    print(f"Optimizer profile: {optimizer_description}")

    run_tag = sanitize_tag(args.run_tag) or f"{args.dataset}_{args.method}_seed{args.seed}"
    checkpoint_dir = PROJECT_ROOT / ("Knee" if args.dataset == "koa" else "Alzheimer_MRI_Loss") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_resnet50_{args.method}_{run_tag}.pt"
    monitor_name = "qwk" if args.dataset == "koa" else "macro_f1"
    best_score = -float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_top = train_one_epoch(
            model, bundle.train_loader, criterion, optimizer, scheduler, device
        )
        valid_loss, valid_top, valid_metrics = evaluate(
            model, bundle.valid_loader, criterion, device
        )
        elapsed = int(time.time() - start)
        print(f"\nEpoch {epoch + 1:02d}/{args.epochs} | Time {elapsed // 60}m {elapsed % 60}s")
        print(
            f"  Train | loss={train_loss:.4f} | top1={train_top['top1'] * 100:.2f}% "
            f"| top2={train_top['top2'] * 100:.2f}% | top3={train_top['top3'] * 100:.2f}%"
        )
        print("  " + metric_line("Valid", valid_loss, valid_top, valid_metrics))

        score = float(valid_metrics[monitor_name])
        if score > best_score + args.early_delta:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                    "monitor": monitor_name,
                    "epoch": epoch + 1,
                    "method": args.method,
                    "num_classes": bundle.num_classes,
                    "optimizer_profile": args.optimizer_profile,
                    "base_lr": args.base_lr,
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
    print(f"\nLoading best model ({checkpoint['monitor']}={checkpoint['best_score']:.4f})")
    test_loss, test_top, test_metrics = evaluate(model, bundle.test_loader, criterion, device)
    print("\n" + metric_line("Test", test_loss, test_top, test_metrics))
    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} "
            f"| ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )
    print(f"Checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
