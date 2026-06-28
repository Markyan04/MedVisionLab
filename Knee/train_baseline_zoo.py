#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train single-model KOA comparison baselines.

This runner mirrors the Knee ResNet50 baseline protocol and swaps only the
backbone. It is intended for filling missing comparison entries such as
Inception V3, DenseNet169, DenseNet201, ViT, Xception, and ResNet50V2.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in (PROJECT_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CLASS_NAMES = ["0_Normal", "1_Doubtful", "2_Mild", "3_Moderate", "4_Severe"]
NUM_CLASSES = len(CLASS_NAMES)
TOPK = (1, 2, 3)

TORCHVISION_MODEL_NAMES = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "Inception V3",
    "densenet121": "DenseNet121",
    "densenet169": "DenseNet169",
    "densenet201": "DenseNet201",
    "mobilenet_v2": "MobileNetV2",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "vit_b_16": "ViT",
}

TIMM_MODEL_CANDIDATES = {
    "xception": ("Xception", ("xception", "legacy_xception")),
    "inception_resnet_v2": ("Inception-ResNetV2", ("inception_resnet_v2",)),
    "resnet50v2": (
        "ResNet50V2",
        (
            "resnetv2_50",
            "resnetv2_50x1_bit",
            "resnetv2_50x1_bit.goog_in21k_ft_in1k",
        ),
    ),
}

MODEL_ALIASES = {
    "inception-v3": "inception_v3",
    "inceptionv3": "inception_v3",
    "densenet-169": "densenet169",
    "densenet-201": "densenet201",
    "mobilenetv2": "mobilenet_v2",
    "mobile_net_v2": "mobilenet_v2",
    "vit": "vit_b_16",
    "vit-b-16": "vit_b_16",
    "resnet50-v2": "resnet50v2",
    "resnet50_v2": "resnet50v2",
    "inception-resnet-v2": "inception_resnet_v2",
    "inceptionresnetv2": "inception_resnet_v2",
}


class EarlyStopping:
    def __init__(self, patience: int, delta: float, save_path: Path) -> None:
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_score: Optional[float] = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, score: float, model: nn.Module) -> None:
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "best_score": score},
                self.save_path,
            )
            print(f"Validation improved. Saved best model to {self.save_path} (QWK={score:.4f})")
            return

        self.num_bad_epochs += 1
        print(f"No improvement. Bad epochs: {self.num_bad_epochs}/{self.patience}")
        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
            print("Early stopping triggered.")


def normalize_model_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return MODEL_ALIASES.get(key, key)


def safe_tag(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw.strip()).strip("._-")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_weights(enum_cls, pretrained: bool):
    return enum_cls.DEFAULT if pretrained else None


def replace_torchvision_classifier(model: nn.Module, model_key: str) -> nn.Module:
    if model_key.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif model_key.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)
    elif model_key == "mobilenet_v2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif model_key.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    elif model_key == "inception_v3":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        if getattr(model, "AuxLogits", None) is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
    elif model_key == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported torchvision model: {model_key}")
    return model


def build_torchvision_model(model_key: str, pretrained: bool) -> Tuple[nn.Module, str]:
    if model_key == "vgg16":
        model = models.vgg16(weights=get_default_weights(models.VGG16_Weights, pretrained))
    elif model_key == "vgg19":
        model = models.vgg19(weights=get_default_weights(models.VGG19_Weights, pretrained))
    elif model_key == "inception_v3":
        model = models.inception_v3(
            weights=get_default_weights(models.Inception_V3_Weights, pretrained),
            aux_logits=True if pretrained else False,
        )
        model.aux_logits = False
        model.AuxLogits = None
    elif model_key == "densenet121":
        model = models.densenet121(weights=get_default_weights(models.DenseNet121_Weights, pretrained))
    elif model_key == "densenet169":
        model = models.densenet169(weights=get_default_weights(models.DenseNet169_Weights, pretrained))
    elif model_key == "densenet201":
        model = models.densenet201(weights=get_default_weights(models.DenseNet201_Weights, pretrained))
    elif model_key == "mobilenet_v2":
        model = models.mobilenet_v2(weights=get_default_weights(models.MobileNet_V2_Weights, pretrained))
    elif model_key == "resnet50":
        model = models.resnet50(weights=get_default_weights(models.ResNet50_Weights, pretrained))
    elif model_key == "resnet101":
        model = models.resnet101(weights=get_default_weights(models.ResNet101_Weights, pretrained))
    elif model_key == "vit_b_16":
        model = models.vit_b_16(weights=get_default_weights(models.ViT_B_16_Weights, pretrained))
    else:
        raise ValueError(f"Unsupported torchvision model: {model_key}")
    return replace_torchvision_classifier(model, model_key), TORCHVISION_MODEL_NAMES[model_key]


def build_timm_model(model_key: str, pretrained: bool) -> Tuple[nn.Module, str]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            f"{model_key} requires timm. Install timm in the Paper environment or run with models that use torchvision."
        ) from exc

    display_name, candidates = TIMM_MODEL_CANDIDATES[model_key]
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            model = timm.create_model(candidate, pretrained=pretrained, num_classes=NUM_CLASSES)
            print(f"Using timm model id: {candidate}")
            return model, display_name
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create timm model for {display_name}. Last error: {last_error}")


def build_model(model_key: str, pretrained: bool) -> Tuple[nn.Module, str]:
    if model_key in TORCHVISION_MODEL_NAMES:
        return build_torchvision_model(model_key, pretrained)
    if model_key in TIMM_MODEL_CANDIDATES:
        return build_timm_model(model_key, pretrained)
    supported = sorted(TORCHVISION_MODEL_NAMES) + sorted(TIMM_MODEL_CANDIDATES)
    raise ValueError(f"Unknown model '{model_key}'. Supported: {', '.join(supported)}")


def output_to_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def calculate_topk_accuracy(logits: torch.Tensor, y: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    with torch.no_grad():
        num_classes = logits.size(1)
        ks = tuple(sorted(set(min(k, num_classes) for k in ks)))
        max_k = max(ks)
        _, top_pred = logits.topk(max_k, dim=1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        out = {}
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            out[f"top{k}"] = correct_k / y.size(0)
        return out


def compute_eval_metrics(y_true, y_pred, y_prob) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    metrics["mae"] = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    metrics["ovr_roc_auc_macro"] = None
    metrics["ovr_pr_auc_macro"] = None
    try:
        y_true_oh = np.eye(NUM_CLASSES)[y_true]
        metrics["ovr_roc_auc_macro"] = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        metrics["ovr_pr_auc_macro"] = average_precision_score(y_true_oh, y_prob, average="macro")
    except Exception:
        pass
    return metrics


def make_dataloaders(data_root: Path, image_size: int, batch_size: int, num_workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Required split directory not found: {split_dir}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(train_dataset.classes)} in {train_dir}")

    def print_stats(name: str, dataset) -> None:
        targets = np.array(dataset.targets)
        class_counts = np.bincount(targets, minlength=NUM_CLASSES)
        print(f"\n{name} samples: {len(dataset)}")
        for i, count in enumerate(class_counts):
            print(f"  {CLASS_NAMES[i]}: {count}")

    print("Class mapping from folders:", train_dataset.class_to_idx)
    print_stats("Train", train_dataset)
    print_stats("Valid", valid_dataset)
    print_stats("Test", test_dataset)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader, train_dataset


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in TOPK}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = output_to_logits(model(x))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_loss += loss.item()
        for key, value in calculate_topk_accuracy(logits, y, TOPK).items():
            epoch_top[key] += value
    epoch_loss /= len(loader)
    for key in epoch_top:
        epoch_top[key] /= len(loader)
    return epoch_loss, epoch_top


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in TOPK}
    all_y = []
    all_pred = []
    all_prob = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = output_to_logits(model(x))
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        epoch_loss += loss.item()
        for key, value in calculate_topk_accuracy(logits, y, TOPK).items():
            epoch_top[key] += value
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    epoch_loss /= len(loader)
    for key in epoch_top:
        epoch_top[key] /= len(loader)
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return epoch_loss, epoch_top, compute_eval_metrics(y_true, y_pred, y_prob)


def split_parameter_groups(model: nn.Module):
    head_keywords = ("fc", "classifier", "head", "heads")
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name.lower() for token in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)
    if not head_params:
        return [{"params": backbone_params}]
    return backbone_params, head_params


def save_summary_csv(summary_dir: Path, model_key: str, run_tag: str, row: Dict[str, object]) -> Path:
    summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    path = summary_dir / f"baseline_zoo_{model_key}{tag}_{timestamp}_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("KNEE_MODEL", "inception_v3"))
    parser.add_argument("--data-root", default=os.getenv("KNEE_DATA_ROOT", str(PROJECT_ROOT / "Knee_Osteoarthritis")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("KNEE_IMAGE_SIZE", "224")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("KNEE_BATCH_SIZE", "32")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("KNEE_EPOCHS", "50")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("KNEE_SEED", os.getenv("GLOBAL_EXPERIMENT_SEED", "1234"))))
    parser.add_argument("--lr-backbone", type=float, default=float(os.getenv("KNEE_LR_BACKBONE", "1e-4")))
    parser.add_argument("--lr-head", type=float, default=float(os.getenv("KNEE_LR_HEAD", "1e-3")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("KNEE_WEIGHT_DECAY", "1e-4")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("KNEE_NUM_WORKERS", "4")))
    parser.add_argument("--patience", type=int, default=int(os.getenv("KNEE_PATIENCE", "10")))
    parser.add_argument("--early-delta", type=float, default=float(os.getenv("KNEE_EARLY_DELTA", "1e-4")))
    parser.add_argument("--run-tag", default=os.getenv("KNEE_RUN_TAG", ""))
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("KNEE_PRETRAINED", "1").lower() not in {"0", "false", "no"},
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_key = normalize_model_name(args.model)
    run_tag = safe_tag(args.run_tag)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting KOA baseline zoo training...")
    print(f"Model key: {model_key}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        "Config | "
        f"seed={args.seed}, batch_size={args.batch_size}, epochs={args.epochs}, "
        f"image_size={args.image_size}, pretrained={args.pretrained}"
    )
    if run_tag:
        print(f"Run tag: {run_tag}")

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Knee data root not found: {data_root}")
    train_loader, valid_loader, test_loader, train_dataset = make_dataloaders(
        data_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    print(
        f"\nSplit sizes | train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    model, display_name = build_model(model_key, args.pretrained)
    model = model.to(device)
    print(f"Display name: {display_name}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_counts = np.bincount(train_dataset.targets, minlength=NUM_CLASSES)
    print(f"\n[INFO] Class counts in training set: {class_counts}")
    print("[INFO] Using Standard CrossEntropyLoss (No class weights).")
    criterion = nn.CrossEntropyLoss()

    groups = split_parameter_groups(model)
    if isinstance(groups, tuple):
        backbone_params, head_params = groups
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params, "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )
        max_lr = [args.lr_backbone, args.lr_head]
    else:
        optimizer = optim.AdamW(groups, lr=args.lr_backbone, weight_decay=args.weight_decay)
        max_lr = args.lr_backbone

    total_steps = args.epochs * len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    checkpoint_tag = f"_{run_tag}" if run_tag else ""
    best_path = THIS_DIR / "checkpoints" / f"best_{model_key}_koa{checkpoint_tag}.pt"
    early_stopping = EarlyStopping(args.patience, args.early_delta, best_path)

    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_top = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        valid_loss, valid_top, valid_metrics = evaluate(model, valid_loader, criterion, device)
        elapsed = int(time.time() - start_time)
        print(f"\nEpoch {epoch + 1:02d}/{args.epochs} | Time {elapsed // 60}m {elapsed % 60}s")
        print(
            f"  Train | loss={train_loss:.4f} | "
            + " | ".join(f"{key}={value * 100:.2f}%" for key, value in train_top.items())
        )
        print(
            f"  Valid | loss={valid_loss:.4f} | "
            + " | ".join(f"{key}={value * 100:.2f}%" for key, value in valid_top.items())
            + f" | acc={valid_metrics['acc'] * 100:.2f}%"
            + f" | bal_acc={valid_metrics['balanced_acc'] * 100:.2f}%"
            + f" | macro_f1={valid_metrics['macro_f1']:.4f}"
            + f" | qwk={valid_metrics['qwk']:.4f}"
            + f" | mae={valid_metrics['mae']:.4f}"
            + f" | weighted_f1={valid_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={valid_metrics['precision_macro']:.4f}"
            + f" | recall_macro={valid_metrics['recall_macro']:.4f}"
        )
        early_stopping(float(valid_metrics["qwk"]), model)
        if early_stopping.early_stop:
            print(f"Training stopped early at epoch {epoch + 1}.")
            break

    print("\nLoading best model and evaluating on test set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best validation QWK: {ckpt['best_score']:.4f}")

    test_loss, test_top, test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"\nTest | loss={test_loss:.4f} | "
        + " | ".join(f"{key}={value * 100:.2f}%" for key, value in test_top.items())
        + f" | acc={test_metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={test_metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={test_metrics['macro_f1']:.4f}"
        + f" | qwk={test_metrics['qwk']:.4f}"
        + f" | mae={test_metrics['mae']:.4f}"
        + f" | weighted_f1={test_metrics['weighted_f1']:.4f}"
        + f" | precision_macro={test_metrics['precision_macro']:.4f}"
        + f" | recall_macro={test_metrics['recall_macro']:.4f}"
    )
    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f}"
            f" | ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )
    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    summary_row = {
        "dataset": "KOA",
        "method": display_name,
        "model_key": model_key,
        "seed": args.seed,
        "test_acc": float(test_metrics["acc"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_qwk": float(test_metrics["qwk"]),
        "test_mae": float(test_metrics["mae"]),
        "test_balanced_acc": float(test_metrics["balanced_acc"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "checkpoint": str(best_path),
    }
    summary_csv = save_summary_csv(THIS_DIR / "logs", model_key, run_tag, summary_row)
    print(f"\nSummary CSV saved: {summary_csv}")
    print(f"KOA baseline zoo completed for {display_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
