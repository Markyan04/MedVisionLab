#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""KOA ResNet50 baseline runner for CE/LS/SORD-CE/DAST loss comparison.

This file intentionally reuses the existing KOA baseline DAST script utilities
for data loading, metrics, training, evaluation, checkpointing, and summaries.
Set KNEE_LOSS to one of: ce, label_smoothing_ce, sord_ce, dast.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in (PROJECT_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medical_losses import (  # noqa: E402
    DistanceAwareSoftTargetLoss,
    LabelSmoothingCrossEntropyLoss,
    OrdinalSoftCrossEntropyLoss,
)


def _load_knee_baseline_module():
    module_path = THIS_DIR / "ResNet_baseline+Loss4.py"
    spec = importlib.util.spec_from_file_location("knee_resnet_baseline_loss4", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load KOA baseline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_knee_baseline_module()

SUPPORTED_LOSSES = ("ce", "label_smoothing_ce", "sord_ce", "dast")
LOSS_ALIASES = {
    "cross_entropy": "ce",
    "ls": "label_smoothing_ce",
    "ls_ce": "label_smoothing_ce",
    "label_smoothing": "label_smoothing_ce",
    "label-smoothing": "label_smoothing_ce",
    "label_smoothing_ce": "label_smoothing_ce",
    "sord": "sord_ce",
    "sord-ce": "sord_ce",
    "sord_ce": "sord_ce",
}
DISPLAY_NAMES = {
    "ce": "CrossEntropyLoss",
    "label_smoothing_ce": "Label Smoothing CE",
    "sord_ce": "SORD-CE",
    "dast": "DAST",
}


def resolve_loss_name() -> str:
    raw = os.getenv("KNEE_LOSS", "label_smoothing_ce").strip().lower()
    loss_name = LOSS_ALIASES.get(raw, raw)
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported KNEE_LOSS={raw!r}. Choose from: {SUPPORTED_LOSSES}")
    return loss_name


def build_criterion(loss_name: str, device: torch.device):
    label_smoothing = float(os.getenv("KNEE_LABEL_SMOOTHING", "0.1"))
    dast_tau = float(os.getenv("KNEE_DAST_TAU", "1.0"))
    dast_gamma = float(os.getenv("KNEE_DAST_GAMMA", "1.5"))
    sord_tau = float(os.getenv("KNEE_SORD_TAU", os.getenv("KNEE_DAST_TAU", "1.0")))

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("KNEE_LABEL_SMOOTHING must satisfy 0 <= value < 1.")
    if dast_tau <= 0:
        raise ValueError("KNEE_DAST_TAU must be > 0.")
    if dast_gamma < 0:
        raise ValueError("KNEE_DAST_GAMMA must be >= 0.")
    if sord_tau <= 0:
        raise ValueError("KNEE_SORD_TAU must be > 0.")

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
        hparams = {"dast_tau": None, "dast_gamma": None, "label_smoothing": None}
    elif loss_name == "label_smoothing_ce":
        criterion = LabelSmoothingCrossEntropyLoss(smoothing=label_smoothing)
        hparams = {
            "dast_tau": None,
            "dast_gamma": None,
            "label_smoothing": label_smoothing,
        }
    elif loss_name == "sord_ce":
        criterion = OrdinalSoftCrossEntropyLoss(num_classes=base.NUM_CLASSES, tau=sord_tau)
        hparams = {"dast_tau": sord_tau, "dast_gamma": 0.0, "label_smoothing": None}
    else:
        criterion = DistanceAwareSoftTargetLoss(
            num_classes=base.NUM_CLASSES,
            tau=dast_tau,
            gamma=dast_gamma,
        )
        hparams = {"dast_tau": dast_tau, "dast_gamma": dast_gamma, "label_smoothing": None}

    return criterion.to(device), hparams


def metric_line(prefix: str, loss_value, top_values, metrics) -> str:
    return (
        f"\n{prefix} | loss={loss_value:.4f} | "
        + " | ".join([f"{k}={v * 100:.2f}%" for k, v in top_values.items()])
        + f" | acc={metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={metrics['macro_f1']:.4f}"
        + f" | qwk={metrics['qwk']:.4f}"
        + f" | mae={metrics['mae']:.4f}"
        + f" | weighted_f1={metrics['weighted_f1']:.4f}"
        + f" | precision_macro={metrics['precision_macro']:.4f}"
        + f" | recall_macro={metrics['recall_macro']:.4f}"
    )


def main() -> None:
    loss_name = resolve_loss_name()
    print(f"Starting Knee Osteoarthritis ResNet50 loss comparison: {DISPLAY_NAMES[loss_name]}")
    print(f"Using device: {base.device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if base.RUN_TAG:
        print(f"Run tag: {base.RUN_TAG}")

    if not os.path.exists(base.DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {base.DATA_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        base.LOGS_DIR / f"ResNet_baseline_loss_compare_{loss_name}{base.RUN_SUFFIX}_{timestamp}_summary.csv"
    ).resolve()

    print(f"Data root: {base.DATA_ROOT}")
    print(
        "Config | "
        f"seed={base.SEED}, img_size={base.IMG_SIZE}, batch_size={base.BATCH_SIZE}, "
        f"epochs={base.EPOCHS}, num_workers={base.NUM_WORKERS}, "
        f"lr_backbone={base.LR_BACKBONE}, lr_head={base.LR_HEAD}, "
        f"weight_decay={base.WEIGHT_DECAY}, patience={base.PATIENCE}, "
        f"early_delta={base.EARLY_STOP_DELTA}"
    )

    train_loader, valid_loader, test_loader, auto_test_loader, train_dataset = base.make_dataloaders()

    print(
        f"\nSplit sizes | train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)}"
        + (f", auto_test={len(auto_test_loader.dataset)}" if auto_test_loader is not None else "")
    )

    print("Loading pretrained ResNet50...")
    weights = base.models.ResNet50_Weights.DEFAULT
    model = base.models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, base.NUM_CLASSES)
    model = model.to(base.device)

    print(f"Trainable parameters: {base.count_parameters(model):,}")

    class_counts = base.np.bincount(train_dataset.targets, minlength=base.NUM_CLASSES)
    print(f"\n[INFO] Class counts in training set: {class_counts}")

    criterion, hparams = build_criterion(loss_name, base.device)
    print(f"[INFO] Using {DISPLAY_NAMES[loss_name]}")
    if loss_name == "label_smoothing_ce":
        print(f"       epsilon={hparams['label_smoothing']:.4f}")
    elif loss_name == "sord_ce":
        print(f"       tau={hparams['dast_tau']:.4f}, gamma=0.0000")
    elif loss_name == "dast":
        print(f"       tau={hparams['dast_tau']:.4f}, gamma={hparams['dast_gamma']:.4f}")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = base.optim.AdamW(
        [
            {"params": backbone_params, "lr": base.LR_BACKBONE},
            {"params": head_params, "lr": base.LR_HEAD},
        ],
        weight_decay=base.WEIGHT_DECAY,
    )

    total_steps = base.EPOCHS * len(train_loader)
    scheduler = base.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[base.LR_BACKBONE, base.LR_HEAD],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_path = base.resolve_checkpoint_path(f"best_resnet50_knee_oa_{loss_name}.pt", base.RUN_SUFFIX)
    early_stopping = base.EarlyStopping(
        patience=base.PATIENCE,
        delta=base.EARLY_STOP_DELTA,
        save_path=str(best_path),
    )

    best_epoch = 0
    best_valid_loss = float("inf")
    best_valid_macro_f1 = float("nan")
    best_valid_qwk = float("nan")
    trained_epochs = 0

    print("Starting training...")
    for epoch in range(base.EPOCHS):
        start_time = time.time()

        train_loss, train_top = base.train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, base.device, topk=base.TOPK
        )
        valid_loss, valid_top, valid_metrics = base.evaluate(
            model, valid_loader, criterion, base.device, topk=base.TOPK
        )

        end_time = time.time()
        epoch_mins, epoch_secs = base.epoch_time(start_time, end_time)
        trained_epochs = epoch + 1

        print(f"\nEpoch {epoch + 1:02d}/{base.EPOCHS} | Time {epoch_mins}m {epoch_secs}s")
        print(
            f"  Train | loss={train_loss:.4f} | "
            + " | ".join([f"{k}={v * 100:.2f}%" for k, v in train_top.items()])
        )
        print(
            f"  Valid | loss={valid_loss:.4f} | "
            + " | ".join([f"{k}={v * 100:.2f}%" for k, v in valid_top.items()])
            + f" | acc={valid_metrics['acc'] * 100:.2f}%"
            + f" | bal_acc={valid_metrics['balanced_acc'] * 100:.2f}%"
            + f" | macro_f1={valid_metrics['macro_f1']:.4f}"
            + f" | qwk={valid_metrics['qwk']:.4f}"
            + f" | mae={valid_metrics['mae']:.4f}"
            + f" | weighted_f1={valid_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={valid_metrics['precision_macro']:.4f}"
            + f" | recall_macro={valid_metrics['recall_macro']:.4f}"
        )

        if valid_metrics["ovr_roc_auc_macro"] is not None:
            print(
                f"         ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f}"
                f" | ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
            )

        improved = early_stopping(valid_metrics["qwk"], model)
        if improved:
            best_epoch = epoch + 1
            best_valid_loss = valid_loss
            best_valid_macro_f1 = valid_metrics["macro_f1"]
            best_valid_qwk = valid_metrics["qwk"]

        if early_stopping.early_stop:
            print(f" Training stopped early at epoch {epoch + 1}.")
            break

    print("\nLoading best model and evaluating on test set...")
    ckpt = torch.load(best_path, map_location=base.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best validation QWK: {ckpt['best_score']:.4f}")

    test_loss, test_top, test_metrics = base.evaluate(
        model, test_loader, criterion, base.device, topk=base.TOPK
    )
    print(metric_line("Test", test_loss, test_top, test_metrics))

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
        "script_name": "ResNet_baseline_loss_compare.py",
        "loss_name": loss_name,
        "status": "success",
        "run_tag": base.RUN_TAG,
        "seed": base.SEED,
        "dast_tau": hparams["dast_tau"],
        "dast_gamma": hparams["dast_gamma"],
        "label_smoothing": hparams["label_smoothing"],
        "trained_epochs": trained_epochs,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_valid_qwk": best_valid_qwk,
        "test_loss": test_loss,
        "test_top1": base.top_value(test_top, "top1"),
        "test_top2": base.top_value(test_top, "top2"),
        "test_top3": base.top_value(test_top, "top3"),
        "test_acc": test_metrics["acc"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_qwk": test_metrics["qwk"],
        "test_mae": test_metrics["mae"],
        "test_ovr_roc_auc_macro": test_metrics["ovr_roc_auc_macro"],
        "test_ovr_pr_auc_macro": test_metrics["ovr_pr_auc_macro"],
        "checkpoint_path": str(best_path),
        "summary_path": str(summary_path),
    }

    if auto_test_loader is not None:
        print("\nEvaluating on auto_test set...")
        auto_test_loss, auto_test_top, auto_test_metrics = base.evaluate(
            model, auto_test_loader, criterion, base.device, topk=base.TOPK
        )
        print(metric_line("AutoTest", auto_test_loss, auto_test_top, auto_test_metrics))
        summary_row.update({
            "auto_test_loss": auto_test_loss,
            "auto_test_top1": base.top_value(auto_test_top, "top1"),
            "auto_test_top2": base.top_value(auto_test_top, "top2"),
            "auto_test_top3": base.top_value(auto_test_top, "top3"),
            "auto_test_acc": auto_test_metrics["acc"],
            "auto_test_balanced_acc": auto_test_metrics["balanced_acc"],
            "auto_test_macro_f1": auto_test_metrics["macro_f1"],
            "auto_test_weighted_f1": auto_test_metrics["weighted_f1"],
            "auto_test_precision_macro": auto_test_metrics["precision_macro"],
            "auto_test_recall_macro": auto_test_metrics["recall_macro"],
            "auto_test_qwk": auto_test_metrics["qwk"],
            "auto_test_mae": auto_test_metrics["mae"],
            "auto_test_ovr_roc_auc_macro": auto_test_metrics["ovr_roc_auc_macro"],
            "auto_test_ovr_pr_auc_macro": auto_test_metrics["ovr_pr_auc_macro"],
        })

    base.write_summary_row(summary_path, summary_row)
    print(f"Summary CSV saved: {summary_path}")
    print(f"Checkpoint saved: {best_path}")
    print(f"\nKnee Osteoarthritis ResNet50 loss comparison ({loss_name}) completed!")


if __name__ == "__main__":
    main()
