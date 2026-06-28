#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a layer3 attention baseline using the existing dataset runners."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_modules import build_attention_module  # noqa: E402


def set_common_env(prefix: str, args: argparse.Namespace) -> None:
    os.environ[f"{prefix}_LOSSES"] = args.loss
    os.environ[f"{prefix}_RUN_TAG"] = args.run_tag
    if args.seed is not None:
        os.environ["GLOBAL_EXPERIMENT_SEED"] = str(args.seed)
        os.environ[f"{prefix}_SEED"] = str(args.seed)
    if args.epochs is not None:
        os.environ[f"{prefix}_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ[f"{prefix}_BATCH_SIZE"] = str(args.batch_size)
    if args.image_size is not None:
        os.environ[f"{prefix}_IMAGE_SIZE"] = str(args.image_size)
    if args.num_workers is not None:
        os.environ[f"{prefix}_NUM_WORKERS"] = str(args.num_workers)
    if args.patience is not None:
        os.environ[f"{prefix}_PATIENCE"] = str(args.patience)
    if args.early_delta is not None:
        os.environ[f"{prefix}_EARLY_DELTA"] = str(args.early_delta)


def load_knee_layer3_mesc_ce_module():
    module_path = PROJECT_ROOT / "Knee" / "ResNet_layer3+MECS+CE.py"
    spec = importlib.util.spec_from_file_location("knee_layer3_mesc_ce", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load KOA layer3 MESC CE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_knee_summary_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run_koa(args: argparse.Namespace) -> None:
    if args.loss != "ce":
        raise ValueError("KOA attention comparison is CE-only; do not mix DAST into this experiment.")

    set_common_env("KNEE", args)
    base = load_knee_layer3_mesc_ce_module()

    class CustomResNet50Attention(base.nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            base_model = base.models.resnet50(weights=base.models.ResNet50_Weights.DEFAULT)
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            # Keep the attribute name "mecs" so KOA uses the same LR grouping as the existing MESC run.
            self.mecs = build_attention_module(args.attention, 1024)
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            self.fc = base.nn.Linear(base_model.fc.in_features, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.mecs(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = base.torch.flatten(x, 1)
            return self.fc(x)

    print(f"Starting ResNet50 layer3 {args.attention} + CE on KOA...")
    print(f"Using device: {base.device}")
    if base.torch.cuda.is_available():
        print(f"CUDA device: {base.torch.cuda.get_device_name(0)}")
    if base.RUN_TAG:
        print(f"Run tag: {base.RUN_TAG}")
    print(f"Config | seed={base.SEED}, batch_size={base.BATCH_SIZE}, epochs={base.EPOCHS}, image_size={base.IMG_SIZE}")

    if not os.path.exists(base.DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {base.DATA_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = base.THIS_DIR / "logs"
    summary_path = logs_dir / f"ResNet_layer3+{args.attention}_{base.RUN_TAG}_{timestamp}_summary.csv"

    train_loader, valid_loader, test_loader, auto_test_loader, train_dataset = base.make_dataloaders()
    print(
        f"\nSplit sizes | train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
        + (f", auto_test={len(auto_test_loader.dataset)}" if auto_test_loader is not None else "")
    )

    print(f"Loading ResNet50 with {args.attention} module at layer3...")
    model = CustomResNet50Attention(num_classes=base.NUM_CLASSES).to(base.device)
    print(f"Trainable parameters: {base.count_parameters(model):,}")

    class_counts = base.np.bincount(train_dataset.targets, minlength=base.NUM_CLASSES)
    print(f"\n[INFO] Class counts in training set: {class_counts}")
    print("[INFO] Using Standard CrossEntropyLoss (No class weights).")
    criterion = base.nn.CrossEntropyLoss()

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name or "mecs" in name:
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
    scheduler = base.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[base.LR_BACKBONE, base.LR_HEAD],
        total_steps=base.EPOCHS * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_path = base.resolve_checkpoint_path(f"best_resnet50_{args.attention}_layer3_knee_oa.pt")
    early_stopping = base.EarlyStopping(
        patience=base.PATIENCE,
        delta=base.EARLY_STOP_DELTA,
        save_path=best_path,
    )

    best_epoch = 0
    best_valid_loss = float("inf")
    best_valid_macro_f1 = float("nan")
    best_valid_qwk = float("nan")
    trained_epochs = 0

    print("Starting training...")
    for epoch in range(base.EPOCHS):
        start_time = base.time.time()
        train_loss, train_top = base.train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, base.device, topk=base.TOPK
        )
        valid_loss, valid_top, valid_metrics = base.evaluate(
            model, valid_loader, criterion, base.device, topk=base.TOPK
        )
        end_time = base.time.time()
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

        previous_best = early_stopping.best_score
        early_stopping(valid_metrics["qwk"], model)
        if previous_best is None or early_stopping.best_score != previous_best:
            best_epoch = epoch + 1
            best_valid_loss = valid_loss
            best_valid_macro_f1 = valid_metrics["macro_f1"]
            best_valid_qwk = valid_metrics["qwk"]

        if early_stopping.early_stop:
            print(f" Training stopped early at epoch {epoch + 1}.")
            break

    print("\nLoading best model and evaluating on test set...")
    ckpt = base.torch.load(best_path, map_location=base.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best validation QWK: {ckpt['best_score']:.4f}")

    test_loss, test_top, test_metrics = base.evaluate(
        model, test_loader, criterion, base.device, topk=base.TOPK
    )
    print(
        f"\nTest | loss={test_loss:.4f} | "
        + " | ".join([f"{k}={v * 100:.2f}%" for k, v in test_top.items()])
        + f" | acc={test_metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={test_metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={test_metrics['macro_f1']:.4f}"
        + f" | qwk={test_metrics['qwk']:.4f}"
        + f" | mae={test_metrics['mae']:.4f}"
        + f" | weighted_f1={test_metrics['weighted_f1']:.4f}"
        + f" | precision_macro={test_metrics['precision_macro']:.4f}"
        + f" | recall_macro={test_metrics['recall_macro']:.4f}"
    )
    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    summary_row = {
        "script_name": f"ResNet_layer3+{args.attention}.py",
        "attention_module": args.attention,
        "loss_name": "ce",
        "status": "success",
        "run_tag": base.RUN_TAG,
        "seed": base.SEED,
        "trained_epochs": trained_epochs,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_valid_qwk": best_valid_qwk,
        "test_loss": test_loss,
        "test_top1": test_top.get("top1"),
        "test_top2": test_top.get("top2"),
        "test_top3": test_top.get("top3"),
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
        auto_test_loss, auto_test_top, auto_test_metrics = base.evaluate(
            model, auto_test_loader, criterion, base.device, topk=base.TOPK
        )
        summary_row.update({
            "auto_test_loss": auto_test_loss,
            "auto_test_top1": auto_test_top.get("top1"),
            "auto_test_top2": auto_test_top.get("top2"),
            "auto_test_top3": auto_test_top.get("top3"),
            "auto_test_acc": auto_test_metrics["acc"],
            "auto_test_macro_f1": auto_test_metrics["macro_f1"],
            "auto_test_qwk": auto_test_metrics["qwk"],
            "auto_test_mae": auto_test_metrics["mae"],
        })

    write_knee_summary_row(summary_path, summary_row)
    print(f"Summary CSV saved: {summary_path}")
    print(f"Checkpoint saved: {best_path}")


def run_ham10000(args: argparse.Namespace) -> None:
    from HAM10000_Loss.ham10000_loss_experiment_common import ResNet50WithInsertedModule, run_ham10000_medical_losses_experiments

    set_common_env("HAM10000", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_ham10000_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_adni(args: argparse.Namespace) -> None:
    from Alzheimer_MRI_Loss.alzheimer_mri_loss_experiment_common import ResNet50WithInsertedModule, run_alzheimer_mri_medical_losses_experiments

    set_common_env("ALZHEIMER", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_alzheimer_mri_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_brain(args: argparse.Namespace) -> None:
    from Brain_Tumor_MRI_Loss.brain_tumor_mri_loss_experiment_common import ResNet50WithInsertedModule, run_brain_tumor_mri_medical_losses_experiments

    set_common_env("BRAIN_MRI", args)

    def build_model(num_classes: int):
        return ResNet50WithInsertedModule(num_classes, build_attention_module(args.attention, 1024), "layer3")

    run_brain_tumor_mri_medical_losses_experiments(
        script_stem=f"ResNet_layer3+{args.attention}",
        model_builder=build_model,
        optimizer_group_divisors=[("conv1", 10), ("bn1", 10), ("layer1", 8), ("layer2", 6), ("layer3", 4), ("inserted_module", 3), ("layer4", 2), ("fc", 1)],
        module_name=args.attention,
        insert_after="layer3",
    )


def run_chest(args: argparse.Namespace) -> None:
    from chest_xray_loss_import import run_chest_attention

    run_chest_attention(args, build_attention_module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("koa", "ham10000", "adni", "brain", "chest"), required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--loss", choices=("ce", "dast"), default="ce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-delta", type=float, default=None)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    if not args.run_tag:
        args.run_tag = f"layer3_{args.attention}_{args.loss}_seed{args.seed}"

    runners = {
        "koa": run_koa,
        "ham10000": run_ham10000,
        "adni": run_adni,
        "brain": run_brain,
        "chest": run_chest,
    }
    runners[args.dataset](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
