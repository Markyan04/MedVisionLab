#!/usr/bin/env python3
"""Controlled 10-seed APTOS 2019 ablation runner.

The four experiments differ only in the use of the layer3 MECS VersionB block
and the DAST loss:

1. resnet_baseline: ResNet50 + cross entropy
2. resnet_layer3_mesc: ResNet50 + layer3 MECS VersionB + cross entropy
3. resnet_dast: ResNet50 + DAST
4. resnet_layer3_mesc_dast: ResNet50 + layer3 MECS VersionB + DAST

Each run writes an independently resumable checkpoint, a best checkpoint,
per-epoch history, a log, and a machine-readable summary.  The ``summarize``
subcommand collects completed runs and reports mean +/- sample standard
deviation without touching the test set during model selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# Required by PyTorch deterministic algorithms for CUDA >= 10.2.  It must be
# configured before torch initializes cuBLAS.  An explicit user value wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MECS_VersionB import MECS_VersionB  # noqa: E402
from medical_losses import DistanceAwareSoftTargetLoss  # noqa: E402


NUM_CLASSES = 5
CLASS_NAMES = ["0", "1", "2", "3", "4"]
TOPK = (1, 2, 3)

EXPERIMENTS: Mapping[str, Mapping[str, Any]] = {
    "resnet_baseline": {"use_mesc": False, "loss": "ce"},
    "resnet_layer3_mesc": {"use_mesc": True, "loss": "ce"},
    "resnet_dast": {"use_mesc": False, "loss": "dast"},
    "resnet_layer3_mesc_dast": {"use_mesc": True, "loss": "dast"},
}

AGGREGATE_METRICS = (
    "test_acc",
    "test_balanced_acc",
    "test_macro_f1",
    "test_weighted_f1",
    "test_qwk",
    "test_mae",
    "test_top1",
    "test_top2",
    "test_top3",
)


class APTOSDataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Any = None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["diagnosis"])


class ResNet50Layer3MECSVersionB(nn.Module):
    """ResNet50 with MECS VersionB inserted after layer3."""

    def __init__(self, num_classes: int, weights: Any) -> None:
        super().__init__()
        base = models.resnet50(weights=weights)

        # Construct the classifier before MECS so its initialization is paired
        # with the baseline classifier for the same seed.
        classifier = nn.Linear(base.fc.in_features, num_classes)
        mesc = MECS_VersionB(in_channels=1024, out_channels=1024)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.mesc = mesc
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv1(inputs)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.mesc(features)
        features = self.layer4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.fc(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="run one experiment/seed")
    train_parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "APTOS-2019"
    )
    train_parser.add_argument(
        "--output-root", type=Path, default=SCRIPT_DIR / "ablation_outputs"
    )
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    train_parser.add_argument("--lr-backbone", type=float, default=1e-4)
    train_parser.add_argument("--lr-new", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--tau", type=float, default=1.0)
    train_parser.add_argument("--gamma", type=float, default=1.5)
    train_parser.add_argument(
        "--weights", choices=("imagenet", "none"), default="imagenet"
    )
    train_parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--force", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument(
        "--keep-last",
        action="store_true",
        help="keep the large resumable last.pt after a run completes",
    )
    train_parser.add_argument("--no-amp", dest="amp", action="store_false")
    train_parser.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false"
    )
    train_parser.set_defaults(amp=True, deterministic=True)

    summary_parser = subparsers.add_parser(
        "summarize", help="aggregate all completed seed summaries"
    )
    summary_parser.add_argument(
        "--output-root", type=Path, default=SCRIPT_DIR / "ablation_outputs"
    )
    summary_parser.add_argument("--expected-seeds", type=int, default=10)

    args = parser.parse_args()
    if args.command == "train":
        positive_values = {
            "image-size": args.image_size,
            "batch-size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr-backbone": args.lr_backbone,
            "lr-new": args.lr_new,
        }
        for name, value in positive_values.items():
            if value <= 0:
                parser.error(f"--{name} must be positive")
        if args.num_workers < 0:
            parser.error("--num-workers must be non-negative")
        if args.weight_decay < 0 or args.early_stop_delta < 0:
            parser.error("--weight-decay and --early-stop-delta must be non-negative")
        if args.tau <= 0 or args.gamma < 0:
            parser.error("--tau must be positive and --gamma must be non-negative")
    elif args.expected_seeds <= 0:
        parser.error("--expected-seeds must be positive")
    return args


def setup_logger(log_path: Path, append: bool) -> logging.Logger:
    logger = logging.getLogger("aptos_ablation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path, mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_stem_to_path_map(image_dir: Path) -> Dict[str, str]:
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths: Dict[str, str] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            if path.stem in paths:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {image_dir}")
            paths[path.stem] = str(path)
    return paths


def prepare_split_dataframe(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    frame = pd.read_csv(csv_path)
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain columns {sorted(required_columns)}")
    if frame["id_code"].duplicated().any():
        duplicates = frame.loc[frame["id_code"].duplicated(), "id_code"].tolist()[:5]
        raise ValueError(f"Duplicate id_code values in {csv_path}: {duplicates}")

    image_paths = build_stem_to_path_map(image_dir)
    frame["id_code"] = frame["id_code"].astype(str)
    frame["path"] = frame["id_code"].map(image_paths)
    missing = frame.loc[frame["path"].isna(), "id_code"].tolist()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} rows in {csv_path.name} have no image in {image_dir}; "
            f"examples: {missing[:5]}"
        )

    frame["diagnosis"] = pd.to_numeric(frame["diagnosis"], errors="raise").astype(int)
    invalid_labels = sorted(set(frame["diagnosis"]) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise ValueError(f"Invalid diagnosis labels in {csv_path}: {invalid_labels}")
    if frame.empty:
        raise ValueError(f"Empty dataset split: {csv_path}")
    return frame[["id_code", "diagnosis", "path"]].reset_index(drop=True)


def resolve_split_csv(data_root: Path, split: str, csv_names: Sequence[str]) -> Path:
    csv_path = next(
        (data_root / name for name in csv_names if (data_root / name).is_file()),
        None,
    )
    if csv_path is None:
        expected = ", ".join(str(data_root / name) for name in csv_names)
        raise FileNotFoundError(f"No CSV found for {split} split; tried: {expected}")
    return csv_path


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> Tuple[
    data.DataLoader,
    data.DataLoader,
    data.DataLoader,
    Dict[str, pd.DataFrame],
    torch.Generator,
]:
    split_specs = {
        # The Kaggle upload uses train_1.csv, while some extracted/repacked
        # copies use train.csv.  Accept both without requiring a rename.
        "train": (("train.csv", "train_1.csv"), "train_images"),
        "valid": (("valid.csv",), "val_images"),
        "test": (("test.csv",), "test_images"),
    }
    frames = {}
    for split, (csv_names, image_dir) in split_specs.items():
        csv_path = resolve_split_csv(data_root, split, csv_names)
        frames[split] = prepare_split_dataframe(csv_path, data_root / image_dir)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    datasets = {
        "train": APTOSDataset(frames["train"], train_transform),
        "valid": APTOSDataset(frames["valid"], eval_transform),
        "test": APTOSDataset(frames["test"], eval_transform),
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    valid_generator = torch.Generator().manual_seed(seed + 10_000)
    test_generator = torch.Generator().manual_seed(seed + 20_000)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        # Restart workers each epoch so their RNG state is fully determined by
        # the saved DataLoader generator when an interrupted run is resumed.
        "persistent_workers": False,
    }
    train_loader = data.DataLoader(
        datasets["train"], shuffle=True, generator=train_generator, **common
    )
    valid_loader = data.DataLoader(
        datasets["valid"], shuffle=False, generator=valid_generator, **common
    )
    test_loader = data.DataLoader(
        datasets["test"], shuffle=False, generator=test_generator, **common
    )
    return train_loader, valid_loader, test_loader, frames, train_generator


def get_torchvision_weights(name: str) -> Any:
    return models.ResNet50_Weights.DEFAULT if name == "imagenet" else None


def build_model(use_mesc: bool, weights_name: str) -> nn.Module:
    weights = get_torchvision_weights(weights_name)
    if use_mesc:
        return ResNet50Layer3MECSVersionB(NUM_CLASSES, weights)
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_criterion(loss_name: str, tau: float, gamma: float) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    return DistanceAwareSoftTargetLoss(NUM_CLASSES, tau=tau, gamma=gamma)


def split_parameter_groups(
    model: nn.Module, use_mesc: bool
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    backbone_parameters: List[nn.Parameter] = []
    new_parameters: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_new = name.startswith("fc.") or (use_mesc and name.startswith("mesc."))
        (new_parameters if is_new else backbone_parameters).append(parameter)
    return backbone_parameters, new_parameters


def calculate_topk_correct(
    logits: torch.Tensor, targets: torch.Tensor, topk: Sequence[int]
) -> Dict[str, int]:
    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices
    return {
        f"top{k}": int(
            predictions[:, : min(k, max_k)].eq(targets[:, None]).any(dim=1).sum()
        )
        for k in topk
    }


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    topk_correct = {f"top{k}": 0 for k in TOPK}

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step after an overflow.  Advancing
        # OneCycleLR in that case produces a scheduler-before-optimizer warning
        # and shifts the schedule by one update.
        if scaler.get_scale() >= scale_before_step:
            scheduler.step()

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += float(loss.detach()) * batch_size
        for key, value in calculate_topk_correct(
            logits.detach(), targets, TOPK
        ).items():
            topk_correct[key] += value

    metrics = {"loss": total_loss / total_samples}
    metrics.update({key: value / total_samples for key, value in topk_correct.items()})
    return metrics


def compute_classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "acc": float(accuracy_score(targets, predictions)),
        "balanced_acc": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(targets, predictions, weights="quadratic")),
        "mae": float(np.mean(np.abs(targets - predictions))),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(NUM_CLASSES))
        ).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[targets]
        metrics["ovr_roc_auc_macro"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
        metrics["ovr_pr_auc_macro"] = float(
            average_precision_score(one_hot, probabilities, average="macro")
        )
    except ValueError:
        metrics["ovr_roc_auc_macro"] = None
        metrics["ovr_pr_auc_macro"] = None
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    topk_correct = {f"top{k}": 0 for k in TOPK}
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += float(loss) * batch_size
        for key, value in calculate_topk_correct(logits, targets, TOPK).items():
            topk_correct[key] += value
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    targets_np = np.concatenate(all_targets)
    predictions_np = np.concatenate(all_predictions)
    probabilities_np = np.concatenate(all_probabilities)
    metrics = compute_classification_metrics(
        targets_np, predictions_np, probabilities_np
    )
    metrics["loss"] = total_loss / total_samples
    metrics.update({key: value / total_samples for key, value in topk_correct.items()})
    return metrics


def format_metrics(prefix: str, metrics: Mapping[str, Any]) -> str:
    fields = [
        f"loss={metrics['loss']:.4f}",
        f"top1={metrics['top1'] * 100:.2f}%",
        f"top2={metrics['top2'] * 100:.2f}%",
        f"top3={metrics['top3'] * 100:.2f}%",
    ]
    for name in ("acc", "balanced_acc", "macro_f1", "qwk", "mae"):
        if name in metrics:
            fields.append(f"{name}={metrics[name]:.4f}")
    return f"{prefix} | " + " | ".join(fields)


def capture_rng_state(train_generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: Mapping[str, Any], train_generator: torch.Generator
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    train_generator.set_state(state["train_generator"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def torch_load_compat(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def atomic_json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            to_builtin(payload), handle, ensure_ascii=False, indent=2, allow_nan=False
        )
    os.replace(temporary, path)


def write_history(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    best_qwk: float,
    best_epoch: int,
    bad_epochs: int,
    history: Sequence[Mapping[str, Any]],
    train_generator: torch.Generator,
    run_config: Mapping[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_qwk": best_qwk,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "history": list(history),
        "rng_state": capture_rng_state(train_generator),
        "run_config": dict(run_config),
    }
    atomic_torch_save(payload, path)


def dry_run_batch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[Tuple[int, ...], float]:
    model.eval()
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    with torch.no_grad(), autocast_context(amp_enabled):
        logits = model(inputs)
        loss = criterion(logits, targets)
    return tuple(logits.shape), float(loss)


def run_training(args: argparse.Namespace) -> None:
    experiment = dict(EXPERIMENTS[args.experiment])
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / args.experiment / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.csv"
    log_path = run_dir / "run.log"

    if summary_path.is_file() and not args.force and not args.dry_run:
        if last_path.is_file() and not args.keep_last:
            last_path.unlink()
        print(f"[SKIP] Completed run already exists: {summary_path}")
        return
    if args.force and not args.dry_run:
        for path in (summary_path, best_path, last_path, history_path):
            if path.exists():
                path.unlink()

    logger = setup_logger(log_path, append=args.resume and last_path.is_file())
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    set_reproducibility(args.seed, args.deterministic)

    logger.info("Experiment=%s | seed=%d", args.experiment, args.seed)
    logger.info("Data root=%s", data_root)
    logger.info("Output=%s", run_dir)
    logger.info(
        "Device=%s | AMP=%s | deterministic=%s | CUBLAS_WORKSPACE_CONFIG=%s",
        device,
        amp_enabled,
        args.deterministic,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    if device.type == "cuda":
        logger.info("GPU=%s", torch.cuda.get_device_name(device))

    train_loader, valid_loader, test_loader, frames, train_generator = (
        build_dataloaders(
            data_root=data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
    )
    for split, frame in frames.items():
        distribution = frame["diagnosis"].value_counts().sort_index().to_dict()
        logger.info("%s samples=%d | classes=%s", split, len(frame), distribution)

    model = build_model(bool(experiment["use_mesc"]), args.weights).to(device)
    criterion = build_criterion(str(experiment["loss"]), args.tau, args.gamma).to(
        device
    )
    backbone_parameters, new_parameters = split_parameter_groups(
        model, bool(experiment["use_mesc"])
    )
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr_backbone},
            {"params": new_parameters, "lr": args.lr_new},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_new],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = make_grad_scaler(amp_enabled)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    new_parameter_count = sum(parameter.numel() for parameter in new_parameters)
    logger.info(
        "Model parameters=%s | new head/MECS parameters=%s | loss=%s",
        f"{total_parameters:,}",
        f"{new_parameter_count:,}",
        experiment["loss"],
    )
    if experiment["loss"] == "dast":
        logger.info("DAST tau=%.4f | gamma=%.4f", args.tau, args.gamma)

    if args.dry_run:
        shape, loss = dry_run_batch(
            model, train_loader, criterion, device=device, amp_enabled=amp_enabled
        )
        logger.info("DRY RUN OK | logits=%s | loss=%.6f", shape, loss)
        return

    run_config = {
        "experiment": args.experiment,
        "seed": args.seed,
        "use_mesc": bool(experiment["use_mesc"]),
        "mesc_version": "MECS_VersionB" if experiment["use_mesc"] else None,
        "mesc_position": "after_layer3" if experiment["use_mesc"] else None,
        "loss": experiment["loss"],
        "tau": args.tau if experiment["loss"] == "dast" else None,
        "gamma": args.gamma if experiment["loss"] == "dast" else None,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "patience": args.patience,
        "early_stop_delta": args.early_stop_delta,
        "lr_backbone": args.lr_backbone,
        "lr_new": args.lr_new,
        "weight_decay": args.weight_decay,
        "weights": args.weights,
        "amp": amp_enabled,
        "deterministic": args.deterministic,
        "data_root": str(data_root),
    }

    start_epoch = 0
    best_qwk = -math.inf
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    if args.resume and last_path.is_file():
        checkpoint = torch_load_compat(last_path, map_location="cpu")
        checkpoint_config = checkpoint.get("run_config", {})
        comparable_keys = (
            "experiment",
            "seed",
            "image_size",
            "batch_size",
            "num_workers",
            "epochs",
            "patience",
            "early_stop_delta",
            "lr_backbone",
            "lr_new",
            "weight_decay",
            "tau",
            "gamma",
            "weights",
            "amp",
            "deterministic",
        )
        mismatches = {
            key: (checkpoint_config.get(key), run_config.get(key))
            for key in comparable_keys
            if checkpoint_config.get(key) != run_config.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume with changed run configuration: {mismatches}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_qwk = float(checkpoint["best_qwk"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint["rng_state"], train_generator)
        logger.info(
            "Resumed after epoch %d | best epoch=%d | best valid QWK=%.6f",
            start_epoch,
            best_epoch + 1,
            best_qwk,
        )

    training_started = time.time()
    if bad_epochs >= args.patience:
        logger.info(
            "Early-stopping state already reached; proceeding to best-checkpoint test"
        )

    for epoch in range(start_epoch, args.epochs):
        if bad_epochs >= args.patience:
            break
        epoch_started = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            amp_enabled,
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled)
        epoch_seconds = time.time() - epoch_started
        valid_qwk = float(valid_metrics["qwk"])
        improved = best_epoch < 0 or (
            math.isfinite(valid_qwk) and valid_qwk > best_qwk + args.early_stop_delta
        )
        if improved:
            best_qwk = valid_qwk
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "run_config": run_config,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "epoch_seconds": epoch_seconds,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[1]["lr"],
            "best_valid_qwk": best_qwk,
            "bad_epochs": bad_epochs,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {
                f"valid_{key}": value
                for key, value in valid_metrics.items()
                if not isinstance(value, (dict, list))
            }
        )
        history.append(row)
        write_history(history, history_path)
        save_last_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_qwk,
            best_epoch,
            bad_epochs,
            history,
            train_generator,
            run_config,
        )

        logger.info("Epoch %02d/%02d | %.1fs", epoch + 1, args.epochs, epoch_seconds)
        logger.info(format_metrics("Train", train_metrics))
        logger.info(format_metrics("Valid", valid_metrics))
        logger.info(
            "Best valid QWK=%.6f at epoch %d | bad epochs=%d/%d",
            best_qwk,
            best_epoch + 1,
            bad_epochs,
            args.patience,
        )

    if not best_path.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {best_path}")
    best_checkpoint = torch_load_compat(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state"])
    valid_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled)
    training_seconds = sum(float(row["epoch_seconds"]) for row in history)

    logger.info("Selected epoch=%d using validation QWK only", best_epoch + 1)
    logger.info(format_metrics("Best-valid", valid_metrics))
    logger.info(format_metrics("Test", test_metrics))
    logger.info("Test confusion matrix=%s", test_metrics["confusion_matrix"])

    summary = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "seed": args.seed,
        "best_epoch": best_epoch + 1,
        "best_valid_qwk": best_qwk,
        "epochs_ran": len(history),
        "training_seconds": training_seconds,
        "current_session_seconds": time.time() - training_started,
        "total_parameters": total_parameters,
        "new_parameters": new_parameter_count,
        "split_sizes": {name: len(frame) for name, frame in frames.items()},
        "run_config": run_config,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path),
        "history_csv": str(history_path),
    }
    atomic_json_dump(summary, summary_path)
    logger.info("Completed summary=%s", summary_path)
    if last_path.is_file() and not args.keep_last:
        last_path.unlink()
        logger.info("Removed completed-run last.pt; best.pt is retained")


def flatten_run_summary(
    summary: Mapping[str, Any], summary_path: Path
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment": summary["experiment"],
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "best_valid_qwk": summary["best_valid_qwk"],
        "epochs_ran": summary["epochs_ran"],
        "training_seconds": summary["training_seconds"],
        "total_parameters": summary["total_parameters"],
        "new_parameters": summary["new_parameters"],
        "summary_path": str(summary_path),
    }
    for split in ("valid", "test"):
        for key, value in summary[f"{split}_metrics"].items():
            if isinstance(value, (int, float)) or value is None:
                row[f"{split}_{key}"] = value
    return row


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_runs(output_root: Path, expected_seeds: int) -> None:
    output_root = output_root.expanduser().resolve()
    summary_paths = sorted(output_root.glob("*/seed_*/summary.json"))
    rows: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") == "completed":
            rows.append(flatten_run_summary(summary, summary_path))
    if not rows:
        raise FileNotFoundError(
            f"No completed summary.json files found under {output_root}"
        )

    runs = pd.DataFrame(rows).sort_values(["experiment", "seed"]).reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_root / "runs.csv", index=False)

    aggregate_rows: List[Dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        subset = runs[runs["experiment"] == experiment]
        if subset.empty:
            continue
        aggregate: Dict[str, Any] = {
            "experiment": experiment,
            "n_seeds": int(subset["seed"].nunique()),
            "seeds": " ".join(str(seed) for seed in sorted(subset["seed"].unique())),
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(output_root / "aggregate.csv", index=False)

    table_rows = []
    for row in aggregate_rows:
        n_seeds = int(row["n_seeds"])
        status = (
            "OK"
            if n_seeds == expected_seeds
            else f"INCOMPLETE ({n_seeds}/{expected_seeds})"
        )
        table_rows.append(
            [
                row["experiment"],
                status,
                f"{row['test_acc_mean']:.4f} +/- {row['test_acc_std']:.4f}",
                f"{row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f}",
                f"{row['test_qwk_mean']:.4f} +/- {row['test_qwk_std']:.4f}",
                f"{row['test_mae_mean']:.4f} +/- {row['test_mae_std']:.4f}",
            ]
        )
    report = "# APTOS 10-seed ablation\n\n" + markdown_table(
        ("Experiment", "Seeds", "Test ACC", "Test Macro-F1", "Test QWK", "Test MAE"),
        table_rows,
    )
    report += (
        "\n\nValues are mean +/- sample standard deviation across completed seeds.\n"
    )
    (output_root / "aggregate.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nRun-level CSV: {output_root / 'runs.csv'}")
    print(f"Aggregate CSV: {output_root / 'aggregate.csv'}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_training(args)
    else:
        summarize_runs(args.output_root, args.expected_seeds)


if __name__ == "__main__":
    main()
