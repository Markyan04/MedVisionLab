#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chest X-ray dataset: standalone ablation script with purpose-focused comments.

本文件从零写出完整 PyTorch 训练流程，不复用项目内已有训练代码。
Chest X-ray 数据默认来自：

    CPN/
      train/<class folders>
      test/<class folders>

注释重点说明每段代码在消融实验中负责什么。

直接运行：
    python chest-x-ray-image_Loss/run_ablation_from_scratch.py
"""

from __future__ import annotations

import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. 写死的实验配置
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# DATASET_NAME 会写入日志和 summary.csv，后续汇总时用它区分数据集。
DATASET_NAME = "Chest X-ray Image"

# Chest X-ray 已有 train/test 两个目录；validation 从 train 中按 seed 切出。
DATA_ROOT = PROJECT_ROOT / "CPN"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

# 五个 seed 对应五次重复实验，每个 seed 下都跑完整四组消融配置。
SEEDS: Sequence[int] = (42, 777, 1234, 2024, 3407)

# 所有方法共享同一套训练超参数，保证比较集中在 MESC/DAST 两个因素上。
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_WORKERS = 2
PATIENCE = 15
EARLY_STOP_DELTA = 1e-4

# validation 从 train 中划出，test 始终只用于最终报告。
VAL_RATIO = 0.10

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4

# DAST 这里沿用类别编号距离构造 soft target；tau/gamma 固定为论文默认设置。
DAST_TAU = 1.0
DAST_GAMMA = 1.5

# 用 macro-F1 选 checkpoint，避免类别不均衡时 accuracy 过度偏向样本多的类别。
MONITOR_METRIC = "macro_f1"

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = THIS_DIR / "logs" / f"from_scratch_ablation_{RUN_STAMP}"
CKPT_DIR = THIS_DIR / "checkpoints" / f"from_scratch_ablation_{RUN_STAMP}"
SUMMARY_CSV = LOG_DIR / "summary.csv"

# 固定 2x2 消融：是否使用 layer3 MESC，是否使用 DAST loss。
EXPERIMENTS: Sequence[Tuple[str, bool, bool]] = (
    ("baseline", False, False),
    ("baseline+dast", False, True),
    ("baseline_layer3+mesc", True, False),
    ("baseline+dast+mesc", True, True),
)


# =============================================================================
# 2. 小工具
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.fp.write(message + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


# =============================================================================
# 3. 数据集
# =============================================================================

class PathImageDataset(data.Dataset):
    """根据路径按需读图，并返回 image tensor 和 label，便于自定义 train/val split。"""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        # X-ray 通常是灰度图；转 3 通道是为了直接使用 ImageNet ResNet50 的 conv1。
        transforms.Grayscale(num_output_channels=3),
        # 统一输入尺寸，保证四组消融配置的输入一致。
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 训练集增强用于提升泛化；验证/测试集不使用随机增强。
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        # ImageNet normalization 与预训练 backbone 的输入分布匹配。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def make_dataloaders(
    seed: int,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, List[str], int]:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"找不到训练目录: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"找不到测试目录: {TEST_DIR}")

    # ImageFolder 负责扫描类别子目录，并建立 class_to_idx 映射。
    raw_train = datasets.ImageFolder(str(TRAIN_DIR))
    raw_test = datasets.ImageFolder(str(TEST_DIR))
    class_names = list(raw_train.classes)
    num_classes = len(class_names)

    labels = [label for _, label in raw_train.samples]
    # stratify 让 validation 类别比例尽量接近 train，减少评估波动。
    train_samples, val_samples = train_test_split(
        raw_train.samples,
        test_size=VAL_RATIO,
        random_state=seed,
        stratify=labels,
    )

    train_transform, eval_transform = build_transforms()
    train_dataset = PathImageDataset(train_samples, transform=train_transform)
    val_dataset = PathImageDataset(val_samples, transform=eval_transform)
    test_dataset = PathImageDataset(raw_test.samples, transform=eval_transform)

    # generator 固定 shuffle 顺序，使同一个 seed 的训练轨迹可追踪。
    generator = torch.Generator().manual_seed(seed)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, class_names, num_classes


# =============================================================================
# 4. MESC 注意力模块
# =============================================================================

def global_median_pooling(x: torch.Tensor) -> torch.Tensor:
    """为通道注意力提供 median 统计量，减少异常高亮区域对权重的影响。"""

    b, c, _, _ = x.shape
    return x.view(b, c, -1).median(dim=2).values.view(b, c, 1, 1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def _branch(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(F.relu(self.fc1(pooled), inplace=True)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # avg/max/median 分别提供整体、最强局部和稳健中心三类通道统计。
        return (
            self._branch(F.adaptive_avg_pool2d(x, 1))
            + self._branch(F.adaptive_max_pool2d(x, 1))
            + self._branch(global_median_pooling(x))
        )


class MESC(nn.Module):
    def __init__(self, channels: int = 1024):
        super().__init__()
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.channel_attention = ChannelAttention(channels)
        self.initial_depth_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels),
        ])
        self.spatial_att_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先用 1x1 conv 混合通道，再计算注意力权重。
        x = self.act(self.pre_conv(x))
        # 通道注意力控制哪些 feature map 被强调。
        x_ca = self.channel_attention(x) * x
        # 多尺度 depthwise 条形卷积关注不同方向和不同尺度的空间结构。
        spatial = self.initial_depth_conv(x_ca)
        spatial = sum(conv(spatial) for conv in self.depth_convs)
        # 残差加回原特征，减少注意力分支带来的信息损失。
        spatial = spatial + x_ca
        return self.post_conv(torch.sigmoid(self.spatial_att_conv(spatial)) * x_ca)


# =============================================================================
# 5. DAST loss
# =============================================================================

class DistanceAwareSoftTargetLoss(nn.Module):
    def __init__(self, num_classes: int, tau: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.register_buffer("class_ids", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 使用类别编号距离生成 soft target；编号越接近，分到的目标概率越高。
        target = target.float().unsqueeze(1)
        distance = torch.abs(self.class_ids.unsqueeze(0) - target)
        soft_target = torch.exp(-distance / self.tau)
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)
        # soft CE 不再只把真实类视为 1，而是给相邻编号类别一部分权重。
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        loss = -(soft_target * log_prob).sum(dim=1)
        if self.gamma > 0:
            # 对模型当前不确定的样本加大权重，类似 focal loss 的作用。
            pt = (prob * soft_target).sum(dim=1).clamp(min=1e-8, max=1.0)
            loss = (1.0 - pt).pow(self.gamma) * loss
        return loss.mean()


# =============================================================================
# 6. 模型
# =============================================================================

def create_torchvision_resnet50() -> nn.Module:
    try:
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception as exc:
        print(f"[WARN] 无法加载 ResNet50 预训练权重，将使用随机初始化。原因: {exc}")
        try:
            return models.resnet50(weights=None)
        except TypeError:
            return models.resnet50(pretrained=False)


class ResNet50AblationModel(nn.Module):
    def __init__(self, num_classes: int, use_mesc: bool):
        super().__init__()
        base = create_torchvision_resnet50()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        # layer3 输出 1024 通道；Identity 让 baseline 与 MESC 版只差这一处。
        self.mesc = MESC(channels=1024) if use_mesc else nn.Identity()
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 结构消融点：MESC 或 Identity。
        x = self.mesc(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc") or name.startswith("mesc"):
            # 新分类头和 MESC 从当前任务学习，给较大学习率。
            head_params.append(param)
        else:
            # 预训练 backbone 小学习率微调，减少破坏通用视觉特征。
            backbone_params.append(param)
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_criterion(use_dast: bool, num_classes: int, device: torch.device) -> nn.Module:
    """loss 消融开关：False 使用 CE，True 使用 DAST。"""

    if use_dast:
        return DistanceAwareSoftTargetLoss(num_classes, tau=DAST_TAU, gamma=DAST_GAMMA).to(device)
    return nn.CrossEntropyLoss().to(device)


# =============================================================================
# 7. 训练/评估
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)              # 前向传播得到每个类别的 logits。
        loss = criterion(logits, labels)    # 按当前消融配置计算 loss。
        optimizer.zero_grad(set_to_none=True)  # 清除上一 batch 的梯度。
        loss.backward()                        # 计算当前 batch 的梯度。
        optimizer.step()                       # 更新模型参数。
        scheduler.step()                       # 每个 batch 更新学习率。
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)  # logit 最大的类别作为预测结果。
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    return total_loss / total_samples, metrics


def metric_to_monitor(metrics: Dict[str, float]) -> float:
    """返回用于保存最佳 checkpoint 的验证指标；异常时回退到 macro-F1。"""

    value = float(metrics.get(MONITOR_METRIC, np.nan))
    if np.isnan(value):
        return float(metrics["macro_f1"])
    return value


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 8. 单次实验
# =============================================================================

def run_one_experiment(
    seed: int,
    method_name: str,
    use_mesc: bool,
    use_dast: bool,
    logger: Logger,
    summary_rows: List[Dict[str, object]],
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log("")
    logger.log("=" * 90)
    logger.log(f"Dataset={DATASET_NAME} | Method={method_name} | seed={seed}")
    logger.log(f"use_mesc={use_mesc} | use_dast={use_dast} | device={device}")
    logger.log("=" * 90)

    # 每个 seed 都重新切分 train/val，并重新初始化训练状态。
    train_loader, val_loader, test_loader, class_names, num_classes = make_dataloaders(seed)
    model = ResNet50AblationModel(num_classes=num_classes, use_mesc=use_mesc).to(device)
    criterion = build_criterion(use_dast=use_dast, num_classes=num_classes, device=device)
    optimizer = build_optimizer(model)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    ckpt_path = CKPT_DIR / f"best_{safe_name(method_name)}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    logger.log(f"Classes: {class_names}")
    logger.log(f"Train/Val/Test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    logger.log(f"Trainable parameters: {count_trainable_parameters(model):,}")
    logger.log(f"Criterion: {criterion.__class__.__name__}")

    best_score = -float("inf")  # 当前 seed/method 的最佳验证分数。
    best_epoch = 0              # 最佳分数对应的 epoch。
    bad_epochs = 0              # 连续未提升次数，用于 early stopping。
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        score = metric_to_monitor(val_metrics)
        # 超过 delta 才认为有提升，避免指标微小抖动频繁覆盖 checkpoint。
        if score > best_score + EARLY_STOP_DELTA:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)  # 保存验证集最优权重。
        else:
            bad_epochs += 1

        logger.log(
            f"Epoch {epoch:03d}/{EPOCHS} | {elapsed:5.1f}s | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['acc'] * 100:.2f}% | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_qwk={val_metrics['qwk']:.4f} | val_mae={val_metrics['mae']:.4f} | "
            f"best_epoch={best_epoch} | bad_epochs={bad_epochs}"
        )

        if bad_epochs >= PATIENCE:
            logger.log(f"Early stopping at epoch {epoch}.")
            break

    # 最终 test 使用验证集选出的最佳 checkpoint，而不是最后一个 epoch。
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.log(
        f"Test | loss={test_loss:.4f} | acc={test_metrics['acc'] * 100:.2f}% | "
        f"balanced_acc={test_metrics['balanced_acc'] * 100:.2f}% | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f} | "
        f"precision_macro={test_metrics['precision_macro']:.4f} | "
        f"recall_macro={test_metrics['recall_macro']:.4f} | "
        f"qwk={test_metrics['qwk']:.4f} | mae={test_metrics['mae']:.4f}"
    )

    # 每行 summary 对应一次 seed+method，后续可直接按 method 做统计。
    summary_rows.append({
        "dataset": DATASET_NAME,
        "method": method_name,
        "seed": seed,
        "use_mesc": use_mesc,
        "use_dast": use_dast,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "test_loss": test_loss,
        **test_metrics,
        "checkpoint": str(ckpt_path),
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_summary(rows: List[Dict[str, object]]) -> None:
    """每完成一次实验就写 summary，避免长时间训练中断后丢失已完成结果。"""

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_DIR / "run.log")
    summary_rows: List[Dict[str, object]] = []
    try:
        logger.log(f"Dataset: {DATASET_NAME}")
        logger.log(f"Train dir: {TRAIN_DIR}")
        logger.log(f"Test dir: {TEST_DIR}")
        logger.log(f"Seeds: {list(SEEDS)}")
        logger.log(f"Experiments: {[name for name, _, _ in EXPERIMENTS]}")
        logger.log(f"Total runs: {len(SEEDS) * len(EXPERIMENTS)}")

        for seed in SEEDS:
            # 外层按 seed 跑，便于观察同一数据划分下四组方法的差异。
            for method_name, use_mesc, use_dast in EXPERIMENTS:
                run_one_experiment(seed, method_name, use_mesc, use_dast, logger, summary_rows)
                save_summary(summary_rows)

        logger.log("")
        logger.log(f"All done. Summary saved to: {SUMMARY_CSV}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
