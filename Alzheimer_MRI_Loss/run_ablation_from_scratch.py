#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ADNI dataset: standalone ablation script with purpose-focused comments.

本文件不复用项目里的 common 训练代码、MESC 文件或 medical_losses 文件。
ADNI 数据没有固定的 train/val/test 目录，所以这里从
Alzheimer_MRI/OriginalDataset 的类别文件夹中收集图片，然后对每个 seed
重新做 stratified train/val/test split。
注释重点解释每段代码在实验流程里承担什么职责。

直接运行：
    python Alzheimer_MRI_Loss/run_ablation_from_scratch.py
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

# DATASET_NAME 写入日志和 summary.csv，方便后续把四个数据集结果合并分析。
DATASET_NAME = "ADNI"

# ADNI 原始数据按类别文件夹存放，没有现成 split，因此后面会按 seed 分层划分。
DATA_ROOT = PROJECT_ROOT / "Alzheimer_MRI" / "OriginalDataset"

# 这里的顺序很重要：DAST、QWK、MAE 都默认类别编号有“病情等级”的含义。
# 如果调整顺序，DAST 里的类别距离含义也会随之改变。
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 五个 seed 对应五次重复实验，用来估计结果稳定性。
SEEDS: Sequence[int] = (42, 777, 1234, 2024, 3407)

# 所有消融配置共享同一组训练超参数，保证只比较 MESC/DAST 两个因素。
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_WORKERS = 2
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4

# 先切出 20% test，再从剩余 80% 中切 10% 做 validation。
TEST_RATIO = 0.20
VAL_RATIO_WITHIN_TRAIN = 0.10

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4

# DAST_TAU 决定软标签扩散范围；DAST_GAMMA 决定难样本加权强度。
DAST_TAU = 1.0
DAST_GAMMA = 1.5

# ADNI 使用 macro-F1 选 checkpoint，避免类别不均衡时 accuracy 过度偏向大类。
MONITOR_METRIC = "macro_f1"

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = THIS_DIR / "logs" / f"from_scratch_ablation_{RUN_STAMP}"
CKPT_DIR = THIS_DIR / "checkpoints" / f"from_scratch_ablation_{RUN_STAMP}"
SUMMARY_CSV = LOG_DIR / "summary.csv"

# 四组配置只改变两个布尔开关：是否插入 MESC，是否使用 DAST。
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
# 3. 数据集：从类别文件夹收集路径，再写 Dataset
# =============================================================================

class PathImageDataset(data.Dataset):
    """保存图片路径和标签，训练时按需读图，避免一次性把所有图片加载进内存。"""

    def __init__(self, samples: Sequence[Tuple[Path, int]], transform=None):
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


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_samples() -> List[Tuple[Path, int]]:
    """按 CLASS_NAMES 的顺序收集图片，确保标签编号和病情顺序一致。"""

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"找不到 ADNI 数据目录: {DATA_ROOT}")

    samples: List[Tuple[Path, int]] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATA_ROOT / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"找不到类别文件夹: {class_dir}")
        files = sorted(path for path in class_dir.rglob("*") if is_image_file(path))
        if not files:
            raise RuntimeError(f"类别文件夹没有图片: {class_dir}")
        # 每个样本只记录路径和整数标签，真正的图像解码放在 Dataset.__getitem__ 中完成。
        samples.extend((path, label) for path in files)
    return samples


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        # ADNI MRI 常见为灰度图，转成 3 通道是为了匹配 ImageNet ResNet50 的输入层。
        transforms.Grayscale(num_output_channels=3),
        # 统一输入尺寸，保证四组消融配置的模型输入完全一致。
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 只在训练集做随机增强，validation/test 保持确定性。
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        # ImageNet normalization 与预训练 backbone 的输入分布保持一致。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def make_dataloaders(seed: int) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    samples = collect_samples()
    labels = [label for _, label in samples]

    # stratify=labels 保证每个 split 中类别比例尽量接近原始数据集。
    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=TEST_RATIO,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [label for _, label in train_val_samples]
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=VAL_RATIO_WITHIN_TRAIN,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_transform, eval_transform = build_transforms()
    train_dataset = PathImageDataset(train_samples, transform=train_transform)
    val_dataset = PathImageDataset(val_samples, transform=eval_transform)
    test_dataset = PathImageDataset(test_samples, transform=eval_transform)

    # generator 固定 shuffle 顺序，使每个 seed 的训练数据顺序可复现。
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
    return train_loader, val_loader, test_loader


# =============================================================================
# 4. MESC 注意力模块
# =============================================================================

def global_median_pooling(x: torch.Tensor) -> torch.Tensor:
    """为通道注意力提供 median 统计量，降低异常亮点对权重估计的影响。"""

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
        # avg/max/median 分别描述整体响应、最强响应和稳健中心趋势。
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
        # 1x1 conv 先做通道混合，再进入通道注意力和空间注意力。
        x = self.act(self.pre_conv(x))
        # 通道注意力决定“哪些 feature map 更值得保留”。
        x_ca = self.channel_attention(x) * x
        # 多个 depthwise 条形卷积负责捕获不同方向、不同尺度的空间上下文。
        spatial = self.initial_depth_conv(x_ca)
        spatial = sum(conv(spatial) for conv in self.depth_convs)
        # 残差保留原特征，避免注意力分支过度抑制有效信息。
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
        # 根据真实类别和所有类别的距离生成软标签，邻近等级会获得更高权重。
        target = target.float().unsqueeze(1)
        distance = torch.abs(self.class_ids.unsqueeze(0) - target)
        soft_target = torch.exp(-distance / self.tau)
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)
        # soft CE 让模型不仅学习“正确/错误”，也学习等级之间的接近关系。
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        loss = -(soft_target * log_prob).sum(dim=1)
        if self.gamma > 0:
            # 类似 focal loss，模型越不确定的样本 loss 权重越大。
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
        # layer3 输出 1024 通道；不用 MESC 时用 Identity，使网络路径只差这一处。
        self.mesc = MESC(channels=1024) if use_mesc else nn.Identity()
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 消融结构变量：这里要么执行 MESC，要么什么都不做。
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
            # 新分类头和新注意力模块学习率较大，便于快速适应 ADNI。
            head_params.append(param)
        else:
            # 预训练 backbone 学习率较小，减少对已有视觉特征的破坏。
            backbone_params.append(param)
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_criterion(use_dast: bool, device: torch.device) -> nn.Module:
    """loss 消融开关：False 使用 CE，True 使用 DAST。"""

    if use_dast:
        return DistanceAwareSoftTargetLoss(NUM_CLASSES, tau=DAST_TAU, gamma=DAST_GAMMA).to(device)
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
        logits = model(images)              # 前向传播得到每类 logits。
        loss = criterion(logits, labels)    # 当前配置对应的训练目标。
        optimizer.zero_grad(set_to_none=True)  # 清空上一 batch 的梯度。
        loss.backward()                        # 根据 loss 计算梯度。
        optimizer.step()                       # 更新模型参数。
        scheduler.step()                       # OneCycleLR 按 batch 调整学习率。
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
        preds = logits.argmax(dim=1)  # logits 最大的位置就是预测类别。
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
    """返回用于 early stopping 的验证指标；异常时回退到 macro-F1。"""

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

    # 每个 seed 重新划分数据和构建 loader，保证五次重复实验互相独立。
    train_loader, val_loader, test_loader = make_dataloaders(seed)
    # 每次实验重新初始化模型/loss/optimizer，避免不同方法之间共享训练状态。
    model = ResNet50AblationModel(num_classes=NUM_CLASSES, use_mesc=use_mesc).to(device)
    criterion = build_criterion(use_dast=use_dast, device=device)
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

    logger.log(f"Train/Val/Test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    logger.log(f"Trainable parameters: {count_trainable_parameters(model):,}")
    logger.log(f"Criterion: {criterion.__class__.__name__}")

    best_score = -float("inf")  # 当前 seed/method 的最佳验证分数。
    best_epoch = 0              # 最佳 checkpoint 对应的 epoch。
    bad_epochs = 0              # 连续没有提升的 epoch 数。
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        score = metric_to_monitor(val_metrics)
        # 只有超过 delta 的提升才覆盖 checkpoint，减少指标微小波动造成的反复保存。
        if score > best_score + EARLY_STOP_DELTA:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)  # 只保存模型权重，方便后续 test 复现。
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

    # 测试集只评估验证集挑出的最佳模型，而不是最后一个 epoch。
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

    # summary.csv 一行对应一次 seed+method，便于后续统计均值和标准差。
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
    """每完成一次实验就写 summary，长任务中断时也能保留已完成结果。"""

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
        logger.log(f"Data root: {DATA_ROOT}")
        logger.log(f"Class order: {CLASS_NAMES}")
        logger.log(f"Seeds: {list(SEEDS)}")
        logger.log(f"Experiments: {[name for name, _, _ in EXPERIMENTS]}")
        logger.log(f"Total runs: {len(SEEDS) * len(EXPERIMENTS)}")

        for seed in SEEDS:
            # 外层按 seed 组织，方便检查同一随机种子下四组配置的差异。
            for method_name, use_mesc, use_dast in EXPERIMENTS:
                run_one_experiment(seed, method_name, use_mesc, use_dast, logger, summary_rows)
                save_summary(summary_rows)

        logger.log("")
        logger.log(f"All done. Summary saved to: {SUMMARY_CSV}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
