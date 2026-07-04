#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""KOA dataset: standalone ablation script with purpose-focused comments.

这个文件故意不复用本项目里的任何训练函数、MESC 文件或 loss 文件。
注释重点说明“这一步在消融实验里负责什么”，尽量围绕实验目的解释。

1. 读取数据集；
2. 定义 MESC 注意力模块；
3. 定义 DAST loss；
4. 定义 ResNet50 baseline / ResNet50 + layer3 MESC；
5. 依次跑 5 个 seed 下的 4 个消融配置；
6. 保存每次实验最优 checkpoint，并把最终 test 指标写入 csv。

直接运行：
    python Knee/run_ablation_from_scratch.py
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)


# =============================================================================
# 1. 写死的实验配置
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# DATASET_NAME 会写入日志和 summary.csv，用来区分不同数据集的实验结果。
DATASET_NAME = "KOA"

# KOA 已经整理成固定的 train/val/test 三个目录，因此这里直接写死三个 split 路径。
DATA_ROOT = PROJECT_ROOT / "Knee_Osteoarthritis"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# 类别顺序对应 KL grade 0-4；DAST 的“类别距离”就依赖这个顺序。
CLASS_NAMES = ["0_Normal", "1_Doubtful", "2_Mild", "3_Moderate", "4_Severe"]
NUM_CLASSES = len(CLASS_NAMES)

# 五个 seed 是论文/实验统计用的重复实验，不在运行时开放修改，避免本脚本变成通用 launcher。
SEEDS: Sequence[int] = (42, 777, 1234, 2024, 3407)

# 下面这些超参数固定为当前实验默认值；四个消融配置共享同一套训练条件。
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_WORKERS = 4
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

# DAST_TAU 控制软标签向邻近等级扩散的范围；DAST_GAMMA 控制难样本加权强度。
DAST_TAU = 1.0
DAST_GAMMA = 1.5

# KOA 是有序分级任务，验证集用 QWK 挑最好 checkpoint 更贴近等级一致性。
MONITOR_METRIC = "qwk"

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = THIS_DIR / "logs" / f"from_scratch_ablation_{RUN_STAMP}"
CKPT_DIR = THIS_DIR / "checkpoints" / f"from_scratch_ablation_{RUN_STAMP}"
SUMMARY_CSV = LOG_DIR / "summary.csv"


# 每一个 tuple 表示一组消融实验，只改变 MESC 和 DAST 两个因素。
# 这样四组结果可以回答三个问题：
#   baseline -> 原始 ResNet50 水平；
#   baseline+dast -> 只看 DAST loss 的影响；
#   baseline_layer3+mesc -> 只看 layer3 后插入 MESC 的影响；
#   baseline+dast+mesc -> 看 DAST 和 MESC 同时使用的影响。
EXPERIMENTS: Sequence[Tuple[str, bool, bool]] = (
    ("baseline", False, False),
    ("baseline+dast", False, True),
    ("baseline_layer3+mesc", True, False),
    ("baseline+dast+mesc", True, True),
)


# =============================================================================
# 2. 小工具：随机种子、日志、文件名
# =============================================================================

def set_seed(seed: int) -> None:
    """固定所有常见随机源，让同一个 seed 对应一条稳定的实验轨迹。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic=True 会让一些 CUDA 算子选择确定性实现。
    # benchmark=False 避免 cuDNN 根据输入动态挑选算法，从而减少随机波动。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader 多进程读取数据时，每个 worker 也要有自己的固定 seed。"""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_name(text: str) -> str:
    """把方法名变成适合做文件名的一小段字符串。"""

    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


class Logger:
    """同时把信息打印到屏幕和 log 文件里。"""

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
# 3. 数据集：ImageFolder 是 PyTorch 最常用的分类数据读取方式
# =============================================================================

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """训练集做轻量增强；验证/测试集只做确定性预处理，保证评估不引入随机扰动。"""

    train_transform = transforms.Compose([
        # 统一尺寸是为了让 ResNet50 的输入形状固定，也方便不同实验配置公平比较。
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 只在训练集做增强，目的是提高泛化；验证/测试集不做这些随机变化。
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        # ToTensor 把 PIL image 转成 [C,H,W] tensor，并把像素缩放到 [0,1]。
        transforms.ToTensor(),
        # 使用 ImageNet 均值方差，是因为 backbone 使用 ImageNet 预训练 ResNet50。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def make_dataloaders(seed: int) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """从 train/val/test 三个目录创建 DataLoader，返回顺序固定为训练/验证/测试。"""

    for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not folder.exists():
            raise FileNotFoundError(f"找不到数据目录: {folder}")

    train_transform, eval_transform = build_transforms()
    # ImageFolder 会把每个子文件夹当成一个类别，并自动生成 class_to_idx。
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transform)

    if len(train_dataset.classes) != NUM_CLASSES:
        raise RuntimeError(
            f"KOA 应该有 {NUM_CLASSES} 个类别目录，但实际读到 {len(train_dataset.classes)} 个: "
            f"{train_dataset.classes}"
        )

    # generator 控制 shuffle 的随机性；worker_init_fn 控制每个 worker 内部的数据增强随机性。
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
# 4. MESC：把通道注意力和多尺度空间注意力写在同一个文件里
# =============================================================================

def global_median_pooling(x: torch.Tensor) -> torch.Tensor:
    """对每个通道的 H*W 个空间位置取中位数，为通道注意力提供抗极值的统计量。"""

    b, c, _, _ = x.shape
    return x.view(b, c, -1).median(dim=2).values.view(b, c, 1, 1)


class ChannelAttention(nn.Module):
    """MESC 的通道注意力：avg/max/median 三种统计量共同决定每个通道的重要性。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def _branch(self, pooled: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(pooled), inplace=True)
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # avg 看整体响应，max 看最强局部响应，median 减少异常亮点对权重的影响。
        avg_weight = self._branch(F.adaptive_avg_pool2d(x, 1))
        max_weight = self._branch(F.adaptive_max_pool2d(x, 1))
        median_weight = self._branch(global_median_pooling(x))
        return avg_weight + max_weight + median_weight


class MESC(nn.Module):
    """Layer3 后使用的 MESC 模块。

    ResNet50 的 layer3 输出通道数是 1024，所以本实验中 channels=1024。
    """

    def __init__(self, channels: int = 1024):
        super().__init__()
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.channel_attention = ChannelAttention(channels)

        self.initial_depth_conv = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
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
        # 先用 1x1 conv 做通道混合，避免后面的 depthwise conv 只处理原始通道。
        x = self.act(self.pre_conv(x))
        # 通道注意力先筛选“哪些特征通道更重要”。
        x_ca = self.channel_attention(x) * x

        # depthwise 多尺度卷积只在空间维度建模，参数量比普通卷积小很多。
        spatial = self.initial_depth_conv(x_ca)
        spatial = sum(conv(spatial) for conv in self.depth_convs)
        # 残差加回 x_ca，避免空间注意力分支破坏原始响应。
        spatial = spatial + x_ca

        # spatial_weight 是逐像素/逐通道的空间权重，用来强调关键区域。
        spatial_weight = torch.sigmoid(self.spatial_att_conv(spatial))
        out = spatial_weight * x_ca
        return self.post_conv(out)


# =============================================================================
# 5. DAST loss：把整数标签变成“距离越近权重越大”的软标签
# =============================================================================

class DistanceAwareSoftTargetLoss(nn.Module):
    """Distance-Aware Soft Target loss.

    普通 CE 的标签是 one-hot，例如真实类别为 2：
        [0, 0, 1, 0, 0]

    DAST 会给相邻等级一点概率，例如：
        [0.07, 0.19, 0.51, 0.19, 0.07]

    tau 控制软标签扩散程度；gamma 类似 focal loss，强调难样本。
    """

    def __init__(self, num_classes: int, tau: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.gamma = gamma
        self.register_buffer("class_ids", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target 原本是 [B] 的整数标签，这里变成 [B,1] 方便和所有类别编号计算距离。
        target = target.float().unsqueeze(1)
        # distance[b,c] 表示第 b 个样本真实等级和类别 c 的等级距离。
        distance = torch.abs(self.class_ids.unsqueeze(0) - target)
        # 距离越近，soft target 权重越大；距离越远，权重指数衰减。
        soft_target = torch.exp(-distance / self.tau)
        soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)

        # 用 soft target 做 cross entropy，而不是只惩罚非真实类别。
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        loss = -(soft_target * log_prob).sum(dim=1)

        if self.gamma > 0:
            # pt 越低表示模型对软标签越不确定；gamma 会提高这类样本的损失权重。
            pt = (prob * soft_target).sum(dim=1).clamp(min=1e-8, max=1.0)
            loss = (1.0 - pt).pow(self.gamma) * loss
        return loss.mean()


# =============================================================================
# 6. 模型：用 torchvision ResNet50，并可选在 layer3 后插入 MESC
# =============================================================================

def create_torchvision_resnet50() -> nn.Module:
    """尽量加载 ImageNet 预训练权重；如果本地没有权重，就退回随机初始化。"""

    try:
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception as exc:
        print(f"[WARN] 无法加载 ResNet50 预训练权重，将使用随机初始化。原因: {exc}")
        try:
            return models.resnet50(weights=None)
        except TypeError:
            return models.resnet50(pretrained=False)


class ResNet50AblationModel(nn.Module):
    """把 torchvision 的 ResNet50 拆开，唯一可变点是 layer3 后是否插入 MESC。"""

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
        # layer3 的输出通道是 1024；不用 MESC 时用 Identity 保持 forward 路径一致。
        self.mesc = MESC(channels=1024) if use_mesc else nn.Identity()
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 消融实验的结构变量只发生在这里：MESC 或 Identity。
        x = self.mesc(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    """给新加的 fc/MESC 较大学习率，给预训练 backbone 较小学习率。"""

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc") or name.startswith("mesc"):
            # fc 和 MESC 是新训练/新插入部分，需要更快适应当前医学数据集。
            head_params.append(param)
        else:
            # backbone 已有 ImageNet 表征，用较小学习率微调，减少破坏预训练特征。
            backbone_params.append(param)

    return optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def build_criterion(use_dast: bool, device: torch.device) -> nn.Module:
    """根据消融配置选择 CE 或 DAST；这是 loss 维度的唯一变量。"""

    if use_dast:
        return DistanceAwareSoftTargetLoss(NUM_CLASSES, tau=DAST_TAU, gamma=DAST_GAMMA).to(device)
    return nn.CrossEntropyLoss().to(device)


# =============================================================================
# 7. 训练和评估循环
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

        logits = model(images)              # 前向传播：得到每个类别的未归一化分数。
        loss = criterion(logits, labels)    # 根据当前配置计算 CE 或 DAST loss。

        optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度，set_to_none=True 更省显存。
        loss.backward()                        # 反向传播：计算所有可训练参数的梯度。
        optimizer.step()                       # 参数更新：把梯度真正应用到模型权重上。
        scheduler.step()                       # OneCycleLR 按 batch 更新学习率。

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

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
        preds = logits.argmax(dim=1)  # 取 logit 最大的类别作为最终预测。

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
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
    """早停时优先使用 QWK；如果出现 NaN，就退回 macro-F1 避免训练无法选模型。"""

    value = float(metrics.get(MONITOR_METRIC, np.nan))
    if np.isnan(value):
        return float(metrics["macro_f1"])
    return value


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 8. 单次实验：给定 seed 和方法，训练 -> 选最好模型 -> test
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

    # 每个 seed 都重新创建 dataloader，保证数据增强 shuffle 与 seed 绑定。
    train_loader, val_loader, test_loader = make_dataloaders(seed)

    # 模型、loss、优化器都在单次实验内部重新创建，避免不同方法之间共享状态。
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
    logger.log(f"Checkpoint: {ckpt_path}")

    best_score = -float("inf")  # 当前方法/seed 下验证集最优分数。
    best_epoch = 0              # 记录最优分数出现在哪个 epoch。
    bad_epochs = 0              # 连续未提升次数，用于 early stopping。

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        score = metric_to_monitor(val_metrics)
        # 只有超过 delta 的提升才算真正变好，避免极小浮动频繁覆盖 checkpoint。
        improved = score > best_score + EARLY_STOP_DELTA
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)  # 只保存最优权重，节省磁盘空间。
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

    # 测试集只评估验证集选出的最好模型，避免使用最后一个 epoch 的偶然状态。
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

    # summary.csv 每一行对应一次 seed+method，后续可以直接做均值/方差统计。
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
    """每完成一次实验就覆盖写一次 summary，长时间训练中断时也能保留已完成结果。"""

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 9. 主程序：5 个 seed x 4 个方法
# =============================================================================

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_DIR / "run.log")
    summary_rows: List[Dict[str, object]] = []
    try:
        logger.log(f"Dataset: {DATASET_NAME}")
        logger.log(f"Data root: {DATA_ROOT}")
        logger.log(f"Seeds: {list(SEEDS)}")
        logger.log(f"Experiments: {[name for name, _, _ in EXPERIMENTS]}")
        logger.log(f"Total runs: {len(SEEDS) * len(EXPERIMENTS)}")

        for seed in SEEDS:
            # 外层按 seed 跑，便于观察同一个随机种子下四组消融配置的差异。
            for method_name, use_mesc, use_dast in EXPERIMENTS:
                run_one_experiment(seed, method_name, use_mesc, use_dast, logger, summary_rows)
                save_summary(summary_rows)

        logger.log("")
        logger.log(f"All done. Summary saved to: {SUMMARY_CSV}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
