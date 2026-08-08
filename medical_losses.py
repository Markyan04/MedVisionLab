from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).float()


def _label_smoothing_one_hot(
    target: torch.Tensor,
    num_classes: int,
    smoothing: float,
) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must satisfy 0 <= smoothing < 1.")
    with torch.no_grad():
        true_dist = torch.zeros(target.size(0), num_classes, device=target.device)
        true_dist.fill_(smoothing / max(num_classes - 1, 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return true_dist


def _soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross entropy with uniform one-hot label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must satisfy 0 <= smoothing < 1.")
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        soft_targets = _label_smoothing_one_hot(
            target,
            logits.size(1),
            self.smoothing,
        )
        return _soft_cross_entropy(logits, soft_targets)


class OrdinalSoftCrossEntropyLoss(nn.Module):
    """Ordinal distance-decayed soft targets followed by soft CE."""

    def __init__(self, num_classes: int, tau: float = 1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        self.num_classes = num_classes
        self.tau = tau
        self.register_buffer(
            "class_ids",
            torch.arange(num_classes, dtype=torch.float),
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_f = target.float().unsqueeze(1)
        dist = torch.abs(self.class_ids.unsqueeze(0) - target_f)
        soft_targets = torch.exp(-dist / self.tau)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        return _soft_cross_entropy(logits, soft_targets)


class DistanceAwareSeverityLoss(nn.Module):
    """DASL: cross-entropy plus ordinal-distribution alignment.

    For each sample, the normalized expected ordinal deviation is

    ``D_ord = sum_k p_k * |k - y| / (K - 1)``.

    DASL minimizes ``-log(p_y) - log(1 - D_ord)``.  The implementation has
    no tunable hyperparameters; the clamp is only a numerical-stability guard.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.num_classes = num_classes
        self.register_buffer(
            "class_ids",
            torch.arange(num_classes, dtype=torch.float),
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise ValueError(
                f"DASL logits must have shape [B, {self.num_classes}], "
                f"got {tuple(logits.shape)}."
            )
        if target.ndim != 1 or target.size(0) != logits.size(0):
            raise ValueError("DASL target must have shape [B].")

        # Evaluate the probability terms in float32 under mixed precision so
        # the alignment score cannot underflow before the logarithm.
        working_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        log_probs = F.log_softmax(working_logits, dim=1)
        probabilities = log_probs.exp()
        target_long = target.long()

        true_log_probability = log_probs.gather(
            1,
            target_long.unsqueeze(1),
        ).squeeze(1)
        normalized_distance = torch.abs(
            self.class_ids.to(probabilities.dtype).unsqueeze(0)
            - target_long.to(probabilities.dtype).unsqueeze(1)
        ) / float(self.num_classes - 1)
        expected_ordinal_deviation = (
            probabilities * normalized_distance
        ).sum(dim=1)
        ordinal_alignment = (1.0 - expected_ordinal_deviation).clamp_min(0.0)

        per_sample = -true_log_probability - torch.log(ordinal_alignment + 1e-8)
        return per_sample.mean()


class KLModulatedOrdinalSoftTargetLoss(nn.Module):
    """Fixed ordinal soft targets with KL-based difficulty modulation.

    For a target grade ``y``, the soft target distribution is fixed as

    ``q_k = exp(-|k-y|) / sum_j exp(-|j-y|)``.

    Given ``p = softmax(logits)``, the per-sample loss is

    ``(1 - exp(-KL(q || p))) * CE(q, p)``.

    The distance temperature and modulation shape are fixed by the formula,
    so this loss exposes no tunable loss hyperparameters.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.num_classes = num_classes

        # Build the fixed table once in float64, then cast to the logits dtype
        # in forward.  This preserves reference-level accuracy in float64
        # tests without changing float32 or mixed-precision training behavior.
        class_ids = torch.arange(num_classes, dtype=torch.double)
        distances = torch.abs(class_ids.unsqueeze(0) - class_ids.unsqueeze(1))
        soft_target_table = torch.exp(-distances)
        soft_target_table = soft_target_table / soft_target_table.sum(
            dim=1,
            keepdim=True,
        )
        self.register_buffer("soft_target_table", soft_target_table)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(1) != self.num_classes:
            raise ValueError(
                f"KL-match logits must have shape [B, {self.num_classes}], "
                f"got {tuple(logits.shape)}."
            )
        if target.ndim != 1 or target.size(0) != logits.size(0):
            raise ValueError("KL-match target must have shape [B].")

        target_long = target.long()
        if torch.any(target_long < 0) or torch.any(target_long >= self.num_classes):
            raise ValueError("KL-match target contains an out-of-range class index.")

        # Keep probability-space calculations in float32 under mixed precision.
        working_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        log_probs = F.log_softmax(working_logits, dim=1)
        soft_targets = self.soft_target_table.to(log_probs.dtype)[target_long]
        log_soft_targets = torch.log(soft_targets)

        soft_target_ce = -(soft_targets * log_probs).sum(dim=1)
        kl_divergence = (
            soft_targets * (log_soft_targets - log_probs)
        ).sum(dim=1).clamp_min(0.0)

        # -expm1(-x) is the stable form of 1 - exp(-x) when KL is small.
        difficulty = -torch.expm1(-kl_divergence)
        return (difficulty * soft_target_ce).mean()


class CoralOrdinalLoss(nn.Module):
    """CORAL loss over the ``K - 1`` cumulative ordinal thresholds.

    For class ``y``, threshold ``k`` is positive when ``y > k``.  The model
    must therefore emit ``num_classes - 1`` logits per sample.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.num_classes = num_classes
        self.register_buffer(
            "threshold_ids",
            torch.arange(num_classes - 1, dtype=torch.long),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        expected = self.num_classes - 1
        if logits.ndim != 2 or logits.size(1) != expected:
            raise ValueError(f"CORAL logits must have shape [B, {expected}].")
        levels = (target.long().unsqueeze(1) > self.threshold_ids.unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, levels)


class CornOrdinalLoss(nn.Module):
    """CORN conditional ordinal loss over ``K - 1`` binary tasks.

    Threshold ``k`` is trained only on samples known to have reached that
    threshold (``y >= k``), matching the conditional training sets in CORN.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        expected = self.num_classes - 1
        if logits.ndim != 2 or logits.size(1) != expected:
            raise ValueError(f"CORN logits must have shape [B, {expected}].")

        target = target.long()
        task_losses = []
        eligible_count = 0
        for threshold in range(expected):
            eligible = target >= threshold
            if not torch.any(eligible):
                continue
            eligible_count += int(eligible.sum().item())
            binary_target = (target[eligible] > threshold).float()
            task_losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[eligible, threshold],
                    binary_target,
                    reduction="sum",
                )
            )
        if not task_losses:
            return logits.sum() * 0.0
        return torch.stack(task_losses).sum() / eligible_count


class ClassBalancedFocalCELoss(nn.Module):
    """Class-balanced focal CE with optional label smoothing."""

    def __init__(
        self,
        class_counts,
        beta: float = 0.9999,
        gamma: float = 2.0,
        smoothing: float = 0.0,
    ):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.smoothing > 0:
            soft_targets = _label_smoothing_one_hot(
                target,
                logits.size(1),
                self.smoothing,
            )
            log_probs = F.log_softmax(logits, dim=1)
            ce_per_sample = -(soft_targets * log_probs).sum(dim=1)
        else:
            ce_per_sample = F.cross_entropy(logits, target, reduction="none")

        sample_weights = self.class_weights[target]
        return (sample_weights * focal_factor * ce_per_sample).mean()


class OrdinalFocalMSELoss(nn.Module):
    """Focal CE plus an expected-grade regression penalty."""

    def __init__(
        self,
        num_classes: int,
        alpha_ce: float = 1.0,
        alpha_mse: float = 0.3,
        gamma: float = 2.0,
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_ce = alpha_ce
        self.alpha_mse = alpha_mse
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float),
            )
        else:
            self.class_weights = None
        self.register_buffer(
            "grade_values",
            torch.arange(num_classes, dtype=torch.float),
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.class_weights,
        )
        focal_ce = (focal_factor * ce).mean()

        pred_grade = (probs * self.grade_values.unsqueeze(0)).sum(dim=1)
        mse = F.mse_loss(pred_grade, target.float())
        return self.alpha_ce * focal_ce + self.alpha_mse * mse


class SymmetricCrossEntropyLoss(nn.Module):
    """Cross entropy plus reverse cross entropy."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        num_classes: int = 5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = _one_hot(target, self.num_classes).clamp(min=1e-4, max=1.0)
        reverse_ce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * reverse_ce


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized cross entropy for noisy-label robustness."""

    def __init__(self, q: float = 0.7):
        super().__init__()
        if not 0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        self.q = q

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-8, max=1.0)
        if abs(self.q - 1.0) < 1e-8:
            return (-torch.log(pt)).mean()
        return ((1.0 - pt.pow(self.q)) / self.q).mean()


class DistanceAwareSoftTargetLoss(nn.Module):
    """
    距离感知软目标损失（DAST）。

    根据真实类别与其他类别之间的序数距离构造软标签，
    距离越近，分配的目标概率越大。

    参数：
        num_classes: 类别数量
        tau: 距离衰减系数，越小则软标签越接近 one-hot
        gamma: Focal 调制系数，设为 0 时不使用 Focal 调制
    """

    def __init__(
        self,
        num_classes: int,
        tau: float = 1.0,
        gamma: float = 0.0
    ):
        super().__init__()

        if tau <= 0:
            raise ValueError("tau 必须大于 0")

        self.num_classes = num_classes
        self.tau = tau
        self.gamma = gamma

        # 类别编号，例如类别数为 5 时得到 [0, 1, 2, 3, 4]
        # buffer 不参与训练，但会随模型自动移动到 CPU 或 GPU
        self.register_buffer(
            "class_ids",
            torch.arange(num_classes, dtype=torch.float)
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            logits: 模型输出，形状为 [B, C]
            target: 真实标签，形状为 [B]

        返回：
            当前 batch 的平均损失
        """

        # 将标签形状从 [B] 转为 [B, 1]，便于广播计算
        target_f = target.float().unsqueeze(1)

        # 计算真实标签与各类别之间的序数距离，形状为 [B, C]
        dist = torch.abs(self.class_ids.unsqueeze(0) - target_f)

        # 根据距离构造指数衰减的软标签
        soft_targets = torch.exp(-dist / self.tau)

        # 归一化，使每个样本的软标签概率之和为 1
        soft_targets = soft_targets / soft_targets.sum(
            dim=1,
            keepdim=True
        )

        # 将模型输出转换为对数概率
        log_probs = F.log_softmax(logits, dim=1)

        # 恢复普通预测概率，用于计算 Focal 权重
        probs = log_probs.exp()

        # 计算每个样本的软标签交叉熵
        per_sample = -(soft_targets * log_probs).sum(dim=1)

        if self.gamma > 0:
            # 计算预测分布与软目标分布的加权匹配程度
            pt_soft = (probs * soft_targets).sum(dim=1)
            pt_soft = pt_soft.clamp(min=1e-8, max=1.0)

            # 降低简单样本权重，增强困难样本的影响
            focal_weight = (1.0 - pt_soft).pow(self.gamma)
            per_sample = focal_weight * per_sample

        # 返回 batch 平均损失
        return per_sample.mean()


class PrototypeConsistencyOrdinalLoss(nn.Module):
    """CE plus prototype consistency and ordinal prototype spacing."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        lambda_proto: float = 0.2,
        lambda_order: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_proto = lambda_proto
        self.lambda_order = lambda_order
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.02)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        proto_target = self.prototypes[target]
        proto_loss = F.mse_loss(features, proto_target)

        if self.num_classes > 1:
            gaps = self.prototypes[1:] - self.prototypes[:-1]
            order_loss = gaps.norm(dim=1).var()
        else:
            order_loss = torch.tensor(0.0, device=logits.device)

        return ce + self.lambda_proto * proto_loss + self.lambda_order * order_loss


class AdaptiveOrdinalMarginLoss(nn.Module):
    """Cross entropy with a class-distance-dependent logit margin."""

    def __init__(
        self,
        num_classes: int,
        margin_base: float = 0.15,
        power: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin_base = margin_base
        self.power = power
        self.register_buffer(
            "class_ids",
            torch.arange(num_classes, dtype=torch.float),
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _ = logits.shape
        target_ids = target.float().unsqueeze(1)
        dist = torch.abs(self.class_ids.unsqueeze(0) - target_ids)
        denominator = max((self.num_classes - 1) ** self.power, 1e-8)
        margins = self.margin_base * (dist.pow(self.power) / denominator)

        adjusted_logits = logits - margins
        row_ids = torch.arange(batch_size, device=logits.device)
        adjusted_logits[row_ids, target] = logits[row_ids, target]
        return F.cross_entropy(adjusted_logits, target)


def build_loss(name: str, **kwargs) -> nn.Module:
    """Construct one of the medical losses by its experiment name."""

    normalized = name.lower()
    if normalized in {"label_smoothing_ce", "ls_ce"}:
        return LabelSmoothingCrossEntropyLoss(**kwargs)
    if normalized in {"sord_ce", "sord"}:
        return OrdinalSoftCrossEntropyLoss(**kwargs)
    if normalized == "dasl":
        return DistanceAwareSeverityLoss(**kwargs)
    if normalized in {"kl_match_ce", "kl_match"}:
        return KLModulatedOrdinalSoftTargetLoss(**kwargs)
    if normalized == "coral":
        return CoralOrdinalLoss(**kwargs)
    if normalized == "corn":
        return CornOrdinalLoss(**kwargs)
    if normalized == "cb_focal_ce":
        return ClassBalancedFocalCELoss(**kwargs)
    if normalized == "ordinal_focal_mse":
        return OrdinalFocalMSELoss(**kwargs)
    if normalized == "sce":
        return SymmetricCrossEntropyLoss(**kwargs)
    if normalized == "gce":
        return GeneralizedCrossEntropyLoss(**kwargs)
    if normalized == "dast":
        return DistanceAwareSoftTargetLoss(**kwargs)
    if normalized == "pcol":
        return PrototypeConsistencyOrdinalLoss(**kwargs)
    if normalized == "aom":
        return AdaptiveOrdinalMarginLoss(**kwargs)
    raise ValueError(f"Unknown loss name: {name}")


if __name__ == "__main__":
    # 固定随机种子，保证测试结果可复现
    torch.manual_seed(42)

    # 8 个样本，5 个类别
    B, C = 8, 5

    # 模拟模型输出和真实标签
    logits = torch.randn(B, C)
    target = torch.randint(0, C, (B,))

    # 计算 DAST 损失
    criterion = DistanceAwareSoftTargetLoss(
        num_classes=C,
        tau=1.0,
        gamma=1.5
    )
    loss = criterion(logits, target)

    print("dast:", float(loss))
