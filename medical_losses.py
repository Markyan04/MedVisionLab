import torch
import torch.nn as nn
import torch.nn.functional as F


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