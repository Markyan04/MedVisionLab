import torch
import torch.nn as nn
import torch.nn.functional as F


def global_median_pooling(x):
    """将每个通道的空间位置展平后取中位数，并恢复成 [B, C, 1, 1]。"""
    median_pooled = torch.median(
        x.flatten(start_dim=2),
        dim=2
    ).values
    median_pooled = median_pooled.view(
        x.size(0),
        x.size(1),
        1,
        1
    )
    return median_pooled


class ChannelAttention_VersionA(nn.Module):
    """
    使用 avg/max/median 三种全局统计量生成通道权重。

    三个分支经过共享 MLP 后先得到未归一化的 logits，
    再对三路 logits 求平均，最后统一执行一次 sigmoid。
    """

    def __init__(
        self,
        input_channels,
        internal_neurons,
        negative_slope=0.1
    ):
        super(ChannelAttention_VersionA, self).__init__()

        self.fc1 = nn.Conv2d(
            input_channels,
            internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True
        )

        self.fc2 = nn.Conv2d(
            internal_neurons,
            input_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )

        # 使用 LeakyReLU 保留负半轴的信息和梯度，
        # 避免普通 ReLU 将三个分支的隐藏特征全部清零。
        self.branch_act = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=False
        )

    def _forward_branch(self, pooled_feature):
        """
        将一种池化描述符送入共享 MLP。

        返回值是 sigmoid 之前的通道 logits，
        三个分支将在 logits 空间完成融合。
        """
        hidden = self.fc1(pooled_feature)
        hidden = self.branch_act(hidden)
        branch_logit = self.fc2(hidden)
        return branch_logit

    def forward(self, inputs, return_branch_attentions=False):
        # 使用三种全局统计量描述每个通道：
        # 平均响应、最强响应和中位响应。
        avg_pool = F.adaptive_avg_pool2d(
            inputs,
            output_size=(1, 1)
        )
        max_pool = F.adaptive_max_pool2d(
            inputs,
            output_size=(1, 1)
        )
        median_pool = global_median_pooling(inputs)

        # 平均池化分支更关注整个通道的整体响应强度。
        avg_logit = self._forward_branch(avg_pool)

        # 最大池化分支强调最显著的局部激活。
        max_logit = self._forward_branch(max_pool)

        # 中位数池化分支对极端激活更稳健，
        # 可以补充平均值和最大值统计。
        median_logit = self._forward_branch(median_pool)

        # 三路先在 sigmoid 之前求平均。
        # 求平均可以避免直接相加导致数值尺度扩大三倍。
        fused_logit = (
            avg_logit
            + max_logit
            + median_logit
        ) / 3.0

        # 只在融合完成后统一执行一次 sigmoid，
        # 得到最终的通道注意力权重。
        out = torch.sigmoid(fused_logit)

        if return_branch_attentions:
            # 单独的分支 sigmoid 结果仅用于诊断和可视化，
            # 实际前向计算只使用融合后的 out。
            return out, {
                "avg_att": torch.sigmoid(avg_logit),
                "max_att": torch.sigmoid(max_logit),
                "med_att": torch.sigmoid(median_logit),
                "avg_logit": avg_logit,
                "max_logit": max_logit,
                "med_logit": median_logit,
                "fused_logit": fused_logit,
                "fused_att": out,
            }

        return out


class MECS_VersionA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channel_attention_reduce=4
    ):
        super(MECS_VersionA, self).__init__()

        assert in_channels == out_channels, (
            "Input and output channels must be the same"
        )

        self.channel_attention = ChannelAttention_VersionA(
            input_channels=in_channels,
            internal_neurons=max(
                1,
                in_channels // channel_attention_reduce
            )
        )

        # groups=in_channels 表示 depthwise conv，
        # 即每个通道分别进行空间卷积。
        self.initial_depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels
        )

        # 这 6 个分支使用不同方向和尺度的条形卷积，
        # 用于捕获多尺度空间上下文。
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                groups=in_channels
            ),
        ])

        # 三个 1×1 卷积分别负责：
        # 输入预处理、空间权重生成和输出映射。
        self.pre_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            padding=0
        )
        self.spatial_att_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            padding=0
        )
        self.post_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            padding=0
        )

        self.act = nn.GELU()

    def forward(
        self,
        inputs,
        return_branch_attentions=False
    ):
        # 先使用 1×1 Conv 进行通道重映射，
        # 再通过 GELU 增加非线性表达能力。
        x = self.pre_conv(inputs)
        x = self.act(x)

        # 根据预处理后的特征生成通道权重，
        # 再将权重逐元素乘回特征图。
        if return_branch_attentions:
            channel_att_vec, branch_attentions = (
                self.channel_attention(
                    x,
                    return_branch_attentions=True
                )
            )
        else:
            channel_att_vec = self.channel_attention(x)

        x_ca = channel_att_vec * x

        # 先使用 5×5 depthwise convolution
        # 进行局部空间信息建模。
        initial_out = self.initial_depth_conv(x_ca)

        # 并行经过 6 个大核 depthwise 分支，
        # 提取不同方向、不同感受野下的空间上下文。
        spatial_outs = [
            conv(initial_out)
            for conv in self.depth_convs
        ]

        # 将所有多尺度分支输出相加，
        # 融合横向、纵向及不同尺度的信息。
        spatial_out = sum(spatial_outs)

        # 将通道增强特征残差加回来，
        # 避免空间分支完全丢失原始响应。
        spatial_out = spatial_out + x_ca

        # 使用 1×1 Conv 和 sigmoid 生成空间注意力图，
        # 再对通道增强后的特征进行逐元素加权。
        spatial_att = torch.sigmoid(
            self.spatial_att_conv(spatial_out)
        )
        out = spatial_att * x_ca

        # 最后使用一次 1×1 Conv 完成输出映射。
        out = self.post_conv(out)

        if return_branch_attentions:
            return out, branch_attentions

        return out