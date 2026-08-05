"""MECS VersionB: median-anchored dynamic routing of three statistics.

This module is independent from ``MECS_old.py``.  It reuses the unchanged
spatial-attention path of MECS VersionA, while replacing only channel-attention
fusion with the avg/max/median router supplied for the VersionB experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from MECS_old import MECS_VersionA


def global_median_pooling(x: torch.Tensor) -> torch.Tensor:
    """Median-pool each channel over spatial positions to ``[B, C, 1, 1]``."""
    median_pooled = torch.median(x.flatten(start_dim=2), dim=2).values
    return median_pooled.view(x.size(0), x.size(1), 1, 1)


class ChannelAttention_VersionB(nn.Module):
    """Dynamically fuse avg/max/median branch logits per sample and channel.

    The shared MLP uses LeakyReLU.  A 12-parameter router receives
    ``[avg-med, max-med, med]`` for each channel and produces three softmax
    weights.  Zero initialization makes the initial fusion exactly equivalent
    to the equal-weight mean used by VersionA.
    """

    def __init__(
        self,
        input_channels: int,
        internal_neurons: int,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            input_channels,
            internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            internal_neurons,
            input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.branch_act = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=False,
        )

        # Shared across samples and channels: 3*3 weights + 3 biases.
        self.router = nn.Linear(in_features=3, out_features=3, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def _forward_branch(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(pooled_feature)
        hidden = self.branch_act(hidden)
        return self.fc2(hidden)

    def _build_router_features(
        self,
        avg_vec: torch.Tensor,
        max_vec: torch.Tensor,
        med_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Build the median-anchored input to the shared 3-to-3 router."""
        return torch.stack(
            (
                avg_vec - med_vec,
                max_vec - med_vec,
                med_vec,
            ),
            dim=-1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        return_branch_attentions: bool = False,
    ):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        avg_logit = self._forward_branch(avg_pool)
        max_logit = self._forward_branch(max_pool)
        median_logit = self._forward_branch(median_pool)

        avg_vec = avg_logit.flatten(1)
        max_vec = max_logit.flatten(1)
        med_vec = median_logit.flatten(1)
        branch_logits = torch.stack((avg_vec, max_vec, med_vec), dim=-1)

        router_features = self._build_router_features(avg_vec, max_vec, med_vec)
        routing_logits = self.router(router_features)
        routing_weights = torch.softmax(routing_logits, dim=-1)

        fused_vec = (routing_weights * branch_logits).sum(dim=-1)
        fused_logit = fused_vec.unsqueeze(-1).unsqueeze(-1)
        fused_attention = torch.sigmoid(fused_logit)

        if not return_branch_attentions:
            return fused_attention

        return fused_attention, {
            "avg_att": torch.sigmoid(avg_logit),
            "max_att": torch.sigmoid(max_logit),
            "med_att": torch.sigmoid(median_logit),
            "avg_logit": avg_logit,
            "max_logit": max_logit,
            "med_logit": median_logit,
            "fused_logit": fused_logit,
            "fused_att": fused_attention,
            "avg_route": routing_weights[..., 0].unsqueeze(-1).unsqueeze(-1),
            "max_route": routing_weights[..., 1].unsqueeze(-1).unsqueeze(-1),
            "med_route": routing_weights[..., 2].unsqueeze(-1).unsqueeze(-1),
            "routing_weights": routing_weights,
            "routing_logits": routing_logits,
        }


class MECS_VersionB(MECS_VersionA):
    """Full MECS block using VersionB channel attention and VersionA spatial path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention_reduce: int = 4,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attention_reduce=channel_attention_reduce,
        )
        self.channel_attention = ChannelAttention_VersionB(
            input_channels=in_channels,
            internal_neurons=max(1, in_channels // channel_attention_reduce),
            negative_slope=negative_slope,
        )


__all__ = [
    "global_median_pooling",
    "ChannelAttention_VersionB",
    "MECS_VersionB",
]
