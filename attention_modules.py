#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reusable layer3 attention modules for controlled ResNet50 comparisons."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from MECS_old import MECS_VersionA, global_median_pooling


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        padding = spatial_kernel // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class MSCABlock(nn.Module):
    """SegNeXt-style multi-scale convolutional spatial attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv0_1 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.conv0_2 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.proj_attn = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.act(self.proj_in(x))
        attn = self.conv0(x)
        attn = (
            attn
            + self.conv0_2(self.conv0_1(attn))
            + self.conv1_2(self.conv1_1(attn))
            + self.conv2_2(self.conv2_1(attn))
        )
        attn = self.proj_attn(attn)
        x = self.proj_out(x * attn)
        return x + shortcut


class MESCVariant(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_stats: Sequence[str] = ("avg", "max", "median"),
        use_channel: bool = True,
        use_spatial: bool = True,
        reduction: int = 4,
    ):
        super().__init__()
        self.channel_stats = tuple(channel_stats)
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.initial_depth_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels),
        ])
        self.spatial_att_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def _descriptor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "avg":
            return F.adaptive_avg_pool2d(x, output_size=(1, 1))
        if name == "max":
            return F.adaptive_max_pool2d(x, output_size=(1, 1))
        if name == "median":
            return global_median_pooling(x)
        raise ValueError(f"Unknown MESC statistic: {name}")

    def _channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for stat in self.channel_stats:
            z = self._descriptor(x, stat)
            outs.append(torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=True))))
        if not outs:
            return torch.ones_like(x[:, :, :1, :1])
        return sum(outs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.pre_conv(inputs))
        x_ca = self._channel_attention(x) * x if self.use_channel else x
        if self.use_spatial:
            initial_out = self.initial_depth_conv(x_ca)
            spatial_out = sum(conv(initial_out) for conv in self.depth_convs) + x_ca
            spatial_att = torch.sigmoid(self.spatial_att_conv(spatial_out))
            out = spatial_att * x_ca
        else:
            out = x_ca
        return self.post_conv(out)


def build_attention_module(name: str, channels: int = 1024) -> nn.Module:
    key = name.strip().lower().replace("-", "_")
    if key == "se":
        return SEBlock(channels)
    if key == "cbam":
        return CBAMBlock(channels)
    if key == "eca":
        return ECABlock(channels)
    if key == "msca":
        return MSCABlock(channels)
    if key in {"mesc", "full"}:
        return MECS_VersionA(channels, channels)
    if key == "avg_only":
        return MESCVariant(channels, channel_stats=("avg",), use_channel=True, use_spatial=True)
    if key == "avg_max":
        return MESCVariant(channels, channel_stats=("avg", "max"), use_channel=True, use_spatial=True)
    if key in {"avg_max_median", "full_mesc"}:
        return MESCVariant(channels, channel_stats=("avg", "max", "median"), use_channel=True, use_spatial=True)
    if key == "spatial_only":
        return MESCVariant(channels, channel_stats=(), use_channel=False, use_spatial=True)
    if key == "channel_only":
        return MESCVariant(channels, channel_stats=("avg", "max", "median"), use_channel=True, use_spatial=False)
    raise ValueError(f"Unknown attention module: {name}")
