"""MECS raw-logit dynamic routing ablation.

This is the strict control for ``MECS_VersionB``.  Both variants use the same
avg/max/median branch logits, shared ``Linear(3, 3)`` router, zero
initialization, softmax weights, weighted fusion, and VersionA spatial path.
The only difference is the router input:

* Raw routing: ``[avg, max, median]``
* VersionB: ``[avg - median, max - median, median]``
"""

from __future__ import annotations

import torch

from MECS_old import MECS_VersionA
from MECS_VersionB import ChannelAttention_VersionB, global_median_pooling


class ChannelAttention_RawRouting(ChannelAttention_VersionB):
    """Use uncentered branch logits as the dynamic router input."""

    def _build_router_features(
        self,
        avg_vec: torch.Tensor,
        max_vec: torch.Tensor,
        med_vec: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack((avg_vec, max_vec, med_vec), dim=-1)


class MECS_RawRouting(MECS_VersionA):
    """Full MECS block with raw-logit dynamic channel routing."""

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
        self.channel_attention = ChannelAttention_RawRouting(
            input_channels=in_channels,
            internal_neurons=max(1, in_channels // channel_attention_reduce),
            negative_slope=negative_slope,
        )


__all__ = [
    "global_median_pooling",
    "ChannelAttention_RawRouting",
    "MECS_RawRouting",
]
