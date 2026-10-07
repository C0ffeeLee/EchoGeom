"""Three-stage spatial-query depth decoder for the selected BV2 model."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from ..config import DecoderConfig
from .blocks import SpatialResidual, group_norm
from .spatial_query import SpatialReadStage


class QueryPyramidDecoder(nn.Module):
    def __init__(self, config=None, fusion_dim=256):
        config = config or DecoderConfig()
        super().__init__()
        self.output_size = config.output_size
        channels = config.channels
        sides = [config.output_size // 8, config.output_size // 4, config.output_size // 2]
        self.stages = nn.ModuleList(
            [
                SpatialReadStage(
                    sides[i],
                    channels[i],
                    fusion_dim,
                    channels[i - 1] if i else None,
                    config.attention_heads,
                    config.readout_depths[i],
                    0.1,
                )
                for i in range(3)
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, padding=1, bias=False),
            group_norm(channels[3]),
            nn.GELU(),
            SpatialResidual(channels[3]),
            SpatialResidual(channels[3]),
            nn.Conv2d(channels[3], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, memories):
        if len(memories) != 3:
            raise ValueError("QueryPyramidDecoder requires exactly 3 acoustic memories")
        x = None
        # Read the deepest acoustic features first, then refine with shallower ones.
        for stage, memory in zip(self.stages, reversed(memories)):
            x = stage(memory, x)
        x = F.interpolate(
            x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        return self.final(x)
