"""Fusion of paired left/right features at encoder readout levels."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import CrossAttention, feed_forward


class EarFusion(nn.Module):
    """Same-depth ears share adapters/interactions; ordered fusion preserves side."""

    def __init__(self, input_dim, dim, heads, drop):
        super().__init__()
        self.adapt = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.cross_norm = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, heads, drop)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = feed_forward(dim, drop)
        self.skip = nn.Linear(dim * 2, dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, left, right):
        left, right = (self.adapt(left), self.adapt(right))
        ln, rn = (self.cross_norm(left), self.cross_norm(right))
        left_x = left + self.cross(ln, rn)
        right_x = right + self.cross(rn, ln)
        left_x = left_x + self.ff(self.ff_norm(left_x))
        right_x = right_x + self.ff(self.ff_norm(right_x))
        return self.skip(torch.cat((left, right), -1)) + self.fuse(torch.cat((left_x, right_x), -1))
