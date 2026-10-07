"""Intermediate cross-ear interaction after encoder blocks 4 and 8."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import CrossAttention


class EncoderEarInteraction(nn.Module):
    """Symmetric cross-ear exchange inserted between pretrained SSAST blocks."""

    def __init__(self, dim, heads, drop, initial_scale):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, heads, drop)
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError("Encoder ear interaction expects matching [B,N,D] tokens")
        left_norm, right_norm = (self.norm(left), self.norm(right))
        left_delta = self.cross(left_norm, right_norm)
        right_delta = self.cross(right_norm, left_norm)
        # Bound the learned residual strength while retaining its sign.
        scale = self.residual_scale.tanh()
        return (left + scale * left_delta, right + scale * right_delta)
