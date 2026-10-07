"""Coordinate queries and progressive cross-attention readout."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import CrossAttention, SpatialResidual, feed_forward


def coordinate_grid(side, dim):
    """Fourier features for image-plane query coordinates."""
    if dim % 4:
        raise ValueError("query channels must be divisible by four")
    freq = torch.logspace(0, math.log2(max(side / 2, 1)), dim // 4, base=2)
    phase = math.pi * torch.linspace(-1, 1, side)[:, None] * freq[None]
    axis = torch.cat((phase.sin(), phase.cos()), -1)
    y = axis[:, None].expand(side, side, -1)
    x = axis[None, :].expand(side, side, -1)
    return torch.cat((y, x), -1).reshape(1, side * side, dim)


class ReadoutBlock(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, heads, drop)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = feed_forward(dim, drop)

    def forward(self, q, memory):
        q = q + self.cross(self.q_norm(q), self.m_norm(memory))
        return q + self.ff(self.ff_norm(q))


class SpatialReadStage(nn.Module):
    def __init__(self, side, dim, memory_dim, previous_dim, heads, depth, drop):
        super().__init__()
        self.side, self.dim = (side, dim)
        query_features = coordinate_grid(side, dim)
        # The coordinate grid is deterministic and moves with the model.
        self.register_buffer("coordinates", query_features, persistent=False)
        self.position = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory_proj = nn.Linear(memory_dim, dim)
        self.previous_proj = nn.Conv2d(previous_dim, dim, 1) if previous_dim else None
        self.blocks = nn.ModuleList([ReadoutBlock(dim, heads, drop) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim)
        self.refine = nn.Sequential(SpatialResidual(dim), SpatialResidual(dim))

    def forward(self, memory, previous=None):
        q = self.position(self.coordinates.to(dtype=memory.dtype)).expand(memory.size(0), -1, -1)
        if self.previous_proj is not None:
            if previous is None:
                raise ValueError("Refinement stage needs previous spatial features")
            previous = F.interpolate(
                previous, size=(self.side, self.side), mode="bilinear", align_corners=False
            )
            # Carry the coarser spatial estimate into this finer query grid.
            q = q + self.previous_proj(previous).flatten(2).transpose(1, 2)
        memory = self.memory_proj(memory)
        for block in self.blocks:
            q = block(q, memory)
        grid = (
            self.output_norm(q)
            .transpose(1, 2)
            .reshape(memory.size(0), self.dim, self.side, self.side)
        )
        grid = self.refine(grid)
        return grid
