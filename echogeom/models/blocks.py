"""Shared attention and residual blocks."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F


def feed_forward(dim, drop=0.0):
    return nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(dim * 2, dim),
        nn.Dropout(drop),
    )


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("attention width must be divisible by head count")
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))

    def forward(self, query, memory):
        b, n, d = query.shape
        q = self.q(query).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = (
            self.kv(memory)
            .reshape(b, memory.size(1), 2, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(b, n, d))


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SpatialResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            group_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.layers(x))
