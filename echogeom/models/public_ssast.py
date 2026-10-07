# Adapted SSAST/DeiT inference. See README.md and licenses/.
# Upstream notices: Copyright 2019 Ross Wightman; Copyright (c) 2022 Yuan Gong.
# Modified for fixed BV2 inference, SDPA attention, and no training initialization.
"""Shared SSAST-Small encoder for binaural spectrograms."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import AudioConfig, BackboneConfig


class PublicMLP(nn.Module):
    def __init__(self, dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PublicAttention(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float) -> None:
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, d // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(b, n, d)))


class PublicBlock(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.attn = PublicAttention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = PublicMLP(dim, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PublicSSAST(nn.Module):
    """Shared SSAST-Small backbone, without external checkpoint downloads."""

    def __init__(self, config=None, audio=None):
        super().__init__()
        config = config or BackboneConfig()
        audio = audio or AudioConfig()
        dim = config.embed_dim
        self.embed_dim, self.num_heads = dim, config.attention_heads
        self.image_size = tuple(audio.image_size)
        self.feature_layers = tuple(config.feature_layers)
        # Preserve the pretrained patch_embed.proj parameter names.
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(1, dim, config.patch_size, stride=config.patch_stride)
        token_count = math.prod(
            (side - config.patch_size) // config.patch_stride + 1 for side in self.image_size
        )
        # SSAST inherits both the class and distillation tokens from DeiT.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count + 2, dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [PublicBlock(dim, config.attention_heads, 0.0) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward_intermediates(self, x, after_block=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, *self.image_size):
            raise ValueError(f"Expected single-ear [B,1,{self.image_size[0]},{self.image_size[1]}]")
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat(
            (
                self.cls_token.expand(x.size(0), -1, -1),
                self.dist_token.expand(x.size(0), -1, -1),
                x,
            ),
            dim=1,
        )
        x = self.pos_drop(x + self.pos_embed)
        outputs = []
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if after_block is not None:
                x = after_block(index, x)
            # Only patch tokens become acoustic memory for the decoder.
            if index in self.feature_layers:
                outputs.append((self.norm(x) if index == len(self.blocks) else x)[:, 2:])
        return outputs
