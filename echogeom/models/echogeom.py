"""Compose the BV2 SSAST backbone, binaural modules, and depth decoder."""

from __future__ import annotations

import torch
from torch import nn

from ..config import InferenceConfig, validate_config
from .binaural_fusion import EarFusion
from .binaural_interaction import EncoderEarInteraction
from .depth_decoder import QueryPyramidDecoder
from .public_ssast import PublicSSAST


class EchoGeom(nn.Module):
    """BV2 paper architecture. Output is normalized depth; multiply by 30 for meters."""

    def __init__(self, config=None):
        super().__init__()
        config = validate_config(config or InferenceConfig())
        backbone = config.backbone
        binaural = config.binaural
        self.image_size = tuple(config.audio.image_size)
        self.backbone = PublicSSAST(backbone, config.audio)
        self.fusions = nn.ModuleList(
            [
                EarFusion(backbone.embed_dim, binaural.fusion_dim, binaural.fusion_heads, 0.1)
                for _ in backbone.feature_layers
            ]
        )
        self.decoder = QueryPyramidDecoder(config.decoder, binaural.fusion_dim)
        self.encoder_interactions = nn.ModuleDict(
            {
                str(layer): EncoderEarInteraction(
                    backbone.embed_dim, binaural.interaction_heads, 0.1, 0.0
                )
                for layer in binaural.interaction_layers
            }
        )

    def encode_memories(self, spectrogram):
        if spectrogram.ndim != 4 or tuple(spectrogram.shape[1:]) != (2, *self.image_size):
            raise ValueError(f"EchoGeom expects [B,2,{self.image_size[0]},{self.image_size[1]}]")
        batch_size = spectrogram.size(0)
        # Each ear passes through the same encoder: [B,2,H,W] -> [2B,1,H,W].
        ears = spectrogram.reshape(batch_size * 2, 1, *spectrogram.shape[-2:])

        def interact_after_block(index, tokens):
            layer = str(index)
            if layer not in self.encoder_interactions:
                return tokens
            interaction = self.encoder_interactions[layer]
            # Keep the two special tokens unchanged during cross-ear exchange.
            patch_tokens = tokens[:, 2:]
            token_count, channels = patch_tokens.shape[1:]
            paired = patch_tokens.reshape(batch_size, 2, token_count, channels)
            left, right = interaction(paired[:, 0], paired[:, 1])
            paired = torch.stack((left, right), dim=1).reshape(
                batch_size * 2, token_count, channels
            )
            return torch.cat((tokens[:, :2], paired), dim=1)

        hook = interact_after_block if self.encoder_interactions else None
        features = self.backbone.forward_intermediates(ears, after_block=hook)
        memories = []
        for fusion, feature in zip(self.fusions, features):
            paired = feature.reshape(batch_size, 2, feature.size(1), feature.size(2))
            memories.append(fusion(paired[:, 0], paired[:, 1]))
        return memories

    def forward(self, spectrogram):
        return self.decoder(self.encode_memories(spectrogram))
