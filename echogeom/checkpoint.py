"""Strict loading of sanitized inference weights."""

import hashlib
from pathlib import Path

import torch

from .config import InferenceConfig, validate_checkpoint_config, validate_config
from .models import EchoGeom

FORMAT = "echogeom_bv2_inference_v1"
ARCHITECTURE = "ssast_small_binaural_bi4_8_spatial_query_128"


def load_model(path, device="cpu", config=None):
    config = validate_config(config or InferenceConfig())
    validate_checkpoint_config(config)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference weights not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected an EchoGeom BV2 inference checkpoint")
    model = EchoGeom(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def checkpoint_sha256(path):
    """Hash the weight file without reading the whole checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
