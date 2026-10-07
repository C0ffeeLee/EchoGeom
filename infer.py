"""Predict a 128x128 depth map in meters from one BV2 stereo WAV."""

import argparse
from pathlib import Path

import numpy as np
import torch
from echogeom.checkpoint import load_model
from echogeom.config import DEFAULT_CONFIG, cli_config, resolve_device, resolve_path
from echogeom.data import load_spectrogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audio", type=Path, help="Override paths.audio")
    parser.add_argument("--checkpoint", type=Path, help="Override paths.checkpoint")
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument("--output", type=Path, help="Override paths.inference_output")
    args = parser.parse_args()
    config = cli_config(args, parser, "inference")
    device = resolve_device(config)
    model = load_model(resolve_path(config.paths.checkpoint), device, config)
    spectrum = (
        load_spectrogram(resolve_path(config.paths.audio), config.audio).unsqueeze(0).to(device)
    )
    with torch.inference_mode():
        # Undo the fixed 30 m depth normalization.
        depth = model(spectrum)[0, 0] * 30.0
    if not torch.isfinite(depth).all():
        raise RuntimeError("The model produced non-finite predictions")
    output = resolve_path(config.paths.inference_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.save(handle, depth.cpu().numpy())
    print(f"Saved depth in meters: {output}")


if __name__ == "__main__":
    main()
