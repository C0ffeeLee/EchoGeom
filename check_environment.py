"""Check runtime dependencies, WAV decoding, and the selected inference checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from echogeom.checkpoint import checkpoint_sha256, load_model
from echogeom.config import DEFAULT_CONFIG, cli_config, resolve_device, resolve_path
from echogeom.data import load_spectrogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument("--checkpoint", type=Path, help="Override paths.checkpoint")
    args = parser.parse_args()
    config = cli_config(args, parser, "check")
    device = resolve_device(config)
    checkpoint = resolve_path(config.paths.checkpoint)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA was requested but is unavailable; use --device cpu or check the CUDA wheel and driver"
        )
    if "soundfile" not in torchaudio.list_audio_backends():
        raise RuntimeError("SoundFile audio backend unavailable; install requirements.txt")

    # This round trip verifies WAV support without requiring dataset files.
    with tempfile.TemporaryDirectory(prefix="echogeom-wav-check-") as directory:
        audio = Path(directory) / "stereo.wav"
        sf.write(
            str(audio),
            np.zeros((config.audio.crop_samples, 2), dtype=np.float32),
            config.audio.sample_rate,
        )
        spectrogram = load_spectrogram(audio, config.audio)
    if (
        tuple(spectrogram.shape) != (2, *config.audio.image_size)
        or not torch.isfinite(spectrogram).all()
    ):
        raise RuntimeError("WAV preprocessing failed")

    model = load_model(checkpoint, device, config)
    with torch.inference_mode():
        prediction = model(spectrogram.unsqueeze(0).to(device))
    if (
        tuple(prediction.shape) != (1, 1, config.decoder.output_size, config.decoder.output_size)
        or not torch.isfinite(prediction).all()
    ):
        raise RuntimeError("Checkpoint inference failed")
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchaudio",
                "torchvision",
                "numpy",
                "pandas",
                "soundfile",
                "PyYAML",
            )
        },
        "device": str(device),
        "configuration": config.to_dict(),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "audio_backend": "soundfile",
        "wav_roundtrip": "passed",
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "output_shape": list(prediction.shape),
        "finite_prediction": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
