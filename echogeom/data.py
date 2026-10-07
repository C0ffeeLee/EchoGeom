"""BatVision V2 evaluation data and the paper's inference preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from .config import AudioConfig


def load_spectrogram(audio_path, config=None):
    config = config or AudioConfig()
    waveform, sample_rate = torchaudio.load(str(audio_path), backend="soundfile")
    if sample_rate != config.sample_rate:
        raise ValueError(f"Expected {config.sample_rate} Hz audio, got {sample_rate}")
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError("Expected stereo audio ordered as left, right")
    if waveform.shape[1] < config.crop_samples:
        raise ValueError(f"Expected at least {config.crop_samples} audio samples")
    waveform = waveform[:, : config.crop_samples]
    spectrum = torchaudio.transforms.Spectrogram(
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        power=1.0 if config.spectrum == "magnitude" else 2.0,
    )(waveform)
    return resize(
        spectrum, config.image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
    )


def load_depth(depth_path, output_size=128):
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError("Expected a 2D BV2 depth array in millimeters")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    # Convert BV2 millimeters to meters, then normalize by the fixed 30 m range.
    depth = np.clip(depth * 1e-3, 0.0, 30.0)
    target = torch.from_numpy(depth.copy()).unsqueeze(0)
    return (
        resize(
            target,
            [output_size, output_size],
            interpolation=InterpolationMode.NEAREST,
            antialias=None,
        )
        / 30.0
    )


class BatVisionV2(Dataset):
    """Read official per-location test.csv or val.csv without augmentation."""

    def __init__(self, root, split="test", audio_config=None, output_size=128):
        self.audio_config = audio_config or AudioConfig()
        self.output_size = output_size
        self.root = Path(root)
        if split not in ("val", "test"):
            raise ValueError("This release supports val and test evaluation only")
        if not self.root.is_dir():
            raise FileNotFoundError(f"BV2 data directory does not exist: {self.root}")
        # Ignore incidental directories; require manifests in audio/depth locations.
        locations = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and ((path / "audio").is_dir() or (path / "depth").is_dir())
        )
        if not locations:
            raise FileNotFoundError("No BV2 location directories found")
        csv_paths = [path / f"{split}.csv" for path in locations]
        missing = [str(path) for path in csv_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing BV2 annotations: {missing}")
        self.rows = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
        required = {"audio path", "audio file name", "depth path", "depth file name"}
        if not required.issubset(self.rows.columns):
            raise ValueError(f"BV2 manifests require columns: {sorted(required)}")
        if self.rows.empty:
            raise ValueError("The selected split is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        audio = self.root / row["audio path"] / row["audio file name"]
        depth = self.root / row["depth path"] / row["depth file name"]
        return {"spectrogram": load_spectrogram(audio, self.audio_config)}, load_depth(
            depth, self.output_size
        )
