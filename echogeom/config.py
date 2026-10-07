"""Load inference options from YAML and apply command-line overrides."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RELEASE_ROOT / "config.yaml"


@dataclass
class PathsConfig:
    data_root: str | None = None
    audio: str | None = None
    checkpoint: str = "weights/echogeom_bv2.pt"
    evaluation_output: str = "results/bv2_test.json"
    inference_output: str = "results/depth.npy"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 0
    split: str = "test"


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    crop_samples: int = 7782
    n_fft: int = 512
    win_length: int = 64
    hop_length: int = 16
    spectrum: str = "magnitude"
    image_size: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class BackboneConfig:
    name: str = "ssast_small"
    embed_dim: int = 384
    num_layers: int = 12
    attention_heads: int = 6
    patch_size: int = 16
    patch_stride: int = 8
    feature_layers: list[int] = field(default_factory=lambda: [4, 8, 12])


@dataclass
class BinauralConfig:
    interaction_layers: list[int] = field(default_factory=lambda: [4, 8])
    interaction_heads: int = 6
    fusion_dim: int = 256
    fusion_heads: int = 8


@dataclass
class DecoderConfig:
    channels: list[int] = field(default_factory=lambda: [256, 192, 96, 48])
    readout_depths: list[int] = field(default_factory=lambda: [2, 1, 1])
    attention_heads: int = 8
    output_size: int = 128


@dataclass
class NormalsConfig:
    enabled: bool = True
    hfov_deg: float = 86.0
    vfov_deg: float = 57.0
    patch_size: int = 5
    min_points: int = 6
    eval_mask: str = "valid_neighborhood"


@dataclass
class InferenceConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    binaural: BinauralConfig = field(default_factory=BinauralConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    normals: NormalsConfig = field(default_factory=NormalsConfig)

    def to_dict(self):
        return asdict(self)


class _ConfigLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    mapping = {}
    # Duplicate keys otherwise overwrite earlier values without a warning.
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        if key in mapping:
            raise ValueError(f"Duplicate configuration key: {key}")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _integer_list(value, name, length=None, allow_empty=False):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise ValueError(f"{name} must be a list" + (f" of length {length}" if length else ""))
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    for item in value:
        _positive(item, name)


def validate_config(config):
    runtime = config.runtime
    audio = config.audio
    backbone = config.backbone
    binaural = config.binaural
    decoder = config.decoder
    normals = config.normals
    for item in fields(config.paths):
        value = getattr(config.paths, item.name)
        nullable = item.name in ("data_root", "audio")
        if not (nullable and value is None) and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"paths.{item.name} must be a nonempty path string"
                + (" or null" if nullable else "")
            )
    if not isinstance(runtime.device, str):
        raise ValueError("runtime.device must be a string")
    if runtime.device != "auto":
        try:
            device = torch.device(runtime.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid runtime.device: {runtime.device}") from error
        if device.type not in ("cpu", "cuda"):
            raise ValueError("runtime.device supports auto, cpu, cuda, and cuda:N")
    _positive(runtime.batch_size, "runtime.batch_size")
    _positive(runtime.num_workers, "runtime.num_workers", 0)
    if runtime.split not in ("test", "val"):
        raise ValueError("runtime.split must be test or val")
    for name in ("sample_rate", "crop_samples", "n_fft", "win_length", "hop_length"):
        _positive(getattr(audio, name), f"audio.{name}")
    if audio.win_length > audio.n_fft or audio.crop_samples <= audio.n_fft // 2:
        raise ValueError("Audio requires win_length <= n_fft and crop_samples > n_fft / 2")
    if audio.spectrum not in ("magnitude", "power"):
        raise ValueError("audio.spectrum must be magnitude or power")
    _integer_list(audio.image_size, "audio.image_size", 2)
    if backbone.name != "ssast_small":
        raise ValueError("Only the ssast_small backbone is included in this release")
    for name in ("embed_dim", "num_layers", "attention_heads", "patch_size", "patch_stride"):
        _positive(getattr(backbone, name), f"backbone.{name}")
    _integer_list(backbone.feature_layers, "backbone.feature_layers", 3)
    if (
        backbone.feature_layers != sorted(set(backbone.feature_layers))
        or backbone.feature_layers[-1] > backbone.num_layers
    ):
        raise ValueError(
            "backbone.feature_layers must contain 3 increasing distinct layers within num_layers"
        )
    if min(audio.image_size) < backbone.patch_size:
        raise ValueError("audio.image_size must be at least backbone.patch_size")
    _integer_list(binaural.interaction_layers, "binaural.interaction_layers", allow_empty=True)
    if binaural.interaction_layers != sorted(set(binaural.interaction_layers)) or any(
        layer > backbone.num_layers for layer in binaural.interaction_layers
    ):
        raise ValueError(
            "binaural.interaction_layers must be increasing distinct layers within num_layers"
        )
    for name in ("interaction_heads", "fusion_dim", "fusion_heads"):
        _positive(getattr(binaural, name), f"binaural.{name}")
    _integer_list(decoder.channels, "decoder.channels", 4)
    _integer_list(decoder.readout_depths, "decoder.readout_depths", 3)
    _positive(decoder.attention_heads, "decoder.attention_heads")
    _positive(decoder.output_size, "decoder.output_size", 8)
    if decoder.output_size % 8:
        raise ValueError("decoder.output_size must be divisible by 8")
    if (
        backbone.embed_dim % backbone.attention_heads
        or backbone.embed_dim % binaural.interaction_heads
    ):
        raise ValueError(
            "backbone.embed_dim must be divisible by backbone and interaction attention head counts"
        )
    if binaural.fusion_dim % binaural.fusion_heads or any(
        c % decoder.attention_heads or c % 4 for c in decoder.channels[:3]
    ):
        raise ValueError(
            "Fusion and decoder widths must be divisible by their head counts; query widths must also be divisible by 4"
        )
    if type(normals.enabled) is not bool:
        raise ValueError("normals.enabled must be true or false")
    for name in ("hfov_deg", "vfov_deg"):
        value = getattr(normals, name)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value < 180:
            raise ValueError(f"normals.{name} must be a finite number between 0 and 180")
    _positive(normals.patch_size, "normals.patch_size", 3)
    _positive(normals.min_points, "normals.min_points", 3)
    if normals.patch_size % 2 == 0 or normals.min_points > normals.patch_size**2:
        raise ValueError("Normal PCA needs an odd patch_size and min_points <= patch_size squared")
    if normals.eval_mask not in ("valid_neighborhood", "valid", "all"):
        raise ValueError("normals.eval_mask must be valid_neighborhood, valid, or all")
    return config


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.load(handle, Loader=_ConfigLoader)
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = InferenceConfig()
    unknown = set(document) - {item.name for item in fields(config)}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for name, values in document.items():
        section = getattr(config, name)
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a YAML mapping")
        unknown = set(values) - {item.name for item in fields(section)}
        if unknown:
            raise ValueError(f"Unknown configuration keys in {name}: {sorted(unknown)}")
        for key, value in values.items():
            setattr(section, key, value)
    return validate_config(config)


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else RELEASE_ROOT / path


def resolve_device(config):
    value = config.runtime.device
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )


def cli_config(args, parser, mode):
    try:
        config = load_config(args.config)
        for name in ("device", "batch_size", "num_workers", "split"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(config.runtime, name, value)
        # CLI paths use the caller's working directory; YAML paths use RELEASE_ROOT.
        for name in ("checkpoint", "data_root", "audio"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(config.paths, name, str(value.expanduser().resolve()))
        if getattr(args, "output", None) is not None:
            key = "evaluation_output" if mode == "evaluation" else "inference_output"
            setattr(config.paths, key, str(args.output.expanduser().resolve()))
        if getattr(args, "no_normals", False):
            config.normals.enabled = False
        validate_config(config)
        required = {"evaluation": "data_root", "inference": "audio"}.get(mode)
        if required and getattr(config.paths, required) is None:
            parser.error(f"Set paths.{required} in YAML or pass --{required.replace('_', '-')}")
        return config
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


def validate_checkpoint_config(config):
    """Check settings such as head count that state_dict shapes cannot validate."""
    expected = InferenceConfig()
    differences = []
    for section in ("backbone", "binaural", "decoder"):
        for key, value in asdict(getattr(expected, section)).items():
            if getattr(getattr(config, section), key) != value:
                differences.append(f"{section}.{key}")
    if config.audio.image_size != expected.audio.image_size:
        differences.append("audio.image_size")
    if differences:
        raise ValueError(
            "Configuration does not match the released checkpoint architecture: "
            + ", ".join(differences)
        )
