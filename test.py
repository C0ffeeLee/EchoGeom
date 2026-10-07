"""Evaluate released EchoGeom weights on BatVision V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from echogeom.checkpoint import load_model
from echogeom.config import (
    DEFAULT_CONFIG,
    NormalsConfig,
    cli_config,
    resolve_device,
    resolve_path,
)
from echogeom.data import BatVisionV2
from echogeom.metrics import (
    average_metric_dicts,
    depth_metrics,
    pointcloud_normal_metrics,
)
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(model, loader, device, with_normals=True, normals_config=None):
    normals = normals_config or NormalsConfig()
    depth_rows, normal_rows = [], []
    for features, target in loader:
        # Undo the fixed 30 m normalization before computing metrics in meters.
        prediction = model(features["spectrogram"].to(device)) * 30.0
        target = target.to(device) * 30.0
        if not torch.isfinite(prediction).all():
            raise RuntimeError("The model produced non-finite depth predictions")
        # Average per-image metrics so each scene sample has equal weight.
        for index in range(target.size(0)):
            depth_rows.append(depth_metrics(prediction[index], target[index]))
            if with_normals:
                metrics, _ = pointcloud_normal_metrics(
                    prediction[index, 0],
                    target[index, 0],
                    hfov_deg=normals.hfov_deg,
                    vfov_deg=normals.vfov_deg,
                    patch_size=normals.patch_size,
                    min_points=normals.min_points,
                    eval_mask=normals.eval_mask,
                )
                normal_rows.append(metrics)
    metrics = average_metric_dicts(depth_rows)
    if not metrics:
        raise RuntimeError("No valid ground-truth depth pixels were evaluated")
    metrics["log_10"] = metrics.pop("log10")
    result = {"metrics": metrics}
    if with_normals:
        result["normal_metrics"] = average_metric_dicts(normal_rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, help="Override paths.data_root")
    parser.add_argument("--checkpoint", type=Path, help="Override paths.checkpoint")
    parser.add_argument("--split", choices=("val", "test"), help="Override runtime.split")
    parser.add_argument("--batch-size", type=int, help="Override runtime.batch_size")
    parser.add_argument("--num-workers", type=int, help="Override runtime.num_workers")
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument(
        "--no-normals", action="store_true", help="Override normals.enabled to false"
    )
    parser.add_argument("--output", type=Path, help="Override paths.evaluation_output")
    args = parser.parse_args()
    config = cli_config(args, parser, "evaluation")
    device = resolve_device(config)
    checkpoint = resolve_path(config.paths.checkpoint)
    model = load_model(checkpoint, device, config)
    dataset = BatVisionV2(
        resolve_path(config.paths.data_root),
        config.runtime.split,
        config.audio,
        config.decoder.output_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluating {config.runtime.split}: {len(dataset)} samples", flush=True)
    result = evaluate(model, loader, device, config.normals.enabled, config.normals)
    output = resolve_path(config.paths.evaluation_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Depth: " + " | ".join(f"{key}={value:.6f}" for key, value in result["metrics"].items()))
    if "normal_metrics" in result:
        print(
            "Normals: "
            + " | ".join(f"{key}={value:.6f}" for key, value in result["normal_metrics"].items())
        )
    print(f"Saved metrics: {output}")


if __name__ == "__main__":
    main()
