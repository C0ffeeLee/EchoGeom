"""Depth and PCA surface-normal metrics for BV2."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def depth_metrics(
    prediction_m: torch.Tensor, target_m: torch.Tensor, min_depth: float = 0.1
) -> dict[str, float]:
    valid = torch.isfinite(target_m) & (target_m > min_depth)
    if not bool(valid.any()):
        return {}
    pred = prediction_m[valid].clamp_min(1e-06)
    target = target_m[valid].clamp_min(1e-06)
    difference = pred - target
    ratio = torch.maximum(pred / target, target / pred)
    return {
        "abs_rel": float((difference.abs() / target).mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "delta1": float((ratio < 1.25).float().mean()),
        "delta2": float((ratio < 1.25**2).float().mean()),
        "delta3": float((ratio < 1.25**3).float().mean()),
        "log10": float((pred.log10() - target.log10()).abs().mean()),
        "mae": float(difference.abs().mean()),
    }


def _backproject(
    depth: torch.Tensor, valid: torch.Tensor, hfov_deg: float, vfov_deg: float
) -> torch.Tensor:
    height, width = depth.shape
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) * 0.5)
    fy = 0.5 * height / math.tan(math.radians(vfov_deg) * 0.5)
    u = torch.arange(width, device=depth.device, dtype=depth.dtype).view(1, width)
    v = torch.arange(height, device=depth.device, dtype=depth.dtype).view(height, 1)
    z = torch.where(valid, depth, torch.zeros_like(depth))
    x = (u - (width - 1) * 0.5) * z / fx
    y = (v - (height - 1) * 0.5) * z / fy
    return torch.stack((x, y, z), dim=-1)


def _pca_normals(
    depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    hfov_deg: float,
    vfov_deg: float,
    patch_size: int,
    min_points: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}")
    if not 3 <= min_points <= patch_size * patch_size:
        raise ValueError(f"min_points must be in [3, {patch_size * patch_size}], got {min_points}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    height, width = depth.shape
    points = _backproject(depth, valid, hfov_deg, vfov_deg)
    count = patch_size * patch_size
    unfolded_points = (
        F.unfold(points.permute(2, 0, 1).unsqueeze(0), patch_size, padding=patch_size // 2)
        .view(1, 3, count, height * width)
        .permute(0, 3, 2, 1)
        .reshape(height * width, count, 3)
    )
    weights = (
        F.unfold(valid.float().view(1, 1, height, width), patch_size, padding=patch_size // 2)
        .view(count, height * width)
        .transpose(0, 1)
    )
    samples = weights.sum(1)
    usable = (samples >= min_points) & valid.flatten()
    normals = torch.zeros((height * width, 3), device=depth.device, dtype=depth.dtype)
    if not bool(usable.any()):
        return (normals.view(height, width, 3), usable.view(height, width))
    indices = torch.nonzero(usable, as_tuple=False).squeeze(1)
    point_windows = unfolded_points[indices]
    point_weights = weights[indices].unsqueeze(-1)
    denominator = samples[indices].view(-1, 1, 1)
    mean = (point_windows * point_weights).sum(1, keepdim=True) / denominator
    # The smallest covariance eigenvector is the local surface normal.
    centered = point_windows - mean
    covariance = (centered * point_weights).transpose(1, 2) @ centered / denominator
    centers = points.reshape(-1, 3)[indices]
    for start in range(0, indices.numel(), chunk_size):
        end = min(start + chunk_size, indices.numel())
        eigenvectors = torch.linalg.eigh(covariance[start:end]).eigenvectors
        normal = eigenvectors[:, :, 0]
        center = centers[start:end]
        # PCA leaves the sign ambiguous; orient each normal toward the camera.
        normal[(normal * center).sum(1) > 0] *= -1
        normals[indices[start:end]] = F.normalize(normal, dim=1)
    return (normals.view(height, width, 3), usable.view(height, width))


def pointcloud_normal_metrics(
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    *,
    min_depth: float = 0.1,
    max_depth: float = 30.0,
    hfov_deg: float = 86.0,
    vfov_deg: float = 57.0,
    patch_size: int = 5,
    min_points: int = 6,
    chunk_size: int = 32768,
    eval_mask: str = "valid_neighborhood",
) -> tuple[dict[str, float], float]:
    if prediction_m.shape != target_m.shape or prediction_m.ndim != 2:
        raise ValueError("pointcloud_normal_metrics expects two [H,W] depth maps")
    prediction_m = prediction_m.float()
    target_m = target_m.float()
    target_valid = torch.isfinite(target_m) & (target_m > min_depth) & (target_m < max_depth)
    pred_valid = (
        torch.isfinite(prediction_m) & (prediction_m > min_depth) & (prediction_m < max_depth)
    )
    pred_normal, pred_normal_valid = _pca_normals(
        prediction_m,
        pred_valid,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        patch_size=patch_size,
        min_points=min_points,
        chunk_size=chunk_size,
    )
    target_normal, target_normal_valid = _pca_normals(
        target_m,
        target_valid,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        patch_size=patch_size,
        min_points=min_points,
        chunk_size=chunk_size,
    )
    mode = eval_mask.lower()
    if mode in {"none", "all", "no_mask"}:
        requested_mask = torch.ones_like(target_valid)
    elif mode == "valid":
        requested_mask = target_valid
    elif mode in {"valid_neighborhood", "neighborhood"}:
        # Require a full valid 3x3 GT neighborhood to avoid boundary artifacts.
        kernel = torch.ones((1, 1, 3, 3), device=target_m.device)
        neighborhood = F.conv2d(target_valid.float()[None, None], kernel, padding=1)[0, 0] == 9
        requested_mask = target_valid & neighborhood
    else:
        raise ValueError(f"Unknown normal evaluation mask: {eval_mask}")
    mask = requested_mask & pred_normal_valid & target_normal_valid
    mask_ratio = float(mask.float().mean())
    if not bool(mask.any()):
        return ({}, mask_ratio)
    cosine = (pred_normal * target_normal).sum(-1).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))[mask]
    return (
        {
            "mean_ae": float(angle.mean()),
            "median_ae": float(angle.median()),
            "rmse_ae": float(angle.square().mean().sqrt()),
            "a11_25": float((angle < 11.25).float().mean()),
            "a22_5": float((angle < 22.5).float().mean()),
            "a30": float((angle < 30).float().mean()),
        },
        mask_ratio,
    )


def average_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    metrics = [item for item in metrics if item]
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum((item[key] for item in metrics)) / len(metrics) for key in keys}
