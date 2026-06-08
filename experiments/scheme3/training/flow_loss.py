#!/usr/bin/env python3
"""Unsupervised optical-flow consistency loss for Scheme 3."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset_loaders import MEAN, STD


def unsupervised_flow_loss(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    left_images: torch.Tensor,
    right_images: torch.Tensor,
    bg_weight: float,
    min_prob: float,
    flow_grids: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    left_prob = left_logits.sigmoid()
    right_prob = right_logits.sigmoid()
    if flow_grids is None:
        right_grid, right_valid = flow_grid_source_to_target(left_images, right_images)
        left_grid, left_valid = flow_grid_source_to_target(right_images, left_images)
    else:
        right_grid = flow_grids["right_grid"].to(left_prob.device, left_prob.dtype)
        right_valid = flow_grids["right_valid"].to(left_prob.device, left_prob.dtype)
        left_grid = flow_grids["left_grid"].to(left_prob.device, left_prob.dtype)
        left_valid = flow_grids["left_valid"].to(left_prob.device, left_prob.dtype)
    right_loss = consistency_loss(warp_with_grid(left_prob, right_grid), right_prob, right_valid, bg_weight, min_prob)
    left_loss = consistency_loss(warp_with_grid(right_prob, left_grid), left_prob, left_valid, bg_weight, min_prob)
    return 0.5 * (left_loss + right_loss)


def consistency_loss(warped: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, bg_weight: float, min_prob: float) -> torch.Tensor:
    valid = valid.to(target.dtype)
    support = torch.maximum(warped.detach(), target.detach())
    foreground = (support >= min_prob).to(target.dtype)
    weights = valid * (foreground + bg_weight * (1.0 - foreground))
    delta = (warped - target).abs()
    return (delta * weights).sum(dim=(1, 2)).div(weights.sum(dim=(1, 2)).clamp_min(1.0)).mean()


def cached_flow_pair_grids(batch: dict, left_images: torch.Tensor, right_images: torch.Tensor, args) -> dict[str, torch.Tensor]:
    rows = [cached_flow_pair_grid_row(entry, right_frame, left_images[idx : idx + 1], right_images[idx : idx + 1], args) for idx, (entry, right_frame) in enumerate(zip(batch["entries_left"], batch["right_frame_numbers"]))]
    return {
        "right_grid": torch.cat([row["right_grid"] for row in rows], dim=0),
        "right_valid": torch.cat([row["right_valid"] for row in rows], dim=0),
        "left_grid": torch.cat([row["left_grid"] for row in rows], dim=0),
        "left_valid": torch.cat([row["left_valid"] for row in rows], dim=0),
    }


def cached_flow_pair_grid_row(entry, right_frame: int, left_image: torch.Tensor, right_image: torch.Tensor, args) -> dict[str, torch.Tensor]:
    path = flow_pair_cache_path(entry, right_frame, args)
    if path.exists():
        try:
            return load_flow_pair_cache(path, left_image.device)
        except (OSError, ValueError, KeyError):
            path.unlink(missing_ok=True)
    right_grid, right_valid = flow_grid_source_to_target(left_image, right_image)
    left_grid, left_valid = flow_grid_source_to_target(right_image, left_image)
    row = {
        "right_grid": right_grid.detach().cpu().float(),
        "right_valid": right_valid.detach().cpu().float(),
        "left_grid": left_grid.detach().cpu().float(),
        "left_valid": left_valid.detach().cpu().float(),
    }
    write_flow_pair_cache(path, row)
    return {key: value.to(left_image.device) for key, value in row.items()}


def flow_pair_cache_path(entry, right_frame: int, args) -> Path:
    key = "|".join(
        [
            "farneback-v1",
            str(args.image_size),
            str(getattr(entry, "take_name", "")),
            str(getattr(entry, "camera_name", "")),
            str(getattr(entry, "frame_number", "")),
            str(int(right_frame)),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return args.flow_cache_dir / digest[:2] / f"{digest}.npz"


def load_flow_pair_cache(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path) as payload:
        return {
            "right_grid": torch.from_numpy(payload["right_grid"].astype(np.float32, copy=False)).to(device),
            "right_valid": torch.from_numpy(payload["right_valid"].astype(np.float32, copy=False)).to(device),
            "left_grid": torch.from_numpy(payload["left_grid"].astype(np.float32, copy=False)).to(device),
            "left_valid": torch.from_numpy(payload["left_valid"].astype(np.float32, copy=False)).to(device),
        }


def write_flow_pair_cache(path: Path, row: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            tmp,
            right_grid=row["right_grid"].numpy().astype(np.float16),
            right_valid=row["right_valid"].numpy().astype(np.uint8),
            left_grid=row["left_grid"].numpy().astype(np.float16),
            left_valid=row["left_valid"].numpy().astype(np.uint8),
        )
        tmp.replace(path)
    except OSError:
        # Flow caching is an optimization; a full/read-only disk must not
        # terminate an otherwise valid training run.
        tmp.unlink(missing_ok=True)


def flow_grid_source_to_target(source_images: torch.Tensor, target_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source_np = normalized_to_gray(source_images)
    target_np = normalized_to_gray(target_images)
    _, height, width = source_np.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    grids, valids = [], []
    for source_gray, target_gray in zip(source_np, target_np):
        flow_target_to_source = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        map_x = grid_x + flow_target_to_source[..., 0]
        map_y = grid_y + flow_target_to_source[..., 1]
        valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        grids.append(np.stack([(2 * map_x / max(width - 1, 1)) - 1, (2 * map_y / max(height - 1, 1)) - 1], axis=-1).astype(np.float32))
        valids.append(valid.astype(np.float32))
    grid = torch.from_numpy(np.stack(grids)).to(source_images.device, source_images.dtype)
    valid = torch.from_numpy(np.stack(valids)).to(source_images.device, source_images.dtype)
    return grid, valid


def normalized_to_gray(images: torch.Tensor) -> np.ndarray:
    rgb = ((images.detach() * STD.to(images.device) + MEAN.to(images.device)).clamp(0, 1) * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return np.stack([cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0) for frame in rgb])


def warp_with_grid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(values.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)
