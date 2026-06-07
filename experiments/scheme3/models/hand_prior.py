#!/usr/bin/env python3
"""Saved hand-segmentor prior for Scheme 3."""

from __future__ import annotations

from pathlib import Path

import torch

from hand_segmentor.model import build_model


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HandPrior:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(str(checkpoint["model_name"]), device, encoder_weights=None)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model
        self.device = device
        self.size = parse_size(checkpoint["image_size"])
        self.threshold = float(checkpoint["threshold"])

    @torch.inference_mode()
    def __call__(self, normalized_images: torch.Tensor) -> torch.Tensor:
        device = normalized_images.device
        mean = MEAN.to(device)
        std = STD.to(device)
        rgb = torch.clamp(normalized_images * std + mean, 0.0, 1.0)
        resized = torch.nn.functional.interpolate(rgb, size=self.size, mode="bilinear", align_corners=False)
        resized = (resized - mean) / std
        logits = self.model(resized.to(self.device))
        if isinstance(logits, dict):
            logits = logits["out"]
        probs = torch.sigmoid(logits).to(device)
        return torch.nn.functional.interpolate(probs, size=normalized_images.shape[-2:], mode="bilinear", align_corners=False)


def parse_size(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)


def dilate_prior(prior: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    if kernel_size <= 1:
        return prior
    pad = kernel_size // 2
    return torch.nn.functional.max_pool2d(prior, kernel_size=kernel_size, stride=1, padding=pad)


def hand_input_features(raw_hand: torch.Tensor, mode: str, kernel_size: int = 15) -> torch.Tensor:
    """Build the canonical hand-conditioning channels for the relevance model."""

    if mode != "raw_ring_outer_distance":
        raise ValueError(f"Unsupported hand input mode: {mode}")
    dilated = dilate_prior(raw_hand, kernel_size=kernel_size)
    ring = torch.clamp(dilated - raw_hand, 0.0, 1.0)
    distance = outer_distance_proximity(raw_hand, max_distance=max(float(kernel_size * 2), 1.0))
    return torch.cat([raw_hand, ring, distance], dim=1)


def hand_input_channels(mode: str) -> int:
    if mode == "raw_ring_outer_distance":
        return 3
    raise ValueError(f"Unsupported hand input mode: {mode}")


def gaussian_proximity(raw_hand: torch.Tensor, max_distance: float = 30.0) -> torch.Tensor:
    radius = max(int(round(max_distance)), 1)
    sigma = max(max_distance / 2.0, 1.0)
    coords = torch.arange(-radius, radius + 1, device=raw_hand.device, dtype=raw_hand.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    horizontal = kernel_1d.view(1, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(raw_hand, horizontal, padding=(0, radius))
    blurred = torch.nn.functional.conv2d(blurred, vertical, padding=(radius, 0))
    return torch.maximum(raw_hand, blurred).clamp(0.0, 1.0)


def outer_distance_proximity(raw_hand: torch.Tensor, max_distance: float = 30.0) -> torch.Tensor:
    """Near-hand proximity with hand pixels suppressed to avoid interior leakage."""

    proximity = gaussian_proximity(raw_hand, max_distance=max_distance)
    return (proximity * (1.0 - raw_hand)).clamp(0.0, 1.0)
