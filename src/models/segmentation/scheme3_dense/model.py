"""Scheme 3 dense model and hand-prior conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from models.segmentation.hand_segmentor.model import build_model as build_hand_model


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class DenseModelConfig:
    image_size: int = 256
    encoder: str = "efficientnet-b4"
    encoder_weights: str | None = "imagenet"
    hand_input_mode: str = "raw_ring_outer_distance"
    hand_kernel_size: int = 15
    hand_prior_power: float = 1.5
    image_feature_mode: str = "none"
    threshold: float = 0.5


class HandPrior:
    """Frozen tensor adapter around the reusable hand segmentor."""

    def __init__(self, checkpoint_path: Path, device: str | torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = build_hand_model(str(checkpoint["model_name"]), device, encoder_weights=None)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.size = parse_size(checkpoint["image_size"])
        self.threshold = float(checkpoint["threshold"])

    @torch.inference_mode()
    def __call__(self, normalized_images: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(normalized_images.device)
        std = IMAGENET_STD.to(normalized_images.device)
        rgb = (normalized_images * std + mean).clamp(0.0, 1.0)
        resized = torch.nn.functional.interpolate(rgb, size=self.size, mode="bilinear", align_corners=False)
        logits = self.model(((resized - mean) / std).to(self.device))
        if isinstance(logits, dict):
            logits = logits["out"]
        probs = torch.sigmoid(logits).to(normalized_images.device)
        return torch.nn.functional.interpolate(probs, size=normalized_images.shape[-2:], mode="bilinear", align_corners=False)


def build_model(config: DenseModelConfig, *, load_encoder_weights: bool = True) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    weights = config.encoder_weights if load_encoder_weights else None
    return smp.UnetPlusPlus(
        encoder_name=config.encoder,
        encoder_weights=weights,
        in_channels=3 + hand_input_channels(config.hand_input_mode) + image_feature_channels(config.image_feature_mode),
        classes=1,
    )


def model_input_tensor(
    images: torch.Tensor,
    hand_prior: HandPrior,
    config: DenseModelConfig,
    image_features: torch.Tensor | None = None,
) -> torch.Tensor:
    return model_input_tensor_from_hand(images, hand_prior(images), config, image_features)


def model_input_tensor_from_hand(
    images: torch.Tensor,
    raw_hand: torch.Tensor,
    config: DenseModelConfig,
    image_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if abs(config.hand_prior_power - 1.0) >= 1e-8:
        raw_hand = raw_hand.clamp(0.0, 1.0).pow(config.hand_prior_power)
    features = image_feature_tensor(images, config.image_feature_mode, image_features)
    return torch.cat([images, hand_input_features(raw_hand, config.hand_input_mode, config.hand_kernel_size), features], dim=1)


def image_feature_channels(mode: str) -> int:
    if mode == "none":
        return 0
    if mode in {"tc_monodepth", "glc_gaze"}:
        return 1
    raise ValueError(f"Unsupported image feature mode: {mode}")


def image_feature_tensor(images: torch.Tensor, mode: str, provided: torch.Tensor | None) -> torch.Tensor:
    if mode == "none":
        return images[:, :0]
    if provided is None:
        raise ValueError(f"{mode} requires an image feature tensor")
    if provided.shape[-2:] != images.shape[-2:]:
        provided = torch.nn.functional.interpolate(provided, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return provided.to(images.device, images.dtype)


def hand_input_features(raw_hand: torch.Tensor, mode: str, kernel_size: int) -> torch.Tensor:
    if mode != "raw_ring_outer_distance":
        raise ValueError(f"Unsupported hand input mode: {mode}")
    dilated = dilate_prior(raw_hand, kernel_size)
    ring = (dilated - raw_hand).clamp(0.0, 1.0)
    distance = outer_distance_proximity(raw_hand, max_distance=max(float(kernel_size * 2), 1.0))
    return torch.cat([raw_hand, ring, distance], dim=1)


def hand_input_channels(mode: str) -> int:
    if mode == "raw_ring_outer_distance":
        return 3
    raise ValueError(f"Unsupported hand input mode: {mode}")


def dilate_prior(prior: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return prior
    return torch.nn.functional.max_pool2d(prior, kernel_size, stride=1, padding=kernel_size // 2)


def outer_distance_proximity(raw_hand: torch.Tensor, max_distance: float) -> torch.Tensor:
    radius = max(int(round(max_distance)), 1)
    sigma = max(max_distance / 2.0, 1.0)
    coords = torch.arange(-radius, radius + 1, device=raw_hand.device, dtype=raw_hand.dtype)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    blurred = torch.nn.functional.conv2d(raw_hand, kernel.view(1, 1, 1, -1), padding=(0, radius))
    blurred = torch.nn.functional.conv2d(blurred, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return (torch.maximum(raw_hand, blurred).clamp(0.0, 1.0) * (1.0 - raw_hand)).clamp(0.0, 1.0)


def parse_size(value: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)
