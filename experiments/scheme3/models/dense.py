#!/usr/bin/env python3
"""Dense relevance model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.hand_prior import hand_input_channels, hand_input_features
from utils import json_safe


def model_input_tensor(images: torch.Tensor, hand_prior, hand_power: float, image_feature_mode: str, hand_input_mode: str, hand_kernel_size: int) -> torch.Tensor:
    return model_input_tensor_from_raw_hand(images, hand_prior(images), hand_power, image_feature_mode, hand_input_mode, hand_kernel_size)


def model_input_tensor_from_raw_hand(images: torch.Tensor, raw_hand: torch.Tensor, hand_power: float, image_feature_mode: str, hand_input_mode: str, hand_kernel_size: int) -> torch.Tensor:
    if image_feature_channels(image_feature_mode) != 0:
        raise ValueError(f"Unsupported image feature mode: {image_feature_mode}")
    hand_source = raw_hand if abs(hand_power - 1.0) < 1e-8 else raw_hand.clamp(0.0, 1.0).pow(hand_power)
    return torch.cat([images, hand_input_features(hand_source, hand_input_mode, hand_kernel_size)], dim=1)


def model_input_channels(args) -> int:
    return 3 + image_feature_channels(args.image_feature_mode) + hand_input_channels(args.hand_input_mode)


def image_feature_channels(mode: str) -> int:
    if mode == "none":
        return 0
    raise ValueError(f"Unsupported image feature mode: {mode}")


def build_model(encoder: str, encoder_weights: str | None, in_channels: int):
    import segmentation_models_pytorch as smp

    weights = None if encoder_weights in ("", "none", "None") else encoder_weights
    return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=1)


def load_init_checkpoint(model, path: Path | None, device: str) -> dict | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict, adaptations = adapt_input_channels(checkpoint["state_dict"], model.state_dict())
    model.load_state_dict(state_dict, strict=True)
    return {"checkpoint": str(path), "threshold": float(checkpoint.get("threshold", 0.0)), "args": json_safe(checkpoint.get("args", {})), "adaptations": adaptations}


def adapt_input_channels(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    state = dict(source_state)
    adaptations = []
    for key, target_weight in target_state.items():
        source_weight = state.get(key)
        if source_weight is None or source_weight.shape == target_weight.shape:
            continue
        if source_weight.ndim != 4 or target_weight.ndim != 4 or source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[2:] != target_weight.shape[2:]:
            continue
        copy_channels = min(source_weight.shape[1], target_weight.shape[1])
        adapted = target_weight.clone()
        adapted[:, :copy_channels] = source_weight[:, :copy_channels].to(adapted.device, adapted.dtype)
        if target_weight.shape[1] > source_weight.shape[1]:
            adapted[:, source_weight.shape[1] :] = 0.0
        state[key] = adapted
        adaptations.append({"key": key, "source_channels": int(source_weight.shape[1]), "target_channels": int(target_weight.shape[1])})
    return state, adaptations
