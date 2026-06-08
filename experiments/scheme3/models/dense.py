#!/usr/bin/env python3
"""Dense relevance model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

from models.hand_prior import hand_input_channels, hand_input_features
from utils import json_safe


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_DEPTH_PRIORS: dict[str, Any] = {}


def model_input_tensor(images: torch.Tensor, hand_prior, hand_power: float, image_feature_mode: str, hand_input_mode: str, hand_kernel_size: int, provided_image_features: torch.Tensor | None = None) -> torch.Tensor:
    return model_input_tensor_from_raw_hand(images, hand_prior(images), hand_power, image_feature_mode, hand_input_mode, hand_kernel_size, provided_image_features)


def model_input_tensor_from_raw_hand(images: torch.Tensor, raw_hand: torch.Tensor, hand_power: float, image_feature_mode: str, hand_input_mode: str, hand_kernel_size: int, provided_image_features: torch.Tensor | None = None) -> torch.Tensor:
    hand_source = raw_hand if abs(hand_power - 1.0) < 1e-8 else raw_hand.clamp(0.0, 1.0).pow(hand_power)
    return torch.cat([images, hand_input_features(hand_source, hand_input_mode, hand_kernel_size), image_features(images, image_feature_mode, provided_image_features)], dim=1)


def model_input_channels(args) -> int:
    return 3 + image_feature_channels(args.image_feature_mode) + hand_input_channels(args.hand_input_mode)


def image_feature_channels(mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "tc_monodepth":
        return 1
    if mode == "glc_gaze":
        return 1
    raise ValueError(f"Unsupported image feature mode: {mode}")


def image_features(images: torch.Tensor, mode: str, provided: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "none":
        return images[:, :0]
    if mode == "glc_gaze":
        if provided is None:
            raise ValueError("GLC gaze mode requires provided image features")
        return provided.to(images.device, images.dtype)
    if mode != "tc_monodepth":
        raise ValueError(f"Unsupported image feature mode: {mode}")
    key = str(images.device)
    estimator = _DEPTH_PRIORS.get(key)
    if estimator is None:
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from src.models.depth.tc_monodepth.adapter import TCMonoDepthEstimator

        estimator = _DEPTH_PRIORS[key] = TCMonoDepthEstimator(device=key)
    rgb_255 = ((images * STD.to(images.device) + MEAN.to(images.device)).clamp(0.0, 1.0) * 255.0).float()
    return estimator.predict_tensor(rgb_255, output_size=tuple(images.shape[-2:])).to(images.device, images.dtype)


def build_model(encoder: str, encoder_weights: str | None, in_channels: int):
    import segmentation_models_pytorch as smp

    weights = None if encoder_weights in ("", "none", "None") else encoder_weights
    return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=1)


def load_init_checkpoint(model, path: Path | None, device: str, new_input_init: str = "zero", new_input_init_scale: float = 0.1) -> dict | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict, adaptations = adapt_input_channels(checkpoint["state_dict"], model.state_dict(), new_input_init, new_input_init_scale)
    model.load_state_dict(state_dict, strict=True)
    return {"checkpoint": str(path), "threshold": float(checkpoint.get("threshold", 0.0)), "args": json_safe(checkpoint.get("args", {})), "adaptations": adaptations}


def adapt_input_channels(source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor], new_input_init: str = "zero", new_input_init_scale: float = 0.1) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
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
            if new_input_init == "rgb_mean":
                seed = source_weight[:, : min(3, source_weight.shape[1])].mean(dim=1, keepdim=True)
                adapted[:, source_weight.shape[1] :] = seed.to(adapted.device, adapted.dtype) * new_input_init_scale
            elif new_input_init == "zero":
                adapted[:, source_weight.shape[1] :] = 0.0
            else:
                raise ValueError(f"Unsupported new input initialization: {new_input_init}")
        state[key] = adapted
        adaptations.append({"key": key, "source_channels": int(source_weight.shape[1]), "target_channels": int(target_weight.shape[1]), "new_input_init": new_input_init, "new_input_init_scale": new_input_init_scale})
    return state, adaptations


def amplify_new_input_gradients(model, feature_channels: int, scale: float) -> str | None:
    """Scale gradients only for newly added trailing input channels."""

    if feature_channels <= 0 or abs(scale - 1.0) < 1e-8:
        return None
    for name, parameter in model.named_parameters():
        if parameter.ndim == 4 and parameter.shape[1] >= 3 + feature_channels:
            def scale_feature_gradient(gradient: torch.Tensor) -> torch.Tensor:
                gradient = gradient.clone()
                gradient[:, -feature_channels:] *= scale
                return gradient

            parameter.register_hook(scale_feature_gradient)
            return name
    raise RuntimeError("Could not find the model input convolution")


def configure_training_stage(model, stage: str, feature_channels: int, learning_rate: float, feature_learning_rate: float, weight_decay: float):
    """Configure a depth-prior learning stage and return its optimizer."""

    handle = getattr(model, "_image_feature_gradient_handle", None)
    if handle is not None:
        handle.remove()
        model._image_feature_gradient_handle = None

    for parameter in model.parameters():
        parameter.requires_grad = stage == "joint"
    if stage == "feature":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("encoder._conv_stem.", "encoder._bn0.", "encoder._blocks.0."))
    elif stage != "joint":
        raise ValueError(f"Unsupported training stage: {stage}")

    input_name, input_weight = input_convolution(model)
    input_weight.requires_grad = True
    scale = feature_learning_rate / learning_rate

    def feature_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        if stage == "feature":
            gradient[:, :-feature_channels] = 0.0
        gradient[:, -feature_channels:] *= scale
        return gradient

    model._image_feature_gradient_handle = input_weight.register_hook(feature_gradient)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # The stem is a shared tensor, so decoupled weight decay would move its
    # gradient-masked RGB/hand slices during the feature-only stage.
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0 if stage == "feature" else weight_decay)
    return optimizer, input_name, sum(parameter.numel() for parameter in trainable)


def image_feature_diagnostics(model, reference_weight: torch.Tensor, feature_channels: int) -> dict[str, float]:
    _, weight = input_convolution(model)
    current = weight.detach().float().cpu()[:, -feature_channels:]
    reference = reference_weight.detach().float().cpu()[:, -feature_channels:]
    delta = current - reference
    return {
        "image_feature_weight_norm": float(current.norm()),
        "image_feature_weight_delta_norm": float(delta.norm()),
        "image_feature_weight_relative_delta": float(delta.norm() / reference.norm().clamp_min(1e-12)),
        "image_feature_weight_cosine_to_init": float(torch.nn.functional.cosine_similarity(current.flatten(), reference.flatten(), dim=0)),
    }


def input_convolution(model) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if parameter.ndim == 4 and parameter.shape[1] >= 3:
            return name, parameter
    raise RuntimeError("Could not find the model input convolution")


def input_channel_norms(model) -> list[float]:
    _, parameter = input_convolution(model)
    return [float(value) for value in parameter.detach().float().square().sum((0, 2, 3)).sqrt().cpu()]
