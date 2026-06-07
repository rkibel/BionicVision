#!/usr/bin/env python3
"""Runtime helpers for Scheme 3 dense checkpoints."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from dataset_loaders import MEAN, STD
from models.dense import build_model, image_feature_channels, model_input_tensor
from models.hand_prior import HandPrior, hand_input_channels


def load_runtime(checkpoint_path: Path, hand_checkpoint: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    cfg = {
        "image_size": int(train_args["image_size"]),
        "image_feature_mode": train_args.get("image_feature_mode", "none"),
        "hand_input_mode": train_args.get("hand_input_mode", "raw_ring_outer_distance"),
        "hand_kernel_size": int(train_args.get("hand_kernel_size", 15)),
        "encoder": train_args.get("encoder", "efficientnet-b4"),
        "encoder_weights": train_args.get("encoder_weights", "imagenet"),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "device": device,
    }
    in_channels = 3 + image_feature_channels(cfg["image_feature_mode"]) + hand_input_channels(cfg["hand_input_mode"])
    model = build_model(cfg["encoder"], cfg["encoder_weights"], in_channels).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, HandPrior(hand_checkpoint, device), cfg


def extract_frames(video_path: Path, start_frame: int, duration_seconds: float) -> tuple[list[np.ndarray], float, list[int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(round(duration_seconds * fps)) if duration_seconds > 0 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) - start_frame
    frames, indices = [], []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for offset in range(max(total, 0)):
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            indices.append(start_frame + offset)
    finally:
        capture.release()
    return frames, fps, indices


def predict_probs(model, hand_prior: HandPrior, frames: list[np.ndarray], indices: list[int], cfg: dict, args) -> dict[int, np.ndarray]:
    output = {}
    for start in range(0, len(frames), args.batch_size):
        batch_frames = frames[start : start + args.batch_size]
        images = preprocess(batch_frames, cfg["image_size"]).to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            model_input = model_input_tensor(images, hand_prior, args.hand_prior_power, cfg["image_feature_mode"], cfg["hand_input_mode"], cfg["hand_kernel_size"])
            probs = torch.sigmoid(model(model_input).squeeze(1)).detach().cpu().float().numpy()
        for local, prob in enumerate(probs):
            output[indices[start + local]] = prob
    return output


def preprocess(frames: list[np.ndarray], image_size: int) -> torch.Tensor:
    rows = []
    for frame in frames:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        rows.append((tensor - MEAN) / STD)
    return torch.stack(rows)


def transform_hand_prior(raw_hand: torch.Tensor, power: float) -> torch.Tensor:
    return raw_hand if abs(power - 1.0) < 1e-8 else raw_hand.clamp(0.0, 1.0).pow(power)
