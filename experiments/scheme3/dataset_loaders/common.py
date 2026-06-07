#!/usr/bin/env python3
"""Shared dataset helpers for Scheme 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ego4d.research.util.masks import decode_mask
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
EGOEXO_ROOT = ROOT / "data/egoexo4d"
EGOHOS_ROOT = ROOT / "data/egohos/data"
DEFAULT_EGOEXO_CAMERA = "aria01_214-1"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class FrameEntry:
    split: str
    take_uid: str
    take_name: str
    camera_name: str
    frame_number: int
    object_tracks: tuple[str, ...]
    hand_tracks: tuple[str, ...]


@dataclass(frozen=True)
class EgoHOSFrameEntry:
    split: str
    image_path: Path
    label_path: Path
    source: str
    frame_id: str


def augment_image_masks(image: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import random

    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        masks = np.ascontiguousarray(masks[..., ::-1])
    image = image.astype(np.float32)
    if random.random() < 0.8:
        image *= random.uniform(0.85, 1.15)
    if random.random() < 0.8:
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * random.uniform(0.85, 1.15) + mean
    if random.random() < 0.5:
        gray = image.mean(axis=2, keepdims=True)
        image = gray + (image - gray) * random.uniform(0.85, 1.20)
    if random.random() < 0.2:
        image += np.random.normal(0.0, 4.0, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8), masks.astype(np.float32, copy=False)


def relation_path(split: str) -> Path:
    return EGOEXO_ROOT / f"annotations/relations_{split}.json"


def load_relations(split: str) -> dict[str, Any]:
    path = relation_path(split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", payload)
    if not isinstance(annotations, dict):
        raise RuntimeError(f"Unexpected relation file format: {path}")
    return annotations


def take_video_path(take_name: str, camera_name: str = DEFAULT_EGOEXO_CAMERA) -> Path:
    return EGOEXO_ROOT / f"takes/{take_name}/frame_aligned_videos/downscaled/448/{camera_name}.mp4"


def is_hand_track(track_id: str) -> bool:
    normalized = " ".join(track_id.lower().replace("_", " ").split())
    return normalized.startswith("left hand") or normalized.startswith("right hand")


def decode_track_mask(take: dict[str, Any], track_id: str, camera_name: str, frame_number: int, image_size: int) -> np.ndarray | None:
    payload = (take.get("object_masks") or {}).get(track_id, {})
    annotation = (payload.get(camera_name, {}) if isinstance(payload, dict) else {}).get("annotation") or {}
    mask_payload = annotation.get(str(frame_number))
    if mask_payload is None:
        return None
    mask = decode_mask(mask_payload).astype(np.uint8)
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)


def read_video_frame(captures: dict[Path, cv2.VideoCapture], path: Path, frame_number: int, image_size: int) -> np.ndarray:
    capture = captures.get(path)
    if capture is None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        captures[path] = capture
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def read_rgb_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def read_egohos_label(path: Path, image_size: int) -> np.ndarray:
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[..., 0]
    return cv2.resize(label.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def egohos_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg"):
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def egohos_source(stem: str) -> str:
    for prefix in ("epic", "ego4d", "thu", "escape", "youtube"):
        if stem.startswith(f"{prefix}_"):
            return prefix
    return stem.split("_", 1)[0].lower()


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "target_masks": [sample["target_masks"] for sample in samples],
        "entries": [sample["entry"] for sample in samples],
        "datasets": [sample.get("dataset", "") for sample in samples],
        "sources": [sample.get("source", "") for sample in samples],
    }


def collate_flow_pairs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    left = [sample["left"] for sample in samples]
    return {
        "images_left": torch.stack([sample["image"] for sample in left]),
        "images_right": torch.stack([sample["right_image"] for sample in samples]),
        "target_masks_left": [sample["target_masks"] for sample in left],
        "entries_left": [sample["entry"] for sample in left],
        "datasets_left": [sample.get("dataset", "") for sample in left],
        "sources_left": [sample.get("source", "") for sample in left],
        "right_frame_numbers": [sample["right_frame_number"] for sample in samples],
    }


def target_union(targets: list[torch.Tensor], device: str | torch.device) -> torch.Tensor:
    rows = []
    for masks in targets:
        if masks.numel():
            rows.append((masks.to(device) >= 0.5).any(dim=0).float())
        else:
            rows.append(torch.zeros((masks.shape[-2], masks.shape[-1]), dtype=torch.float32, device=device))
    return torch.stack(rows)
