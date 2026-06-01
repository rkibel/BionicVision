#!/usr/bin/env python3
"""Shared utilities for VISOR hand segmentation experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DATA_ROOT = ROOT / "data/epic_kitchens/visor"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.epic_kitchens.annotations import load_visor_annotations, rasterize_object

TRAIN_VIDEOS = (
    "P01_03",
    "P01_07",
    "P01_103",
    "P01_104",
    "P01_107",
    "P02_01",
    "P02_07",
    "P02_107",
    "P02_128",
    "P03_03",
    "P03_22",
    "P03_11",
    "P03_13",
    "P03_17",
    "P03_10",
    "P04_12",
    "P04_13",
    "P04_21",
    "P04_26",
    "P04_33",
    "P06_01",
    "P06_110",
    "P06_12",
    "P06_14",
    "P07_08",
    "P07_101",
    "P07_103",
    "P07_110",
    "P09_02",
    "P09_07",
    "P09_103",
    "P09_104",
    "P09_106",
    "P18_01",
    "P18_02",
    "P18_07",
)
VAL_VIDEOS = (
    "P04_06",
    "P06_03",
    "P06_10",
    "P25_09",
    "P37_102",
)
TEST_VIDEOS = (
    "P03_120",
    "P06_108",
    "P08_17",
    "P22_107",
)
SPLITS = {
    "train": TRAIN_VIDEOS,
    "val": VAL_VIDEOS,
    "test": TEST_VIDEOS,
}
VIDEO_SPLIT = {video_id: split for split, videos in SPLITS.items() for video_id in videos}
HAND_LABEL_TOKENS = ("hand", "glove", "mitten")


@dataclass(frozen=True)
class FrameKey:
    video_id: str
    frame_index: int

    @property
    def stem(self) -> str:
        return f"{self.video_id}_frame_{self.frame_index:010d}"

    @classmethod
    def from_stem(cls, stem: str) -> "FrameKey":
        video_id, frame_text = stem.split("_frame_", maxsplit=1)
        return cls(video_id, int(frame_text))


def annotation_path(video_id: str) -> Path:
    return DATA_ROOT / "dense_annotations" / video_id / f"{video_id}_interpolations.json"


def sparse_frame_dir(video_id: str) -> Path:
    return DATA_ROOT / "sparse_rgb_frames" / video_id


def sparse_frame_path(key: FrameKey) -> Path:
    return sparse_frame_dir(key.video_id) / f"{key.stem}.jpg"


def available_video_ids() -> list[str]:
    annotation_videos = {path.parent.name for path in (DATA_ROOT / "dense_annotations").glob("*/*_interpolations.json")}
    image_videos = {path.name for path in (DATA_ROOT / "sparse_rgb_frames").glob("*") if path.is_dir()}
    return sorted(annotation_videos & image_videos)


@lru_cache(maxsize=4)
def load_frames_by_index(video_id: str):
    frames = load_visor_annotations(annotation_path(video_id))
    merged = {}
    for frame in frames:
        if frame.frame_index in merged:
            annotations = merged[frame.frame_index].annotations + frame.annotations
            merged[frame.frame_index] = type(frame)(**{**frame.__dict__, "annotations": annotations})
        else:
            merged[frame.frame_index] = frame
    return merged


def split_frame_keys(split: str, *, hand_only: bool = True) -> list[FrameKey]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    keys = []
    for video_id in SPLITS[split]:
        if not annotation_path(video_id).exists() or not sparse_frame_dir(video_id).exists():
            continue
        frames_by_index = load_frames_by_index(video_id)
        for path in sorted(sparse_frame_dir(video_id).glob(f"{video_id}_frame_*.jpg")):
            key = FrameKey.from_stem(path.stem)
            frame = frames_by_index.get(key.frame_index)
            if frame is None:
                continue
            has_hand = any(is_hand_like_label(annotation.name) for annotation in frame.annotations)
            if hand_only and not has_hand:
                continue
            keys.append(key)
    return keys


def hand_ground_truth_mask(video_id: str, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    frame = load_frames_by_index(video_id)[frame_index]
    mask = np.zeros(shape, dtype=bool)
    for annotation in frame.annotations:
        if is_hand_like_label(annotation.name):
            mask |= rasterize_object(annotation, shape).astype(bool)
    return mask


def rasterize_hand_mask(frame, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for annotation in frame.annotations:
        if is_hand_like_label(annotation.name):
            mask |= rasterize_object(annotation, shape).astype(bool)
    return mask


def is_hand_like_label(label: str) -> bool:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return any(token in normalized for token in HAND_LABEL_TOKENS)


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
    intersection = int(np.count_nonzero(pred & gt))
    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))
    union = int(np.count_nonzero(pred | gt))
    return {
        "iou": intersection / union if union else None,
        "precision": intersection / pred_count if pred_count else None,
        "recall": intersection / gt_count if gt_count else None,
    }


def summarize_frame_metrics(records: list[dict]) -> dict[str, float | int]:
    metrics = [record["metrics"] for record in records]
    detections = [record["detections"] for record in records]
    return {
        "frames": len(records),
        "mean_iou": mean_metric(metrics, "iou"),
        "mean_precision": mean_metric(metrics, "precision"),
        "mean_recall": mean_metric(metrics, "recall"),
        "mean_detections": float(np.mean(detections)) if detections else 0.0,
        "empty_frames": sum(1 for count in detections if count == 0),
    }


def mean_metric(metrics: list[dict[str, float | None]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if metric[key] is not None]
    return float(np.mean(values)) if values else 0.0


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
