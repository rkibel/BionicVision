#!/usr/bin/env python3
"""Small helpers shared by the active-object HITL editor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DATA_ROOT = ROOT / "data/epic_kitchens/visor"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.epic_kitchens.annotations import EpicFrame, load_visor_annotations  # noqa: E402

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
EVAL_VIDEOS = ("P04_06", "P06_03", "P06_10", "P25_09", "P37_102")
TEST_VIDEOS = ("P03_120", "P06_108", "P08_17", "P22_107")
SPLITS = {"train": TRAIN_VIDEOS, "eval": EVAL_VIDEOS, "val": EVAL_VIDEOS, "test": TEST_VIDEOS}


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


@lru_cache(maxsize=1)
def load_frames_by_index(video_id: str) -> dict[int, EpicFrame]:
    frames = load_visor_annotations(annotation_path(video_id))
    merged: dict[int, EpicFrame] = {}
    for frame in frames:
        if frame.frame_index in merged:
            annotations = merged[frame.frame_index].annotations + frame.annotations
            merged[frame.frame_index] = type(frame)(**{**frame.__dict__, "annotations": annotations})
        else:
            merged[frame.frame_index] = frame
    return merged


def split_frame_keys(split: str) -> list[FrameKey]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    keys: list[FrameKey] = []
    for video_id in SPLITS[split]:
        frame_dir = sparse_frame_dir(video_id)
        if not annotation_path(video_id).exists() or not frame_dir.exists():
            continue
        for path in sorted(frame_dir.glob(f"{video_id}_frame_*.jpg")):
            keys.append(FrameKey.from_stem(path.stem))
    return keys


def annotation_shape(frame: EpicFrame) -> tuple[int, int]:
    if frame.annotation_size:
        width, height = frame.annotation_size
        return height, width
    return 480, 854


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
