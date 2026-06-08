"""EPIC-KITCHENS VISOR samples for hand-segmentor evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .annotations import VisorFrame, VisorObject, load_visor_annotations, rasterize_object


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data/epic_kitchens/visor"

VISOR_SPLITS = {
    "train": (
        "P01_03", "P01_07", "P01_103", "P01_104", "P01_107", "P02_01", "P02_07", "P02_107", "P02_128",
        "P03_03", "P03_22", "P03_11", "P03_13", "P03_17", "P03_10", "P04_12", "P04_13", "P04_21",
        "P04_26", "P04_33", "P06_01", "P06_110", "P06_12", "P06_14", "P07_08", "P07_101", "P07_103",
        "P07_110", "P09_02", "P09_07", "P09_103", "P09_104", "P09_106", "P18_01", "P18_02", "P18_07",
    ),
    "val": ("P04_06", "P06_03", "P06_10", "P25_09", "P37_102"),
    "test": ("P03_120", "P06_108", "P08_17", "P22_107"),
}
HAND_LABEL_TOKENS = ("hand", "glove", "mitten")


@dataclass(frozen=True)
class VisorEvaluationItem:
    image_path: Path
    split: str
    video_id: str
    frame_index: int
    annotations: tuple[VisorObject, ...]

    @property
    def identifier(self) -> str:
        return self.image_path.stem


def list_visor_items(
    splits: tuple[str, ...] = ("test",),
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    max_frames: int | None = None,
) -> list[VisorEvaluationItem]:
    """Return annotated sparse VISOR frames containing a hand-like label."""

    items = []
    for split in splits:
        if split not in VISOR_SPLITS:
            raise ValueError(f"Unknown VISOR split: {split}")
        for video_id in VISOR_SPLITS[split]:
            annotation_path = data_root / "dense_annotations" / video_id / f"{video_id}_interpolations.json"
            frame_dir = data_root / "sparse_rgb_frames" / video_id
            if not annotation_path.exists() or not frame_dir.exists():
                continue
            frames = _merge_frames(load_visor_annotations(annotation_path))
            for image_path in sorted(frame_dir.glob(f"{video_id}_frame_*.jpg")):
                frame_index = int(image_path.stem.split("_frame_", maxsplit=1)[1])
                frame = frames.get(frame_index)
                if frame is None or not any(is_hand_like_label(obj.name) for obj in frame.annotations):
                    continue
                items.append(VisorEvaluationItem(image_path, split, video_id, frame_index, frame.annotations))
                if max_frames is not None and len(items) >= max_frames:
                    return items
    return items


def load_visor_sample(item: VisorEvaluationItem) -> tuple[np.ndarray, np.ndarray]:
    """Load one BGR frame and its binary VISOR hand mask."""

    image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(item.image_path)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for annotation in item.annotations:
        if is_hand_like_label(annotation.name):
            mask |= rasterize_object(annotation, image.shape[:2]).astype(bool)
    return image, mask


def is_hand_like_label(label: str) -> bool:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return any(token in normalized for token in HAND_LABEL_TOKENS)


def _merge_frames(frames: list[VisorFrame]) -> dict[int, VisorFrame]:
    merged = {}
    for frame in frames:
        if frame.frame_index in merged:
            existing = merged[frame.frame_index]
            frame = VisorFrame(frame.video_id, frame.frame_name, frame.frame_index, existing.annotations + frame.annotations)
        merged[frame.frame_index] = frame
    return merged
