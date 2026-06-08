"""EgoHOS samples for hand-segmentor evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data/egohos/data"
HAND_LABEL_IDS = (1, 2)


@dataclass(frozen=True)
class EgoHOSEvaluationItem:
    image_path: Path
    label_path: Path
    split: str

    @property
    def identifier(self) -> str:
        return self.image_path.stem


def list_egohos_items(
    splits: tuple[str, ...] = ("val", "test_indomain", "test_outdomain"),
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    max_frames: int | None = None,
) -> list[EgoHOSEvaluationItem]:
    """Return EgoHOS image/label pairs for evaluation."""

    items = []
    for split in splits:
        image_dir = data_root / split / "image"
        label_dir = data_root / split / "label"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing EgoHOS split directories: {image_dir} and {label_dir}")
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.png"
            if label_path.exists():
                items.append(EgoHOSEvaluationItem(image_path, label_path, split))
                if max_frames is not None and len(items) >= max_frames:
                    return items
    if not items:
        raise RuntimeError(f"No EgoHOS image/label pairs found for splits: {splits}")
    return items


def load_egohos_sample(item: EgoHOSEvaluationItem) -> tuple[np.ndarray, np.ndarray]:
    """Load one BGR frame and its left/right-hand mask."""

    image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(item.label_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(item.image_path)
    if label is None:
        raise FileNotFoundError(item.label_path)
    if label.shape != image.shape[:2]:
        label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, np.isin(label, HAND_LABEL_IDS)
