"""PyTorch datasets for Ego-Exo4D object unions and temporal pairs."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .relations import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    EgoExoFrameEntry,
    build_frame_entries,
    decode_track_mask,
    take_video_path,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class EgoExoObjectDataset(Dataset):
    """Annotated egocentric frames paired with unions of non-hand object masks."""

    def __init__(
        self,
        split: str,
        *,
        data_root: Path = DEFAULT_DATA_ROOT,
        camera_name: str = DEFAULT_CAMERA,
        image_size: int = 256,
        max_samples: int | None = None,
        seed: int = 17,
        shuffle: bool = False,
        min_area_ratio: float = 0.00035,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.min_area_ratio = min_area_ratio
        self.relations, self.entries = build_frame_entries(split, camera_name=camera_name, data_root=self.data_root)
        if shuffle:
            random.Random(seed).shuffle(self.entries)
        if max_samples is not None:
            self.entries = self.entries[:max_samples]
        self._captures: dict[Path, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.sample(self.entries[index])

    def sample(self, entry: EgoExoFrameEntry) -> dict[str, Any]:
        image = self.read_frame(entry, entry.frame_number)
        take = self.relations[entry.take_uid]
        masks = [
            mask
            for track_id in entry.object_tracks
            if (mask := decode_track_mask(take, track_id, entry.camera_name, entry.frame_number, self.image_size)) is not None
            and mask.mean() >= self.min_area_ratio
        ]
        target = np.any(np.stack(masks), axis=0) if masks else np.zeros((self.image_size, self.image_size), dtype=bool)
        return {
            "image": normalize_rgb(image),
            "target": torch.from_numpy(target.astype(np.float32)),
            "entry": entry,
            "dataset": "egoexo",
            "source": "egoexo",
        }

    def read_frame(self, entry: EgoExoFrameEntry, frame_number: int) -> np.ndarray:
        path = take_video_path(entry.take_name, entry.camera_name, data_root=self.data_root)
        capture = self._captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise FileNotFoundError(path)
            self._captures[path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_number} from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


class EgoExoFlowPairDataset(Dataset):
    """Annotated Ego-Exo frames paired with nearby frames for temporal training."""

    def __init__(
        self,
        split: str,
        *,
        offsets: tuple[int, ...] = (1, -1, 2, -2, 5, -5, 10, -10),
        max_samples: int | None = None,
        seed: int = 17,
        **dataset_kwargs,
    ) -> None:
        self.base = EgoExoObjectDataset(split, shuffle=False, max_samples=None, **dataset_kwargs)
        self.pairs = build_flow_pairs(self.base, offsets)
        random.Random(seed).shuffle(self.pairs)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry, neighbor = self.pairs[index]
        left = self.base.sample(entry)
        return {
            "left_image": left["image"],
            "right_image": normalize_rgb(self.base.read_frame(entry, neighbor)),
            "target": left["target"],
            "entry": entry,
            "right_frame_number": neighbor,
        }

    def close(self) -> None:
        self.base.close()


def build_flow_pairs(dataset: EgoExoObjectDataset, offsets: tuple[int, ...]) -> list[tuple[EgoExoFrameEntry, int]]:
    frame_counts: dict[tuple[str, str], int] = {}
    pairs = []
    for entry in dataset.entries:
        key = (entry.take_name, entry.camera_name)
        if key not in frame_counts:
            path = take_video_path(*key, data_root=dataset.data_root)
            capture = cv2.VideoCapture(str(path))
            frame_counts[key] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
            capture.release()
        for offset in offsets:
            neighbor = entry.frame_number + offset
            if offset and 0 <= neighbor < frame_counts[key]:
                pairs.append((entry, neighbor))
    return pairs


def normalize_rgb(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD
