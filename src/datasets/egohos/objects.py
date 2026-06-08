"""EgoHOS interacting-object unions for Scheme 3 dense segmentation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .evaluation import DEFAULT_DATA_ROOT, EgoHOSEvaluationItem, list_egohos_items


OBJECT_LABEL_IDS = (3, 4, 5, 6, 7, 8)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class EgoHOSObjectDataset(Dataset):
    """EgoHOS frames paired with the union of selected object classes."""

    def __init__(
        self,
        split: str,
        *,
        data_root: Path = DEFAULT_DATA_ROOT,
        image_size: int = 256,
        object_ids: tuple[int, ...] = OBJECT_LABEL_IDS,
        max_samples: int | None = None,
        seed: int = 17,
        shuffle: bool = False,
    ) -> None:
        self.image_size = image_size
        self.object_ids = tuple(object_ids)
        self.items = list_egohos_items((split,), data_root=data_root)
        if shuffle:
            random.Random(seed).shuffle(self.items)
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(item.label_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(item.image_path)
        if label is None:
            raise FileNotFoundError(item.label_path)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        target = np.isin(label, self.object_ids).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        return {
            "image": (tensor - IMAGENET_MEAN) / IMAGENET_STD,
            "target": torch.from_numpy(target),
            "entry": item,
            "dataset": "egohos",
            "source": egohos_source(item),
        }


def egohos_source(item: EgoHOSEvaluationItem) -> str:
    for prefix in ("epic", "ego4d", "thu", "escape", "youtube"):
        if item.identifier.startswith(f"{prefix}_"):
            return prefix
    return item.identifier.split("_", 1)[0].lower()
