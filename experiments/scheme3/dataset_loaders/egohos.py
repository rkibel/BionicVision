#!/usr/bin/env python3
"""EgoHOS dataset for Scheme 3."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loaders.common import (
    EGOHOS_ROOT,
    MEAN,
    STD,
    EgoHOSFrameEntry,
    augment_image_masks,
    egohos_image_path,
    egohos_source,
    read_egohos_label,
    read_rgb_image,
)


class EgoHOSMaskDataset(Dataset):
    """EgoHOS frames with interacting-object labels as dense object masks."""

    OBJECT_IDS = (3, 4, 5, 6, 7, 8)

    def __init__(
        self,
        split: str = "train",
        root: Path = EGOHOS_ROOT,
        image_size: int = 256,
        max_samples: int = 0,
        seed: int = 17,
        sources: tuple[str, ...] = (),
        balance_sources: bool = False,
        min_area_ratio: float = 0.00035,
        shuffle: bool = True,
        preserve_order: bool = False,
        augment: bool = False,
        object_ids: tuple[int, ...] = OBJECT_IDS,
        filter_min_object_area_ratio: float = 0.0,
    ) -> None:
        self.split = split
        self.root = Path(root)
        self.image_size = image_size
        self.min_area_ratio = min_area_ratio
        self.augment = augment
        self.object_ids = tuple(int(row) for row in object_ids)
        self.sources = tuple(source.strip().lower() for source in sources if source.strip())
        self.entries = build_egohos_entries(self.root, split, self.sources)
        if filter_min_object_area_ratio > 0:
            self.entries = [entry for entry in self.entries if egohos_object_area(entry, self.image_size, self.object_ids) >= filter_min_object_area_ratio]
        if balance_sources and max_samples > 0:
            self.entries = balance_egohos_sources(self.entries, max_samples, seed)
        elif shuffle:
            random.Random(seed).shuffle(self.entries)
        if max_samples > 0 and not balance_sources:
            self.entries = self.entries[:max_samples]
        if preserve_order:
            self.entries = sorted(self.entries, key=lambda item: (item.source, item.frame_id))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image = read_rgb_image(entry.image_path, self.image_size)
        label = read_egohos_label(entry.label_path, self.image_size)
        object_mask = np.isin(label, self.object_ids).astype(np.float32)
        mask_array = object_mask[None] if object_mask.mean() >= self.min_area_ratio else np.zeros((0, self.image_size, self.image_size), dtype=np.float32)
        if self.augment:
            image, mask_array = augment_image_masks(image, mask_array)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"image": (image_t - MEAN) / STD, "target_masks": torch.from_numpy(mask_array), "entry": entry, "dataset": "egohos", "source": entry.source}


def build_egohos_entries(root: Path, split: str, sources: tuple[str, ...]) -> list[EgoHOSFrameEntry]:
    image_dir = root / split / "image"
    label_dir = root / split / "label"
    if not image_dir.exists() or not label_dir.exists():
        return []
    entries = []
    for label_path in sorted(label_dir.glob("*.png")):
        source = egohos_source(label_path.stem)
        if sources and source not in sources:
            continue
        image_path = egohos_image_path(image_dir, label_path.stem)
        if image_path is not None:
            entries.append(EgoHOSFrameEntry(split, image_path, label_path, source, label_path.stem))
    return entries


def egohos_object_area(entry: EgoHOSFrameEntry, image_size: int, object_ids: tuple[int, ...]) -> float:
    return float(np.isin(read_egohos_label(entry.label_path, image_size), object_ids).mean())


def balance_egohos_sources(entries: list[EgoHOSFrameEntry], max_samples: int, seed: int) -> list[EgoHOSFrameEntry]:
    by_source: dict[str, list[EgoHOSFrameEntry]] = {}
    rng = random.Random(seed)
    for entry in entries:
        by_source.setdefault(entry.source, []).append(entry)
    originals = {source: list(rows) for source, rows in by_source.items()}
    for rows in by_source.values():
        rng.shuffle(rows)
    selected = []
    sources = sorted(by_source)
    while len(selected) < max_samples and sources:
        for source in sources:
            rows = by_source[source]
            if not rows:
                rows.extend(originals[source])
                rng.shuffle(rows)
            if rows and len(selected) < max_samples:
                selected.append(rows.pop())
    rng.shuffle(selected)
    return selected
