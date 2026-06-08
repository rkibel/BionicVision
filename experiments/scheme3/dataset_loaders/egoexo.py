#!/usr/bin/env python3
"""Ego-Exo4D datasets for Scheme 3."""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_loaders.common import (
    DEFAULT_EGOEXO_CAMERA,
    MEAN,
    STD,
    FrameEntry,
    augment_image_masks,
    decode_track_mask,
    is_hand_track,
    load_image_feature,
    load_relations,
    read_video_frame,
    take_video_path,
)


class EgoExoMaskDataset(Dataset):
    """Annotated Ego-Exo frames with object-union masks."""

    def __init__(
        self,
        split: str,
        camera_name: str = DEFAULT_EGOEXO_CAMERA,
        image_size: int = 256,
        max_samples: int = 0,
        seed: int = 17,
        exclude_target_window: bool = True,
        target_only: bool = False,
        target_take: str = "",
        target_start: int = -1,
        target_end: int = -1,
        shuffle: bool = True,
        preserve_order: bool = False,
        max_objects: int = 24,
        min_area_ratio: float = 0.00035,
        augment: bool = False,
        image_feature_mode: str = "none",
        image_feature_cache=None,
    ) -> None:
        self.split = split
        self.camera_name = camera_name
        self.image_size = image_size
        self.max_objects = max_objects
        self.min_area_ratio = min_area_ratio
        self.augment = augment
        self.image_feature_mode = image_feature_mode
        self.image_feature_cache = image_feature_cache
        self.relations = load_relations(split)
        self.entries = build_frame_entries(self.relations, split, camera_name, exclude_target_window, target_only, target_take, target_start, target_end)
        if shuffle:
            random.Random(seed).shuffle(self.entries)
        if max_samples > 0:
            self.entries = self.entries[:max_samples]
        if preserve_order:
            self.entries = sorted(self.entries, key=lambda item: (item.take_name, item.frame_number))
        self._captures = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.sample(self.entries[index])

    def sample(self, entry: FrameEntry) -> dict[str, Any]:
        take = self.relations[entry.take_uid]
        frame = read_video_frame(self._captures, take_video_path(entry.take_name, entry.camera_name), entry.frame_number, self.image_size)
        masks = []
        for track_id in entry.object_tracks:
            mask = decode_track_mask(take, track_id, entry.camera_name, entry.frame_number, self.image_size)
            if mask is not None and mask.mean() >= self.min_area_ratio:
                masks.append(mask)
        masks = sorted(masks, key=lambda item: float(item.sum()), reverse=True)[: self.max_objects]
        mask_array = np.stack(masks).astype(np.float32) if masks else np.zeros((0, self.image_size, self.image_size), dtype=np.float32)
        identity = f"{take_video_path(entry.take_name, entry.camera_name)}|{entry.frame_number}"
        image_feature = load_image_feature(self.image_feature_cache, self.image_feature_mode, "egoexo", identity, self.image_size)
        if self.augment:
            augmented = np.concatenate([mask_array, image_feature[None]], axis=0) if image_feature is not None else mask_array
            frame, augmented = augment_image_masks(frame, augmented)
            mask_array, image_feature = (augmented[:-1], augmented[-1]) if image_feature is not None else (augmented, None)
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return {"image": (image - MEAN) / STD, "image_feature": torch.from_numpy(image_feature[None]) if image_feature is not None else None, "target_masks": torch.from_numpy(mask_array), "entry": entry, "dataset": "egoexo", "source": "egoexo"}

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


class EgoExoFlowPairDataset(Dataset):
    """Annotated frames paired with nearby raw video frames for temporal loss."""

    def __init__(
        self,
        split: str,
        camera_name: str = DEFAULT_EGOEXO_CAMERA,
        image_size: int = 256,
        max_samples: int = 0,
        seed: int = 17,
        exclude_target_window: bool = True,
        target_take: str = "",
        target_start: int = -1,
        target_end: int = -1,
        frame_offsets: tuple[int, ...] = (1, -1, 2, -2),
        image_feature_mode: str = "none",
        image_feature_cache=None,
    ) -> None:
        self.base = EgoExoMaskDataset(
            split,
            camera_name,
            image_size,
            max_samples=0,
            seed=seed,
            exclude_target_window=exclude_target_window,
            target_take=target_take,
            target_start=target_start,
            target_end=target_end,
            shuffle=False,
            image_feature_mode=image_feature_mode,
            image_feature_cache=image_feature_cache,
        )
        self.pairs = build_flow_pairs(self.base.entries, frame_offsets)
        random.Random(seed).shuffle(self.pairs)
        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry, neighbor = self.pairs[index]
        left = self.base.sample(entry)
        frame = read_video_frame(self.base._captures, take_video_path(entry.take_name, entry.camera_name), neighbor, self.base.image_size)
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        identity = f"{take_video_path(entry.take_name, entry.camera_name)}|{neighbor}"
        feature = load_image_feature(self.base.image_feature_cache, self.base.image_feature_mode, "egoexo", identity, self.base.image_size)
        return {"left": left, "right_image": (image - MEAN) / STD, "right_image_feature": torch.from_numpy(feature[None]) if feature is not None else None, "right_frame_number": int(neighbor)}

    def close(self) -> None:
        self.base.close()


def build_frame_entries(
    relations: dict[str, Any],
    split: str,
    camera_name: str,
    exclude_target_window: bool,
    target_only: bool,
    target_take: str,
    target_start: int,
    target_end: int,
) -> list[FrameEntry]:
    entries = []
    for take_uid, take in relations.items():
        take_name = str(take.get("take_name") or "")
        if not take_video_path(take_name, camera_name).exists():
            continue
        by_frame: dict[int, dict[str, list[str]]] = {}
        for track_id, payload in (take.get("object_masks") or {}).items():
            annotation = (payload.get(camera_name, {}) if isinstance(payload, dict) else {}).get("annotation") or {}
            kind = "hand_tracks" if is_hand_track(str(track_id)) else "object_tracks"
            for frame_text in annotation:
                try:
                    frame_number = int(frame_text)
                except ValueError:
                    continue
                has_target = bool(target_take) and target_start >= 0 and target_end > target_start
                in_target = has_target and take_name == target_take and target_start <= frame_number < target_end
                if target_only and not in_target:
                    continue
                if exclude_target_window and in_target:
                    continue
                by_frame.setdefault(frame_number, {"object_tracks": [], "hand_tracks": []})[kind].append(str(track_id))
        for frame_number, tracks in by_frame.items():
            if tracks["object_tracks"]:
                entries.append(FrameEntry(split, str(take_uid), take_name, camera_name, frame_number, tuple(sorted(tracks["object_tracks"])), tuple(sorted(tracks["hand_tracks"]))))
    return sorted(entries, key=lambda item: (item.take_name, item.frame_number))


def build_flow_pairs(entries: list[FrameEntry], offsets: tuple[int, ...]) -> list[tuple[FrameEntry, int]]:
    frame_counts: dict[tuple[str, str], int] = {}
    pairs = []
    for entry in entries:
        key = (entry.take_name, entry.camera_name)
        if key not in frame_counts:
            capture = cv2.VideoCapture(str(take_video_path(*key)))
            frame_counts[key] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
            capture.release()
        for offset in offsets:
            neighbor = entry.frame_number + int(offset)
            if offset and neighbor >= 0 and (frame_counts[key] <= 0 or neighbor < frame_counts[key]):
                pairs.append((entry, neighbor))
    return pairs
