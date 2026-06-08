"""Ego-Exo4D relation annotations and mask decoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ego4d.research.util.masks import decode_mask


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data/egoexo4d"
DEFAULT_CAMERA = "aria01_214-1"


@dataclass(frozen=True)
class EgoExoFrameEntry:
    split: str
    take_uid: str
    take_name: str
    camera_name: str
    frame_number: int
    object_tracks: tuple[str, ...]


def load_relations(split: str, *, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    path = data_root / "annotations" / f"relations_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", payload)
    if not isinstance(annotations, dict):
        raise RuntimeError(f"Unexpected relation file format: {path}")
    return annotations


def take_video_path(
    take_name: str,
    camera_name: str = DEFAULT_CAMERA,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    return data_root / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448" / f"{camera_name}.mp4"


def is_hand_track(track_id: str) -> bool:
    normalized = " ".join(track_id.lower().replace("_", " ").split())
    return normalized.startswith("left hand") or normalized.startswith("right hand")


def decode_track_mask(
    take: dict[str, Any],
    track_id: str,
    camera_name: str,
    frame_number: int,
    image_size: int,
) -> np.ndarray | None:
    payload = (take.get("object_masks") or {}).get(track_id, {})
    annotation = (payload.get(camera_name, {}) if isinstance(payload, dict) else {}).get("annotation") or {}
    encoded = annotation.get(str(frame_number))
    if encoded is None:
        return None
    mask = decode_mask(encoded).astype(np.uint8)
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)


def build_frame_entries(
    split: str,
    *,
    camera_name: str = DEFAULT_CAMERA,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[dict[str, Any], list[EgoExoFrameEntry]]:
    relations = load_relations(split, data_root=data_root)
    entries = []
    for take_uid, take in relations.items():
        take_name = str(take.get("take_name") or "")
        if not take_video_path(take_name, camera_name, data_root=data_root).exists():
            continue
        by_frame: dict[int, list[str]] = {}
        for track_id, payload in (take.get("object_masks") or {}).items():
            if is_hand_track(str(track_id)):
                continue
            annotation = (payload.get(camera_name, {}) if isinstance(payload, dict) else {}).get("annotation") or {}
            for frame_text in annotation:
                try:
                    frame_number = int(frame_text)
                except ValueError:
                    continue
                by_frame.setdefault(frame_number, []).append(str(track_id))
        entries.extend(
            EgoExoFrameEntry(split, str(take_uid), take_name, camera_name, frame_number, tuple(sorted(track_ids)))
            for frame_number, track_ids in by_frame.items()
            if track_ids
        )
    return relations, sorted(entries, key=lambda item: (item.take_name, item.frame_number))
