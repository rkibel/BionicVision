"""Minimal EPIC-KITCHENS VISOR annotation parsing for hand evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VisorObject:
    name: str
    segments: tuple[tuple[tuple[float, float], ...], ...]
    annotation_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class VisorFrame:
    video_id: str
    frame_name: str
    frame_index: int
    annotations: tuple[VisorObject, ...]


def load_visor_annotations(path: str | Path) -> list[VisorFrame]:
    """Load the frame and polygon fields needed for hand evaluation."""

    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotation_size = _annotation_size(data)
    frames = []
    for datapoint in sorted(data.get("video_annotations", []), key=_image_sort_key):
        image = datapoint.get("image", {})
        frame_name = str(image.get("name", ""))
        image_path = str(image.get("image_path", ""))
        video_id = str(image.get("video") or _infer_video_id(frame_name, image_path, json_path))
        annotations = tuple(
            VisorObject(
                name=str(obj.get("name", obj.get("class", "unknown"))),
                segments=tuple(tuple(tuple(point) for point in polygon) for polygon in obj.get("segments", [])),
                annotation_size=annotation_size,
            )
            for obj in datapoint.get("annotations", [])
        )
        frames.append(VisorFrame(video_id, frame_name, parse_frame_index(frame_name), annotations))
    return frames


def rasterize_object(obj: VisorObject, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize one VISOR polygon annotation into a binary mask."""

    height, width = shape
    scale_x = scale_y = 1.0
    if obj.annotation_size is not None:
        source_width, source_height = obj.annotation_size
        if source_width > 0 and source_height > 0:
            scale_x = width / source_width
            scale_y = height / source_height

    polygons = []
    for segment in obj.segments:
        polygon = np.asarray(segment, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3:
            continue
        polygon = polygon.copy()
        polygon[:, 0] = np.clip(polygon[:, 0] * scale_x, 0, max(width - 1, 0))
        polygon[:, 1] = np.clip(polygon[:, 1] * scale_y, 0, max(height - 1, 0))
        polygons.append(np.rint(polygon).astype(np.int32))

    mask = np.zeros(shape, dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def parse_frame_index(frame_name: str) -> int:
    digits = "".join(ch for ch in Path(frame_name).stem.split("_")[-1] if ch.isdigit())
    return int(digits) if digits else -1


def _annotation_size(data: dict[str, Any]) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\s*x\s*(\d+)", json.dumps(data.get("info", {})))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _image_sort_key(datapoint: dict[str, Any]) -> tuple[str, int]:
    image = datapoint.get("image", {})
    return str(image.get("image_path", "")), parse_frame_index(str(image.get("name", "")))


def _infer_video_id(frame_name: str, image_path: str, json_path: Path) -> str:
    if image_path:
        return image_path.split("/")[0]
    if "_" in frame_name:
        return "_".join(frame_name.split("_")[:2])
    return json_path.stem
