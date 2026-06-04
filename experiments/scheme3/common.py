#!/usr/bin/env python3
"""Shared helpers for Ego-Exo4D relation-track experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


EGOEXO_ROOT = Path("data/egoexo4d")
OUTPUT_DIR = Path("outputs/experiments/scheme3")


@dataclass(frozen=True)
class RelationFile:
    path: Path
    split: str
    data: dict[str, Any]


def find_relation_files(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*.json"):
        name = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if "relation" in name or "relations" in parts:
            candidates.append(path)
    return sorted(candidates)


def load_relation_files(root: Path) -> list[RelationFile]:
    files = []
    for path in find_relation_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data = unwrap_relations(data)
        if not looks_like_relations(data):
            continue
        files.append(RelationFile(path=path, split=infer_split(path), data=data))
    return files


def looks_like_relations(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    sample = next(iter(data.values()))
    return isinstance(sample, dict) and ("object_masks" in sample or "object_names" in sample)


def unwrap_relations(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("annotations"), dict):
        return data["annotations"]
    return data


def infer_split(path: Path) -> str:
    lower_parts = [part.lower() for part in path.parts]
    for split in ("train", "val", "test"):
        if split in lower_parts or split in path.name.lower():
            return split
    return "unknown"


def iter_takes(files: list[RelationFile]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for relation_file in files:
        for take_uid, take in relation_file.data.items():
            if isinstance(take, dict):
                yield relation_file.split, take_uid, take


def ego_cameras_for_object(object_payload: dict[str, Any]) -> list[str]:
    cameras = []
    for camera_name, camera_payload in object_payload.items():
        if camera_name.startswith("aria") and isinstance(camera_payload, dict):
            if camera_payload.get("annotation"):
                cameras.append(camera_name)
    return sorted(cameras)


def mask_frame_numbers(camera_payload: dict[str, Any]) -> list[int]:
    annotation = camera_payload.get("annotation", {})
    frames = []
    for key in annotation.keys():
        try:
            frames.append(int(key))
        except ValueError:
            continue
    return sorted(frames)


def object_track_rows(split: str, take_uid: str, take: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    masks = take.get("object_masks", {}) or {}
    video_fps = take_video_fps(take)
    for track_id, object_payload in masks.items():
        if not isinstance(object_payload, dict):
            continue
        for camera_name in ego_cameras_for_object(object_payload):
            camera_payload = object_payload[camera_name]
            frames = mask_frame_numbers(camera_payload)
            if not frames:
                continue
            rows.append(
                {
                    "split": split,
                    "take_uid": take_uid,
                    "take_name": take.get("take_name"),
                    "scenario": take.get("scenario"),
                    "track_id": track_id,
                    "track_kind": track_kind(track_id),
                    "camera_name": camera_name,
                    "annotation_fps": camera_payload.get("annotation_fps"),
                    "video_fps": video_fps,
                    "frame_count": len(frames),
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "frames": frames,
                }
            )
    return rows


def track_kind(track_id: str) -> str:
    if is_hand_track(track_id):
        return "hand"
    return "object"


def is_hand_track(track_id: str) -> bool:
    normalized = track_id.lower().replace("_", " ").strip()
    normalized = " ".join(normalized.split())
    return normalized.startswith("left hand") or normalized.startswith("right hand")


def object_name_box_rows(take: dict[str, Any]) -> list[dict[str, Any]]:
    object_names = take.get("object_names", {}) or {}
    rows = []
    for box in object_names.get("annotation", []) or []:
        if not isinstance(box, dict):
            continue
        track_id = str(box.get("track_id") or "")
        try:
            frame_number = int(box["frameNumber"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "track_id": track_id,
                "track_kind": track_kind(track_id),
                "frame_number": frame_number,
                "x": float(box.get("x", 0.0)),
                "y": float(box.get("y", 0.0)),
                "width": float(box.get("width", 0.0)),
                "height": float(box.get("height", 0.0)),
                "original_width": float(box.get("original_width", 0.0) or 0.0),
                "original_height": float(box.get("original_height", 0.0) or 0.0),
                "interpolated": bool(box.get("interpolated", False)),
            }
        )
    return rows


def take_video_fps(take: dict[str, Any]) -> float:
    object_names = take.get("object_names", {}) or {}
    try:
        fps = float(object_names.get("annotation_fps"))
        if fps > 0:
            return fps
    except (TypeError, ValueError):
        pass
    return 30.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
