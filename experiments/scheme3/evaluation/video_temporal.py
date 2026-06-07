#!/usr/bin/env python3
"""Video temporal and flow-aligned metrics for Scheme 3."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from ego4d.research.util.masks import decode_mask

from dataset_loaders import is_hand_track, load_relations
from utils import mean
from evaluation.metrics import binary_iou


def postprocess_probs(
    probs: dict[int, np.ndarray],
    indices: list[int],
    threshold: float,
    ema_alpha: float,
    keep_threshold: float,
    close_kernel: int,
) -> dict[int, torch.Tensor]:
    probs = bidirectional_ema(probs, indices, ema_alpha) if ema_alpha > 0 else probs
    previous = None
    masks = {}
    for frame_number in indices:
        current = probs[frame_number] >= threshold
        if previous is not None and keep_threshold > 0:
            current = current | (previous & (probs[frame_number] >= keep_threshold))
        if close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            current = cv2.morphologyEx(current.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        masks[frame_number] = torch.from_numpy(current)
        previous = current
    return masks


def bidirectional_ema(probs: dict[int, np.ndarray], indices: list[int], alpha: float) -> dict[int, np.ndarray]:
    forward = one_way_ema(probs, indices, alpha)
    backward = one_way_ema(probs, list(reversed(indices)), alpha)
    return {frame: ((forward[frame] + backward[frame]) * 0.5).astype(np.float32) for frame in indices}


def one_way_ema(probs: dict[int, np.ndarray], indices: list[int], alpha: float) -> dict[int, np.ndarray]:
    output, previous = {}, None
    for frame in indices:
        previous = probs[frame] if previous is None else alpha * previous + (1.0 - alpha) * probs[frame]
        output[frame] = previous.astype(np.float32)
    return output


def load_gt_by_frame(split: str, take_name: str, camera_name: str, start_frame: int, duration_seconds: float, image_size: int) -> dict[int, torch.Tensor]:
    end_frame = start_frame + int(round(duration_seconds * 30.0))
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    for take in load_relations(split).values():
        if str(take.get("take_name") or "") != take_name:
            continue
        for track_id, payload in (take.get("object_masks") or {}).items():
            if is_hand_track(str(track_id)):
                continue
            annotation = (payload.get(camera_name, {}) if isinstance(payload, dict) else {}).get("annotation") or {}
            for frame_text, mask_payload in annotation.items():
                try:
                    frame_number = int(frame_text)
                except ValueError:
                    continue
                if start_frame <= frame_number < end_frame:
                    mask = decode_mask(mask_payload).astype(np.uint8)
                    masks_by_frame.setdefault(frame_number, []).append(cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool))
    return {frame: torch.from_numpy(np.any(np.stack(masks), axis=0)) for frame, masks in sorted(masks_by_frame.items()) if masks}


def evaluate_sparse(pred: dict[int, torch.Tensor], gt: dict[int, torch.Tensor], warp_maps: list[tuple[np.ndarray, np.ndarray] | None], positions: dict[int, int]) -> dict:
    frames = sorted(frame for frame in gt if frame in pred)
    mask_ious = [binary_iou(pred[frame], gt[frame]) for frame in frames]
    pred_raw = pair_ious(pred, frames)
    gt_raw = pair_ious(gt, frames)
    pred_flow = flow_pair_ious(pred, frames, warp_maps, positions)
    gt_flow = flow_pair_ious(gt, frames, warp_maps, positions)
    return {
        "selected_union_mean_iou": mean(mask_ious),
        "selected_temporal_union_iou": mean(pred_raw),
        "gt_temporal_union_iou": mean(gt_raw),
        "selected_flow_temporal_union_iou": mean(pred_flow),
        "gt_flow_temporal_union_iou": mean(gt_flow),
        "flow_temporal_iou_gap_to_gt": mean(pred_flow) - mean(gt_flow),
        "selected_mean_area": mean([float(pred[frame].float().mean()) for frame in frames]),
        "frames": len(frames),
        "temporal_pairs": len(pred_raw),
        "flow_temporal_pairs": len(pred_flow),
    }


def evaluate_full_fps(
    pred: dict[int, torch.Tensor],
    indices: list[int],
    warp_maps: list[tuple[np.ndarray, np.ndarray] | None],
    positions: dict[int, int],
    horizons: tuple[int, ...] = (1,),
) -> dict:
    frames = [frame for frame in indices if frame in pred]
    by_horizon = {}
    for horizon in tuple(sorted({row for row in horizons if row > 0})) or (1,):
        raw_rows = pair_ious(pred, frames, horizon)
        flow_rows = flow_pair_ious(pred, frames, warp_maps, positions, horizon)
        by_horizon[str(horizon)] = {
            "temporal_union_iou": mean(raw_rows),
            "flow_temporal_union_iou": mean(flow_rows),
            "temporal_pairs": len(flow_rows),
        }
    adjacent = by_horizon.get("1") or next(iter(by_horizon.values()))
    return {
        "full_fps_temporal_union_iou": adjacent["temporal_union_iou"],
        "full_fps_flow_temporal_union_iou": adjacent["flow_temporal_union_iou"],
        "full_fps_mean_area": mean([float(pred[frame].float().mean()) for frame in frames]),
        "full_fps_frames": len(frames),
        "full_fps_temporal_pairs": adjacent["temporal_pairs"],
        "full_fps_flow_temporal_by_horizon": by_horizon,
    }


def pair_ious(masks: dict[int, torch.Tensor], frames: list[int], horizon: int = 1) -> list[float]:
    return [
        binary_iou(masks[left], masks[right])
        for left, right in zip(frames[:-horizon], frames[horizon:])
        if torch.logical_or(masks[left], masks[right]).any()
    ]


def flow_pair_ious(
    masks: dict[int, torch.Tensor],
    frames: list[int],
    warp_maps: list[tuple[np.ndarray, np.ndarray] | None],
    positions: dict[int, int],
    horizon: int = 1,
) -> list[float]:
    rows = []
    for left, right in zip(frames[:-horizon], frames[horizon:]):
        if left not in positions or right not in positions:
            continue
        warped = torch.from_numpy(warp_mask_between_positions(masks[left].cpu().numpy().astype(bool), positions[left], positions[right], warp_maps))
        if torch.logical_or(warped, masks[right]).any():
            rows.append(binary_iou(warped, masks[right]))
    return rows


def precompute_flow_warp_maps(frames: list[np.ndarray], image_size: int) -> list[tuple[np.ndarray, np.ndarray] | None]:
    grays = [resize_gray(frame, image_size) for frame in frames]
    if not grays:
        return []
    height, width = grays[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    maps: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    for previous_gray, current_gray in zip(grays[:-1], grays[1:]):
        flow_current_to_previous = cv2.calcOpticalFlowFarneback(current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        maps.append((grid_x + flow_current_to_previous[..., 0], grid_y + flow_current_to_previous[..., 1]))
    return maps


def resize_gray(frame: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), (3, 3), 0)


def warp_mask_between_positions(mask: np.ndarray, source_position: int, target_position: int, warp_maps: list[tuple[np.ndarray, np.ndarray] | None]) -> np.ndarray:
    aligned = mask
    for position in range(source_position + 1, target_position + 1):
        aligned = warp_mask(aligned, warp_maps[position])
    return aligned


def warp_mask(mask: np.ndarray, maps: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if maps is None:
        return mask
    map_x, map_y = maps
    return cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def parse_ints(value: str) -> tuple[int, ...]:
    rows = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not rows or any(row <= 0 for row in rows):
        raise ValueError("Expected a comma-separated list of positive frame horizons")
    return rows
