"""Raw and optical-flow-corrected temporal mask consistency for dense schemes."""

from __future__ import annotations

import cv2
import numpy as np

from .metrics import mask_metrics, mean


DEFAULT_HORIZONS = (1, 2, 5, 10, 15, 30)


def evaluate_temporal_masks(
    frames_bgr: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    flow_size: int = 256,
) -> dict[str, object]:
    if len(frames_bgr) != len(masks):
        raise ValueError(f"Frame/mask count mismatch: {len(frames_bgr)} != {len(masks)}")
    resized_masks = [
        cv2.resize(mask.astype(np.uint8), (flow_size, flow_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in masks
    ]
    warp_maps = precompute_flow_warp_maps(frames_bgr, flow_size)
    by_horizon = {}
    for horizon in sorted({value for value in horizons if value > 0}):
        raw, corrected = [], []
        for left in range(0, len(resized_masks) - horizon):
            right = left + horizon
            if np.any(resized_masks[left] | resized_masks[right]):
                raw.append(mask_metrics(resized_masks[left], resized_masks[right])["iou"])
            warped = warp_mask_between_positions(resized_masks[left], left, right, warp_maps)
            if np.any(warped | resized_masks[right]):
                corrected.append(mask_metrics(warped, resized_masks[right])["iou"])
        by_horizon[str(horizon)] = {
            "raw_pairs": len(raw),
            "flow_corrected_pairs": len(corrected),
            "raw_temporal_iou": mean(raw),
            "flow_corrected_temporal_iou": mean(corrected),
            "raw_prediction_iou": mean(raw),
            "flow_warped_prediction_iou": mean(corrected),
        }
    return {
        "frames": len(frames_bgr),
        "flow_size": flow_size,
        "metric": "IoU between predicted masks at t+h and either the raw prediction at t or the prediction at t warped to t+h by optical flow.",
        "horizons": by_horizon,
    }


def precompute_flow_warp_maps(frames_bgr: list[np.ndarray], flow_size: int) -> list[tuple[np.ndarray, np.ndarray] | None]:
    grays = [resize_gray(frame, flow_size) for frame in frames_bgr]
    if not grays:
        return []
    x, y = np.meshgrid(np.arange(flow_size, dtype=np.float32), np.arange(flow_size, dtype=np.float32))
    maps: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    for previous, current in zip(grays[:-1], grays[1:]):
        flow_current_to_previous = cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        maps.append((x + flow_current_to_previous[..., 0], y + flow_current_to_previous[..., 1]))
    return maps


def resize_gray(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def warp_mask_between_positions(
    mask: np.ndarray,
    source_position: int,
    target_position: int,
    warp_maps: list[tuple[np.ndarray, np.ndarray] | None],
) -> np.ndarray:
    aligned = mask.astype(bool)
    for position in range(source_position + 1, target_position + 1):
        maps = warp_maps[position]
        if maps is not None:
            aligned = cv2.remap(
                aligned.astype(np.uint8),
                maps[0],
                maps[1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
    return aligned
