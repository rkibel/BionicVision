#!/usr/bin/env python3
"""Run a non-oracle 30s hand/object tracking pipeline on an egocentric video."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ego4d.research.util.masks import decode_mask

ROOT = Path(__file__).resolve().parents[2]
GROUNDED_SAM2 = ROOT / "external/model_sources/segmentation/Grounded-SAM-2"
if str(GROUNDED_SAM2) not in sys.path:
    sys.path.insert(0, str(GROUNDED_SAM2))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from torchvision.ops import box_convert  # noqa: E402

from common import EGOEXO_ROOT, OUTPUT_DIR, load_relation_files, write_json  # noqa: E402


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class Track:
    track_id: int
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    seed_score: float = 0.0
    seed_source: str = "sam"
    seed_label: str = ""
    seed_confidence: float = 0.0

    @property
    def first_frame(self) -> int:
        return min(self.masks)

    @property
    def last_frame(self) -> int:
        return max(self.masks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "unsupervised_30s_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=5, help="Video-frame stride; 5 gives 6 FPS on 30 FPS videos.")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--chunk-frames", type=int, default=30)
    parser.add_argument("--max-seeds", type=int, default=14)
    parser.add_argument("--max-display-objects", type=int, default=0, help="Optional safety cap after thresholding; 0 means no cap.")
    parser.add_argument("--object-score-threshold", type=float, default=0.5)
    parser.add_argument("--seed-mode", choices=("interaction", "coverage", "hybrid"), default="hybrid")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.00055)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.24)
    parser.add_argument("--area-prior-ratio", type=float, default=0.022)
    parser.add_argument("--coverage-area-penalty-weight", type=float, default=0.50)
    parser.add_argument("--small-object-boost", type=float, default=0.0)
    parser.add_argument("--proposal-nms-iou", type=float, default=0.72)
    parser.add_argument("--score-model", type=Path)
    parser.add_argument("--load-track-cache", type=Path, help="Reuse a saved non-oracle detector/SAM track cache.")
    parser.add_argument("--track-gap-fill-frames", type=int, default=0, help="Carry scored track masks across short missing-frame gaps.")
    parser.add_argument("--save-score-cache", action="store_true")
    parser.add_argument("--save-track-cache", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--calibrate-threshold-with-gt", action="store_true")
    parser.add_argument("--threshold-grid", default="0.05:1.80:0.05")
    parser.add_argument("--calibration-temporal-weight", type=float, default=0.10)
    parser.add_argument("--calibration-count-penalty", type=float, default=0.015)
    parser.add_argument("--sam-points-per-side", type=int, default=24)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.70)
    parser.add_argument("--sam-stability-thresh", type=float, default=0.80)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=96)
    parser.add_argument("--dino-prompt", default="")
    parser.add_argument("--dino-box-threshold", type=float, default=0.22)
    parser.add_argument("--dino-text-threshold", type=float, default=0.18)
    parser.add_argument("--dino-max-boxes", type=int, default=12)
    parser.add_argument("--dino-checkpoint", type=Path, default=GROUNDED_SAM2 / "gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino-config", type=Path, default=GROUNDED_SAM2 / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--carryover-tracks", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--sam2-checkpoint", type=Path, default=GROUNDED_SAM2 / "checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--hand-checkpoint", type=Path, default=ROOT / "outputs/experiments/scheme3/hand_segmentor/best.pt")
    parser.add_argument("--hand-threshold", type=float)
    parser.add_argument("--take-uid")
    parser.add_argument("--camera-name", default="aria01_214-1")
    parser.add_argument("--gt-window-start", type=int)
    parser.add_argument("--gt-window-end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    set_reproducible_seed(args.seed)

    run_name = args.run_name or f"{args.input.stem}_f{args.start_frame}_s{args.stride}"
    run_dir = args.output_dir / run_name
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames, source_fps, original_indices = extract_frames(args.input, args.start_frame, args.duration_seconds, args.stride, args.width, args.height)
    sampled_fps = source_fps / args.stride
    frames_dir = run_dir / "frames"
    write_frames(frames, frames_dir)

    hand_model, hand_size, threshold = load_hand_model(args.hand_checkpoint, args.device)
    if args.hand_threshold is not None:
        threshold = args.hand_threshold
    hand_probs, hand_masks, hand_proximity = predict_hands(hand_model, frames, hand_size, threshold, args.device)

    if args.load_track_cache:
        tracks, per_frame_scores = load_track_cache(args.load_track_cache, len(frames), frames[0].shape[:2])
    else:
        sam_image_model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
        mask_generator = SAM2AutomaticMaskGenerator(
            sam_image_model,
            points_per_side=args.sam_points_per_side,
            points_per_batch=80,
            pred_iou_thresh=args.sam_pred_iou_thresh,
            stability_score_thresh=args.sam_stability_thresh,
            min_mask_region_area=args.sam_min_mask_region_area,
            multimask_output=True,
        )
        video_predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
        grounding_model = load_grounding_model(args) if args.dino_prompt else None

        tracks = run_chunked_sam2_tracking(frames, frames_dir, hand_masks, hand_proximity, mask_generator, video_predictor, grounding_model, args, run_dir)
        per_frame_scores = score_tracks(tracks, frames, hand_masks, hand_proximity, len(frames))
    if args.score_model:
        model_threshold = apply_score_model(args.score_model, tracks, per_frame_scores, args.device)
        if model_threshold is not None:
            args.object_score_threshold = float(model_threshold)
    if args.track_gap_fill_frames > 0:
        fill_short_track_gaps(tracks, per_frame_scores, args.track_gap_fill_frames, len(frames))
    threshold_calibration = None
    if args.calibrate_threshold_with_gt:
        require_gt_args(args)
        threshold_calibration = calibrate_object_threshold(args, tracks, per_frame_scores, original_indices, frames[0].shape[:2])
        args.object_score_threshold = float(threshold_calibration["best_threshold"])
    overlay_path = None
    contact_sheet_path = None
    if not args.skip_render:
        overlay_frames = render_overlays(frames, hand_masks, tracks, per_frame_scores, original_indices, args)
        overlay_path = run_dir / "overlay.mp4"
        contact_sheet_path = run_dir / "contact_sheet.jpg"
        write_video(overlay_path, overlay_frames, sampled_fps)
        transcode_for_browser(overlay_path)
        write_contact_sheet(contact_sheet_path, overlay_frames)

    evaluation = None
    if args.take_uid and args.gt_window_start is not None and args.gt_window_end is not None:
        evaluation = evaluate_against_egoexo(args, tracks, per_frame_scores, original_indices, frames[0].shape[:2])
    score_cache_path = None
    if args.save_score_cache:
        require_gt_args(args)
        score_cache_path = save_score_cache(args, tracks, per_frame_scores, original_indices, frames[0].shape[:2], run_dir)
    track_cache_path = None
    if args.save_track_cache:
        require_gt_args(args)
        track_cache_path = save_track_cache(args, tracks, per_frame_scores, original_indices, frames[0].shape[:2], run_dir)

    manifest = {
        "input": str(args.input),
        "frames": len(frames),
        "source_fps": source_fps,
        "sampled_fps": sampled_fps,
        "start_frame": args.start_frame,
        "duration_seconds": args.duration_seconds,
        "stride": args.stride,
        "track_count": len(tracks),
        "object_score_threshold": args.object_score_threshold,
        "track_gap_fill_frames": args.track_gap_fill_frames,
        "mean_selected_objects": float(np.mean([len(selected_object_ids(per_frame_scores, i, args.object_score_threshold, args.max_display_objects)) for i in range(len(frames))])),
        "hand_threshold": threshold,
        "threshold_calibration": threshold_calibration,
        "score_model": str(args.score_model) if args.score_model else None,
        "load_track_cache": str(args.load_track_cache.resolve()) if args.load_track_cache else None,
        "score_cache": str(score_cache_path.resolve()) if score_cache_path else None,
        "track_cache": str(track_cache_path.resolve()) if track_cache_path else None,
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "dino_prompt": args.dino_prompt,
        "seed": args.seed,
        "overlay_video": str(overlay_path.resolve()) if overlay_path else None,
        "contact_sheet": str(contact_sheet_path.resolve()) if contact_sheet_path else None,
        "evaluation": evaluation,
    }
    write_json(run_dir / "manifest.json", manifest)
    if not args.keep_intermediates:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(run_dir / "chunks", ignore_errors=True)
    print(json.dumps(manifest, indent=2))


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_frames(input_path: Path, start_frame: int, duration_seconds: float, stride: int, width: int, height: int):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_to_scan = int(round(duration_seconds * fps))
    frames = []
    original_indices = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for offset in range(total_to_scan):
            ok, bgr = capture.read()
            if not ok:
                break
            original_index = start_frame + offset
            if offset % stride != 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            original_indices.append(original_index)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("No frames extracted")
    return frames, fps, original_indices


def write_frames(frames: list[np.ndarray], frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{index:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def load_hand_model(checkpoint_path: Path, device: str):
    import segmentation_models_pytorch as smp

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    if model_name == "smp-unetpp-efficientnet-b4":
        model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
    elif model_name == "smp-unetpp-resnet101":
        model = smp.UnetPlusPlus(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=1)
    elif model_name == "smp-deeplabv3plus-resnet101":
        model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unsupported hand model: {model_name}")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    size = parse_size(checkpoint["image_size"])
    return model, size, float(checkpoint["threshold"])


def parse_size(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)


@torch.inference_mode()
def predict_hands(model, frames: list[np.ndarray], size: tuple[int, int], threshold: float, device: str):
    probs = []
    masks = []
    proximities = []
    batch = []
    batch_shapes = []
    for frame in frames:
        resized = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image = ((image - MEAN) / STD).unsqueeze(0)
        batch.append(image)
        batch_shapes.append(frame.shape[:2])
        if len(batch) == 4:
            flush_hand_batch(model, batch, batch_shapes, probs, masks, proximities, threshold, device)
            batch, batch_shapes = [], []
    if batch:
        flush_hand_batch(model, batch, batch_shapes, probs, masks, proximities, threshold, device)
    return probs, masks, proximities


def flush_hand_batch(model, batch, batch_shapes, probs, masks, proximities, threshold: float, device: str) -> None:
    images = torch.cat(batch, dim=0).to(device)
    logits = model(images)
    if isinstance(logits, dict):
        logits = logits["out"]
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    for prob_small, shape in zip(probabilities, batch_shapes):
        prob = cv2.resize(prob_small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask = prob >= threshold
        proximity = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        probs.append(prob)
        masks.append(mask)
        proximities.append(proximity)


def run_chunked_sam2_tracking(
    frames: list[np.ndarray],
    frames_dir: Path,
    hand_masks: list[np.ndarray],
    hand_proximity: list[np.ndarray],
    mask_generator,
    video_predictor,
    grounding_model,
    args,
    run_dir: Path,
) -> list[Track]:
    tracks: list[Track] = []
    next_track_id = 1
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    for chunk_start in range(0, len(frames), args.chunk_frames):
        chunk_end = min(len(frames), chunk_start + args.chunk_frames)
        carryover = carryover_seed_masks(tracks, chunk_start, limit=args.max_seeds // 2) if args.carryover_tracks else []
        new_seed_limit = max(args.max_seeds - len(carryover), args.max_seeds // 2)
        seed_records = seed_masks(
            frames[chunk_start],
            hand_masks[chunk_start],
            hand_proximity[chunk_start],
            mask_generator,
            grounding_model,
            new_seed_limit,
            args,
            suppress=carryover,
        )
        seed_records = carryover + seed_records
        if not seed_records:
            continue
        chunk_dir = materialize_chunk(frames_dir, chunks_dir, chunk_start, chunk_end)
        chunk_segments = propagate_chunk(video_predictor, chunk_dir, seed_records, args.device)
        for local_id, seed in enumerate(seed_records, start=1):
            masks = {
                chunk_start + local_frame: mask.astype(bool)
                for local_frame, obj_masks in chunk_segments.items()
                for obj_id, mask in obj_masks.items()
                if obj_id == local_id and np.count_nonzero(mask) > 32
            }
            if len(masks) < max(3, (chunk_end - chunk_start) // 4):
                continue
            match = track_by_id(tracks, seed.get("track_id")) if seed.get("track_id") is not None else match_track(tracks, masks, chunk_start)
            if match is None:
                match = Track(
                    track_id=next_track_id,
                    seed_score=float(seed["score"]),
                    seed_source=str(seed.get("source", "sam")),
                    seed_label=str(seed.get("label", "")),
                    seed_confidence=float(seed.get("confidence", 0.0)),
                )
                next_track_id += 1
                tracks.append(match)
            for frame_idx, mask in masks.items():
                if frame_idx not in match.masks or np.count_nonzero(mask) > np.count_nonzero(match.masks[frame_idx]):
                    match.masks[frame_idx] = mask
    return tracks


def carryover_seed_masks(tracks: list[Track], chunk_start: int, limit: int) -> list[dict]:
    seeds = []
    if chunk_start <= 0:
        return seeds
    for track in sorted(tracks, key=lambda row: row.seed_score, reverse=True):
        if track.last_frame < chunk_start - 2:
            continue
        mask = track.masks[track.last_frame]
        if np.count_nonzero(mask) <= 32:
            continue
        seeds.append(
            {
                "mask": mask,
                "score": float(track.seed_score),
                "bbox": mask_bbox_xywh(mask),
                "track_id": track.track_id,
                "source": "carryover",
                "label": track.seed_label,
                "confidence": track.seed_confidence,
            }
        )
        if len(seeds) >= limit:
            break
    return seeds


def track_by_id(tracks: list[Track], track_id: int | None) -> Track | None:
    if track_id is None:
        return None
    for track in tracks:
        if track.track_id == track_id:
            return track
    return None


def seed_masks(
    frame: np.ndarray,
    hand_mask: np.ndarray,
    hand_proximity: np.ndarray,
    mask_generator,
    grounding_model,
    max_seeds: int,
    args,
    suppress: list[dict] | None = None,
) -> list[dict]:
    suppress = suppress or []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        proposals = mask_generator.generate(frame)
    candidates = []
    image_area = frame.shape[0] * frame.shape[1]
    ring = hand_proximity & ~hand_mask
    for proposal in proposals:
        mask = np.asarray(proposal["segmentation"], dtype=bool)
        area = int(np.count_nonzero(mask))
        area_ratio = area / image_area
        if area_ratio < args.min_mask_area_ratio or area_ratio > args.max_mask_area_ratio:
            continue
        hand_overlap = float(hand_mask[mask].mean()) if area else 0.0
        if hand_overlap > 0.45:
            continue
        proximity = float(hand_proximity[mask].mean()) if area else 0.0
        ring_score = float(ring[mask].mean()) if area else 0.0
        bbox = proposal["bbox"]
        if is_background_like_box(bbox, frame.shape[1], frame.shape[0], area_ratio):
            continue
        if any(mask_iou(mask, existing["mask"]) > 0.45 for existing in suppress):
            continue
        edge_penalty = 0.2 if touches_border_xywh(bbox, frame.shape[1], frame.shape[0]) else 0.0
        area_penalty = 0.09 * abs(math.log(max(area_ratio, 1e-6) / max(args.area_prior_ratio, 1e-6)))
        small_bonus = args.small_object_boost * max(0.0, min(1.0, (0.010 - area_ratio) / 0.010))
        quality = float(proposal["predicted_iou"]) + 0.7 * float(proposal["stability_score"])
        if args.seed_mode == "interaction":
            score = quality + 1.25 * ring_score + 0.35 * proximity + small_bonus - area_penalty - edge_penalty
        elif args.seed_mode == "coverage":
            score = (
                quality
                + 0.15 * min(area_ratio / 0.035, 1.0)
                + small_bonus
                - args.coverage_area_penalty_weight * area_penalty
                - edge_penalty
            )
        else:
            score = (
                quality
                + 0.70 * ring_score
                + 0.18 * proximity
                + 0.08 * min(area_ratio / 0.035, 1.0)
                + small_bonus
                - 0.75 * area_penalty
                - edge_penalty
            )
        candidates.append({"mask": mask, "score": score, "bbox": bbox, "track_id": None, "source": "sam", "label": "", "confidence": 0.0})
    if grounding_model is not None:
        candidates.extend(dino_seed_boxes(frame, hand_mask, hand_proximity, grounding_model, args, suppress))
    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    keep = []
    for candidate in candidates:
        if all(mask_iou(candidate["mask"], other["mask"]) < args.proposal_nms_iou for other in keep):
            keep.append(candidate)
        if len(keep) >= max_seeds:
            break
    return keep


def load_grounding_model(args):
    from grounding_dino.groundingdino.util.inference import load_model

    return load_model(
        model_config_path=str(args.dino_config),
        model_checkpoint_path=str(args.dino_checkpoint),
        device=args.device,
    )


def dino_seed_boxes(frame: np.ndarray, hand_mask: np.ndarray, hand_proximity: np.ndarray, grounding_model, args, suppress: list[dict]) -> list[dict]:
    from grounding_dino.groundingdino.datasets import transforms as T
    from grounding_dino.groundingdino.util.inference import predict
    from PIL import Image

    image_pil = Image.fromarray(frame)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    with torch.inference_mode(), groundingdino_python_attention_fallback():
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=args.dino_prompt,
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
            device=args.device,
        )
    if boxes.numel() == 0:
        return []
    height, width = frame.shape[:2]
    boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()
    order = np.argsort(-confidences)[: args.dino_max_boxes]
    records = []
    for index in order:
        x1, y1, x2, y2 = clip_box_xyxy(boxes_xyxy[index], width, height)
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            continue
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        area_ratio = np.count_nonzero(mask) / mask.size
        if area_ratio < args.min_mask_area_ratio or area_ratio > args.max_mask_area_ratio:
            continue
        hand_overlap = float(hand_mask[mask].mean()) if np.any(mask) else 0.0
        if hand_overlap > 0.55:
            continue
        if any(mask_iou(mask, existing["mask"]) > 0.45 for existing in suppress):
            continue
        proximity = float(hand_proximity[mask].mean()) if np.any(mask) else 0.0
        edge_penalty = 0.2 if touches_border_xywh(mask_bbox_xywh(mask), width, height) else 0.0
        score = 1.35 + float(confidences[index]) + 0.12 * min(area_ratio / 0.035, 1.0) + 0.08 * proximity - edge_penalty
        records.append(
            {
                "mask": mask,
                "score": score,
                "bbox": mask_bbox_xywh(mask),
                "box_xyxy": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                "label": str(labels[index]) if index < len(labels) else "",
                "confidence": float(confidences[index]),
                "source": "dino",
                "track_id": None,
            }
        )
    return records


@contextlib.contextmanager
def groundingdino_python_attention_fallback():
    try:
        from grounding_dino.groundingdino.models.GroundingDINO import ms_deform_attn
    except ImportError:
        yield
        return
    if hasattr(ms_deform_attn, "_C"):
        yield
        return
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original_cuda_available


def clip_box_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    return x1, y1, x2, y2


def touches_border_xywh(bbox, width: int, height: int) -> bool:
    x, y, w, h = bbox
    return x <= 2 or y <= 2 or x + w >= width - 3 or y + h >= height - 3


def mask_bbox_xywh(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def is_background_like_box(bbox, width: int, height: int, area_ratio: float) -> bool:
    x, y, w, h = bbox
    width_frac = w / width
    height_frac = h / height
    if width_frac > 0.86 or height_frac > 0.86:
        return True
    if touches_border_xywh(bbox, width, height) and area_ratio > 0.018:
        return True
    if (x <= 2 and x + w >= width - 3) or (y <= 2 and y + h >= height - 3):
        return True
    return False


def materialize_chunk(frames_dir: Path, chunks_dir: Path, start: int, end: int) -> Path:
    chunk_dir = chunks_dir / f"{start:05d}_{end:05d}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    for local_idx, global_idx in enumerate(range(start, end)):
        source = (frames_dir / f"{global_idx:05d}.jpg").resolve()
        target = chunk_dir / f"{local_idx:05d}.jpg"
        os.symlink(source, target)
    return chunk_dir


def propagate_chunk(video_predictor, chunk_dir: Path, seed_records: list[dict], device: str) -> dict[int, dict[int, np.ndarray]]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        state = video_predictor.init_state(video_path=str(chunk_dir), offload_video_to_cpu=True, offload_state_to_cpu=False)
        for obj_id, seed in enumerate(seed_records, start=1):
            if "box_xyxy" in seed:
                video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=seed["box_xyxy"],
                )
            else:
                video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=seed["mask"],
                )
        segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(state):
            frame_segments = {}
            for index, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[index] > 0.0).detach().cpu().numpy()
                frame_segments[int(obj_id)] = np.squeeze(mask).astype(bool)
            segments[int(out_frame_idx)] = frame_segments
    return segments


def match_track(tracks: list[Track], masks: dict[int, np.ndarray], chunk_start: int) -> Track | None:
    first_frame = min(masks)
    first_mask = masks[first_frame]
    best_track = None
    best_score = 0.0
    for track in tracks:
        if track.last_frame < chunk_start - 3 or track.last_frame > chunk_start + 2:
            continue
        ref_mask = track.masks[track.last_frame]
        score = mask_iou(first_mask, ref_mask)
        if score > best_score:
            best_score = score
            best_track = track
    return best_track if best_score >= 0.18 else None


def score_tracks(
    tracks: list[Track],
    frames: list[np.ndarray],
    hand_masks: list[np.ndarray],
    hand_proximity: list[np.ndarray],
    frame_count: int,
) -> dict[int, dict[int, dict]]:
    per_frame: dict[int, dict[int, dict]] = {i: {} for i in range(frame_count)}
    memory: dict[int, float] = {track.track_id: 0.0 for track in tracks}
    height, width = hand_masks[0].shape
    track_stats = {}
    for track in tracks:
        span = max(track.last_frame - track.first_frame + 1, 1)
        track_stats[track.track_id] = {
            "span": span / max(frame_count, 1),
            "visible_fraction": len(track.masks) / max(frame_count, 1),
            "first": track.first_frame,
            "last": track.last_frame,
        }
    for frame_idx in range(frame_count):
        for track in tracks:
            mask = track.masks.get(frame_idx)
            if mask is None:
                memory[track.track_id] *= 0.86
                continue
            hand_ring = hand_proximity[frame_idx] & ~hand_masks[frame_idx]
            hand_overlap = float(hand_masks[frame_idx][mask].mean()) if np.any(mask) else 0.0
            proximity = float(hand_proximity[frame_idx][mask].mean()) if np.any(mask) else 0.0
            ring = float(hand_ring[mask].mean()) if np.any(mask) else 0.0
            active = max(0.0, 1.35 * ring + 0.25 * proximity - 0.95 * hand_overlap)
            memory[track.track_id] = max(memory[track.track_id] * 0.92, active)
            area = np.count_nonzero(mask) / mask.size
            bbox = mask_bbox_xywh(mask)
            bbox_cx = (bbox[0] + 0.5 * bbox[2]) / width
            bbox_cy = (bbox[1] + 0.5 * bbox[3]) / height
            bbox_w = bbox[2] / width
            bbox_h = bbox[3] / height
            bbox_aspect = min(bbox_w / max(bbox_h, 1e-6), 8.0) / 8.0
            border_touch = 1.0 if touches_border_xywh(bbox, width, height) else 0.0
            stats = track_stats[track.track_id]
            track_progress = (frame_idx - stats["first"]) / max(stats["last"] - stats["first"], 1)
            prev_mask = track.masks.get(frame_idx - 1)
            next_mask = track.masks.get(frame_idx + 1)
            prev_iou = mask_iou(mask, prev_mask) if prev_mask is not None else 0.0
            next_iou = mask_iou(mask, next_mask) if next_mask is not None else 0.0
            centroid_motion = centroid_delta(mask, prev_mask, width, height) if prev_mask is not None else 0.0
            seed_semantics = semantic_seed_features(track)
            color_features = masked_color_features(frames[frame_idx], mask)
            object_score = 0.35 * memory[track.track_id] + 0.25 * active + 0.18 * track.seed_score + 0.10 * min(area / 0.04, 1.0) + 0.12 * proximity
            per_frame[frame_idx][track.track_id] = {
                "active": float(active),
                "memory": float(memory[track.track_id]),
                "ring": float(ring),
                "proximity": float(proximity),
                "hand_overlap": float(hand_overlap),
                "area": float(area),
                "seed_score": float(track.seed_score),
                "bbox_cx": float(bbox_cx),
                "bbox_cy": float(bbox_cy),
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "bbox_aspect": float(bbox_aspect),
                "border_touch": float(border_touch),
                "track_progress": float(track_progress),
                "track_span": float(stats["span"]),
                "visible_fraction": float(stats["visible_fraction"]),
                "prev_iou": float(prev_iou),
                "next_iou": float(next_iou),
                "centroid_motion": float(centroid_motion),
                **seed_semantics,
                **color_features,
                "object_score": float(object_score),
            }
    return per_frame


FEATURE_NAMES = (
    "active",
    "memory",
    "ring",
    "proximity",
    "hand_overlap",
    "area",
    "seed_score",
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "bbox_aspect",
    "border_touch",
    "track_progress",
    "track_span",
    "visible_fraction",
    "prev_iou",
    "next_iou",
    "centroid_motion",
    "dino_seed",
    "dino_confidence",
    "label_food",
    "label_tool",
    "label_container",
    "label_surface",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "rgb_std",
    "saturation",
    "value",
    "colorfulness",
)


def masked_color_features(frame: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = frame[mask]
    if pixels.size == 0:
        return {"rgb_r": 0.0, "rgb_g": 0.0, "rgb_b": 0.0, "rgb_std": 0.0, "saturation": 0.0, "value": 0.0, "colorfulness": 0.0}
    normalized = pixels.astype(np.float32) / 255.0
    mean = normalized.mean(axis=0)
    std = float(normalized.std())
    max_channel = normalized.max(axis=1)
    min_channel = normalized.min(axis=1)
    saturation = float(((max_channel - min_channel) / np.maximum(max_channel, 1e-6)).mean())
    value = float(max_channel.mean())
    rg = normalized[:, 0] - normalized[:, 1]
    yb = 0.5 * (normalized[:, 0] + normalized[:, 1]) - normalized[:, 2]
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {
        "rgb_r": float(mean[0]),
        "rgb_g": float(mean[1]),
        "rgb_b": float(mean[2]),
        "rgb_std": std,
        "saturation": saturation,
        "value": value,
        "colorfulness": colorfulness,
    }


def semantic_seed_features(track: Track) -> dict[str, float]:
    label = track.seed_label.lower()
    return {
        "dino_seed": 1.0 if track.seed_source == "dino" else 0.0,
        "dino_confidence": float(track.seed_confidence),
        "label_food": 1.0 if any(token in label for token in ("food", "vegetable", "tomato", "cucumber", "egg")) else 0.0,
        "label_tool": 1.0 if any(token in label for token in ("knife", "spoon", "fork", "utensil", "peeler")) else 0.0,
        "label_container": 1.0 if any(token in label for token in ("bowl", "cup", "plate", "bottle", "container", "pan", "pot", "package", "lid")) else 0.0,
        "label_surface": 1.0 if any(token in label for token in ("board", "cutting board")) else 0.0,
    }


def centroid_delta(mask: np.ndarray, other: np.ndarray, width: int, height: int) -> float:
    bbox = mask_bbox_xywh(mask)
    other_bbox = mask_bbox_xywh(other)
    cx = bbox[0] + 0.5 * bbox[2]
    cy = bbox[1] + 0.5 * bbox[3]
    other_cx = other_bbox[0] + 0.5 * other_bbox[2]
    other_cy = other_bbox[1] + 0.5 * other_bbox[3]
    return min(math.hypot(cx - other_cx, cy - other_cy) / math.hypot(width, height), 1.0)


def score_feature_vector(score: dict, names=FEATURE_NAMES) -> list[float]:
    return [float(score.get(name, 0.0)) for name in names]


def apply_score_model(path: Path, tracks: list[Track], per_frame_scores: dict[int, dict[int, dict]], device: str) -> float | None:
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("model_kind") == "calibrated_track_policy":
        return apply_calibrated_track_policy(payload, per_frame_scores, device)
    if isinstance(payload, dict) and payload.get("model_kind") == "union_linear":
        weights = np.asarray(payload["weights"], dtype=np.float32)
        feature_names = list(payload.get("feature_names", FEATURE_NAMES))
        feature_mean = np.asarray(payload.get("feature_mean", np.zeros_like(weights)), dtype=np.float32)
        feature_std = np.asarray(payload.get("feature_std", np.ones_like(weights)), dtype=np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        for frame_scores in per_frame_scores.values():
            for score in frame_scores.values():
                features = np.asarray(score_feature_vector_with_heuristic(score, feature_names), dtype=np.float32)
                score["object_score"] = float(((features - feature_mean) / feature_std) @ weights)
        return None
    if isinstance(payload, dict) and payload.get("model_kind") == "augmented_sklearn":
        model = payload["model"]
        feature_names = list(payload.get("feature_names", list(FEATURE_NAMES) + ["heuristic_object_score"]))
        for frame_scores in per_frame_scores.values():
            if not frame_scores:
                continue
            track_ids = list(frame_scores)
            features = np.asarray(
                [score_feature_vector_with_heuristic(frame_scores[track_id], feature_names) for track_id in track_ids],
                dtype=np.float32,
            )
            scores = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else model.predict(features)
            for track_id, score in zip(track_ids, scores):
                frame_scores[track_id]["object_score"] = float(score)
        return None
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feature_names = list(payload.get("feature_names", FEATURE_NAMES)) if isinstance(payload, dict) else list(FEATURE_NAMES)
    for frame_scores in per_frame_scores.values():
        if not frame_scores:
            continue
        track_ids = list(frame_scores)
        features = np.asarray([score_feature_vector(frame_scores[track_id], feature_names) for track_id in track_ids], dtype=np.float32)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(features)[:, 1]
        else:
            scores = model.predict(features)
        for track_id, score in zip(track_ids, scores):
            frame_scores[track_id]["object_score"] = float(score)
    return None


def apply_calibrated_track_policy(payload: dict, per_frame_scores: dict[int, dict[int, dict]], device: str) -> float:
    set_scores = score_with_set_selector(Path(payload["set_selector_checkpoint"]), per_frame_scores, device)
    raw_scores = collect_score_rows(per_frame_scores, "object_score")
    set_norm = normalize_by_frame(set_scores, per_frame_scores)
    raw_norm = normalize_by_frame(raw_scores, per_frame_scores)
    set_weight = float(payload.get("set_weight", 0.5))
    blended = {
        key: set_weight * set_norm.get(key, 0.0) + (1.0 - set_weight) * raw_norm.get(key, 0.0)
        for key in set(raw_norm) | set(set_norm)
    }
    smoothed = smooth_track_scores(blended, per_frame_scores, float(payload.get("smoothing_alpha", 0.95)))
    track_scores = aggregate_track_scores(smoothed, per_frame_scores, float(payload.get("track_quantile", 0.9)))
    threshold = float(payload["threshold"])
    suppressed = static_background_tracks(payload.get("static_background_gate", {}), per_frame_scores, track_scores, threshold)
    for frame_scores in per_frame_scores.values():
        for track_id, score in frame_scores.items():
            track_id = int(track_id)
            score["object_score"] = -1.0 if track_id in suppressed else float(track_scores.get(track_id, 0.0))
    return threshold


def score_with_set_selector(checkpoint_path: Path, per_frame_scores: dict[int, dict[int, dict]], device: str) -> dict[tuple[int, int], float]:
    from train_set_selector import SetSelector

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SetSelector(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), int(checkpoint["num_layers"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    feature_names = list(FEATURE_NAMES) + ["heuristic_object_score"]
    outputs: dict[tuple[int, int], float] = {}
    with torch.inference_mode():
        for frame_idx, frame_scores in per_frame_scores.items():
            if not frame_scores:
                continue
            track_ids = list(frame_scores)
            features = np.asarray([score_feature_vector_with_heuristic(frame_scores[track_id], feature_names) for track_id in track_ids], dtype=np.float32)
            features = (features - mean) / std
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(tensor)).detach().cpu().numpy()
            for track_id, score in zip(track_ids, scores):
                outputs[(int(frame_idx), int(track_id))] = float(score)
    return outputs


def collect_score_rows(per_frame_scores: dict[int, dict[int, dict]], name: str) -> dict[tuple[int, int], float]:
    return {
        (int(frame_idx), int(track_id)): float(score.get(name, 0.0))
        for frame_idx, frame_scores in per_frame_scores.items()
        for track_id, score in frame_scores.items()
    }


def normalize_by_frame(scores: dict[tuple[int, int], float], per_frame_scores: dict[int, dict[int, dict]]) -> dict[tuple[int, int], float]:
    normalized: dict[tuple[int, int], float] = {}
    for frame_idx, frame_scores in per_frame_scores.items():
        keys = [(int(frame_idx), int(track_id)) for track_id in frame_scores]
        values = np.asarray([scores.get(key, 0.0) for key in keys], dtype=np.float32)
        if values.size == 0:
            continue
        low = float(values.min())
        high = float(values.max())
        scaled = (values - low) / max(high - low, 1e-6)
        for key, value in zip(keys, scaled):
            normalized[key] = float(value)
    return normalized


def smooth_track_scores(scores: dict[tuple[int, int], float], per_frame_scores: dict[int, dict[int, dict]], alpha: float) -> dict[tuple[int, int], float]:
    by_track: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for frame_idx, frame_scores in per_frame_scores.items():
        for track_id in frame_scores:
            key = (int(frame_idx), int(track_id))
            by_track.setdefault(int(track_id), []).append((int(frame_idx), key))
    smoothed = dict(scores)
    for rows in by_track.values():
        rows.sort()
        previous = None
        previous_frame = None
        for frame_idx, key in rows:
            value = float(smoothed.get(key, 0.0))
            if previous is not None and previous_frame is not None:
                value = max(value, previous * (alpha ** max(frame_idx - previous_frame, 1)))
            smoothed[key] = value
            previous = value
            previous_frame = frame_idx
    return smoothed


def aggregate_track_scores(scores: dict[tuple[int, int], float], per_frame_scores: dict[int, dict[int, dict]], quantile: float) -> dict[int, float]:
    by_track: dict[int, list[float]] = {}
    for frame_idx, frame_scores in per_frame_scores.items():
        for track_id in frame_scores:
            key = (int(frame_idx), int(track_id))
            by_track.setdefault(int(track_id), []).append(float(scores.get(key, 0.0)))
    return {track_id: float(np.quantile(values, quantile)) for track_id, values in by_track.items() if values}


def static_background_tracks(gate: dict, per_frame_scores: dict[int, dict[int, dict]], track_scores: dict[int, float], threshold: float) -> set[int]:
    if not gate:
        return set()
    area_min = float(gate.get("area_min", 0.008))
    active_quantile = float(gate.get("active_quantile", 0.90))
    active_max = float(gate.get("active_max", 0.20))
    span_min = float(gate.get("span_min", 0.25))
    by_track: dict[int, list[dict]] = {}
    for frame_scores in per_frame_scores.values():
        for track_id, score in frame_scores.items():
            by_track.setdefault(int(track_id), []).append(score)
    suppressed = set()
    for track_id, rows in by_track.items():
        if track_scores.get(track_id, 0.0) < threshold:
            continue
        area = float(np.mean([row.get("area", 0.0) for row in rows]))
        active = float(np.quantile([row.get("active", 0.0) for row in rows], active_quantile))
        span = float(np.mean([row.get("track_span", 0.0) for row in rows]))
        if area >= area_min and active <= active_max and span >= span_min:
            suppressed.add(track_id)
    return suppressed


def score_feature_vector_with_heuristic(score: dict, feature_names: list[str]) -> list[float]:
    values = []
    for name in feature_names:
        if name == "heuristic_object_score":
            values.append(float(score["object_score"]))
        else:
            values.append(float(score.get(name, 0.0)))
    return values


def load_track_cache(path: Path, frame_count: int, shape: tuple[int, int]) -> tuple[list[Track], dict[int, dict[int, dict]]]:
    payload = np.load(path, allow_pickle=True)
    height = int(payload["height"][0])
    width = int(payload["width"][0])
    if (height, width) != shape:
        raise RuntimeError(f"Track cache shape {(height, width)} does not match video shape {shape}")
    feature_names = [str(name) for name in payload["feature_names"]]
    features = payload["features"].astype(np.float32)
    object_scores = payload["object_scores"].astype(np.float32)
    frame_indices = payload["frame_indices"].astype(np.int32)
    track_ids = payload["track_ids"].astype(np.int32)
    pred_masks = payload["pred_masks"].astype(np.uint8)

    tracks_by_id: dict[int, Track] = {}
    per_frame_scores: dict[int, dict[int, dict]] = {i: {} for i in range(frame_count)}
    for row, (frame_idx, track_id) in enumerate(zip(frame_indices, track_ids)):
        frame_idx = int(frame_idx)
        track_id = int(track_id)
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        track = tracks_by_id.setdefault(track_id, Track(track_id=track_id, seed_source="cache"))
        mask = unpack_mask(pred_masks[row], height, width)
        track.masks[frame_idx] = mask
        score = {name: float(features[row, index]) for index, name in enumerate(feature_names)}
        score["object_score"] = float(object_scores[row])
        per_frame_scores[frame_idx][track_id] = score
    return sorted(tracks_by_id.values(), key=lambda track: track.track_id), per_frame_scores


def selected_object_ids(per_frame_scores: dict[int, dict[int, dict]], frame_idx: int, threshold: float, max_display: int) -> list[int]:
    scores = per_frame_scores.get(frame_idx, {})
    ranked = sorted(
        ((track_id, score) for track_id, score in scores.items() if score["object_score"] >= threshold),
        key=lambda item: (item[1]["object_score"], item[1]["active"]),
        reverse=True,
    )
    if max_display > 0:
        ranked = ranked[:max_display]
    return [track_id for track_id, _ in ranked]


def fill_short_track_gaps(tracks: list[Track], per_frame_scores: dict[int, dict[int, dict]], max_gap: int, frame_count: int) -> None:
    if max_gap <= 0:
        return
    for track in tracks:
        present = sorted(track.masks)
        if len(present) < 2:
            continue
        for left, right in zip(present[:-1], present[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_score = per_frame_scores.get(left, {}).get(track.track_id)
            right_score = per_frame_scores.get(right, {}).get(track.track_id)
            if left_score is None and right_score is None:
                continue
            source_score = dict(left_score or right_score or {})
            for frame_idx in range(left + 1, right):
                if frame_idx < 0 or frame_idx >= frame_count or frame_idx in track.masks:
                    continue
                if frame_idx - left <= right - frame_idx:
                    track.masks[frame_idx] = track.masks[left].copy()
                    source_score = dict(left_score or source_score)
                else:
                    track.masks[frame_idx] = track.masks[right].copy()
                    source_score = dict(right_score or source_score)
                per_frame_scores.setdefault(frame_idx, {})[track.track_id] = dict(source_score)


def render_overlays(
    frames: list[np.ndarray],
    hand_masks: list[np.ndarray],
    tracks: list[Track],
    per_frame_scores: dict[int, dict[int, dict]],
    original_indices: list[int],
    args,
) -> list[np.ndarray]:
    track_by_id = {track.track_id: track for track in tracks}
    output = []
    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()
        hand = hand_masks[frame_idx]
        canvas[hand] = blend(canvas[hand], np.asarray([60, 210, 255]), 0.58)
        draw_contour(canvas, hand, (60, 230, 255), 2)
        for track_id in selected_object_ids(per_frame_scores, frame_idx, args.object_score_threshold, args.max_display_objects):
            track = track_by_id[track_id]
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue
            color = np.asarray([255, 70, 65])
            alpha = 0.58
            canvas[mask] = blend(canvas[mask], color, alpha)
            draw_contour(canvas, mask, tuple(int(v) for v in color), 2)
            label_track(canvas, mask, f"{track_id}", tuple(int(v) for v in color))
        selected_count = len(selected_object_ids(per_frame_scores, frame_idx, args.object_score_threshold, args.max_display_objects))
        put_label(canvas, f"frame {original_indices[frame_idx]} | tracks {len(tracks)} | selected {selected_count} | cyan hands | red score >= {args.object_score_threshold:.3f}")
        output.append(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return output


def blend(pixels: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(pixels.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def draw_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def label_track(image: np.ndarray, mask: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    x = int(np.median(xs))
    y = int(np.percentile(ys, 20))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def put_label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (6, 6), (image.shape[1] - 6, 34), (0, 0, 0), -1)
    cv2.putText(image, text[:100], (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)


def write_video(path: Path, frames_bgr: list[np.ndarray], fps: float) -> None:
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height))
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def transcode_for_browser(path: Path) -> None:
    tmp = path.with_name(path.stem + ".h264.tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        check=True,
    )
    tmp.replace(path)


def write_contact_sheet(path: Path, frames_bgr: list[np.ndarray], cols: int = 4) -> None:
    if not frames_bgr:
        return
    indices = np.linspace(0, len(frames_bgr) - 1, min(16, len(frames_bgr))).round().astype(int)
    thumbs = [cv2.resize(frames_bgr[i], (320, 320), interpolation=cv2.INTER_AREA) for i in indices]
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 320, cols * 320, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet[row * 320 : row * 320 + 320, col * 320 : col * 320 + 320] = thumb
    cv2.imwrite(str(path), sheet)


def evaluate_against_egoexo(args, tracks: list[Track], per_frame_scores: dict[int, dict[int, dict]], original_indices: list[int], shape: tuple[int, int]) -> dict:
    gt_by_frame = load_egoexo_gt_by_frame(args, original_indices, shape)
    selected = evaluate_selection_for_threshold(args, tracks, per_frame_scores, gt_by_frame, shape, args.object_score_threshold)
    proposal = evaluate_proposal_coverage(tracks, gt_by_frame)
    return {**proposal, **selected}


def load_egoexo_gt_by_frame(args, original_indices: list[int], shape: tuple[int, int]) -> dict[int, list[np.ndarray]]:
    take = load_take(args.take_uid)
    gt_by_frame = {}
    for track_id, object_payload in (take.get("object_masks") or {}).items():
        camera = object_payload.get(args.camera_name, {})
        annotation = camera.get("annotation") or {}
        for sampled_idx, original_idx in enumerate(original_indices):
            if original_idx < args.gt_window_start or original_idx >= args.gt_window_end:
                continue
            payload = annotation.get(str(original_idx))
            if payload is None:
                continue
            mask = decode_mask(payload).astype(np.uint8)
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            gt_by_frame.setdefault(sampled_idx, []).append(mask)
    return gt_by_frame


def evaluate_proposal_coverage(tracks: list[Track], gt_by_frame: dict[int, list[np.ndarray]]) -> dict:
    ious = []
    hits_03 = 0
    hits_05 = 0
    total = 0
    size_bins = {
        "small": {"ious": [], "hits_03": 0, "hits_05": 0},
        "medium": {"ious": [], "hits_03": 0, "hits_05": 0},
        "large": {"ious": [], "hits_03": 0, "hits_05": 0},
    }
    for frame_idx, gt_masks in gt_by_frame.items():
        pred_masks = [track.masks[frame_idx] for track in tracks if frame_idx in track.masks]
        for gt in gt_masks:
            best = max((mask_iou(gt, pred) for pred in pred_masks), default=0.0)
            bin_name = size_bin(gt)
            size_bins[bin_name]["ious"].append(best)
            size_bins[bin_name]["hits_03"] += int(best >= 0.3)
            size_bins[bin_name]["hits_05"] += int(best >= 0.5)
            ious.append(best)
            hits_03 += int(best >= 0.3)
            hits_05 += int(best >= 0.5)
            total += 1
    result = {
        "gt_masks": total,
        "frames_with_gt": len(gt_by_frame),
        "mean_best_iou": float(np.mean(ious)) if ious else 0.0,
        "recall_at_0.3": hits_03 / total if total else 0.0,
        "recall_at_0.5": hits_05 / total if total else 0.0,
    }
    for name, values in size_bins.items():
        count = len(values["ious"])
        result[f"{name}_gt_masks"] = count
        result[f"{name}_mean_best_iou"] = float(np.mean(values["ious"])) if count else 0.0
        result[f"{name}_recall_at_0.3"] = values["hits_03"] / count if count else 0.0
        result[f"{name}_recall_at_0.5"] = values["hits_05"] / count if count else 0.0
    return result


def size_bin(mask: np.ndarray) -> str:
    area_ratio = np.count_nonzero(mask) / mask.size
    if area_ratio < 0.005:
        return "small"
    if area_ratio < 0.025:
        return "medium"
    return "large"


def evaluate_selection_for_threshold(
    args,
    tracks: list[Track],
    per_frame_scores: dict[int, dict[int, dict]],
    gt_by_frame: dict[int, list[np.ndarray]],
    shape: tuple[int, int],
    threshold: float,
) -> dict:
    selected_union_ious = []
    selected_counts = []
    selected_unions = []
    for frame_idx, gt_masks in gt_by_frame.items():
        selected_ids = selected_object_ids(per_frame_scores, frame_idx, threshold, args.max_display_objects)
        selected_masks = [track.masks[frame_idx] for track in tracks if track.track_id in selected_ids and frame_idx in track.masks]
        selected_union = union_masks(selected_masks, shape)
        gt_union = union_masks(gt_masks, shape)
        selected_union_ious.append(mask_iou(selected_union, gt_union))
        selected_counts.append(len(selected_masks))
        selected_unions.append(selected_union)
    temporal_ious = [mask_iou(a, b) for a, b in zip(selected_unions[:-1], selected_unions[1:]) if np.any(a | b)]
    return {
        "selected_union_mean_iou": float(np.mean(selected_union_ious)) if selected_union_ious else 0.0,
        "selected_mean_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "selected_temporal_union_iou": float(np.mean(temporal_ious)) if temporal_ious else 0.0,
    }


def calibrate_object_threshold(args, tracks: list[Track], per_frame_scores: dict[int, dict[int, dict]], original_indices: list[int], shape: tuple[int, int]) -> dict:
    gt_by_frame = load_egoexo_gt_by_frame(args, original_indices, shape)
    candidates = []
    for threshold in parse_grid(args.threshold_grid):
        metrics = evaluate_selection_for_threshold(args, tracks, per_frame_scores, gt_by_frame, shape, threshold)
        objective = (
            metrics["selected_union_mean_iou"]
            + args.calibration_temporal_weight * metrics["selected_temporal_union_iou"]
            - args.calibration_count_penalty * metrics["selected_mean_count"]
        )
        candidates.append({"threshold": threshold, "objective": float(objective), **metrics})
    best = max(candidates, key=lambda row: row["objective"]) if candidates else {"threshold": args.object_score_threshold}
    return {
        "objective": "selected_union_iou + temporal_weight * temporal_union_iou - count_penalty * selected_count",
        "temporal_weight": args.calibration_temporal_weight,
        "count_penalty": args.calibration_count_penalty,
        "grid": args.threshold_grid,
        "best_threshold": float(best["threshold"]),
        "best": best,
        "top_candidates": sorted(candidates, key=lambda row: row["objective"], reverse=True)[:8],
    }


def save_score_cache(
    args,
    tracks: list[Track],
    per_frame_scores: dict[int, dict[int, dict]],
    original_indices: list[int],
    shape: tuple[int, int],
    run_dir: Path,
) -> Path:
    gt_by_frame = load_egoexo_gt_by_frame(args, original_indices, shape)
    track_by_id = {track.track_id: track for track in tracks}
    rows = []
    labels = []
    ious = []
    frame_indices = []
    track_ids = []
    for frame_idx, frame_scores in per_frame_scores.items():
        gt_masks = gt_by_frame.get(frame_idx, [])
        for track_id, score in frame_scores.items():
            track = track_by_id[track_id]
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue
            best_iou = max((mask_iou(mask, gt) for gt in gt_masks), default=0.0)
            rows.append(score_feature_vector(score))
            labels.append(float(best_iou >= 0.3))
            ious.append(best_iou)
            frame_indices.append(frame_idx)
            track_ids.append(track_id)
    path = run_dir / "score_cache.npz"
    np.savez_compressed(
        path,
        features=np.asarray(rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        iou=np.asarray(ious, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        track_ids=np.asarray(track_ids, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return path


def save_track_cache(
    args,
    tracks: list[Track],
    per_frame_scores: dict[int, dict[int, dict]],
    original_indices: list[int],
    shape: tuple[int, int],
    run_dir: Path,
) -> Path:
    gt_by_frame = load_egoexo_gt_by_frame(args, original_indices, shape)
    track_by_id = {track.track_id: track for track in tracks}
    rows = []
    frame_indices = []
    track_ids = []
    object_scores = []
    pred_masks = []
    ious = []
    for frame_idx, frame_scores in per_frame_scores.items():
        gt_masks = gt_by_frame.get(frame_idx, [])
        for track_id, score in frame_scores.items():
            track = track_by_id[track_id]
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue
            rows.append(score_feature_vector(score))
            frame_indices.append(frame_idx)
            track_ids.append(track_id)
            object_scores.append(float(score["object_score"]))
            pred_masks.append(pack_mask(mask))
            ious.append(max((mask_iou(mask, gt) for gt in gt_masks), default=0.0))

    gt_unions = []
    gt_frame_indices = []
    for frame_idx, gt_masks in sorted(gt_by_frame.items()):
        gt_unions.append(pack_mask(union_masks(gt_masks, shape)))
        gt_frame_indices.append(frame_idx)

    path = run_dir / "track_cache.npz"
    np.savez_compressed(
        path,
        features=np.asarray(rows, dtype=np.float32),
        object_scores=np.asarray(object_scores, dtype=np.float32),
        iou=np.asarray(ious, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        track_ids=np.asarray(track_ids, dtype=np.int32),
        pred_masks=np.asarray(pred_masks, dtype=np.uint8),
        gt_unions=np.asarray(gt_unions, dtype=np.uint8),
        gt_frame_indices=np.asarray(gt_frame_indices, dtype=np.int32),
        original_indices=np.asarray(original_indices, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
        height=np.asarray([shape[0]], dtype=np.int32),
        width=np.asarray([shape[1]], dtype=np.int32),
    )
    return path


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(-1).astype(np.uint8))


def unpack_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.unpackbits(mask, count=height * width).reshape(height, width).astype(bool)


def parse_grid(value: str) -> list[float]:
    start, stop, step = (float(part) for part in value.split(":"))
    if step <= 0:
        raise ValueError("--threshold-grid step must be positive")
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def require_gt_args(args) -> None:
    if not args.take_uid or args.gt_window_start is None or args.gt_window_end is None:
        raise SystemExit("--calibrate-threshold-with-gt requires --take-uid, --gt-window-start, and --gt-window-end")


def load_take(take_uid: str) -> dict:
    for relation_file in load_relation_files(EGOEXO_ROOT):
        if take_uid in relation_file.data:
            return relation_file.data[take_uid]
    raise RuntimeError(f"take_uid not found: {take_uid}")


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return union


if __name__ == "__main__":
    main()
