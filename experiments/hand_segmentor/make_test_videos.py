#!/usr/bin/env python3
"""Create test-split VISOR videos and ground-truth visualizations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

import cv2
import numpy as np

from common import ROOT, is_hand_like_label, load_frames_by_index, read_rgb, sparse_frame_path, split_frame_keys, write_json
from datasets.epic_kitchens.annotations import object_id, rasterize_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/epic_kitchens/video_snippets/test_set")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames-per-video", type=int)
    args = parser.parse_args()

    keys_by_video = group_keys(split_frame_keys(args.split, hand_only=True))
    input_dir = args.output_dir / "inputs"
    gt_dir = args.output_dir / "ground_truth"
    labels_dir = args.output_dir / "labels"
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "fps": args.fps, "videos": {}}
    for video_id, keys in keys_by_video.items():
        selected = even_sample(keys, args.max_frames_per_video) if args.max_frames_per_video else keys
        paths = write_video_set(video_id, selected, input_dir, gt_dir, labels_dir, args.fps)
        manifest["videos"][video_id] = {
            "frames": len(selected),
            "first_frame": selected[0].frame_index if selected else None,
            "last_frame": selected[-1].frame_index if selected else None,
            "frame_indices": [key.frame_index for key in selected],
            **{name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        }
    write_json(args.output_dir / "manifest.json", manifest)
    print(args.output_dir / "manifest.json")


def group_keys(keys):
    grouped = {}
    for key in keys:
        grouped.setdefault(key.video_id, []).append(key)
    return dict(sorted(grouped.items()))


def even_sample(keys, count):
    if count is None or len(keys) <= count:
        return keys
    indices = np.linspace(0, len(keys) - 1, count).round().astype(int)
    return [keys[int(index)] for index in indices]


def write_video_set(video_id, keys, input_dir, gt_dir, labels_dir, fps):
    if not keys:
        raise ValueError(f"No frames for {video_id}")
    first_image = read_rgb(sparse_frame_path(keys[0]))
    height, width = first_image.shape[:2]
    input_path = input_dir / f"{video_id}_test_frames.mp4"
    hand_mask_path = gt_dir / f"{video_id}_test_hand_gt_masks.mp4"
    hand_overlay_path = gt_dir / f"{video_id}_test_hand_gt_overlay.mp4"
    all_mask_path = gt_dir / f"{video_id}_test_all_gt_masks.mp4"
    all_overlay_path = gt_dir / f"{video_id}_test_all_gt_overlay.mp4"
    labels_path = labels_dir / f"{video_id}_test_labels.json"
    remove_stale_outputs(video_id, gt_dir)
    input_writer = video_writer(input_path, fps, width, height)
    hand_mask_writer = video_writer(hand_mask_path, fps, width, height)
    hand_overlay_writer = video_writer(hand_overlay_path, fps, width, height)
    all_mask_writer = video_writer(all_mask_path, fps, width, height)
    all_overlay_writer = video_writer(all_overlay_path, fps, width, height)
    label_payload = {"video_id": video_id, "frames": []}
    frames_by_index = load_frames_by_index(video_id)
    try:
        for key in keys:
            rgb = read_rgb(sparse_frame_path(key))
            frame = frames_by_index[key.frame_index]
            hand_mask, all_mask, color_mask, labels = ground_truth_masks_and_labels(frame, rgb.shape[:2])
            input_writer.write(to_bgr(rgb))
            hand_mask_writer.write(to_bgr(mask_rgb(hand_mask)))
            hand_overlay_writer.write(to_bgr(mask_overlay(rgb, hand_mask, (0, 220, 80))))
            all_mask_writer.write(to_bgr(mask_rgb(all_mask)))
            all_overlay_writer.write(to_bgr(color_overlay(rgb, color_mask)))
            label_payload["frames"].append({"frame_index": key.frame_index, "objects": labels})
    finally:
        input_writer.release()
        hand_mask_writer.release()
        hand_overlay_writer.release()
        all_mask_writer.release()
        all_overlay_writer.release()
    for path in (input_path, hand_mask_path, hand_overlay_path, all_mask_path, all_overlay_path):
        transcode_for_vscode(path)
    write_json(labels_path, label_payload)
    return {
        "input_video": input_path,
        "hand_gt_mask_video": hand_mask_path,
        "hand_gt_overlay_video": hand_overlay_path,
        "all_gt_mask_video": all_mask_path,
        "all_gt_overlay_video": all_overlay_path,
        "labels_json": labels_path,
    }


def remove_stale_outputs(video_id, gt_dir):
    for path in gt_dir.glob(f"{video_id}_test*_gt_*.mp4"):
        path.unlink(missing_ok=True)


def video_writer(path, fps, width, height):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def transcode_for_vscode(path):
    tmp = path.with_name(path.stem + ".h264.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)


def mask_rgb(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = (255, 255, 255)
    return out


def ground_truth_masks_and_labels(frame, shape):
    hand_mask = np.zeros(shape, dtype=bool)
    all_mask = np.zeros(shape, dtype=bool)
    color_mask = np.zeros((*shape, 3), dtype=np.uint8)
    labels = []
    for annotation in frame.annotations:
        mask = rasterize_object(annotation, shape).astype(bool)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        oid = object_id(annotation)
        color = instance_color(oid)
        color_mask[mask] = color
        all_mask |= mask
        if is_hand_like_label(annotation.name):
            hand_mask |= mask
        labels.append(
            {
                "name": annotation.name,
                "track_id": annotation.track_id,
                "object_id": oid,
                "relation": annotation.relation,
                "mask_type": annotation.mask_type,
                "area_pixels": area,
                "is_hand_like": is_hand_like_label(annotation.name),
                "color_rgb": list(color),
            }
        )
    return hand_mask, all_mask, color_mask, labels


def instance_color(identifier):
    digest = hashlib.md5(str(identifier).encode("utf-8")).digest()
    return (80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176)


def mask_overlay(image, mask, color):
    out = image.astype(np.float32)
    green = np.zeros_like(out)
    green[mask] = color
    out[mask] = 0.45 * out[mask] + 0.55 * green[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def color_overlay(image, color_mask):
    mask = np.any(color_mask > 0, axis=2)
    out = image.astype(np.float32)
    out[mask] = 0.45 * out[mask] + 0.55 * color_mask[mask].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
