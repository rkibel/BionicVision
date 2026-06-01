#!/usr/bin/env python3
"""Write visual overlays for a trained hand segmentor."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ROOT, FrameKey, split_frame_keys, write_json
from train_supervised_segmentor import (
    build_model,
    cache_path,
    ensure_cache,
    frame_keys_for_splits,
    parse_size,
    predict_probs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mobilenetv3", "resnet50"), default="resnet50")
    parser.add_argument("--image-size", default="512x912")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/cache_512x912")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/deeplab_r50_512/test_overlays")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    size = parse_size(args.image_size)
    keys = frame_keys_for_splits(args.split, hand_only=True)
    sampled = even_sample(keys, args.samples)
    ensure_cache(args.cache_dir, size, sampled)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_model(args.model, args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = float(checkpoint["threshold"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlays = []
    records = []
    with torch.inference_mode():
        for key in sampled:
            data = np.load(cache_path(args.cache_dir, key))
            image = data["image"].copy()
            gt = data["mask"].astype(bool)
            image_t = image_to_tensor(image).to(args.device)
            pred = predict_probs(model, image_t, tta_flip=args.tta_flip) >= threshold
            overlay = color_overlay(image, pred, gt)
            label = f"{key.video_id} {key.frame_index}"
            overlay = put_label(overlay, label)
            path = args.output_dir / f"{key.stem}_overlay.png"
            cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlays.append(overlay)
            records.append({"frame": key.stem, "path": str(path.relative_to(ROOT))})

    sheet = contact_sheet(overlays, columns=4)
    sheet_path = args.output_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    write_json(args.output_dir / "manifest.json", {"split": args.split, "threshold": threshold, "samples": records})
    print(sheet_path)


def even_sample(keys: list[FrameKey], count: int) -> list[FrameKey]:
    if len(keys) <= count:
        return keys
    indices = np.linspace(0, len(keys) - 1, count).round().astype(int)
    return [keys[int(index)] for index in indices]


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    from train_supervised_segmentor import MEAN, STD

    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def color_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    colors = np.zeros_like(out)
    colors[pred & gt] = (0, 220, 80)
    colors[pred & ~gt] = (255, 60, 40)
    colors[~pred & gt] = (40, 120, 255)
    active = pred | gt
    out[active] = 0.45 * out[active] + 0.55 * colors[active]
    return np.clip(out, 0, 255).astype(np.uint8)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(images: list[np.ndarray], columns: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to place in contact sheet")
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return sheet


if __name__ == "__main__":
    main()
