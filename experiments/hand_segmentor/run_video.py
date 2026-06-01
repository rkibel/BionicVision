#!/usr/bin/env python3
"""Run the trained hand segmentor on a video or frame directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from common import ROOT, write_json
from make_overlays import contact_sheet, image_to_tensor, put_label
from train_supervised_segmentor import build_model, parse_size, predict_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input MP4/video file or directory of images.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/video_runs")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt")
    parser.add_argument("--model", choices=("mobilenetv3", "resnet50"), default="resnet50")
    parser.add_argument("--image-size", default="512x912")
    parser.add_argument("--threshold", type=float, help="Override the checkpoint threshold.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--write-masks", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    size = parse_size(args.image_size)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    threshold = float(args.threshold if args.threshold is not None else checkpoint["threshold"])
    model = build_model(args.model, args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    run_dir = args.output_dir / args.input.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = run_dir / "masks"
    if args.write_masks:
        mask_dir.mkdir(exist_ok=True)

    source = FrameSource(args.input, stride=args.stride, max_frames=args.max_frames)
    total_output_frames = args.max_frames or (source.total_frames // args.stride if source.total_frames else 240)
    sample_every = max(1, total_output_frames // 24)
    writer = None
    samples = []
    records = []
    pending = []
    with torch.inference_mode():
        for item in source:
            pending.append(item)
            if len(pending) >= args.batch_size:
                writer = flush_batch(pending, model, size, threshold, sample_every, args, run_dir, mask_dir, samples, records, writer, source.fps)
                pending = []
        if pending:
            writer = flush_batch(pending, model, size, threshold, sample_every, args, run_dir, mask_dir, samples, records, writer, source.fps)
    if writer is not None:
        writer.release()
        transcode_for_vscode(run_dir / "overlay.mp4")
    source.close()

    sheet_path = None
    if samples:
        sheet = contact_sheet(samples, columns=min(4, len(samples)))
        sheet_path = run_dir / "contact_sheet.png"
        cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    manifest = {
        "input": str(args.input),
        "frames": len(records),
        "threshold": threshold,
        "overlay_video": display_path(run_dir / "overlay.mp4"),
        "contact_sheet": display_path(sheet_path) if sheet_path else None,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(run_dir / "overlay.mp4")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


class FrameSource:
    def __init__(self, path: Path, *, stride: int, max_frames: int | None):
        self.path = path
        self.stride = stride
        self.max_frames = max_frames
        self.index = 0
        self.emitted = 0
        self.capture = None
        self.files = None
        self.fps = 30.0
        self.total_frames = 0
        if path.is_dir():
            self.files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            self.total_frames = len(self.files)
        else:
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if fps and fps > 0 else 30.0
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_frames is not None and self.emitted >= self.max_frames:
            raise StopIteration
        while True:
            frame = self.read_next()
            if frame is None:
                raise StopIteration
            index = self.index
            self.index += 1
            if index % self.stride == 0:
                self.emitted += 1
                return {"index": index, "rgb": frame}

    def read_next(self) -> np.ndarray | None:
        if self.files is not None:
            if self.index >= len(self.files):
                return None
            bgr = cv2.imread(str(self.files[self.index]), cv2.IMREAD_COLOR)
        else:
            ok, bgr = self.capture.read()
            if not ok:
                return None
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def flush_batch(items, model, size, threshold, sample_every, args, run_dir, mask_dir, samples, records, writer, fps):
    images = [resize_rgb(item["rgb"], size) for item in items]
    batch = torch.cat([image_to_tensor(image) for image in images], dim=0).to(args.device)
    probs = predict_batch(model, batch, tta_flip=args.tta_flip)
    for item, small_rgb, prob in zip(items, images, probs):
        pred_small = prob >= threshold
        pred = cv2.resize(pred_small.astype(np.uint8), (item["rgb"].shape[1], item["rgb"].shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = prediction_overlay(item["rgb"], pred, alpha=args.alpha)
        overlay = put_label(overlay, f"frame {item['index']}")
        if writer is None:
            height, width = overlay.shape[:2]
            writer = cv2.VideoWriter(str(run_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (width, height))
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if args.write_masks:
            cv2.imwrite(str(mask_dir / f"frame_{item['index']:06d}.png"), pred.astype(np.uint8) * 255)
        if len(samples) < 24 and len(records) % sample_every == 0:
            samples.append(cv2.resize(overlay, (small_rgb.shape[1], small_rgb.shape[0]), interpolation=cv2.INTER_AREA))
        records.append({"frame_index": item["index"], "mask_pixels": int(np.count_nonzero(pred))})
        if len(records) % 100 == 0:
            print(f"processed {len(records)} frames", flush=True)
    return writer


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def predict_batch(model, images: torch.Tensor, *, tta_flip: bool) -> np.ndarray:
    logits = model(images)["out"]
    if tta_flip:
        flipped = torch.flip(images, dims=(-1,))
        flip_logits = torch.flip(model(flipped)["out"], dims=(-1,))
        logits = 0.5 * (logits + flip_logits)
    return torch.sigmoid(logits).detach().cpu().numpy()[:, 0]


def prediction_overlay(image: np.ndarray, pred: np.ndarray, *, alpha: float) -> np.ndarray:
    out = image.astype(np.float32)
    color = np.zeros_like(out)
    color[pred] = (0, 220, 80)
    out[pred] = (1.0 - alpha) * out[pred] + alpha * color[pred]
    return np.clip(out, 0, 255).astype(np.uint8)


def transcode_for_vscode(path: Path) -> None:
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


if __name__ == "__main__":
    main()
