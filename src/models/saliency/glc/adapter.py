"""Frozen GLC egocentric-gaze prior backed by the official checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from src.models.base import ModelSpec
except ImportError:
    from models.base import ModelSpec


ROOT = Path(__file__).resolve().parents[4]
GLC_ROOT = ROOT / "external/model_sources/saliency/GLC"
DEFAULT_CHECKPOINT = ROOT / "external/model_weights/glc_ego4d.pyth"
DEFAULT_CONFIG = GLC_ROOT / "configs/Ego4d/MVIT_B_16x4_CONV.yaml"

GLC_SPEC = ModelSpec(name="glc_gaze", task="saliency", required_packages=("torch", "fairscale", "fvcore"))


def assert_glc_available() -> None:
    required = [GLC_ROOT / "slowfast/models/custom_video_model_builder.py", DEFAULT_CONFIG, DEFAULT_CHECKPOINT]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"GLC external files are missing: {', '.join(str(path) for path in missing)}")


@dataclass
class GLCGazeEstimator:
    """Predict temporally informed gaze heatmaps from past-ending clips."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "cuda"

    def __post_init__(self) -> None:
        assert_glc_available()
        import torch

        if str(GLC_ROOT) not in sys.path:
            sys.path.insert(0, str(GLC_ROOT))
        from slowfast.config.defaults import get_cfg
        from slowfast.models.custom_video_model_builder import GLC_Gaze

        cfg = get_cfg()
        cfg.merge_from_file(str(DEFAULT_CONFIG))
        cfg.NUM_GPUS = 0
        self._torch = torch
        self.device_obj = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model = GLC_Gaze(cfg)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(self.device_obj).eval().requires_grad_(False)

    def predict_clips(self, clips_rgb: np.ndarray, output_size: tuple[int, int] = (256, 256)):
        """Predict `[B, T, H, W]` gaze maps for RGB uint8 clips."""

        torch = self._torch
        clips = torch.from_numpy(np.asarray(clips_rgb)).to(self.device_obj, dtype=torch.float32)
        clips = clips.permute(0, 4, 1, 2, 3)
        clips = torch.nn.functional.interpolate(clips, size=(8, 256, 256), mode="trilinear", align_corners=False)
        clips = ((clips / 255.0 - 0.45) / 0.225).contiguous()
        with torch.inference_mode():
            logits = self.model([clips]).squeeze(1)
            maps = torch.softmax(logits.flatten(-2) / 2.0, dim=-1).view_as(logits)
            flat = maps.flatten(2)
            lo = flat.amin(dim=2).view(maps.shape[0], maps.shape[1], 1, 1)
            span = (flat.amax(dim=2).view(maps.shape[0], maps.shape[1], 1, 1) - lo).clamp_min(1e-6)
            maps = (maps - lo) / span
            maps = torch.nn.functional.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)
        return maps.cpu()

    def predict_video_path(self, video_path: Path, frame_numbers: list[int], batch_size: int = 8) -> dict[int, np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        output = {}
        try:
            for start in range(0, len(frame_numbers), batch_size):
                targets = frame_numbers[start : start + batch_size]
                clips = np.stack([read_past_clip(capture, frame) for frame in targets])
                maps = self.predict_clips(clips)[:, -1].numpy()
                output.update({frame: gaze.astype(np.float32) for frame, gaze in zip(targets, maps)})
        finally:
            capture.release()
        return output

    def predict_frames(self, frames_rgb: list[np.ndarray], batch_size: int = 8) -> list[np.ndarray]:
        output = []
        for start in range(0, len(frames_rgb), batch_size):
            targets = range(start, min(start + batch_size, len(frames_rgb)))
            clips = np.stack([[frames_rgb[max(target - offset, 0)] for offset in range(56, -1, -8)] for target in targets])
            output.extend(self.predict_clips(clips)[:, -1].numpy())
        return output

    def predict_image(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.predict_clips(np.stack([[image_rgb] * 8]))[0, -1].numpy()


def read_past_clip(capture: cv2.VideoCapture, target_frame: int) -> np.ndarray:
    frames = []
    for frame_number in range(target_frame - 56, target_frame + 1, 8):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_number}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames)
