"""Reusable EgoHOS hand-segmentation checkpoint adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from models.base import ModelSpec
from models.segmentation.hand_segmentor.model import build_model, normalize_images


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT / "external/model_weights/hand_segmentor.pt"

HAND_SEGMENTOR_SPEC = ModelSpec(
    name="egohos_hand_segmentor",
    task="segmentation",
    required_packages=("torch", "segmentation_models_pytorch", "opencv-python"),
)

@dataclass
class HandSegmentor:
    """Predict hand probabilities and masks from BGR frames."""

    checkpoint: str | Path = DEFAULT_CHECKPOINT
    device: str = "cuda"

    def __post_init__(self) -> None:
        import torch

        self.checkpoint = Path(self.checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Hand-segmentor checkpoint not found: {self.checkpoint}")

        self._torch = torch
        self.device_obj = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint, map_location=self.device_obj, weights_only=False)
        self.image_size = parse_size(checkpoint["image_size"])
        self.threshold = float(checkpoint["threshold"])

        self.model = build_model(str(checkpoint["model_name"]), self.device_obj, encoder_weights=None)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval().requires_grad_(False)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 hand-probability map matching the input frame size."""

        return self.predict_batch([image_bgr])[0]

    def predict_mask(self, image_bgr: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
        """Return a binary uint8 hand mask matching the input frame size."""

        selected_threshold = self.threshold if threshold is None else threshold
        return (self.predict(image_bgr) >= selected_threshold).astype(np.uint8) * 255

    def predict_batch(self, images_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Return float32 hand-probability maps for a batch of BGR frames."""

        if not images_bgr:
            return []
        torch = self._torch
        height, width = self.image_size
        tensors = []
        output_sizes = []
        for image in images_bgr:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected an HxWx3 BGR image, got shape {image.shape}")
            output_sizes.append(image.shape[:2])
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            tensors.append(torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1))

        images = torch.stack(tensors).to(self.device_obj, dtype=torch.float32) / 255.0
        images = normalize_images(images)
        with torch.inference_mode():
            logits = self.model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            probs = torch.sigmoid(logits)

        outputs = []
        for prob, output_size in zip(probs, output_sizes):
            resized = torch.nn.functional.interpolate(
                prob.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(resized[0, 0].cpu().numpy().astype(np.float32))
        return outputs


def parse_size(value: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)


def predict_image(
    image_path: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cuda",
) -> np.ndarray:
    """Load an image and return its binary hand mask."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return HandSegmentor(checkpoint=checkpoint, device=device).predict_mask(image)
