"""Inference adapter for Scheme 3 dense checkpoints."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as DEFAULT_HAND_CHECKPOINT

from .model import DenseModelConfig, HandPrior, IMAGENET_MEAN, IMAGENET_STD, build_model, model_input_tensor_from_hand


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT / "external/model_weights/scheme3.pt"


class Scheme3DenseSegmentor:
    """Predict interacting-object probabilities using RGB and a frozen hand prior."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        hand_checkpoint: str | Path = DEFAULT_HAND_CHECKPOINT,
        device: str = "cuda",
        sequence_feature_context: bool = False,
    ) -> None:
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        args = payload.get("args", {})
        self.config = DenseModelConfig(
            image_size=int(args.get("image_size", 256)),
            encoder=str(args.get("encoder", "efficientnet-b4")),
            encoder_weights=args.get("encoder_weights", "imagenet"),
            hand_input_mode=str(args.get("hand_input_mode", "raw_ring_outer_distance")),
            hand_kernel_size=int(args.get("hand_kernel_size", 15)),
            hand_prior_power=float(args.get("hand_prior_power", 1.5)),
            image_feature_mode=str(args.get("image_feature_mode", "none")),
            threshold=float(payload.get("threshold", 0.5)),
        )
        self.model = build_model(self.config, load_encoder_weights=False).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().requires_grad_(False)
        self.hand_prior = HandPrior(Path(hand_checkpoint), self.device)
        self.feature_prior = build_feature_prior(self.config.image_feature_mode, self.device.type)
        self.sequence_feature_context = sequence_feature_context
        self._feature_history_bgr: list[np.ndarray] = []

    @property
    def threshold(self) -> float:
        return self.config.threshold

    @torch.inference_mode()
    def predict_batch(self, images_bgr: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not images_bgr:
            return [], []
        images = preprocess_bgr(images_bgr, self.config.image_size).to(self.device)
        raw_hand = self.hand_prior(images)
        image_features = self.predict_feature_prior(images_bgr, images)
        model_input = model_input_tensor_from_hand(images, raw_hand, self.config, image_features)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            object_probs = torch.sigmoid(self.model(model_input)).float()
        return resize_probabilities(object_probs, images_bgr), resize_probabilities(raw_hand, images_bgr)

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        objects, hands = self.predict_batch([image_bgr])
        return objects[0], hands[0]

    def reset_sequence(self) -> None:
        self._feature_history_bgr.clear()

    def predict_feature_prior(self, images_bgr: list[np.ndarray], normalized_images: torch.Tensor):
        if self.config.image_feature_mode == "glc_gaze" and self.sequence_feature_context:
            context = [*self._feature_history_bgr, *images_bgr]
            rgb_context = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in context]
            maps = self.feature_prior.predict_frames(rgb_context)[-len(images_bgr) :]
            self._feature_history_bgr = [image.copy() for image in context[-7:]]
            return torch.from_numpy(np.stack(maps)).unsqueeze(1).to(normalized_images.device, normalized_images.dtype)
        return predict_feature_prior(
            self.feature_prior,
            self.config.image_feature_mode,
            images_bgr,
            normalized_images,
            self.config.image_size,
        )


def preprocess_bgr(images_bgr: list[np.ndarray], image_size: int) -> torch.Tensor:
    rows = []
    for image in images_bgr:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got {image.shape}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rows.append(torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0)
    images = torch.stack(rows)
    return (images - IMAGENET_MEAN) / IMAGENET_STD


def resize_probabilities(probs: torch.Tensor, images_bgr: list[np.ndarray]) -> list[np.ndarray]:
    return [
        cv2.resize(prob[0].detach().cpu().numpy().astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        for prob, image in zip(probs, images_bgr)
    ]


def build_feature_prior(mode: str, device: str):
    if mode == "none":
        return None
    if mode == "tc_monodepth":
        from models.depth.tc_monodepth.adapter import TCMonoDepthEstimator

        return TCMonoDepthEstimator(device=device)
    if mode == "glc_gaze":
        from models.saliency.glc.adapter import GLCGazeEstimator

        return GLCGazeEstimator(device=device)
    raise ValueError(f"Unsupported image feature mode: {mode}")


def predict_feature_prior(prior, mode: str, images_bgr: list[np.ndarray], normalized_images: torch.Tensor, image_size: int):
    if mode == "none":
        return None
    if mode == "tc_monodepth":
        mean = IMAGENET_MEAN.to(normalized_images.device)
        std = IMAGENET_STD.to(normalized_images.device)
        rgb_255 = ((normalized_images * std + mean).clamp(0.0, 1.0) * 255.0).float()
        return prior.predict_tensor(rgb_255, output_size=(image_size, image_size)).to(normalized_images.device)
    if mode == "glc_gaze":
        # Independent-image evaluation should not leak temporal context across
        # unrelated batch rows. Video callers enable sequence_feature_context.
        maps = [prior.predict_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images_bgr]
        return torch.from_numpy(np.stack(maps)).unsqueeze(1).to(normalized_images.device, normalized_images.dtype)
    raise ValueError(f"Unsupported image feature mode: {mode}")
