"""TCMonoDepth adapter backed by the checkout under `external/`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from src.models.base import ModelSpec
except ImportError:
    from models.base import ModelSpec


ROOT = Path(__file__).resolve().parents[4]
TC_MONODEPTH_ROOT = ROOT / "external" / "model_sources" / "depth" / "TCMonoDepth"
DEFAULT_CHECKPOINT = TC_MONODEPTH_ROOT / "weights" / "_ckpt_small.pt.tar"

TC_MONODEPTH_SPEC = ModelSpec(
    name="tc_monodepth",
    task="depth",
    required_packages=("torch", "torchvision"),
)


def assert_tc_monodepth_available() -> None:
    """Raise a helpful error if the external TCMonoDepth checkout is incomplete."""

    required = [
        TC_MONODEPTH_ROOT / "networks" / "TCSmallNet.py",
        TC_MONODEPTH_ROOT / "networks" / "transforms.py",
        DEFAULT_CHECKPOINT,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        details = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"TCMonoDepth external files are missing: {details}")


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    min_value = float(values[finite].min())
    max_value = float(values[finite].max())
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - min_value) / (max_value - min_value), 0, 1).astype(np.float32)


@dataclass
class TCMonoDepthEstimator:
    """Small TCMonoDepth wrapper for frame-by-frame depth prediction."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "cuda"
    resize_size: int = 256

    def __post_init__(self) -> None:
        assert_tc_monodepth_available()
        import torch
        from torchvision.transforms import Compose

        if str(TC_MONODEPTH_ROOT) not in sys.path:
            sys.path.insert(0, str(TC_MONODEPTH_ROOT))
        from networks import TCSmallNet
        from networks.transforms import PrepareForNet, Resize

        self._torch = torch
        self.device_obj = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model = TCSmallNet(SimpleNamespace())
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device_obj).eval().requires_grad_(False)
        self.transform = Compose(
            [
                Resize(
                    self.resize_size,
                    self.resize_size,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="lower_bound",
                ),
                PrepareForNet(),
            ]
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a normalized TCMonoDepth prediction in `[0, 1]`."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model_input = self.transform({"image": image_rgb})["image"]
        tensor = self._torch.from_numpy(model_input).to(self.device_obj).unsqueeze(0)
        prediction = self.predict_tensor(tensor, output_size=image_rgb.shape[:2])
        return prediction[0, 0].cpu().numpy().astype(np.float32)

    def predict_tensor(self, images_rgb_255, *, output_size: tuple[int, int] | None = None):
        """Predict normalized depth for a batched RGB tensor in `[0, 255]`."""

        torch = self._torch
        target_size = output_size or tuple(images_rgb_255.shape[-2:])
        with torch.inference_mode(), torch.autocast(device_type=self.device_obj.type, enabled=False):
            prediction = self.model(images_rgb_255.to(self.device_obj, dtype=torch.float32))
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=target_size,
                mode="bicubic",
                align_corners=False,
            )
            flat = prediction.flatten(1)
            minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
            span = (flat.amax(dim=1).view(-1, 1, 1, 1) - minimum).clamp_min(1e-6)
            return ((prediction - minimum) / span).clamp(0.0, 1.0)


def predict_image(image_path: str | Path, *, device: str = "cuda") -> np.ndarray:
    """Load an image and predict normalized TCMonoDepth."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return TCMonoDepthEstimator(device=device).predict(image)
