"""PyTorch EgoHOS dataset for hand-segmentor training."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .evaluation import EgoHOSEvaluationItem, load_egohos_sample


class EgoHOSTrainingDataset(Dataset):
    def __init__(
        self,
        items: list[EgoHOSEvaluationItem],
        image_size: tuple[int, int],
        *,
        augment: bool,
    ) -> None:
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image_bgr, mask = load_egohos_sample(item)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            image_rgb, mask = augment_pair(image_rgb, mask)

        height, width = self.image_size
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        return image, torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def augment_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = random_zoom_pair(image, mask)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.85:
        image = np.clip(image.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-28, 28), 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.25:
        kernel = random.choice((3, 5))
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return image, mask


def random_zoom_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = random.uniform(0.82, 1.18)
    new_height = max(8, int(round(height * scale)))
    new_width = max(8, int(round(width * scale)))
    image_scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if scale >= 1.0:
        top = random.randint(0, new_height - height)
        left = random.randint(0, new_width - width)
        return image_scaled[top : top + height, left : left + width], mask_scaled[top : top + height, left : left + width]

    top = random.randint(0, height - new_height)
    left = random.randint(0, width - new_width)
    image_out = np.zeros_like(image)
    mask_out = np.zeros_like(mask)
    image_out[top : top + new_height, left : left + new_width] = image_scaled
    mask_out[top : top + new_height, left : left + new_width] = mask_scaled
    return image_out, mask_out
