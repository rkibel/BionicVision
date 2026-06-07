#!/usr/bin/env python3
"""Small shared utilities for Scheme 3."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_grid(value: str) -> list[float]:
    if ":" not in value:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    start, stop, step = (float(part) for part in value.split(":"))
    rows, current = [], start
    while current <= stop + 1e-9:
        rows.append(round(current, 6))
        current += step
    return rows


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets or any(offset == 0 for offset in offsets):
        raise ValueError("--flow-pair-offsets must contain non-zero integers")
    return offsets


def parse_csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def parse_label_ids(value: str) -> tuple[int, ...]:
    rows = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not rows:
        raise ValueError("Expected at least one EgoHOS label ID")
    return rows


def parse_weight_map(value: str) -> dict[str, float]:
    rows = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Expected source:weight in --source-loss-weights, got {part!r}")
        key, weight = part.split(":", maxsplit=1)
        rows[key.strip().lower()] = float(weight)
    return rows


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
