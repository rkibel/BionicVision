#!/usr/bin/env python3
"""Shared paths and defaults for Scheme 3."""

from pathlib import Path

from dataset_loaders import DEFAULT_EGOEXO_CAMERA


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiments/scheme3"
HAND_CHECKPOINT = OUTPUT_DIR / "hand_segmentor/best.pt"
DEFAULT_FLOW_CACHE_DIR = OUTPUT_DIR / "cache/flow_grids"
CURRENT_DENSE_CHECKPOINT = OUTPUT_DIR / "checkpoints/best.pt"
NEXT_DENSE_CHECKPOINT = OUTPUT_DIR / "checkpoints/candidate.pt"
DEFAULT_BENCHMARK_TAKE = "sfu_cooking_008_3"
DEFAULT_BENCHMARK_CAMERA = DEFAULT_EGOEXO_CAMERA
DEFAULT_BENCHMARK_START = 3150
DEFAULT_BENCHMARK_END = 4050
SUPERVISED_LOSSES = ("bce_dice", "bce_tversky")
