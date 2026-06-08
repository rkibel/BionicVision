"""Ego-Exo4D relation-mask data access for dense object segmentation."""

from .dataset import EgoExoFlowPairDataset, EgoExoObjectDataset
from .relations import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    EgoExoFrameEntry,
    build_frame_entries,
    is_hand_track,
    load_relations,
    take_video_path,
)

__all__ = [
    "DEFAULT_CAMERA",
    "DEFAULT_DATA_ROOT",
    "EgoExoFlowPairDataset",
    "EgoExoFrameEntry",
    "EgoExoObjectDataset",
    "build_frame_entries",
    "is_hand_track",
    "load_relations",
    "take_video_path",
]
