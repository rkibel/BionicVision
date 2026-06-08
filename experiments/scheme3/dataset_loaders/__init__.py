#!/usr/bin/env python3
"""Public dataset imports for Scheme 3.

The implementation lives in dataset-specific modules so this file can stay as
the stable import surface for training, evaluation, and rendering scripts.
"""

from __future__ import annotations

from dataset_loaders.common import (
    DEFAULT_EGOEXO_CAMERA,
    EGOEXO_ROOT,
    EGOHOS_ROOT,
    MEAN,
    ROOT,
    STD,
    EgoHOSFrameEntry,
    FrameEntry,
    augment_image_masks,
    collate_flow_pairs,
    collate_samples,
    decode_track_mask,
    egohos_image_path,
    egohos_source,
    is_hand_track,
    image_feature_cache_path,
    load_image_feature,
    load_relations,
    read_egohos_label,
    read_rgb_image,
    read_video_frame,
    relation_path,
    take_video_path,
    target_union,
)
from dataset_loaders.egoexo import EgoExoFlowPairDataset, EgoExoMaskDataset, build_flow_pairs, build_frame_entries
from dataset_loaders.egohos import EgoHOSMaskDataset, balance_egohos_sources, build_egohos_entries


__all__ = [
    "DEFAULT_EGOEXO_CAMERA",
    "EGOEXO_ROOT",
    "EGOHOS_ROOT",
    "MEAN",
    "ROOT",
    "STD",
    "FrameEntry",
    "EgoHOSFrameEntry",
    "EgoExoMaskDataset",
    "EgoExoFlowPairDataset",
    "EgoHOSMaskDataset",
    "augment_image_masks",
    "balance_egohos_sources",
    "build_egohos_entries",
    "build_flow_pairs",
    "build_frame_entries",
    "collate_flow_pairs",
    "collate_samples",
    "decode_track_mask",
    "egohos_image_path",
    "egohos_source",
    "is_hand_track",
    "image_feature_cache_path",
    "load_image_feature",
    "load_relations",
    "read_egohos_label",
    "read_rgb_image",
    "read_video_frame",
    "relation_path",
    "take_video_path",
    "target_union",
]
