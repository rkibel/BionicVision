"""Shared evaluation for dense object-mask schemes 3, 4, and 5."""

from .metrics import evaluate_supervised_masks
from .temporal import evaluate_temporal_masks

__all__ = ["evaluate_supervised_masks", "evaluate_temporal_masks"]
