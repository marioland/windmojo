"""Evaluation module for windmojo."""

from windmojo.evaluation.metrics import calculate_metrics

__all__ = ["calculate_metrics", "create_validation_diagrams"]


def create_validation_diagrams(*args, **kwargs):
    """Lazy import to avoid pulling in pandas/matplotlib at module level."""
    from windmojo.evaluation.diagrams import create_validation_diagrams as _create

    return _create(*args, **kwargs)
