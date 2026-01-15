"""
Data loading utilities for JEPA training.
"""

# Check for torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .transition_dataset import (
        Vocabulary,
        TransitionDataset,
        TransitionCollator,
        create_dataloader,
        load_all_transitions,
        ACTION_TYPES,
        ACTION_TO_ID,
    )

__all__ = [
    "TORCH_AVAILABLE",
]

if TORCH_AVAILABLE:
    __all__.extend([
        "Vocabulary",
        "TransitionDataset",
        "TransitionCollator",
        "create_dataloader",
        "load_all_transitions",
        "ACTION_TYPES",
        "ACTION_TO_ID",
    ])
