"""Reusable components for direction-regularized LoRA.

Heavy ML dependencies are imported lazily so that data utilities can be used
for preprocessing without importing PyTorch, Transformers, or PEFT.
"""

from .data import load_jsonl, split_records

__all__ = [
    "DirectionRegularizedLoRAConfig",
    "direction_anchor_loss",
    "extract_directions",
    "initialize_directional_b",
    "load_directions",
    "load_jsonl",
    "save_directions",
    "split_records",
]


def __getattr__(name):
    if name in {"extract_directions", "load_directions", "save_directions"}:
        from . import directions

        return getattr(directions, name)
    if name in {
        "DirectionRegularizedLoRAConfig",
        "direction_anchor_loss",
        "initialize_directional_b",
    }:
        from . import adapter

        return getattr(adapter, name)
    raise AttributeError(name)
