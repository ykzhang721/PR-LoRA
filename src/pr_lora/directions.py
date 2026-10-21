"""Per-layer contrastive directions from transformer block outputs."""

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import answer_text


def extract_directions(model, tokenizer, records, device, batch_size=1, max_length=512):
    """Normalize mean(desired) - mean(undesired) at the last non-padding token.

    Read the residual after each decoder block, before the model's final norm.
    Only selected tokens are copied to CPU; full activations are not retained.
    Defaults and float32 NumPy aggregation match the original experiment.
    """
    if not records or batch_size < 1 or max_length < 2:
        raise ValueError("Nonempty records, positive batch size and max_length >= 2 are required")
    layers = model.model.layers
    collected = [[[] for _ in layers] for _ in range(2)]
    was_training = model.training
    model.eval()
    try:
        for start in tqdm(range(0, len(records), batch_size), desc="extract directions"):
            batch = records[start : start + batch_size]
            for slot, field in enumerate(
                ("answer_not_matching_behavior", "answer_matching_behavior")
            ):
                texts = [item["question"] + answer_text(item[field]) for item in batch]
                encoded = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(device)
                mask = encoded["attention_mask"].bool()
                positions = torch.arange(mask.shape[1], device=mask.device)
                last = positions.expand_as(mask).masked_fill(~mask, -1).max(dim=1).values
                if (last < 0).any():
                    raise ValueError("A sequence has no non-padding tokens")
                hooks = []

                def make_hook(index):
                    def capture(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        rows = torch.arange(hidden.shape[0], device=hidden.device)
                        selected = hidden[rows, last.to(hidden.device)].detach().float().cpu()
                        collected[slot][index].extend(selected.numpy())

                    return capture

                try:
                    for index, layer in enumerate(layers):
                        hooks.append(layer.register_forward_hook(make_hook(index)))
                    with torch.no_grad():
                        model(**encoded, use_cache=False)
                finally:
                    for hook in hooks:
                        hook.remove()
    finally:
        model.train(was_training)

    directions = {}
    for index in range(len(layers)):
        vector = np.mean(collected[0][index], axis=0) - np.mean(collected[1][index], axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm):
            raise ValueError(f"Non-finite direction at layer {index}")
        if norm > 1e-8:
            directions[index] = torch.from_numpy(vector / norm)
    if not directions:
        raise ValueError("All contrastive directions are zero")
    return directions


def save_directions(directions, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": {str(k): v.cpu() for k, v in directions.items()},
            "metadata": metadata or {},
        },
        path,
    )


def load_directions(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {int(k): torch.as_tensor(v).float() for k, v in payload["directions"].items()}
