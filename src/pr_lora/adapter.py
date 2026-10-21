"""PR-LoRA initialization and regularization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DirectionRegularizedLoRAConfig:
    rank: int = 4
    lora_alpha: int | None = None
    dropout: float = 0.0
    anchor_lambda: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    @property
    def alpha(self) -> int:
        return self.rank if self.lora_alpha is None else self.lora_alpha

    def peft(self):
        from peft import LoraConfig

        if self.rank < 1 or self.alpha <= 0 or self.anchor_lambda < 0:
            raise ValueError(
                "Rank and LoRA alpha must be positive; anchor lambda must be non-negative"
            )
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            target_modules=list(self.target_modules),
            task_type="CAUSAL_LM",
        )


def initialize_directional_b(
    model: torch.nn.Module,
    directions: dict[int, torch.Tensor],
    freeze: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy compatible layer directions into every LoRA-B column.

    Returns a CPU copy of each initialized tensor for use by
    :func:`direction_anchor_loss`. Incompatible output dimensions keep PEFT's
    standard initialization and are not regularized.
    """
    for index, vector in directions.items():
        if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() < 1e-8:
            raise ValueError(f"Invalid direction for layer {index}")
        if not torch.isclose(vector.float().norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError(f"Direction for layer {index} must have unit norm")
    targets: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" not in name or not name.endswith(".weight"):
                continue
            layer_index = _layer_index(name)
            if layer_index is None or layer_index not in directions:
                continue
            direction = directions[layer_index].to(parameter.device, parameter.dtype)
            if parameter.shape[0] != direction.numel():
                continue
            initialized = direction[:, None].expand_as(parameter).clone()
            parameter.copy_(initialized)
            parameter.requires_grad_(not freeze)
            targets[name] = initialized.float().cpu()
    if not targets:
        raise ValueError(
            "No LoRA B matrices matched the supplied layer indices and vector dimensions"
        )
    return targets


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def direction_anchor_loss(
    model: torch.nn.Module,
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the sum of squared drift from the directional initialization."""
    parameters = dict(model.named_parameters())
    if not targets:
        return next(model.parameters()).new_zeros(())
    loss = next(model.parameters()).new_zeros(())
    for name, target in targets.items():
        if name not in parameters:
            raise ValueError(f"Missing anchored parameter: {name}")
        parameter = parameters[name]
        loss = loss + (parameter - target.to(parameter.device, parameter.dtype)).pow(2).sum()
    return loss
