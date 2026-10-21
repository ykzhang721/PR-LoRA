import copy

import pytest
import torch
from peft import PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from pr_lora.adapter import (
    DirectionRegularizedLoRAConfig,
    direction_anchor_loss,
    initialize_directional_b,
)
from pr_lora.directions import extract_directions, load_directions, save_directions


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_extraction_matches_unpadded_block_outputs(tiny_model, tokenizer, records, padding_side):
    tokenizer.padding_side = padding_side
    single = extract_directions(tiny_model, tokenizer, records, "cpu")
    batched = extract_directions(tiny_model, tokenizer, records, "cpu", batch_size=2)
    for layer in single:
        torch.testing.assert_close(single[layer], batched[layer], atol=2e-6, rtol=2e-5)
        assert abs(single[layer].norm().item() - 1) < 1e-6
    assert tiny_model.training
    assert all(not layer._forward_hooks for layer in tiny_model.model.layers)


def test_hook_cleanup_on_failure(tiny_model, tokenizer, records, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional")

    monkeypatch.setattr(tiny_model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        extract_directions(tiny_model, tokenizer, records, "cpu")
    assert tiny_model.training
    assert all(not layer._forward_hooks for layer in tiny_model.model.layers)


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("frozen", [False, True])
def test_training_save_reload_and_merge(
    tiny_model, tokenizer, records, tmp_path, architecture, frozen
):
    if architecture == "llama":
        tiny_model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=24,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=0,
            )
        )
    base = copy.deepcopy(tiny_model)
    directions = extract_directions(tiny_model, tokenizer, records, "cpu")
    save_directions(directions, tmp_path / "directions.pt", {"test": True})
    directions = load_directions(tmp_path / "directions.pt")
    model = get_peft_model(tiny_model, DirectionRegularizedLoRAConfig().peft())
    targets = initialize_directional_b(model, directions, freeze=frozen)
    assert len(targets) == 4  # two layers, Q/O only; GQA K/V have smaller outputs
    for name, parameter in model.named_parameters():
        if "lora_B" in name and name not in targets:
            assert torch.count_nonzero(parameter) == 0
    assert direction_anchor_loss(model, targets).item() == 0
    batch = tokenizer(["User: Is this true? Assistant: No"], return_tensors="pt")
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        loss = model(**batch, labels=batch["input_ids"]).loss + 0.1 * direction_anchor_loss(
            model, targets
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            p.grad is not None and p.grad.norm() > 0
            for n, p in model.named_parameters()
            if "lora_A" in n
        )
        optimizer.step()
        optimizer.zero_grad()
    drift = direction_anchor_loss(model, targets).item()
    assert drift == 0 if frozen else drift > 0
    model.eval()
    with torch.no_grad():
        expected = model(**batch).logits
    model.save_pretrained(tmp_path / "adapter")
    restored = PeftModel.from_pretrained(base, tmp_path / "adapter").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch).logits, expected)
        merged = restored.merge_and_unload()
        torch.testing.assert_close(merged(**batch).logits, expected, atol=2e-6, rtol=2e-5)
