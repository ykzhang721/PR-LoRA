#!/usr/bin/env python3
"""Train PR-LoRA or a matched standard-LoRA control."""

from __future__ import annotations

import argparse
import json
import random
import hashlib
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

from pr_lora.adapter import (
    DirectionRegularizedLoRAConfig,
    direction_anchor_loss,
    initialize_directional_b,
)
from pr_lora.data import answer_text, load_jsonl
from pr_lora.directions import load_directions


class SupervisedDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int):
        self.examples = []
        for record in records:
            text = f"User: {record['question']}\nAssistant: {answer_text(record['answer_not_matching_behavior'])}"
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def parse_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return choices[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"dtype must be one of {sorted(choices)}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--directions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=["pr_lora", "standard", "frozen"], default="pr_lora")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--anchor-lambda", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--weight-decay", type=float, help="Default: 0.1 for frozen, 0.01 otherwise"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", type=parse_dtype)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    args = parser.parse_args()
    if args.method != "standard" and args.directions is None:
        parser.error("--directions is required for direction-based methods")
    if min(args.steps, args.batch_size) < 1 or args.max_length < 2:
        parser.error("steps and batch-size must be positive, max-length must be >= 2")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    if args.weight_decay is None:
        args.weight_decay = 0.1 if args.method == "frozen" else 0.01

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype).to(args.device)
    model.config.use_cache = False

    cfg = DirectionRegularizedLoRAConfig(
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        anchor_lambda=args.anchor_lambda,
        target_modules=tuple(args.target_modules),
    )
    model = get_peft_model(model, cfg.peft())
    targets = {}
    if args.method != "standard":
        targets = initialize_directional_b(
            model, load_directions(args.directions), freeze=args.method == "frozen"
        )
    anchored_names = list(targets)
    if args.method == "frozen":
        targets = {}
    model.train()
    data = SupervisedDataset(load_jsonl(args.train_data), tokenizer, args.max_length)
    loader = DataLoader(data, batch_size=min(args.batch_size, len(data)), shuffle=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    step = 0
    history = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                labels=batch["labels"].to(args.device),
            )
            anchor = direction_anchor_loss(model, targets) if targets else output.loss.new_zeros(())
            loss = output.loss + args.anchor_lambda * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            history.append({"step": step, "sft_loss": output.loss.item(), "anchor": anchor.item()})
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(f"step={step} task_loss={output.loss.item():.6f} anchor={anchor.item():.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "initialized_modules": anchored_names,
                "source_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
                "lora_alpha": cfg.alpha,
                "training_history": history,
            },
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
