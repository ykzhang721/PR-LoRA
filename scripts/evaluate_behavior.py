#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on contrastive behavior pairs."""

from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pr_lora.data import answer_text, load_jsonl


def parse_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return choices[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"dtype must be one of {sorted(choices)}") from exc


def mean_loss(model, tokenizer, text: str, device: str, max_length: int) -> float:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(
        device
    )
    with torch.no_grad():
        loss = float(model(**encoded, labels=encoded["input_ids"]).loss.item())
    if not math.isfinite(loss):
        raise ValueError("Non-finite sequence loss; check inputs, precision and max-length")
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", type=parse_dtype)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype).to(args.device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
    model.eval()

    records = load_jsonl(args.data)
    matches = 0
    rows = []
    for record in tqdm(records, desc="evaluate"):
        matching = mean_loss(
            model,
            tokenizer,
            f"{record['question']}{answer_text(record['answer_matching_behavior'])}",
            args.device,
            args.max_length,
        )
        opposing = mean_loss(
            model,
            tokenizer,
            f"{record['question']}{answer_text(record['answer_not_matching_behavior'])}",
            args.device,
            args.max_length,
        )
        is_match = matching < opposing
        matches += int(is_match)
        rows.append(
            {
                "matching_loss": matching,
                "opposing_loss": opposing,
                "undesirable_preferred": is_match,
            }
        )
    result = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "data": str(args.data),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "max_length": args.max_length,
        "dtype": str(args.dtype),
        "num_items": len(records),
        "undesirable_behavior_score": matches / len(records),
        "lower_is_better": True,
        "items": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
