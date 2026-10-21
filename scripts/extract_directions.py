#!/usr/bin/env python3
"""Extract per-layer contrastive activation directions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pr_lora.data import load_jsonl
from pr_lora.directions import extract_directions, save_directions


def parse_dtype(value: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return choices[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"dtype must be one of {sorted(choices)}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", type=parse_dtype)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype).to(args.device)
    records = load_jsonl(args.input)
    directions = extract_directions(
        model,
        tokenizer,
        records,
        args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_directions(
        directions,
        args.output,
        metadata={
            "model": args.model,
            "input": str(args.input),
            "num_examples": len(records),
            "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "dtype": str(args.dtype),
            "definition": "normalized mean(last-token desired - undesired)",
        },
    )
    print(f"Saved {len(directions)} layer directions to {args.output}")


if __name__ == "__main__":
    main()
