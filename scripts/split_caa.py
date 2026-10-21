#!/usr/bin/env python3
"""Create deterministic direction, training, and evaluation splits."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from pr_lora.data import load_jsonl, split_indices, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direction-size", type=int)
    parser.add_argument("--evaluation-size", type=int)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    indices = split_indices(
        len(records),
        args.direction_size,
        args.train_size,
        args.evaluation_size,
        args.seed,
        args.train_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in indices.items():
        write_jsonl([records[i] for i in rows], args.output_dir / f"{name}.jsonl")
    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "train_seed": args.train_seed,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "counts": {name: len(rows) for name, rows in indices.items()},
        "source_row_indices": indices,
    }
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_row_indices"}, indent=2))


if __name__ == "__main__":
    main()
