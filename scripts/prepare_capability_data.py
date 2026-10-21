#!/usr/bin/env python3
"""Prepare MMLU and TruthfulQA caches used by the original paper scripts."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/data"))
    parser.add_argument("--dataset", choices=["mmlu", "truthfulqa", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in ("mmlu", "both"):
        ds = load_dataset("cais/mmlu", "all")
        out = args.output_dir / "mmlu"
        out.mkdir(parents=True, exist_ok=True)
        for split in ("test", "validation"):
            rows = [
                {key: item[key] for key in ("question", "choices", "answer", "subject")}
                for item in ds[split]
            ]
            (out / f"{split}.json").write_text(json.dumps(rows) + "\n")
            print(f"MMLU {split}: {len(rows)} items")
    if args.dataset in ("truthfulqa", "both"):
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        out = args.output_dir / "truthfulqa"
        out.mkdir(parents=True, exist_ok=True)
        rows = [{key: item[key] for key in ("question", "mc1_targets")} for item in ds]
        (out / "mc1.json").write_text(json.dumps(rows) + "\n")
        print(f"TruthfulQA MC1: {len(rows)} items")


if __name__ == "__main__":
    main()
