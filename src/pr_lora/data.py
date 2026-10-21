"""Data loading and deterministic split utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records and validate the required fields."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            missing = {
                "question",
                "answer_matching_behavior",
                "answer_not_matching_behavior",
            } - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {sorted(missing)}")
            if not isinstance(record["question"], str) or not record["question"].strip():
                raise ValueError(f"{path}:{line_number} has an empty or non-string question")
            for field in ("answer_matching_behavior", "answer_not_matching_behavior"):
                answer_text(record[field])
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def answer_text(value: Any) -> str:
    """Normalize CAA answer fields, which may be strings or one-item lists."""
    if isinstance(value, list):
        if not value:
            raise ValueError("Answer lists must not be empty")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Answers must be non-empty strings")
    return value.strip()


def split_indices(
    total: int,
    direction_size: int | None = None,
    train_size: int = 50,
    evaluation_size: int | None = None,
    split_seed: int = 42,
    train_seed: int = 42,
) -> dict[str, list[int]]:
    """Reproduce the paper's source-row partitions and per-seed SFT sampling.

    Shuffle all rows with split_seed, reserve the direction block followed by
    the evaluation block, then reshuffle the remaining pool with train_seed.
    Unused training-pool rows are intentionally not added to evaluation.
    """
    nd = min(500, total * 3 // 10) if direction_size is None else direction_size
    ne = min(300, total * 3 // 10) if evaluation_size is None else evaluation_size
    if min(nd, ne, train_size) < 1 or nd + ne + train_size > total:
        raise ValueError(
            f"Cannot allocate {nd} direction, {ne} eval and {train_size} train rows from {total}"
        )
    rows = list(range(total))
    random.Random(split_seed).shuffle(rows)
    pool = rows[nd + ne :]
    random.Random(train_seed).shuffle(pool)
    return {"direction": rows[:nd], "eval": rows[nd : nd + ne], "train": pool[:train_size]}


def split_records(
    records: Iterable[dict[str, Any]],
    direction_size: int | None = None,
    train_size: int = 50,
    seed: int = 42,
    evaluation_size: int | None = None,
    train_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return disjoint direction, train, and evaluation partitions."""
    records = list(records)
    indices = split_indices(
        len(records), direction_size, train_size, evaluation_size, seed, train_seed
    )
    return tuple([records[i] for i in indices[name]] for name in ("direction", "train", "eval"))


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
