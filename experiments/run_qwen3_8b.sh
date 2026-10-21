#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DEVICE="${DEVICE:-cuda:0}"
BEHAVIOR="${BEHAVIOR:?Set BEHAVIOR, e.g. sycophancy}"
TRAIN_SEED="${TRAIN_SEED:-42}"
METHOD="${METHOD:-pr_lora}"
DTYPE="${DTYPE:-bfloat16}"
RUN="${BEHAVIOR}/seed${TRAIN_SEED}"

python scripts/split_caa.py \
  --input "data/raw/${BEHAVIOR}.jsonl" \
  --output-dir "data/splits/${RUN}" \
  --train-size 50 \
  --seed 42 --train-seed "$TRAIN_SEED"

python scripts/extract_directions.py \
  --model "$MODEL" \
  --input "data/splits/${RUN}/direction.jsonl" \
  --output "directions/${RUN}.pt" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

python scripts/train_pr_lora.py \
  --model "$MODEL" \
  --train-data "data/splits/${RUN}/train.jsonl" \
  --directions "directions/${RUN}.pt" \
  --output "adapters/${METHOD}/${RUN}" \
  --method "$METHOD" --seed "$TRAIN_SEED" \
  --rank 4 \
  --steps 20 \
  --learning-rate 5e-4 \
  --anchor-lambda 0.1 \
  --device "$DEVICE" \
  --dtype "$DTYPE"

python scripts/evaluate_behavior.py \
  --model "$MODEL" \
  --data "data/splits/${RUN}/eval.jsonl" \
  --adapter "adapters/${METHOD}/${RUN}" \
  --output "results/${METHOD}/${RUN}.json" \
  --device "$DEVICE" \
  --dtype "$DTYPE"
