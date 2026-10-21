#!/usr/bin/env python3
"""PiSSA baseline: LoRA with PiSSA initialization (principal singular vectors of W).
Tests whether data-agnostic structured init achieves behavioral alignment.
Compares PiSSA init vs random init vs direction init (PR-LoRA) on 3 behaviors, 3 seeds.
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)
os.environ["WANDB_DISABLED"] = "true"

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/pissa_baseline"
RANK = 4
SEEDS = [42, 123, 456]
STEPS = 20
LR = 5e-4
N_TRAIN = 50

BEHS = {
    "sycophancy": "sycophancy_on_nlp_survey.jsonl",
    "power_seeking": "power-seeking-inclination.jsonl",
    "corrigibility": "corrigible-less-HHH.jsonl",
}


def _s(v):
    return v[0].strip() if isinstance(v, list) else v.strip()


def load_jsonl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def get_data_split(beh_name):
    fp = EVAL_DIR / BEHS[beh_name]
    if not fp.exists():
        return [], [], []
    data = load_jsonl(fp)
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    total = len(sh)
    nd = min(500, total * 3 // 10)
    ne = min(300, total * 3 // 10)
    return sh[:nd], sh[nd : nd + ne], sh[nd + ne :]


def eval_beh(model, tok, items, device):
    model.eval()
    match = 0
    for item in items:
        q = item["question"]
        m, n = _s(item["answer_matching_behavior"]), _s(item["answer_not_matching_behavior"])
        em = tok(f"{q}{m}", return_tensors="pt", truncation=True, max_length=1024).to(device)
        en = tok(f"{q}{n}", return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            lm = model(**em, labels=em["input_ids"]).loss.item()
            ln = model(**en, labels=en["input_ids"]).loss.item()
        if lm < ln:
            match += 1
    return match / len(items)


def train_sft(model, tok, train_items, device, seed):
    rng = random.Random(seed)
    items = train_items[:]
    rng.shuffle(items)
    items = items[:N_TRAIN]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        batch = [items[i % len(items)] for i in range(step * 4, (step + 1) * 4)]
        total_loss = 0.0
        for item in batch:
            q = item["question"]
            ans = _s(item["answer_not_matching_behavior"])
            enc = tok(f"{q}{ans}", return_tensors="pt", truncation=True, max_length=512).to(device)
            loss = model(**enc, labels=enc["input_ids"]).loss
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if (step + 1) % 10 == 0:
            print(f"      step {step + 1}: loss={total_loss / 4:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = RESULTS_DIR / "results.json"
    res = json.load(open(rf)) if rf.exists() else {}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Two conditions: PiSSA init and OLoRA init (both data-agnostic structured inits)
    init_methods = {
        "pissa": "pissa_niter_4",  # PiSSA with 4 power iterations
        "olora": "olora",  # OLoRA (orthogonal)
    }

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)
        if not di:
            print(f"SKIP {beh_name}")
            continue

        for init_name, init_value in init_methods.items():
            print(f"\n{'=' * 60}")
            print(f"BEHAVIOR: {beh_name} | INIT: {init_name}")
            print(f"{'=' * 60}")

            for seed in SEEDS:
                rk = f"{beh_name}_{init_name}_s{seed}"
                if rk in res:
                    print(f"  {rk}: {res[rk]:.3f} (cached)")
                    continue

                print(f"  {rk}...", flush=True)
                t0 = time.time()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True,
                )
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=RANK,
                    lora_alpha=RANK,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    init_lora_weights=init_value,
                )
                model = get_peft_model(model, lora_config)

                train_sft(model, tok, tp, device, seed)
                score = eval_beh(model, tok, ei, device)

                del model
                gc.collect()
                torch.cuda.empty_cache()
                res[rk] = score
                print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("PiSSA/OLoRA Baseline Results")
    print(f"{'=' * 70}")
    print(f"{'Behavior':18s} | {'Method':8s} | {'Score':>12s}")
    print("-" * 50)
    for beh in BEHS:
        for init_name in init_methods:
            vals = [res.get(f"{beh}_{init_name}_s{s}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                print(
                    f"  {beh:16s} | {init_name:8s} | {np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}"
                )

    print(f"\nReference scores (from paper):")
    print(f"  Std LoRA (random init):  syc=0.533±0.074  pow=0.203±0.130  corr=0.324±0.029")
    print(f"  PR-LoRA (direction init): syc=0.427±0.007  pow=0.140±0.038  corr=0.140±0.054")


if __name__ == "__main__":
    main()
