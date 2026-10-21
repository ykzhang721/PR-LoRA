#!/usr/bin/env python3
"""Qwen3-4B single-direction evaluation: adds a third model scale point.
Tests sycophancy, power_seeking, corrigibility with 3 seeds + MMLU.
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)
os.environ["WANDB_DISABLED"] = "true"

MODEL_PATH = os.environ.get("PR_LORA_QWEN4B", "Qwen/Qwen3-4B")
SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
MMLU_DIR = SCRIPT_DIR / "benchmarks/data/mmlu"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/qwen4b_single"
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


def eval_mmlu(model, tok, device):
    fp = MMLU_DIR / "test.json"
    if not fp.exists():
        return None
    data = json.load(open(fp))
    correct = 0
    total = 0
    model.eval()
    for item in data:
        q = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        losses = []
        for c in choices:
            enc = tok(f"{q} {c}", return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                loss = model(**enc, labels=enc["input_ids"]).loss.item()
            losses.append(loss)
        if np.argmin(losses) == answer:
            correct += 1
        total += 1
    return correct / total


def extract_directions(model, tok, dir_items, device):
    model.eval()
    directions = {}
    # Qwen3-4B has 36 layers, use layers 10-28 for steering
    for layer_idx in range(10, 29):
        diffs = []
        for item in dir_items[:200]:
            q = item["question"]
            pos = _s(item["answer_not_matching_behavior"])
            neg = _s(item["answer_matching_behavior"])
            enc_pos = tok(f"{q}{pos}", return_tensors="pt", truncation=True, max_length=512).to(
                device
            )
            enc_neg = tok(f"{q}{neg}", return_tensors="pt", truncation=True, max_length=512).to(
                device
            )
            with torch.no_grad():
                h_pos = (
                    model(**enc_pos, output_hidden_states=True)
                    .hidden_states[layer_idx][:, -1, :]
                    .squeeze()
                )
                h_neg = (
                    model(**enc_neg, output_hidden_states=True)
                    .hidden_states[layer_idx][:, -1, :]
                    .squeeze()
                )
            diffs.append(h_pos - h_neg)
        mean_diff = torch.stack(diffs).mean(dim=0)
        directions[layer_idx] = mean_diff / mean_diff.norm()
    return directions


def train_pr_lora(model, tok, train_items, directions, device, seed):
    rng = random.Random(seed)
    items = train_items[:]
    rng.shuffle(items)
    items = items[:N_TRAIN]
    b_init = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            for li in directions:
                if f"layers.{li}." in name:
                    d = directions[li]
                    if d.shape[0] == param.shape[0]:
                        with torch.no_grad():
                            for j in range(param.shape[1]):
                                param[:, j] = d
                    b_init[name] = param.clone().detach()
                    break
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        batch = [items[i % len(items)] for i in range(step * 4, (step + 1) * 4)]
        sft_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for item in batch:
            q = item["question"]
            ans = _s(item["answer_not_matching_behavior"])
            enc = tok(f"{q}{ans}", return_tensors="pt", truncation=True, max_length=512).to(device)
            sft_loss = sft_loss + model(**enc, labels=enc["input_ids"]).loss
        reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in b_init:
                reg = reg + torch.norm(param - b_init[name]) ** 2
        (sft_loss / 4 + 0.1 * reg).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()


def train_std_lora(model, tok, train_items, device, seed):
    rng = random.Random(seed)
    items = train_items[:]
    rng.shuffle(items)
    items = items[:N_TRAIN]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        batch = [items[i % len(items)] for i in range(step * 4, (step + 1) * 4)]
        for item in batch:
            q = item["question"]
            ans = _s(item["answer_not_matching_behavior"])
            enc = tok(f"{q}{ans}", return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**enc, labels=enc["input_ids"]).loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()


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

    # Eval baseline first
    for beh_name in BEHS:
        rk = f"{beh_name}_baseline"
        if rk not in res:
            di, ei, tp = get_data_split(beh_name)
            print(f"\n=== {beh_name} baseline ===")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            score = eval_beh(model, tok, ei, device)
            res[rk] = score
            print(f"  -> {score:.3f}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

    # Eval steering (approximate with alpha=8 on layers 10-28)
    # Skip steering for now — focus on LoRA comparisons

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)

        for method in ["std_lora", "pr_lora"]:
            for seed in SEEDS:
                rk = f"{beh_name}_{method}_s{seed}"
                if rk in res:
                    print(f"  {rk}: {res[rk]:.3f} (cached)")
                    continue

                print(f"\n  {rk}...", flush=True)
                t0 = time.time()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                if method == "pr_lora":
                    # Extract directions
                    base = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=torch.bfloat16,
                        device_map=device,
                        trust_remote_code=True,
                    )
                    directions = extract_directions(base, tok, di, device)
                    del base
                    gc.collect()
                    torch.cuda.empty_cache()

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True,
                )
                model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=RANK,
                        lora_alpha=RANK,
                        lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    ),
                )

                if method == "pr_lora":
                    train_pr_lora(model, tok, tp, directions, device, seed)
                else:
                    train_std_lora(model, tok, tp, device, seed)

                score = eval_beh(model, tok, ei, device)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                res[rk] = score
                print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

    # MMLU for PR-LoRA sycophancy
    rk = "mmlu_pr_lora_syc"
    if rk not in res:
        print("\n=== MMLU (PR-LoRA sycophancy) ===")
        di, ei, tp = get_data_split("sycophancy")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        directions = extract_directions(base, tok, di, device)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=RANK,
                lora_alpha=RANK,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        train_pr_lora(model, tok, tp, directions, device, 42)
        mmlu = eval_mmlu(model, tok, device)
        if mmlu:
            res[rk] = mmlu
            print(f"  -> MMLU={mmlu:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)

    rk = "mmlu_baseline"
    if rk not in res:
        print("\n=== MMLU (baseline) ===")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        mmlu = eval_mmlu(model, tok, device)
        if mmlu:
            res[rk] = mmlu
            print(f"  -> MMLU={mmlu:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("Qwen3-4B Single-Direction Results")
    print(f"{'=' * 60}")
    for beh in BEHS:
        base = res.get(f"{beh}_baseline", "?")
        std_vals = [res.get(f"{beh}_std_lora_s{s}") for s in SEEDS]
        pr_vals = [res.get(f"{beh}_pr_lora_s{s}") for s in SEEDS]
        std_vals = [v for v in std_vals if v is not None]
        pr_vals = [v for v in pr_vals if v is not None]
        print(
            f"  {beh:18s}: base={base:.3f}"
            + (f"  std={np.mean(std_vals):.3f}±{np.std(std_vals, ddof=1):.3f}" if std_vals else "")
            + (f"  pr={np.mean(pr_vals):.3f}±{np.std(pr_vals, ddof=1):.3f}" if pr_vals else "")
        )
    print(f"\n  MMLU baseline: {res.get('mmlu_baseline', '?')}")
    print(f"  MMLU PR-LoRA:  {res.get('mmlu_pr_lora_syc', '?')}")


if __name__ == "__main__":
    main()
