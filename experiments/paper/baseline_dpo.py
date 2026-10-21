#!/usr/bin/env python3
"""DPO-LoRA baseline: train LoRA with DPO on same behavioral data as PR-LoRA.
Converts CAA contrastive pairs to preference pairs.
Tests sycophancy, power_seeking, corrigibility on Qwen3-8B, 3 seeds.
"""

import json, os, random, sys, time, warnings, gc, copy
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
# Manual DPO implementation (trl requires transformers<4.47, incompatible with Qwen3)

sys.stdout.reconfigure(line_buffering=True)
os.environ["WANDB_DISABLED"] = "true"

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/dpo_baseline"
RANK = 4
SEEDS = [42, 123, 456]

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


def run_dpo_manual(model, ref_model, tok, train_items, eval_items, device, seed, beta=0.1):
    """Manual DPO training since trl DPOTrainer may have compatibility issues."""
    from torch.utils.data import DataLoader

    rng = random.Random(seed)
    items_copy = train_items[:]
    rng.shuffle(items_copy)
    items_copy = items_copy[:50]

    pairs = []
    for item in items_copy:
        q = item["question"]
        good = _s(item["answer_not_matching_behavior"])
        bad = _s(item["answer_matching_behavior"])
        pairs.append((q, good, bad))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)

    def get_logprobs(mdl, prompt, response):
        text = f"{prompt}{response}"
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        prompt_len = prompt_enc["input_ids"].shape[1]

        with torch.no_grad() if mdl is ref_model else torch.enable_grad():
            outputs = mdl(**enc)
            logits = outputs.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            response_mask = torch.zeros_like(token_log_probs)
            if prompt_len - 1 < token_log_probs.shape[1]:
                response_mask[:, prompt_len - 1 :] = 1.0
            return (token_log_probs * response_mask).sum()

    model.train()
    for step in range(20):
        batch_pairs = [pairs[i % len(pairs)] for i in range(step * 4, (step + 1) * 4)]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for prompt, chosen, rejected in batch_pairs:
            pi_chosen = get_logprobs(model, prompt, chosen)
            pi_rejected = get_logprobs(model, prompt, rejected)

            with torch.no_grad():
                ref_chosen = get_logprobs(ref_model, prompt, chosen)
                ref_rejected = get_logprobs(ref_model, prompt, rejected)

            log_ratio_chosen = pi_chosen - ref_chosen
            log_ratio_rejected = pi_rejected - ref_rejected
            dpo_loss = -torch.nn.functional.logsigmoid(
                beta * (log_ratio_chosen - log_ratio_rejected)
            )
            total_loss = total_loss + dpo_loss

        total_loss = total_loss / len(batch_pairs)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()

        if (step + 1) % 10 == 0:
            print(f"    step {step + 1}: loss={total_loss.item():.4f}")

    return eval_beh(model, tok, eval_items, device)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = args.device

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = RESULTS_DIR / "results.json"
    res = json.load(open(rf)) if rf.exists() else {}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)
        if not di:
            print(f"SKIP {beh_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"BEHAVIOR: {beh_name} (DPO-LoRA baseline)")
        print(f"{'=' * 60}")

        for seed in SEEDS:
            rk = f"{beh_name}_dpo_s{seed}"
            if rk in res:
                print(f"  {rk}: {res[rk]:.3f} (cached)")
                continue

            print(f"  {rk}...", flush=True)
            t0 = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            ref_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            ref_model.eval()

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

            score = run_dpo_manual(model, ref_model, tok, tp, ei, device, seed)

            del model, ref_model
            gc.collect()
            torch.cuda.empty_cache()
            res[rk] = score
            print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("DPO-LoRA Baseline Results")
    print(f"{'=' * 70}")
    for beh in BEHS:
        vals = [res.get(f"{beh}_dpo_s{s}") for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"  {beh:18s}: {np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}")


if __name__ == "__main__":
    main()
