#!/usr/bin/env python3
"""Critical ablation: disentangle direction initialization vs L2 regularization.

Tests 4 conditions on 3 core behaviors (sycophancy, power_seeking, corrigibility), 3 seeds:
  1. rand_init_l2:   Random unit B init + L2 reg toward random init (lambda=0.1)
  2. dir_init_nol2:  Direction B init + NO L2 reg (lambda=0)
  3. dir_init_l2:    Direction B init + L2 reg (= PR-LoRA, for reference)
  4. std_lora:       Random B init + no reg (= standard LoRA, for reference)

This ablation establishes whether the behavioral direction or the regularization
structure drives PR-LoRA's improvement over standard LoRA.
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
DIR_CACHE = SCRIPT_DIR / "benchmarks/results/all_dirs_8b"
OLD_DIR_CACHE = SCRIPT_DIR / "benchmarks/results/align_v6"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/ablation_dir_vs_reg"
LR = 5e-4
PR_LAMBDA = 0.1
RANK = 4
STEPS = 20
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


class SDS(Dataset):
    def __init__(s, data, tok, ml=384):
        s.ex = []
        for item in data:
            e = tok(
                f"User: {item['question']}\nAssistant: {item['answer']}",
                truncation=True,
                max_length=ml,
                return_tensors="pt",
                padding="max_length",
            )
            ids = e["input_ids"].squeeze(0)
            mask = e["attention_mask"].squeeze(0)
            lab = ids.clone()
            lab[mask == 0] = -100
            s.ex.append({"input_ids": ids, "attention_mask": mask, "labels": lab})

    def __len__(s):
        return len(s.ex)

    def __getitem__(s, i):
        return s.ex[i]


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


def load_direction(beh_name):
    for cache in [DIR_CACHE, OLD_DIR_CACHE]:
        dc = cache / f"{beh_name}_dir.pt"
        if dc.exists():
            return torch.load(dc, weights_only=False)
    return None


def generate_random_directions(direction, seed):
    """Generate random unit directions with same shape as real directions."""
    rng = np.random.RandomState(seed + 9999)
    rand_dir = {}
    for li, d in direction.items():
        v = rng.randn(len(d)).astype(np.float32)
        v = v / np.linalg.norm(v)
        rand_dir[li] = v
    return rand_dir


def run_one(model, tok, direction, td, ei, device, method, seed, b_init_vectors):
    """Train and evaluate one configuration.

    method: 'rand_init_l2' | 'dir_init_nol2' | 'dir_init_l2' | 'std_lora'
    b_init_vectors: the vectors to init B (real direction or random)
    """
    ds = SDS(td, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    use_direction_init = method in ["dir_init_l2", "dir_init_nol2", "rand_init_l2"]
    use_l2_reg = method in ["dir_init_l2", "rand_init_l2"]

    b_targets = {}
    if use_direction_init:
        for name, param in model.named_parameters():
            if "lora_B" not in name or "weight" not in name:
                continue
            parts = name.split(".")
            li = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    li = int(parts[i + 1])
            if li is None or li not in b_init_vectors:
                continue
            d = b_init_vectors[li]
            if param.shape[0] != len(d) or param.shape[1] != RANK:
                continue
            B = np.stack([d] * RANK, axis=1)
            with torch.no_grad():
                param.copy_(torch.tensor(B, dtype=param.dtype).to(param.device))
            if use_l2_reg:
                b_targets[name] = torch.tensor(B, dtype=torch.float32)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    model.train()
    step = 0
    for _ in range(500):
        for batch in loader:
            if step >= STEPS:
                break
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss
            if use_l2_reg and b_targets:
                b_reg = torch.tensor(0.0, device=device)
                for bname, param in model.named_parameters():
                    if bname in b_targets:
                        target = b_targets[bname].to(param.device).to(param.dtype)
                        b_reg = b_reg + ((param - target) ** 2).sum()
                loss = loss + PR_LAMBDA * b_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
        if step >= STEPS:
            break

    score = eval_beh(model, tok, ei, device)
    return score


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

    methods = ["rand_init_l2", "dir_init_nol2", "dir_init_l2", "std_lora"]

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)
        if not di:
            print(f"SKIP {beh_name}")
            continue

        direction = load_direction(beh_name)
        if direction is None:
            print(f"SKIP {beh_name} (no direction)")
            continue

        print(f"\n{'=' * 60}")
        print(f"BEHAVIOR: {beh_name}")
        print(f"{'=' * 60}")

        for seed in SEEDS:
            rng2 = random.Random(seed)
            tp2 = tp[:]
            rng2.shuffle(tp2)
            td = [
                {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
                for s in tp2[:50]
            ]

            rand_dir = generate_random_directions(direction, seed)

            for meth in methods:
                rk = f"{beh_name}_{meth}_s{seed}"
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

                if meth == "rand_init_l2":
                    b_init = rand_dir
                elif meth in ["dir_init_nol2", "dir_init_l2"]:
                    b_init = direction
                else:
                    b_init = direction  # won't be used

                score = run_one(model, tok, direction, td, ei, device, meth, seed, b_init)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                res[rk] = score
                print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"ABLATION: Direction Init vs L2 Regularization")
    print(f"{'=' * 80}")
    print(
        f"  {'behavior':<18} {'std_lora':>10} {'rand+L2':>10} {'dir+noL2':>10} {'dir+L2(PR)':>10}"
    )
    print(f"  {'-' * 60}")
    for beh in BEHS:
        vals = {}
        for meth in methods:
            scores = [res.get(f"{beh}_{meth}_s{s}") for s in SEEDS]
            scores = [v for v in scores if v is not None]
            vals[meth] = f"{np.mean(scores):.3f}±{np.std(scores):.3f}" if scores else "N/A"
        print(
            f"  {beh:<18} {vals['std_lora']:>10} {vals['rand_init_l2']:>10} {vals['dir_init_nol2']:>10} {vals['dir_init_l2']:>10}"
        )


if __name__ == "__main__":
    main()
