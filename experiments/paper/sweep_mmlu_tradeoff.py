#!/usr/bin/env python3
"""Sweep λ and steps for PR-LoRA r4 on Qwen3-8B.
Measure both sycophancy alignment AND MMLU for each config.
Goal: find Pareto-optimal config (strong alignment + minimal MMLU drop).
"""

import json, os, random, sys, time, warnings
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
EVAL_DIR = ROOT / "benchmarks/data/anthropic_evals"
DIR_CACHE = ROOT / "benchmarks/results/all_dirs_8b"
RESULTS_DIR = ROOT / "benchmarks/results/mmlu_tradeoff"
LR = 5e-4

LAMBDAS = [0.0, 0.01, 0.03, 0.1]
STEPS_LIST = [10, 20, 40]
RANK = 4
SEED = 42


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


def eval_mmlu(model, tok, device, n_per_subject=5):
    mmlu_dir = ROOT / "benchmarks/data/mmlu"
    test_data = json.load(open(mmlu_dir / "test.json"))
    by_subject = {}
    for item in test_data:
        by_subject.setdefault(item["subject"], []).append(item)

    correct = 0
    total = 0
    subjects = sorted(by_subject.keys())
    for subj in subjects:
        items = by_subject[subj][:n_per_subject]
        for item in items:
            q = item["question"]
            choices = item["choices"]
            labels = ["A", "B", "C", "D"]
            prompt = f"Question: {q}\n"
            for i, c in enumerate(choices):
                prompt += f"{labels[i]}. {c}\n"
            prompt += "Answer:"
            best_label = -1
            best_logp = float("-inf")
            for i, label in enumerate(labels):
                enc = tok(
                    f"{prompt} {label}", return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                with torch.no_grad():
                    loss = model(**enc, labels=enc["input_ids"]).loss.item()
                logp = -loss
                if logp > best_logp:
                    best_logp = logp
                    best_label = i
            if best_label == item["answer"]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0


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

    direction = torch.load(DIR_CACHE / "sycophancy_dir.pt", weights_only=False)

    # Data split (same as other experiments)
    data = load_jsonl(EVAL_DIR / "sycophancy_on_nlp_survey.jsonl")
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    nd = min(500, len(sh) * 3 // 10)
    ne = min(300, len(sh) * 3 // 10)
    ei = sh[nd : nd + ne]
    tp = sh[nd + ne :]
    random.seed(SEED)
    rng2 = random.Random(SEED)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    td = [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:50]
    ]

    # Baseline MMLU (cached from previous)
    if "baseline_mmlu" not in res:
        print("Evaluating baseline MMLU...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        res["baseline_mmlu"] = eval_mmlu(model, tok, device)
        res["baseline_syc"] = eval_beh(model, tok, ei, device)
        del model
        torch.cuda.empty_cache()
        print(f"  baseline: syc={res['baseline_syc']:.3f} mmlu={res['baseline_mmlu']:.4f}")
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)

    # Sweep: λ × steps
    for lam in LAMBDAS:
        for steps in STEPS_LIST:
            rk = f"pr_r{RANK}_lam{lam}_st{steps}"
            if f"{rk}_syc" in res and f"{rk}_mmlu" in res:
                print(f"  {rk}: syc={res[f'{rk}_syc']:.3f} mmlu={res[f'{rk}_mmlu']:.4f} (cached)")
                continue

            print(f"\n  {rk} (λ={lam}, steps={steps})...", flush=True)
            t0 = time.time()

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
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

            # Init B to persona direction
            b_targets = {}
            if lam > 0:
                for name, param in model.named_parameters():
                    if "lora_B" not in name or "weight" not in name:
                        continue
                    parts = name.split(".")
                    li = None
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            li = int(parts[i + 1])
                    if li is None or li not in direction:
                        continue
                    d = direction[li]
                    if param.shape[0] != len(d) or param.shape[1] != RANK:
                        continue
                    B = np.stack([d] * RANK, axis=1)
                    with torch.no_grad():
                        param.copy_(torch.tensor(B, dtype=param.dtype).to(param.device))
                    b_targets[name] = torch.tensor(B, dtype=torch.float32)
                    param.requires_grad = True

            # Train
            ds = SDS(td, tok)
            loader = DataLoader(ds, batch_size=4, shuffle=True)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
            model.train()
            step = 0
            for _ in range(500):
                for batch in loader:
                    if step >= steps:
                        break
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    if lam > 0 and b_targets:
                        b_reg = torch.tensor(0.0, device=device)
                        for name, param in model.named_parameters():
                            if name in b_targets:
                                target = b_targets[name].to(param.device).to(param.dtype)
                                b_reg = b_reg + ((param - target) ** 2).sum()
                        loss = out.loss + lam * b_reg
                    else:
                        loss = out.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                if step >= steps:
                    break

            # Evaluate both
            syc_score = eval_beh(model, tok, ei, device)
            mmlu_score = eval_mmlu(model, tok, device)
            del model
            torch.cuda.empty_cache()

            res[f"{rk}_syc"] = syc_score
            res[f"{rk}_mmlu"] = mmlu_score

            # Compute cos if applicable
            print(f"    syc={syc_score:.3f} mmlu={mmlu_score:.4f} ({time.time() - t0:.0f}s)")
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"PARETO SWEEP: λ × steps → sycophancy + MMLU (Qwen3-8B, rank={RANK})")
    print(f"{'=' * 80}")
    base_syc = res.get("baseline_syc", 0)
    base_mmlu = res.get("baseline_mmlu", 0)
    print(f"  baseline: syc={base_syc:.3f} mmlu={base_mmlu:.4f}")
    print(f"\n  {'config':<28} {'syc':>6} {'Δsyc':>7} {'mmlu':>7} {'Δmmlu':>8} {'Pareto?':>8}")
    print(f"  {'-' * 68}")

    configs = []
    for lam in LAMBDAS:
        for steps in STEPS_LIST:
            rk = f"pr_r{RANK}_lam{lam}_st{steps}"
            syc = res.get(f"{rk}_syc")
            mmlu = res.get(f"{rk}_mmlu")
            if syc is not None and mmlu is not None:
                configs.append((rk, lam, steps, syc, mmlu))

    # Find Pareto front
    pareto = []
    for i, (rk, lam, steps, syc, mmlu) in enumerate(configs):
        dominated = False
        for j, (rk2, lam2, steps2, syc2, mmlu2) in enumerate(configs):
            if i == j:
                continue
            if syc2 <= syc and mmlu2 >= mmlu and (syc2 < syc or mmlu2 > mmlu):
                dominated = True
                break
        if not dominated:
            pareto.append(rk)

    for rk, lam, steps, syc, mmlu in configs:
        d_syc = syc - base_syc
        d_mmlu = mmlu - base_mmlu
        is_pareto = "★" if rk in pareto else ""
        print(f"  {rk:<28} {syc:>6.3f} {d_syc:>+6.3f} {mmlu:>7.4f} {d_mmlu:>+7.4f} {is_pareto:>8}")


if __name__ == "__main__":
    main()
