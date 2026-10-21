#!/usr/bin/env python3
"""P0.2 Rebuttal: Full MMLU evaluation (all ~14k items instead of 5-per-subject).

Evaluates capability preservation with proper confidence intervals.
Runs for baseline, std_r4, pr_r4 on:
  - Qwen3-8B: 10 seeds
  - LLaMA-3.1-8B: 5 seeds

Usage:
  python rebuttal_p0_full_mmlu.py --device cuda:0 --model qwen3_8b
  python rebuttal_p0_full_mmlu.py --device cuda:0 --model llama31_8b
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)

EVAL_DIR = ROOT / "benchmarks/data/anthropic_evals"
RESULTS_DIR = ROOT / "benchmarks/results/rebuttal_full_mmlu"
LR = 5e-4
PR_LAMBDA = 0.1
RANK = 4
STEPS = 20

MODELS = {
    "qwen3_8b": {
        "path": os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B"),
        "n_layers": 36,
        "steer_layers": list(range(14, 32)),
        "dir_cache": (ROOT / "benchmarks/results/all_dirs_8b"),
        "seeds": [42, 123, 456, 789, 1024, 2048, 3407, 5555, 7777, 9999],
    },
    "llama31_8b": {
        "path": os.environ.get("PR_LORA_LLAMA8B", "meta-llama/Llama-3.1-8B-Instruct"),
        "n_layers": 32,
        "steer_layers": list(range(12, 28)),
        "dir_cache": (ROOT / "benchmarks/results/pr_lora_llama8b"),
        "seeds": [42, 123, 456, 789, 1024],
    },
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


def get_acts(model, tok, text, device, n_layers):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    acts = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):

        def mk(l):
            def fn(mod, i, o):
                h = o[0] if isinstance(o, tuple) else o
                acts[l] = h[:, -1, :].detach().cpu().float().numpy().flatten()

            return fn

        hooks.append(layer.register_forward_hook(mk(li)))
    with torch.no_grad():
        model(**inp)
    for h in hooks:
        h.remove()
    return acts


def extract_direction(model, tok, items, device, n_layers):
    model.eval()
    pa = {i: [] for i in range(n_layers)}
    na = {i: [] for i in range(n_layers)}
    for i, item in enumerate(items):
        q = item["question"]
        for text, tgt in [
            (f"{q}{_s(item['answer_not_matching_behavior'])}", pa),
            (f"{q}{_s(item['answer_matching_behavior'])}", na),
        ]:
            a = get_acts(model, tok, text, device, n_layers)
            for li in a:
                tgt[li].append(a[li])
        if (i + 1) % 100 == 0:
            print(f"      dir:{i + 1}/{len(items)}", flush=True)
    d = {}
    for li in range(n_layers):
        if pa[li] and na[li]:
            v = np.mean(pa[li], 0) - np.mean(na[li], 0)
            nm = np.linalg.norm(v)
            if nm > 1e-8:
                d[li] = v / nm
    return d


def eval_mmlu_full(model, tok, device):
    """Evaluate on ALL MMLU test items (not sampled)."""
    mmlu_dir = ROOT / "benchmarks/data/mmlu"
    if not (mmlu_dir / "test.json").exists():
        print("  Downloading MMLU...", flush=True)
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", cache_dir=str(ROOT / "benchmarks/data/hf_cache"))
        mmlu_dir.mkdir(parents=True, exist_ok=True)
        for split in ["test", "validation"]:
            items = [
                {
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "subject": row["subject"],
                }
                for row in ds[split]
            ]
            with open(mmlu_dir / f"{split}.json", "w") as f:
                json.dump(items, f)
        print(f"  MMLU saved: {len(ds['test'])} test items")

    test_data = json.load(open(mmlu_dir / "test.json"))
    print(f"  Evaluating on {len(test_data)} MMLU items...", flush=True)

    model.eval()
    correct = 0
    total = 0
    labels = ["A", "B", "C", "D"]
    t0 = time.time()

    for idx, item in enumerate(test_data):
        prompt = f"Question: {item['question']}\n"
        for i, c in enumerate(item["choices"]):
            prompt += f"({labels[i]}) {c}\n"
        prompt += "Answer: ("

        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
        toks = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        if int(np.argmax([logits[t].item() for t in toks])) == item["answer"]:
            correct += 1
        total += 1

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(test_data) - idx - 1)
            print(
                f"    {idx + 1}/{len(test_data)} ({correct / total:.4f}) ETA {eta:.0f}s", flush=True
            )

    score = correct / total
    print(f"  Full MMLU: {score:.4f} ({correct}/{total}, {time.time() - t0:.0f}s)")
    return score


def train_and_eval_mmlu(model_name, device, method, seed, direction, train_data, res, rf, mcfg):
    rk = f"{model_name}_{method}_mmlu_s{seed}"
    if rk in res:
        print(f"  {rk}: {res[rk]:.4f} (cached)")
        return

    print(f"\n  {rk}...", flush=True)
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
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

    b_targets = {}
    if method == "pr_r4":
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

    # Train on sycophancy data (same as paper)
    rng2 = random.Random(seed)
    td_shuffled = train_data[:]
    rng2.shuffle(td_shuffled)
    td = td_shuffled[:50]
    tok_train = AutoTokenizer.from_pretrained(mcfg["path"], trust_remote_code=True)
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token
    ds = SDS(td, tok_train)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
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
            if b_targets:
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

    # Eval full MMLU
    tok_eval = AutoTokenizer.from_pretrained(mcfg["path"], trust_remote_code=True)
    if tok_eval.pad_token is None:
        tok_eval.pad_token = tok_eval.eos_token
    score = eval_mmlu_full(model, tok_eval, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    res[rk] = score
    print(f"    -> {score:.4f} ({time.time() - t0:.0f}s)")
    with open(rf, "w") as f:
        json.dump(res, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="qwen3_8b", choices=list(MODELS.keys()))
    args = parser.parse_args()
    device = args.device
    model_name = args.model
    mcfg = MODELS[model_name]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = RESULTS_DIR / f"{model_name}.json"
    res = json.load(open(rf)) if rf.exists() else {}

    tok = AutoTokenizer.from_pretrained(mcfg["path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. Baseline full MMLU
    bl_key = f"{model_name}_baseline_mmlu"
    if bl_key not in res:
        print(f"Evaluating baseline full MMLU on {model_name}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        res[bl_key] = eval_mmlu_full(model, tok, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)
    print(f"  Baseline MMLU: {res[bl_key]:.4f}")

    # 2. Load sycophancy direction + training data
    dc = mcfg["dir_cache"] / "sycophancy_dir.pt"
    if not dc.exists():
        print(f"ERROR: no cached direction at {dc}. Run supplement/main experiments first.")
        sys.exit(1)
    direction = torch.load(dc, weights_only=False)

    syc_data = load_jsonl(EVAL_DIR / "sycophancy_on_nlp_survey.jsonl")
    rng = random.Random(42)
    syc_shuffled = syc_data[:]
    rng.shuffle(syc_shuffled)
    nd = min(500, len(syc_shuffled) * 3 // 10)
    ne = min(300, len(syc_shuffled) * 3 // 10)
    train_pool = syc_shuffled[nd + ne :]
    train_data = [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in train_pool
    ]

    # 3. Run std_r4 and pr_r4 with all seeds
    for seed in mcfg["seeds"]:
        for method in ["std_r4", "pr_r4"]:
            train_and_eval_mmlu(
                model_name, device, method, seed, direction, train_data, res, rf, mcfg
            )

    # ============================================================
    # SUMMARY with bootstrap CIs
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"FULL MMLU RESULTS ({model_name})")
    print(f"{'=' * 70}")

    bl = res.get(bl_key, 0)
    for method in ["std_r4", "pr_r4"]:
        vals = [res.get(f"{model_name}_{method}_mmlu_s{s}") for s in mcfg["seeds"]]
        vals = [v for v in vals if v is not None]
        if vals:
            mu = np.mean(vals)
            std = np.std(vals)
            # Bootstrap 95% CI
            n_boot = 10000
            boot_means = [
                np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)
            ]
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            delta = mu - bl
            print(
                f"  {method}: {mu:.4f} ± {std:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]  Δ={delta:+.4f}"
            )
        else:
            print(f"  {method}: no results yet")

    print(f"  baseline: {bl:.4f}")
    with open(rf, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
