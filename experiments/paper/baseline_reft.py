#!/usr/bin/env python3
"""ReFT baseline: LoReFT on same behavioral data as PR-LoRA.
Tests sycophancy, power_seeking, corrigibility on Qwen3-8B, 3 seeds.
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/reft_baseline"
SEEDS = [42, 123, 456]
STEER_LAYERS = list(range(14, 32))

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

    print(
        "Using manual LoReFT implementation (pyreft requires transformers 4.45.1, incompatible with Qwen3)"
    )

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)
        if not di:
            print(f"SKIP {beh_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"BEHAVIOR: {beh_name} (ReFT baseline)")
        print(f"{'=' * 60}")

        for seed in SEEDS:
            rk = f"{beh_name}_reft_s{seed}"
            if rk in res:
                print(f"  {rk}: {res[rk]:.3f} (cached)")
                continue

            print(f"  {rk}...", flush=True)
            t0 = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            rng2 = random.Random(seed)
            tp2 = tp[:]
            rng2.shuffle(tp2)
            train_data = tp2[:50]

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )

            score = _run_manual_loreft(model, tok, train_data, ei, device, seed)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            res[rk] = score
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("ReFT Baseline Results")
    print(f"{'=' * 70}")
    for beh in BEHS:
        vals = [res.get(f"{beh}_reft_s{s}") for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"  {beh:18s}: {np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}")


def _run_manual_loreft(model, tok, train_data, eval_items, device, seed):
    """Manual LoReFT: learn low-rank intervention on residual stream at steering layers."""
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn

    hidden_size = model.config.hidden_size

    class LoReFTIntervention(nn.Module):
        def __init__(self, d, r=4):
            super().__init__()
            self.R = nn.Parameter(torch.randn(r, d) * 0.1)
            self.W = nn.Parameter(torch.randn(r, d) * 0.1)
            self.b = nn.Parameter(torch.zeros(r))

        def forward(self, h):
            proj = torch.matmul(h, self.W.T) + self.b
            intervention = torch.matmul(proj, self.R)
            return h + intervention

    interventions = nn.ModuleDict(
        {
            str(l): LoReFTIntervention(hidden_size, r=4).to(device).to(torch.bfloat16)
            for l in STEER_LAYERS
        }
    )

    hooks = []
    for li in STEER_LAYERS:
        interv = interventions[str(li)]

        def make_hook(intervention):
            def hook_fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = intervention(h)
                return (h,) + out[1:] if isinstance(out, tuple) else h

            return hook_fn

        hooks.append(model.model.layers[li].register_forward_hook(make_hook(interv)))

    class SDS(Dataset):
        def __init__(self, data, tok, ml=384):
            self.ex = []
            for item in data:
                q = item["question"]
                a = _s(item["answer_not_matching_behavior"])
                e = tok(
                    f"User: {q}\nAssistant: {a}",
                    truncation=True,
                    max_length=ml,
                    return_tensors="pt",
                    padding="max_length",
                )
                ids = e["input_ids"].squeeze(0)
                mask = e["attention_mask"].squeeze(0)
                lab = ids.clone()
                lab[mask == 0] = -100
                self.ex.append({"input_ids": ids, "attention_mask": mask, "labels": lab})

        def __len__(self):
            return len(self.ex)

        def __getitem__(self, i):
            return self.ex[i]

    ds = SDS(train_data, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    for p in model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(interventions.parameters(), lr=2e-3, weight_decay=0.01)

    model.train()
    step = 0
    max_steps = 50
    for _ in range(500):
        for batch in loader:
            if step >= max_steps:
                break
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(interventions.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
        if step >= max_steps:
            break

    score = eval_beh(model, tok, eval_items, device)

    for h in hooks:
        h.remove()
    return score


if __name__ == "__main__":
    main()
