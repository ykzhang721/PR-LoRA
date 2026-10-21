#!/usr/bin/env python3
"""Multi-direction v6 method on Qwen3-8B: frozen B + bias=α*/10 + L2 on A.
Same approach that worked on 1.7B (comp_r3).
B=[d_syc|d_power|d_syc_phil], each col frozen, bias per-col = α*/10.
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
RESULTS_DIR = ROOT / "benchmarks/results/multi_v6_8b"
LR = 5e-4
L2_WEIGHT = 0.1
RANK = 3
STEPS = 20
BIAS_SCALE = 0.1 * (4096 / 2048)  # adaptive for 8B
SEEDS = [42, 123, 456]

TARGET_BEHS = ["sycophancy", "power_seeking", "sycophancy_phil"]
ALPHAS = {
    "sycophancy": 8.0,
    "power_seeking": 8.0,
    "sycophancy_phil": 8.0,
}

ALL_BEHS = {
    "sycophancy": "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_phil": "sycophancy_on_philpapers2020.jsonl",
    "power_seeking": "power-seeking-inclination.jsonl",
    "survival": "survival-instinct.jsonl",
    "myopia": "myopic-reward.jsonl",
    "corrigibility": "corrigible-less-HHH.jsonl",
    "coordination": "coordinate-other-ais.jsonl",
}


def _s(v):
    return v[0].strip() if isinstance(v, list) else v.strip()


def load_jsonl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


class LoRABiasWrapper(nn.Module):
    def __init__(s, m, init_bias):
        super().__init__()
        s.lora_a = m
        if isinstance(init_bias, (int, float)):
            init_bias = torch.full((m.weight.shape[0],), init_bias)
        s.gate_bias = nn.Parameter(init_bias.float())

    @property
    def weight(s):
        return s.lora_a.weight

    def forward(s, x):
        return s.lora_a(x) + s.gate_bias


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
    test_data = json.load(open(ROOT / "benchmarks/data/mmlu/test.json"))
    by_subject = {}
    for item in test_data:
        by_subject.setdefault(item["subject"], []).append(item)
    correct = 0
    total = 0
    for subj in sorted(by_subject.keys()):
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
                if -loss > best_logp:
                    best_logp = -loss
                    best_label = i
            if best_label == item["answer"]:
                correct += 1
            total += 1
    return correct / total


def get_eval_items(beh_name):
    fp = EVAL_DIR / ALL_BEHS[beh_name]
    if not fp.exists():
        return []
    data = load_jsonl(fp)
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    nd = min(500, len(sh) * 3 // 10)
    ne = min(300, len(sh) * 3 // 10)
    return sh[nd : nd + ne]


def get_train_data(beh_name, seed, n=50):
    fp = EVAL_DIR / ALL_BEHS[beh_name]
    data = load_jsonl(fp)
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    nd = min(500, len(sh) * 3 // 10)
    ne = min(300, len(sh) * 3 // 10)
    tp = sh[nd + ne :]
    rng2 = random.Random(seed)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    return [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:n]
    ]


def inject_multi_B_frozen(model, directions_list, rank):
    """Freeze B columns: col i = direction i."""
    for name, param in model.named_parameters():
        if "lora_B" not in name or "weight" not in name:
            continue
        parts = name.split(".")
        li = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                li = int(parts[i + 1])
        if li is None or param.shape[1] != rank:
            continue

        for col_idx, direction in enumerate(directions_list):
            if col_idx >= rank:
                break
            if li not in direction:
                continue
            d = direction[li]
            if len(d) != param.shape[0]:
                continue
            with torch.no_grad():
                param[:, col_idx] = torch.tensor(d, dtype=param.dtype).to(param.device)
        param.requires_grad = False


def add_bias_wrapper(model, alpha_list):
    """Wrap lora_A with per-column bias = α_i * BIAS_SCALE."""
    bias_val = torch.tensor([a * BIAS_SCALE for a in alpha_list], dtype=torch.float32)
    for name, module in list(model.named_modules()):
        if "lora_A" in name and isinstance(module, nn.Linear):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[-1],
                LoRABiasWrapper(module, bias_val).to(next(module.parameters()).device),
            )


def train_v6(model, data, tok, device, steps):
    """Train only gate A (with L2 weight_decay) — B is frozen."""
    ds = SDS(data, tok)
    loader = DataLoader(ds, batch_size=min(4, len(data)), shuffle=True)
    gA = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_A" in n and "gate_bias" not in n
    ]
    rest = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not ("lora_A" in n and "gate_bias" not in n)
    ]
    if not gA and not rest:
        return
    opt = torch.optim.AdamW(
        [
            {"params": gA, "lr": LR, "weight_decay": L2_WEIGHT},
            {"params": rest, "lr": LR, "weight_decay": 0.01},
        ]
    )
    model.train()
    step = 0
    for _ in range(500):
        for batch in loader:
            if step >= steps:
                return
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1


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

    directions = []
    alpha_list = []
    for beh in TARGET_BEHS:
        directions.append(torch.load(DIR_CACHE / f"{beh}_dir.pt", weights_only=False))
        alpha_list.append(ALPHAS[beh])
    print(f"Directions: {TARGET_BEHS}, alphas: {alpha_list}, bias_scale: {BIAS_SCALE}")

    all_eval = {beh: get_eval_items(beh) for beh in ALL_BEHS}

    for seed in SEEDS:
        mixed_td = []
        for beh in TARGET_BEHS:
            mixed_td.extend(get_train_data(beh, seed, n=50))
        random.Random(seed).shuffle(mixed_td)

        for meth in ["v6_comp_r3", "std_r3"]:
            pfx = f"{meth}_s{seed}"

            all_done = all(f"{pfx}_{beh}" in res for beh in ALL_BEHS) and f"{pfx}_mmlu" in res
            if all_done:
                print(f"\n  {pfx}: all cached")
                continue

            print(f"\n  Training {pfx}...", flush=True)
            t0 = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

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

            if meth == "v6_comp_r3":
                inject_multi_B_frozen(model, directions, RANK)
                add_bias_wrapper(model, alpha_list)
                train_v6(model, mixed_td, tok, device, STEPS)
            else:
                # Standard r3: train normally
                ds = SDS(mixed_td, tok)
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
                        out.loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()
                        step += 1
                    if step >= STEPS:
                        break

            print(f"  Training done ({time.time() - t0:.0f}s). Evaluating...", flush=True)

            for beh in ALL_BEHS:
                rk = f"{pfx}_{beh}"
                if rk in res:
                    print(f"    {beh}: {res[rk]:.3f} (cached)")
                    continue
                if not all_eval[beh]:
                    continue
                score = eval_beh(model, tok, all_eval[beh], device)
                res[rk] = score
                print(f"    {beh}: {score:.3f}")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

            rk_mmlu = f"{pfx}_mmlu"
            if rk_mmlu not in res:
                mmlu = eval_mmlu(model, tok, device)
                res[rk_mmlu] = mmlu
                print(f"    MMLU: {mmlu:.4f}")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

            del model
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"MULTI-DIRECTION v6 (frozen B + bias) vs std_r3 (Qwen3-8B, steps={STEPS})")
    print(f"{'=' * 80}")

    # Load previous PR-LoRA results for 3-way comparison
    pr_res_path = ROOT / "benchmarks/results/multi_pr_lora/results.json"
    pr_res = json.load(open(pr_res_path)) if pr_res_path.exists() else {}

    print(f"\n  TARGET behaviors:")
    for beh in TARGET_BEHS:
        for meth, src, prefix in [
            ("std_r3", res, "std_r3"),
            ("v6_comp", res, "v6_comp_r3"),
            ("pr_comp", pr_res, "pr_comp_r3"),
        ]:
            vals = [src.get(f"{prefix}_s{s}_{beh}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                m = np.mean(vals)
                print(f"    {beh:20s} {meth:10s}: {m:.3f} ±{np.std(vals):.3f}")

    print(f"\n  NON-TARGET behaviors:")
    non_targets = [b for b in ALL_BEHS if b not in TARGET_BEHS]
    for beh in non_targets:
        for meth, src, prefix in [
            ("std_r3", res, "std_r3"),
            ("v6_comp", res, "v6_comp_r3"),
            ("pr_comp", pr_res, "pr_comp_r3"),
        ]:
            vals = [src.get(f"{prefix}_s{s}_{beh}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                m = np.mean(vals)
                print(f"    {beh:20s} {meth:10s}: {m:.3f} ±{np.std(vals):.3f}")

    print(f"\n  MMLU:")
    for meth, src, prefix in [
        ("std_r3", res, "std_r3"),
        ("v6_comp", res, "v6_comp_r3"),
        ("pr_comp", pr_res, "pr_comp_r3"),
    ]:
        vals = [src.get(f"{prefix}_s{s}_mmlu") for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if vals:
            m = np.mean(vals)
            print(f"    {meth:10s}: {m:.4f} ±{np.std(vals):.4f}")


if __name__ == "__main__":
    main()
