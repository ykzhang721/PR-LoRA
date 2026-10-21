#!/usr/bin/env python3
"""P1: SVD analysis of trained BA product for Frozen-B vs PR-LoRA.

Trains both methods once (seed=42, sycophancy), then extracts and analyzes
the BA matrix to show:
- Frozen-B: effective rank exactly 1 (all B columns identical and frozen)
- PR-LoRA: effective rank ~2-3 (B columns diverge slightly, expanding subspace)

This supports the corrected Frozen-B ablation interpretation in §3 of the rebuttal.

Usage:
  python rebuttal_p1_svd_analysis.py --device cuda:1
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
EVAL_DIR = ROOT / "benchmarks/data/anthropic_evals"
DIR_CACHE = ROOT / "benchmarks/results/all_dirs_8b"
RESULTS_DIR = ROOT / "benchmarks/results/rebuttal_svd_analysis"
LR = 5e-4
PR_LAMBDA = 0.1
RANK = 4
STEPS = 20
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


def get_data():
    data = load_jsonl(EVAL_DIR / "sycophancy_on_nlp_survey.jsonl")
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    total = len(sh)
    nd = min(500, total * 3 // 10)
    ne = min(300, total * 3 // 10)
    tp = sh[nd + ne :]
    rng2 = random.Random(SEED)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    return [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:50]
    ]


def train_and_extract(method, direction, train_data, tok, device):
    """Train a model, then extract all BA products and compute SVD."""
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

    b_targets = {}
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
        if method == "frozen_r4":
            param.requires_grad = False
        else:
            b_targets[name] = torch.tensor(B, dtype=torch.float32)
            param.requires_grad = True

    # Train
    ds = SDS(train_data, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if method == "frozen_r4":
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.1)
    else:
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
            if method == "pr_r4" and b_targets:
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

    # Extract BA products and compute SVD
    model.eval()
    ba_stats = []
    b_cosines = []

    lora_pairs = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and "weight" in name:
            key = name.replace("lora_B", "LORA").replace(".weight", "")
            lora_pairs.setdefault(key, {})["B"] = param.detach().float().cpu()
        elif "lora_A" in name and "weight" in name:
            key = name.replace("lora_A", "LORA").replace(".weight", "")
            lora_pairs.setdefault(key, {})["A"] = param.detach().float().cpu()

    for key in sorted(lora_pairs):
        if "A" not in lora_pairs[key] or "B" not in lora_pairs[key]:
            continue
        A = lora_pairs[key]["A"]  # (r, d_in)
        B = lora_pairs[key]["B"]  # (d_out, r)
        BA = B @ A  # (d_out, d_in)

        U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
        S = S.numpy()
        total = np.sum(S**2)
        if total < 1e-12:
            continue

        # Effective rank (exponential of entropy of normalized singular values)
        p = (S**2) / total
        p = p[p > 1e-10]
        eff_rank = np.exp(-np.sum(p * np.log(p)))

        # Top-1 and top-2 variance fraction
        top1_frac = S[0] ** 2 / total
        top2_frac = (S[0] ** 2 + S[1] ** 2) / total if len(S) > 1 else top1_frac

        # B column cosine with initial direction
        parts = key.split(".")
        li = None
        for i, p_name in enumerate(parts):
            if p_name == "layers" and i + 1 < len(parts):
                li = int(parts[i + 1])
        if li is not None and li in direction:
            d_vec = torch.tensor(direction[li], dtype=torch.float32)
            if B.shape[0] == len(d_vec):
                for col in range(B.shape[1]):
                    cos = torch.nn.functional.cosine_similarity(
                        B[:, col].unsqueeze(0), d_vec.unsqueeze(0)
                    ).item()
                    b_cosines.append(cos)

        # Inter-column cosine of B
        col_cos = []
        for i in range(B.shape[1]):
            for j in range(i + 1, B.shape[1]):
                c = torch.nn.functional.cosine_similarity(
                    B[:, i].unsqueeze(0), B[:, j].unsqueeze(0)
                ).item()
                col_cos.append(c)

        ba_stats.append(
            {
                "module": key.split(".")[-2] if "." in key else key,
                "sv": S.tolist(),
                "eff_rank": float(eff_rank),
                "top1_frac": float(top1_frac),
                "top2_frac": float(top2_frac),
                "b_col_inter_cos": float(np.mean(col_cos)) if col_cos else 1.0,
            }
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_eff_rank": float(np.mean([s["eff_rank"] for s in ba_stats])),
        "median_eff_rank": float(np.median([s["eff_rank"] for s in ba_stats])),
        "mean_top1_frac": float(np.mean([s["top1_frac"] for s in ba_stats])),
        "mean_top2_frac": float(np.mean([s["top2_frac"] for s in ba_stats])),
        "mean_b_col_inter_cos": float(np.mean([s["b_col_inter_cos"] for s in ba_stats])),
        "mean_b_direction_cos": float(np.mean(b_cosines)) if b_cosines else 0,
        "min_b_direction_cos": float(np.min(b_cosines)) if b_cosines else 0,
        "n_modules": len(ba_stats),
    }


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

    direction = torch.load(DIR_CACHE / "sycophancy_dir.pt", weights_only=False)
    train_data = get_data()

    for method in ["frozen_r4", "pr_r4"]:
        if method in res:
            print(f"  {method}: cached")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Training {method} and extracting BA SVD...")
        print(f"{'=' * 60}", flush=True)
        t0 = time.time()
        stats = train_and_extract(method, direction, train_data, tok, device)
        res[method] = stats
        print(f"  Done ({time.time() - t0:.0f}s)")
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)

    # Also analyze standard LoRA for comparison
    if "std_r4" not in res:
        print(f"\n  Training std_r4 for comparison...", flush=True)
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
        ds = SDS(train_data, tok)
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

        model.eval()
        lora_pairs = {}
        for name, param in model.named_parameters():
            if "lora_B" in name and "weight" in name:
                key = name.replace("lora_B", "LORA").replace(".weight", "")
                lora_pairs.setdefault(key, {})["B"] = param.detach().float().cpu()
            elif "lora_A" in name and "weight" in name:
                key = name.replace("lora_A", "LORA").replace(".weight", "")
                lora_pairs.setdefault(key, {})["A"] = param.detach().float().cpu()

        ba_stats = []
        for key in sorted(lora_pairs):
            if "A" not in lora_pairs[key] or "B" not in lora_pairs[key]:
                continue
            A = lora_pairs[key]["A"]
            B = lora_pairs[key]["B"]
            BA = B @ A
            U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
            S = S.numpy()
            total = np.sum(S**2)
            if total < 1e-12:
                continue
            p = (S**2) / total
            p = p[p > 1e-10]
            eff_rank = np.exp(-np.sum(p * np.log(p)))
            ba_stats.append({"eff_rank": float(eff_rank)})

        res["std_r4"] = {
            "mean_eff_rank": float(np.mean([s["eff_rank"] for s in ba_stats])),
            "n_modules": len(ba_stats),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        with open(rf, "w") as f:
            json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("SVD ANALYSIS: Frozen-B vs PR-LoRA vs Std LoRA")
    print(f"{'=' * 70}")
    print(f"  {'metric':<28} {'Frozen-B':>10} {'PR-LoRA':>10} {'Std LoRA':>10}")
    print(f"  {'-' * 60}")

    fr = res["frozen_r4"]
    pr = res["pr_r4"]
    print(
        f"  {'Mean effective rank':<28} {fr['mean_eff_rank']:>10.3f} {pr['mean_eff_rank']:>10.3f} {res['std_r4']['mean_eff_rank']:>10.3f}"
    )
    print(
        f"  {'Mean top-1 SV fraction':<28} {fr['mean_top1_frac']:>10.3f} {pr['mean_top1_frac']:>10.3f} {'—':>10}"
    )
    print(
        f"  {'Mean B inter-column cos':<28} {fr['mean_b_col_inter_cos']:>10.3f} {pr['mean_b_col_inter_cos']:>10.3f} {'—':>10}"
    )
    print(
        f"  {'Mean B-direction cos':<28} {fr['mean_b_direction_cos']:>10.3f} {pr['mean_b_direction_cos']:>10.3f} {'—':>10}"
    )
    print(
        f"  {'Min B-direction cos':<28} {fr['min_b_direction_cos']:>10.3f} {pr['min_b_direction_cos']:>10.3f} {'—':>10}"
    )

    print(f"\n  Interpretation:")
    print(
        f"  - Frozen-B: eff_rank ≈ {fr['mean_eff_rank']:.1f} (all B columns identical → rank-1 BA)"
    )
    print(
        f"  - PR-LoRA:  eff_rank ≈ {pr['mean_eff_rank']:.1f} (B columns diverge slightly → richer subspace)"
    )
    print(
        f"  - Std LoRA: eff_rank ≈ {res['std_r4']['mean_eff_rank']:.1f} (random init, full rank usage)"
    )
    print(
        f"  - PR-LoRA B stays aligned with direction: mean cos = {pr['mean_b_direction_cos']:.3f}"
    )


if __name__ == "__main__":
    main()
