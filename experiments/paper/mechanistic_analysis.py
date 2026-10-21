#!/usr/bin/env python3
"""Mechanistic analysis for paper:
1. Training dynamics: behavioral score at each step (PR-LoRA vs Std vs Frozen)
2. Logit lens: project behavioral directions onto vocabulary
3. Case study: generate responses before/after PR-LoRA
4. B drift: cos(B, B_init) during training
5. Pareto frontier: MMLU vs behavioral score
6. Direction quality vs method effectiveness
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
EVAL_DIR = ROOT / "benchmarks/data/anthropic_evals"
DIR_CACHE = ROOT / "benchmarks/results/all_dirs_8b"
OLD_DIR_CACHE = ROOT / "benchmarks/results/rank_ablation_8b"
OUT_DIR = ROOT / "benchmarks/results/mechanistic"
LR = 5e-4
PR_LAMBDA = 0.1
RANK = 4


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
    fp = EVAL_DIR / f"{'sycophancy_on_nlp_survey' if beh_name == 'sycophancy' else beh_name}.jsonl"
    if beh_name == "sycophancy":
        fp = EVAL_DIR / "sycophancy_on_nlp_survey.jsonl"
    elif beh_name == "power_seeking":
        fp = EVAL_DIR / "power-seeking-inclination.jsonl"
    if not fp.exists():
        return [], [], []
    data = load_jsonl(fp)
    rng = random.Random(42)
    sh = data[:]
    rng.shuffle(sh)
    nd = min(500, len(sh) * 3 // 10)
    ne = min(300, len(sh) * 3 // 10)
    return sh[:nd], sh[nd : nd + ne], sh[nd + ne :]


def load_direction(beh_name):
    for cache in [DIR_CACHE, OLD_DIR_CACHE]:
        dc = cache / f"{beh_name}_dir.pt"
        if dc.exists():
            return torch.load(dc, weights_only=False)
    return None


def analysis_1_training_dynamics(tok, device):
    """Plot behavioral score at each training step for 3 methods."""
    print("\n=== ANALYSIS 1: Training Dynamics ===", flush=True)

    direction = load_direction("sycophancy")
    _, ei, tp = get_data_split("sycophancy")
    eval_subset = ei[:100]  # faster eval
    random.seed(42)
    rng2 = random.Random(42)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    td = [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:50]
    ]

    results = {}
    steps_to_eval = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30]

    for meth in ["frozen", "std", "pr_lora"]:
        print(f"  {meth}...", flush=True)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

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
        if meth in ["frozen", "pr_lora"]:
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
                if meth == "frozen":
                    param.requires_grad = False
                else:
                    b_targets[name] = torch.tensor(B, dtype=torch.float32)
                    param.requires_grad = True

        ds = SDS(td, tok)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        params = [p for p in model.parameters() if p.requires_grad]
        wd = 0.1 if meth == "frozen" else 0.01
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=wd)

        scores = []
        step = 0
        if 0 in steps_to_eval:
            scores.append((0, eval_beh(model, tok, eval_subset, device)))
            print(f"    step 0: {scores[-1][1]:.3f}")

        model.train()
        for _ in range(100):
            for batch in loader:
                if step >= max(steps_to_eval):
                    break
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = out.loss
                if meth == "pr_lora" and b_targets:
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

                if step in steps_to_eval:
                    s = eval_beh(model, tok, eval_subset, device)
                    scores.append((step, s))
                    print(f"    step {step}: {s:.3f}")
            if step >= max(steps_to_eval):
                break

        results[meth] = scores
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"frozen": "#e74c3c", "std": "#3498db", "pr_lora": "#2ecc71"}
    labels = {"frozen": "Frozen-B Persona", "std": "Standard LoRA", "pr_lora": "PR-LoRA (ours)"}
    for meth in ["frozen", "std", "pr_lora"]:
        steps, scores = zip(*results[meth])
        ax.plot(
            steps, scores, "o-", color=colors[meth], label=labels[meth], linewidth=2, markersize=5
        )
    ax.axhline(y=0.460, color="gray", linestyle="--", alpha=0.7, label="Steering (CAA)")
    ax.axhline(y=0.977, color="gray", linestyle=":", alpha=0.5, label="Baseline")
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Sycophancy Score (lower = better)", fontsize=12)
    ax.set_title("Training Dynamics: Sycophancy on Qwen3-8B", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.1, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_dynamics.pdf", dpi=150)
    plt.savefig(OUT_DIR / "training_dynamics.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'training_dynamics.pdf'}")

    json.dump(
        results,
        open(OUT_DIR / "training_dynamics.json", "w"),
        indent=2,
        default=lambda x: list(x) if hasattr(x, "__iter__") else x,
    )


def analysis_2_logit_lens(tok, device):
    """Project behavioral directions onto vocabulary to reveal semantic meaning."""
    print("\n=== ANALYSIS 2: Logit Lens ===", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    embed = model.model.embed_tokens.weight.detach().float().cpu()  # (vocab, d)
    lm_head = model.lm_head.weight.detach().float().cpu()  # (vocab, d)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    behs = ["sycophancy", "power_seeking", "corrigibility"]
    results = {}

    for beh in behs:
        direction = load_direction(beh)
        if direction is None:
            continue
        layer = 24  # middle-late layer
        if layer not in direction:
            continue
        d = torch.tensor(direction[layer], dtype=torch.float32)

        # Project onto lm_head
        logits = lm_head @ d  # (vocab,)
        top_k = 15
        top_idx = logits.topk(top_k).indices.tolist()
        bot_idx = logits.topk(top_k, largest=False).indices.tolist()

        top_tokens = [(tok.decode([i]).strip(), logits[i].item()) for i in top_idx]
        bot_tokens = [(tok.decode([i]).strip(), logits[i].item()) for i in bot_idx]

        results[beh] = {"top": top_tokens, "bottom": bot_tokens}
        print(f"\n  {beh} (layer {layer}):")
        print(f"    Top (promoting): {[t[0] for t in top_tokens[:10]]}")
        print(f"    Bottom (opposing): {[t[0] for t in bot_tokens[:10]]}")

    json.dump(results, open(OUT_DIR / "logit_lens.json", "w"), indent=2)

    # Plot
    fig, axes = plt.subplots(1, len(behs), figsize=(5 * len(behs), 5))
    if len(behs) == 1:
        axes = [axes]
    for ax, beh in zip(axes, behs):
        if beh not in results:
            continue
        top = results[beh]["top"][:10]
        bot = results[beh]["bottom"][:10]
        all_items = list(reversed(bot)) + top
        tokens = [t[0][:12] for t in all_items]
        values = [t[1] for t in all_items]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
        ax.barh(range(len(tokens)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel("Logit projection", fontsize=10)
        ax.set_title(f"{beh}", fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.5)
    plt.suptitle("Logit Lens: What Do Behavioral Directions Encode?", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "logit_lens.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(OUT_DIR / "logit_lens.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'logit_lens.pdf'}")


def analysis_3_case_study(tok, device):
    """Generate actual responses for sycophancy prompts."""
    print("\n=== ANALYSIS 3: Case Study ===", flush=True)

    direction = load_direction("sycophancy")
    _, ei, tp = get_data_split("sycophancy")

    # Pick 3 diverse prompts
    prompts = []
    for item in ei[:50]:
        q = item["question"]
        if (
            len(q) > 100
            and len(q) < 500
            and "agree" in q.lower()
            or "opinion" in q.lower()
            or "think" in q.lower()
        ):
            prompts.append(item)
            if len(prompts) >= 3:
                break
    if len(prompts) < 3:
        prompts = ei[:3]

    random.seed(42)
    rng2 = random.Random(42)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    td = [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:50]
    ]

    cases = []
    for item in prompts:
        case = {
            "question": item["question"],
            "matching": _s(item["answer_matching_behavior"]),
            "not_matching": _s(item["answer_not_matching_behavior"]),
            "responses": {},
        }

        for meth in ["baseline", "pr_lora"]:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )

            if meth == "pr_lora":
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
                    b_targets[name] = torch.tensor(B, dtype=torch.float32)
                    param.requires_grad = True

                ds = SDS(td, tok)
                loader = DataLoader(ds, batch_size=4, shuffle=True)
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
                model.train()
                step = 0
                for _ in range(500):
                    for batch in loader:
                        if step >= 20:
                            break
                        out = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device),
                        )
                        b_reg = torch.tensor(0.0, device=device)
                        for bname, param in model.named_parameters():
                            if bname in b_targets:
                                target = b_targets[bname].to(param.device).to(param.dtype)
                                b_reg = b_reg + ((param - target) ** 2).sum()
                        loss = out.loss + PR_LAMBDA * b_reg
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()
                        step += 1
                    if step >= 20:
                        break

            model.eval()
            prompt = item["question"]
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=False, temperature=1.0)
            response = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            case["responses"][meth] = response[:500]
            print(f"  [{meth}] {response[:100]}...")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        cases.append(case)

    json.dump(cases, open(OUT_DIR / "case_study.json", "w"), indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_DIR / 'case_study.json'}")


def analysis_4_b_drift(tok, device):
    """Track cos(B, B_init) during training."""
    print("\n=== ANALYSIS 4: B Drift Trajectory ===", flush=True)

    direction = load_direction("sycophancy")
    _, _, tp = get_data_split("sycophancy")
    random.seed(42)
    rng2 = random.Random(42)
    tp2 = tp[:]
    rng2.shuffle(tp2)
    td = [
        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
        for s in tp2[:50]
    ]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
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
        b_targets[name] = torch.tensor(B, dtype=torch.float32)
        param.requires_grad = True

    ds = SDS(td, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

    drift_log = []  # [(step, mean_cos, min_cos)]
    steps_to_check = list(range(0, 41))

    def measure_drift(current_model):
        cos_vals = []
        for name, param in current_model.named_parameters():
            if name in b_targets:
                target = b_targets[name].float().cpu()
                trained = param.detach().float().cpu()
                for col in range(min(RANK, trained.shape[1])):
                    cos = torch.nn.functional.cosine_similarity(
                        trained[:, col].unsqueeze(0), target[:, col].unsqueeze(0)
                    ).item()
                    cos_vals.append(cos)
        return np.mean(cos_vals), np.min(cos_vals)

    mean_c, min_c = measure_drift(model)
    drift_log.append((0, mean_c, min_c))

    model.train()
    step = 0
    for _ in range(500):
        for batch in loader:
            if step >= 40:
                break
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            b_reg = torch.tensor(0.0, device=device)
            for bname, param in model.named_parameters():
                if bname in b_targets:
                    target = b_targets[bname].to(param.device).to(param.dtype)
                    b_reg = b_reg + ((param - target) ** 2).sum()
            loss = out.loss + PR_LAMBDA * b_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step in steps_to_check:
                mean_c, min_c = measure_drift(model)
                drift_log.append((step, mean_c, min_c))
                if step % 10 == 0:
                    print(f"    step {step}: mean_cos={mean_c:.6f} min_cos={min_c:.6f}")
        if step >= 40:
            break

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Plot
    steps_arr = [x[0] for x in drift_log]
    mean_arr = [x[1] for x in drift_log]
    min_arr = [x[2] for x in drift_log]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        steps_arr,
        mean_arr,
        "o-",
        color="#2ecc71",
        label="Mean cos(B, B_init)",
        linewidth=2,
        markersize=4,
    )
    ax.fill_between(steps_arr, min_arr, mean_arr, alpha=0.2, color="#2ecc71")
    ax.plot(
        steps_arr,
        min_arr,
        "--",
        color="#27ae60",
        label="Min cos(B, B_init)",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Cosine Similarity with Initial Direction", fontsize=12)
    ax.set_title("B Matrix Drift During PR-LoRA Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.98, 1.002)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "b_drift.pdf", dpi=150)
    plt.savefig(OUT_DIR / "b_drift.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'b_drift.pdf'}")

    json.dump(drift_log, open(OUT_DIR / "b_drift.json", "w"))


def analysis_5_pareto(tok, device):
    """Plot Pareto frontier: MMLU vs behavioral score."""
    print("\n=== ANALYSIS 5: Pareto Frontier ===", flush=True)

    # Collect data from previous experiments
    points = []
    # From Pareto sweep
    sweep_path = ROOT / "benchmarks/results/mmlu_tradeoff/results.json"
    if sweep_path.exists():
        sw = json.load(open(sweep_path))
        configs = [
            ("Std r4 st10", "pr_r4_lam0.0_st10"),
            ("Std r4 st20", "pr_r4_lam0.0_st20"),
            ("Std r4 st40", "pr_r4_lam0.0_st40"),
            ("PR $\\lambda$=0.01 st20", "pr_r4_lam0.01_st20"),
            ("PR $\\lambda$=0.01 st40", "pr_r4_lam0.01_st40"),
            ("PR $\\lambda$=0.1 st10", "pr_r4_lam0.1_st10"),
            ("PR $\\lambda$=0.1 st20", "pr_r4_lam0.1_st20"),
            ("PR $\\lambda$=0.1 st40", "pr_r4_lam0.1_st40"),
        ]
        for label, key in configs:
            syc = sw.get(f"{key}_syc")
            mmlu = sw.get(f"{key}_mmlu")
            if syc is not None and mmlu is not None:
                is_pr = "PR" in label
                points.append((label, syc, mmlu, is_pr))

    # Add baseline and steering
    points.append(("Baseline", 0.977, 0.765, False))
    points.append(("Steering", 0.460, 0.765, False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, syc, mmlu, is_pr in points:
        color = "#2ecc71" if is_pr else "#3498db"
        marker = "s" if is_pr else "o"
        if label in ["Baseline", "Steering"]:
            color = "#95a5a6"
            marker = "^"
        ax.scatter(
            syc, mmlu, color=color, marker=marker, s=80, zorder=5, edgecolors="black", linewidth=0.5
        )
        offset = (5, 5) if syc < 0.8 else (-40, 5)
        ax.annotate(
            label.replace("$\\lambda$", "λ"),
            (syc, mmlu),
            textcoords="offset points",
            xytext=offset,
            fontsize=7,
            alpha=0.8,
        )

    ax.axhline(y=0.765, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=0.460, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Sycophancy Score (lower = better alignment)", fontsize=12)
    ax.set_ylabel("MMLU (higher = better capability)", fontsize=12)
    ax.set_title("Pareto Frontier: Behavioral Alignment vs. Capability", fontsize=13)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#2ecc71",
            markersize=8,
            label="PR-LoRA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=8,
            label="Standard LoRA",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#95a5a6",
            markersize=8,
            label="No training",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pareto.pdf", dpi=150)
    plt.savefig(OUT_DIR / "pareto.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'pareto.pdf'}")


def analysis_6_direction_quality(tok, device):
    """Scatter: steering effectiveness vs PR-LoRA effectiveness across 7 behaviors."""
    print("\n=== ANALYSIS 6: Direction Quality vs Method Effectiveness ===", flush=True)

    supp_path = ROOT / "benchmarks/results/supplement/results.json"
    if not supp_path.exists():
        print("  No supplement data, skipping")
        return
    supp = json.load(open(supp_path))

    behs = [
        "sycophancy",
        "sycophancy_phil",
        "power_seeking",
        "survival",
        "myopia",
        "corrigibility",
        "coordination",
    ]
    seeds = [42, 123, 456, 789, 1024]

    fig, ax = plt.subplots(figsize=(7, 5))
    steer_deltas = []
    pr_deltas = []

    for beh in behs:
        bl = supp.get(f"{beh}_baseline", 0)
        st = supp.get(f"{beh}_steering", 0)
        pr_vals = [supp.get(f"{beh}_pr_r4_s{s}") for s in seeds]
        pr_vals = [v for v in pr_vals if v is not None]
        if not pr_vals:
            continue
        pr_m = np.mean(pr_vals)

        steer_d = bl - st  # positive = steering works well
        pr_d = bl - pr_m  # positive = PR-LoRA works well

        steer_deltas.append(steer_d)
        pr_deltas.append(pr_d)

        color = "#2ecc71" if pr_m < st else "#e74c3c"
        ax.scatter(steer_d, pr_d, color=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(beh[:8], (steer_d, pr_d), textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Diagonal
    lim = max(max(steer_deltas), max(pr_deltas)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="PR-LoRA = Steering")
    ax.set_xlabel("Steering Effectiveness (baseline - steering)", fontsize=12)
    ax.set_ylabel("PR-LoRA Effectiveness (baseline - PR-LoRA)", fontsize=12)
    ax.set_title("Direction Quality Predicts Method Effectiveness", fontsize=13)

    # Correlation
    r = np.corrcoef(steer_deltas, pr_deltas)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ecc71",
            markersize=8,
            label="PR-LoRA beats steering",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=8,
            label="Steering wins",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "direction_quality.pdf", dpi=150)
    plt.savefig(OUT_DIR / "direction_quality.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'direction_quality.pdf'}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    analysis_1_training_dynamics(tok, device)
    analysis_2_logit_lens(tok, device)
    analysis_3_case_study(tok, device)
    analysis_4_b_drift(tok, device)
    analysis_5_pareto(tok, device)
    analysis_6_direction_quality(tok, device)

    print("\n\n=== ALL ANALYSES COMPLETE ===")
    print(f"Output directory: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
