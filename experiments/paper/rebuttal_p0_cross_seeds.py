#!/usr/bin/env python3
"""P0.4 Rebuttal: Extra seeds for cross-model experiments (3→5 seeds).

Adds seeds [789, 1024] to existing [42, 123, 456] for:
  - LLaMA-3.1-8B-Instruct: sycophancy, power_seeking, corrigibility
  - Qwen3-4B: sycophancy, power_seeking, corrigibility

Uses same methods as paper: baseline, steering, std_r4, pr_r4.

Usage:
  python rebuttal_p0_cross_seeds.py --device cuda:0 --model llama31_8b
  python rebuttal_p0_cross_seeds.py --device cuda:0 --model qwen3_4b
"""

import json, os, random, sys, time, warnings, gc
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = ROOT
EVAL_DIR = SCRIPT_DIR / "benchmarks/data/anthropic_evals"
RESULTS_DIR = SCRIPT_DIR / "benchmarks/results/rebuttal_cross_seeds"

LR = 5e-4
PR_LAMBDA = 0.1
RANK = 4
STEPS = 20
ALL_SEEDS = [42, 123, 456, 789, 1024]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]

MODELS = {
    "llama31_8b": {
        "path": os.environ.get("PR_LORA_LLAMA8B", "meta-llama/Llama-3.1-8B-Instruct"),
        "n_layers": 32,
        "steer_layers": list(range(12, 28)),
        "dir_cache": SCRIPT_DIR / "benchmarks/results/pr_lora_llama8b",
    },
    "qwen3_4b": {
        "path": os.environ.get("PR_LORA_QWEN4B", "Qwen/Qwen3-4B"),
        "n_layers": 36,
        "steer_layers": list(range(10, 28)),
        "dir_cache": SCRIPT_DIR / "benchmarks/results/qwen4b_single",
    },
}

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


def eval_steer(model, tok, items, sv, alpha, steer_layers, device):
    model.eval()
    match = 0
    for item in items:
        q = item["question"]
        m, n = _s(item["answer_matching_behavior"]), _s(item["answer_not_matching_behavior"])
        losses = {}
        for lab, text in [("m", f"{q}{m}"), ("n", f"{q}{n}")]:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            hooks = []
            for li, layer in enumerate(model.model.layers):

                def mk(l):
                    def fn(mod, i, o):
                        h = o[0] if isinstance(o, tuple) else o
                        if l in sv and l in steer_layers:
                            v = torch.tensor(sv[l], dtype=h.dtype, device=h.device)
                            h = h + alpha * v.unsqueeze(0).unsqueeze(0)
                        return (h,) + o[1:] if isinstance(o, tuple) else h

                    return fn

                hooks.append(layer.register_forward_hook(mk(li)))
            with torch.no_grad():
                losses[lab] = model(**enc, labels=enc["input_ids"]).loss.item()
            for h in hooks:
                h.remove()
        if losses["m"] < losses["n"]:
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


def load_direction(beh_name, model_name, tok, device, mcfg):
    dc = mcfg["dir_cache"] / f"{beh_name}_dir.pt"
    if dc.exists():
        return torch.load(dc, weights_only=False)
    print(f"  Extracting direction for {beh_name} on {model_name}...", flush=True)
    di, _, _ = get_data_split(beh_name)
    if not di:
        return None
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    direction = extract_direction(model, tok, di, device, mcfg["n_layers"])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mcfg["dir_cache"].mkdir(parents=True, exist_ok=True)
    torch.save(direction, dc)
    return direction


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    device = args.device
    model_name = args.model
    mcfg = MODELS[model_name]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = RESULTS_DIR / f"{model_name}.json"
    res = json.load(open(rf)) if rf.exists() else {}

    # Import previous 3-seed results
    for prev_dir in [mcfg["dir_cache"]]:
        prev_rf = prev_dir / "results.json"
        if prev_rf.exists():
            prev = json.load(open(prev_rf))
            for k, v in prev.items():
                if k not in res:
                    res[k] = v
            print(f"  Imported {len(prev)} cached results from {prev_rf}")

    tok = AutoTokenizer.from_pretrained(mcfg["path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for beh_name in BEHS:
        di, ei, tp = get_data_split(beh_name)
        if not di:
            print(f"SKIP {beh_name}")
            continue

        direction = load_direction(beh_name, model_name, tok, device, mcfg)
        if direction is None:
            print(f"SKIP {beh_name} (no dir)")
            continue

        # Baseline
        if f"{beh_name}_baseline" not in res:
            model = AutoModelForCausalLM.from_pretrained(
                mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            res[f"{beh_name}_baseline"] = eval_beh(model, tok, ei, device)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

        # Best alpha
        if f"{beh_name}_alpha" not in res:
            model = AutoModelForCausalLM.from_pretrained(
                mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            best_s = 1.0
            best_a = 1.0
            for a in ALPHAS:
                s = eval_steer(model, tok, ei[:100], direction, a, mcfg["steer_layers"], device)
                if s < best_s:
                    best_s = s
                    best_a = a
            res[f"{beh_name}_alpha"] = best_a
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

        if f"{beh_name}_steering" not in res:
            model = AutoModelForCausalLM.from_pretrained(
                mcfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
            res[f"{beh_name}_steering"] = eval_steer(
                model, tok, ei, direction, res[f"{beh_name}_alpha"], mcfg["steer_layers"], device
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)

        bl = res[f"{beh_name}_baseline"]
        st = res[f"{beh_name}_steering"]
        print(f"\n{'=' * 60}")
        print(f"{model_name} / {beh_name}: baseline={bl:.3f} steering={st:.3f}")
        print(f"{'=' * 60}")

        for seed in ALL_SEEDS:
            rng2 = random.Random(seed)
            tp2 = tp[:]
            rng2.shuffle(tp2)
            td = [
                {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
                for s in tp2[:50]
            ]

            for meth in ["std_r4", "pr_r4"]:
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
                    mcfg["path"],
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

                b_targets = {}
                if meth == "pr_r4":
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

                score = eval_beh(model, tok, ei, device)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                res[rk] = score
                print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"CROSS-MODEL 5-SEED RESULTS ({model_name})")
    print(f"{'=' * 80}")
    print(
        f"  {'behavior':<18} {'base':>6} {'steer':>6} | {'std_r4':>14} {'PR-LoRA':>14} | {'Δ steer':>10}"
    )
    print(f"  {'-' * 80}")

    for beh in BEHS:
        bl = res.get(f"{beh}_baseline", 0)
        st = res.get(f"{beh}_steering", 0)
        for meth in ["std_r4", "pr_r4"]:
            vals = [res.get(f"{beh}_{meth}_s{s}") for s in ALL_SEEDS]
            vals = [v for v in vals if v is not None]
            if meth == "std_r4":
                std_v = vals
            else:
                pr_v = vals

        def fmt(vals):
            if not vals:
                return "---"
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

        delta = np.mean(pr_v) - st if pr_v else 0
        mark = "WIN" if delta < -0.01 else ("LOSE" if delta > 0.01 else "~")
        print(
            f"  {beh:<18} {bl:>6.3f} {st:>6.3f} | {fmt(std_v):>14} {fmt(pr_v):>14} | {delta:>+8.3f} {mark}"
        )

    with open(rf, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {rf}")


if __name__ == "__main__":
    main()
