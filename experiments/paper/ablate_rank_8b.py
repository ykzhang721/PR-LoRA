#!/usr/bin/env python3
"""Quick rank ablation on Qwen3-8B: rank=1/2/4/8 on sycophancy + power_seeking."""

import json, os, random, sys, time, warnings
from pathlib import Path
from paper_paths import ROOT
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("PR_LORA_QWEN8B", "Qwen/Qwen3-8B")
EVAL_DIR = ROOT / "benchmarks/data/anthropic_evals"
RESULTS_DIR = ROOT / "benchmarks/results/rank_ablation_8b"
SEEDS = [42, 123, 456]
LR = 5e-4
L2 = 0.1
N_EVAL = 300
STEER_LAYERS = list(range(14, 32))
RANKS = [1, 2, 4, 8]
BEHS = {
    "sycophancy": "sycophancy_on_nlp_survey.jsonl",
    "power_seeking": "power-seeking-inclination.jsonl",
    "sycophancy_phil": "sycophancy_on_philpapers2020.jsonl",
    "myopia": "myopic-reward.jsonl",
}


def _s(v):
    return v[0].strip() if isinstance(v, list) else v.strip()


def load_jsonl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def load_model(d):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=d, trust_remote_code=True
    )


def get_acts(model, tok, text, device):
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


def extract_dir(model, tok, items, device):
    model.eval()
    pa = {}
    na = {}
    for li in range(36):
        pa[li] = []
        na[li] = []
    for i, item in enumerate(items):
        q = item["question"]
        for text, tgt in [
            (f"{q}{_s(item['answer_not_matching_behavior'])}", pa),
            (f"{q}{_s(item['answer_matching_behavior'])}", na),
        ]:
            a = get_acts(model, tok, text, device)
            for li in a:
                tgt[li].append(a[li])
        if (i + 1) % 100 == 0:
            print(f"      dir:{i + 1}/{len(items)}", flush=True)
    d = {}
    for li in range(36):
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


def eval_steer(model, tok, items, sv, alpha, device):
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
                        if l in sv and l in STEER_LAYERS:
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


class LBW(nn.Module):
    def __init__(s, m, ib):
        super().__init__()
        s.lora_a = m
        if isinstance(ib, (int, float)):
            ib = torch.full((m.weight.shape[0],), ib)
        s.gate_bias = nn.Parameter(ib.float())

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


def inject_B(model, direction, rank):
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
        if param.shape[0] == len(d) and param.shape[1] == rank:
            B = np.stack([d] * rank, axis=1)
            if rank > 1:
                for c in range(1, rank):
                    noise = np.random.randn(len(d)) * 0.1
                    noise = noise - d * np.dot(noise, d)
                    noise = noise / max(np.linalg.norm(noise), 1e-8) * 0.1
                    B[:, c] = d + noise
                    B[:, c] = B[:, c] / np.linalg.norm(B[:, c])
            with torch.no_grad():
                param.copy_(torch.tensor(B, dtype=param.dtype).to(param.device))
            param.requires_grad = False


def train_m(model, data, tok, device, steps):
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
            {"params": gA, "lr": LR, "weight_decay": L2},
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

    for beh, fname in BEHS.items():
        data = load_jsonl(EVAL_DIR / fname)
        rng = random.Random(42)
        sh = data[:]
        rng.shuffle(sh)
        nd = min(500, len(sh) * 3 // 10)
        ne = min(N_EVAL, len(sh) * 3 // 10)
        di = sh[:nd]
        ei = sh[nd : nd + ne]
        tp = sh[nd + ne :]

        # Direction
        dc = RESULTS_DIR / f"{beh}_dir.pt"
        if dc.exists():
            direction = torch.load(dc, weights_only=False)
        else:
            model = load_model(device)
            direction = extract_dir(model, tok, di, device)
            del model
            torch.cuda.empty_cache()
            torch.save(direction, dc)

        # Best alpha
        ak = f"{beh}_alpha"
        if ak not in res:
            model = load_model(device)
            bs = 1.0
            ba = 1.0
            for a in [0.5, 1, 2, 4, 8]:
                s = eval_steer(model, tok, ei[:100], direction, a, device)
                if s < bs:
                    bs = s
                    ba = a
            del model
            torch.cuda.empty_cache()
            res[ak] = ba
            with open(rf, "w") as f:
                json.dump(res, f, indent=2)
        best_alpha = res[ak]

        print(f"\n{'=' * 70}\n{beh} — rank ablation (α={best_alpha})\n{'=' * 70}")

        # Baseline + steering (once)
        for meth in ["baseline", "steering"]:
            rk = f"{beh}_{meth}"
            if rk not in res:
                model = load_model(device)
                if meth == "baseline":
                    res[rk] = eval_beh(model, tok, ei, device)
                else:
                    res[rk] = eval_steer(model, tok, ei, direction, best_alpha, device)
                del model
                torch.cuda.empty_cache()
                print(f"  {meth}: {res[rk]:.3f}")
                with open(rf, "w") as f:
                    json.dump(res, f, indent=2)

        # For each rank: standard + persona
        for rank in RANKS:
            bias_scale = 0.1 * (4096 / 2048)  # adaptive
            steps = max(10, int(10 * rank))  # more steps for higher rank

            for mt in ["std", "persona"]:
                for seed in SEEDS:
                    rk = f"{beh}_r{rank}_{mt}_s{seed}"
                    if rk in res:
                        print(f"  [{beh} r={rank}] {mt} s={seed}: {res[rk]:.3f} (cached)")
                        continue
                    print(f"  [{beh} r={rank}] {mt} s={seed} (steps={steps})...", flush=True)
                    t0 = time.time()

                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rng2 = random.Random(seed)
                    tp2 = tp[:]
                    rng2.shuffle(tp2)
                    td = [
                        {"question": s["question"], "answer": _s(s["answer_not_matching_behavior"])}
                        for s in tp2[:50]
                    ]

                    model = load_model(device)
                    model = get_peft_model(
                        model,
                        LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            r=rank,
                            lora_alpha=rank,
                            lora_dropout=0.0,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        ),
                    )

                    if mt == "persona":
                        inject_B(model, direction, rank)
                        bias_val = best_alpha * bias_scale
                        for name, module in list(model.named_modules()):
                            if "lora_A" in name and isinstance(module, nn.Linear):
                                parts = name.split(".")
                                parent = model
                                for p in parts[:-1]:
                                    parent = getattr(parent, p)
                                setattr(
                                    parent,
                                    parts[-1],
                                    LBW(module, torch.full((module.weight.shape[0],), bias_val)).to(
                                        next(module.parameters()).device
                                    ),
                                )

                    train_m(model, td, tok, device, steps)
                    score = eval_beh(model, tok, ei, device)
                    del model
                    torch.cuda.empty_cache()
                    res[rk] = score
                    print(f"    -> {score:.3f} ({time.time() - t0:.0f}s)")
                    with open(rf, "w") as f:
                        json.dump(res, f, indent=2)

    # Summary
    print(f"\n{'=' * 80}\nRANK ABLATION SUMMARY\n{'=' * 80}")
    for beh in BEHS:
        bl = res.get(f"{beh}_baseline", 0)
        st = res.get(f"{beh}_steering", 0)
        print(f"\n  {beh}: baseline={bl:.3f}, steering={st:.3f}")
        print(f"  {'rank':>6} {'std':>8} {'persona':>8} {'Δsteer':>8} {'Δstd':>8}")
        for rank in RANKS:
            std_v = [res.get(f"{beh}_r{rank}_std_s{s}") for s in SEEDS]
            std_v = [x for x in std_v if x]
            per_v = [res.get(f"{beh}_r{rank}_persona_s{s}") for s in SEEDS]
            per_v = [x for x in per_v if x]
            if std_v and per_v:
                sm = np.mean(std_v)
                pm = np.mean(per_v)
                ds = pm - st
                dr = pm - sm
                ms = "★" if ds < -0.005 else ("✗" if ds > 0.005 else "≈")
                mr = "★" if dr < -0.005 else ("✗" if dr > 0.005 else "≈")
                print(f"  r={rank:>4} {sm:>8.3f} {pm:>8.3f} {ds:>+7.3f}{ms} {dr:>+7.3f}{mr}")


if __name__ == "__main__":
    main()
