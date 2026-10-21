# Paper experiments

These scripts preserve the research training and evaluation loops while
replacing machine-specific paths with public model IDs and a shared workspace.
Use the clean `scripts/` interface for saved adapters and new applications;
use these experiment drivers for the original sweeps and per-seed result keys.
Scripts retain their research filenames so results can be traced to their origin.

## Setup and primary comparison

Run commands from the repository root after installing `.[experiments]`:

```bash
python scripts/download_data.py --original-names --output-dir benchmarks/data/anthropic_evals
python scripts/prepare_capability_data.py
python experiments/paper/rebuttal_p0_extra_seeds.py --device cuda:0
python scripts/summarize_results.py --input benchmarks/results/rebuttal_extra_seeds/results.json
```

The main driver evaluates seven behaviors, the base model, steering, and
Frozen-B/standard/PR-LoRA across ten seeds. It extracts directions if absent,
stores them in `benchmarks/results/all_dirs_8b`, and resumes completed result
keys. `--beh sycophancy` restricts computation to one behavior. With an empty
workspace it computes all ten seeds; the old five-seed cache is optional.

The default root is this repository. Set `PR_LORA_WORKDIR` to an absolute
path to separate multiple experimental runs; prepare `benchmarks/data` under
that same root. Do not mix cached results from different hyperparameters.
Model locations can be overridden with `PR_LORA_QWEN8B`, `PR_LORA_QWEN4B`
and `PR_LORA_LLAMA8B`; each accepts a local model directory or Hugging Face ID.
Access to LLaMA requires the corresponding model permissions.

## Experiment map

All filenames below are relative to this directory. Each supports `--help`.
Direction-dependent analyses require running the primary driver first.

| Experiment | Script | Notes |
|---|---|---|
| Main seven-behavior table | `rebuttal_p0_extra_seeds.py` | 10 seeds; base, steering, Frozen-B, standard, PR |
| Full MMLU | `rebuttal_p0_full_mmlu.py` | `--model qwen3_8b`; 14,042 items; sycophancy adapters |
| TruthfulQA and sampled MMLU | `p1_extra_eval.py` | 5 seeds; first 200 TruthfulQA validation items; 5 MMLU items per subject |
| DPO low-update baseline | `baseline_dpo.py` | 50 pairs, 20 updates, 3 seeds; manual DPO loss |
| DPO data/update controls | `rebuttal_p0_dpo_fair.py` | Includes the settings listed below |
| LoReFT baseline | `baseline_reft.py` | Manual residual intervention used in the experiments; 50 steps |
| PiSSA baseline | `baseline_pissa.py` | PEFT PiSSA initialization, rank 4 |
| Direction/random × anchor ablation | `ablate_direction_vs_reg.py` | Four configurations, 3 seeds |
| Training dynamics, drift, logit lens | `mechanistic_analysis.py` | Also plots sweep and five-seed direction-quality artifacts |
| Effective rank of BA | `rebuttal_p1_svd_analysis.py` | Frozen-B, standard, PR; module-level SVD |
| Rank ablation | `ablate_rank_8b.py` | Original rank sweep and direction caches |
| Step/anchor sweep | `sweep_mmlu_tradeoff.py` | 12 settings, seed 42; sampled MMLU |
| LLaMA and Qwen3-4B | `rebuttal_p0_cross_seeds.py` | `--model llama31_8b` or `--model qwen3_4b` |
| Qwen3-4B original comparison | `eval_qwen4b_single.py` | Separate historical experiment |
| Joint control, rank 3 | `bench_multi_pr_lora.py` | Three directions with one column each |
| Joint control, rank 6, V/O | `sweep_vo_r6_steps.py` | Three directions, two columns each; 20/30/40/60 steps |
| Joint control, frozen-B variant | `bench_multi_v6_8b.py` | Exploratory bias/gating variant |
| Open-ended evaluation | `rebuttal_p2_generative_eval.py` | 50 prompts per behavior; external judge |

For example, after the primary direction caches are available:

```bash
python experiments/paper/rebuttal_p0_full_mmlu.py --model qwen3_8b --device cuda:0
python experiments/paper/ablate_direction_vs_reg.py --device cuda:0
python experiments/paper/rebuttal_p1_svd_analysis.py --device cuda:0
python experiments/paper/baseline_dpo.py --device cuda:0
python experiments/paper/sweep_vo_r6_steps.py --device cuda:0
```

The reusable CLI also supports `Qwen/Qwen3-1.7B`, using the architecture's
actual decoder count. Model-specific paper drivers contain their own layer
ranges; do not point the 8B driver at a smaller model merely by changing its path.

## Evaluation details

**Steering.** The original Qwen3-8B driver adds the per-layer direction at
zero-indexed layers 14–31 and selects an alpha from `[.5, 1, 2, 4, 8]` on the
first 100 evaluation items. It then reports the score on the full evaluation
split, including those items. These steering scores are descriptive, not an
independent held-out tuning estimate. The release preserves this protocol
and its scores rather than silently changing the selection set.

**Statistics.** The main summary uses population standard deviations
(`ddof=0`), paired two-sided tests between PR and standard LoRA on all ten
shared seeds, and Holm adjustment across all seven behaviors. The macro test
first averages seven behaviors within each seed. Earlier exploratory drivers
may print other historical comparisons; use `scripts/summarize_results.py`
for the final main table and tests. Summary input must contain all ten seeds.

**Capability.** Full MMLU uses the prompt `Question: ...\n(A) ...\n(B) ...\n(C) ...\n(D) ...\nAnswer: (`
and selects the highest next-token logit among A/B/C/D, with maximum input
length 512. TruthfulQA MC1 in `p1_extra_eval.py` selects the choice with
the lowest mean full-sequence loss on `Q: {question}\nA: {choice}` (maximum
length 512). Its MMLU results and those of the sweeps use five items per subject,
so they are distinct from the full-MMLU results.

**DPO and LoReFT.** The DPO budget script runs `(pairs, steps)` settings
`(50,25)`, `(200,20)`, `(50,100)`, `(200,100)` and an additional `(550,275)`
budget experiment, subject to available training items. The first four are
not a complete factorial grid because 25 and 20 steps differ. The manual
LoReFT implementation uses 50 steps and learning rate `2e-3`; the primary
SFT adapters and DPO low-update baseline use 20 steps.

**Step/anchor analysis.** The 12 configurations are the Cartesian product
of steps `[10,20,40]` and lambda `[0,.01,.03,.1]`, all at seed 42. In this
historical driver, lambda zero also disables direction initialization, so
those three configurations are standard LoRA controls. This differs from
setting `--anchor-lambda 0` in the reusable PR-LoRA training command, which
retains the direction initialization. Use the four-way component ablation
to isolate initialization from anchoring.

**Open-ended generation.** Prompts and exact judge instructions are in the
generation script. The base model and both sycophancy adapters are tested on
sycophancy and corrigibility prompts. Generation is greedy, uses at most 256
new tokens, and disables Qwen3 thinking. The judge receives only the prompt
and response, without a method name; temperature is 0 and the output limit
is 10 tokens. Set `JUDGE_API_KEY`, `JUDGE_API_URL` (a chat-completions endpoint)
and `JUDGE_MODEL` before running. The historical model label is
`deepseek-v4-flash`. This command sends prompts and generated responses to
the configured provider; judge availability/version can affect results.
Failed requests or unrecognized labels now stop the run instead of becoming
numeric scores. Human agreement statistics are not recomputed by this script.

## Scope of the cleanup

[PROVENANCE.json](PROVENANCE.json) records source and exported file hashes.
The public versions replace local paths, remove forced offline mode and
embedded service credentials, use a shared direction-cache location, and
format the code. Unused DPO/pyreft branches were removed. A redundant MMLU
dataset construction before pad-token setup was removed; the drift helper
now accepts its model explicitly. The main summary was updated to the final
paired PR-versus-standard analysis. Generation now requires explicit service
configuration and rejects invalid judge responses.

The small CPU tests exercise the reusable implementation and CLI. These
original full-size GPU experiment drivers were syntax-checked, not rerun
during release preparation. Numerical reproducibility also depends on model
revision, software versions, accelerator kernels and precision.

Legacy direction caches use NumPy-containing `.pt` files and are loaded with
`weights_only=False`; use caches produced by these scripts. The reusable
CLI uses a separate tensor-only format and `weights_only=True`. The two cache
formats should not be interchanged.
