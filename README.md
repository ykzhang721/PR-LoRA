# PR-LoRA

Code for **PR-LoRA: Direction-Regularized Low-Rank Adaptation for Behavioral Alignment in Large Language Models**.

PR-LoRA uses contrastive activation directions to initialize compatible LoRA
output factors (`B`) and anchors them with an L2 penalty during training.
The input factors (`A`) remain freely trainable, retaining input-dependent
updates. For Qwen3-8B, the prior initializes the Q/O adapters; K/V adapters
retain standard LoRA initialization because their output dimensions differ.

The repository provides a reusable implementation, runnable training and
evaluation commands, the original paper experiment scripts adapted to public
paths, and recorded numerical results.

## Results

Primary Qwen3-8B evaluation, 10 adapter seeds. Scores are the fraction of
items preferring the undesirable answer; **lower is better for all seven behaviors**.

| Behavior | Base | Steering | Frozen-B | Standard LoRA | PR-LoRA |
|---|---:|---:|---:|---:|---:|
| Sycophancy | .977 | .460 | .775 | .506 | **.422** |
| Sycophancy-philosophy | .967 | **.353** | .690 | .489 | .362 |
| Power-seeking | .756 | .385 | .361 | .221 | **.145** |
| Corrigibility | .533 | .419 | .275 | .335 | **.185** |
| Survival instinct | .446 | **.379** | .421 | .380 | .481 |
| Myopic reward | .500 | .500 | .522 | **.466** | .508 |
| Coordination | .593 | .472 | .458 | **.441** | .514 |
| Macro average | .682 | .424 | .500 | .405 | **.374** |

The macro-average reduction versus standard LoRA is **3.17 percentage points
(7.82% relative)**, calculated from unrounded scores. Recompute the table,
standard deviations and paired tests directly from the recorded measurements:

```bash
python scripts/summarize_results.py
```

These are the recorded paper results, not new measurements from the release
tests. [Experiment documentation](experiments/paper/README.md) describes
the separate data budgets, steering selection, baselines and capability checks.

## Installation

```bash
git clone https://github.com/ykzhang721/PR-LoRA.git
cd PR-LoRA
python3.11 -m venv .venv
source .venv/bin/activate
# Install PyTorch for your CUDA environment first, then:
pip install -e '.[test,experiments]'
```

The release tests use Python 3.11, PyTorch 2.8.0, Transformers 4.51.3 and
PEFT 0.15.2. Dependency ranges are in [pyproject.toml](pyproject.toml).
The 8B experiments need a CUDA device with enough memory for the base model
and training activations; the default recipe uses bfloat16 without quantization.
Qwen3 and LLaMA decoder architectures are supported. Other architectures may
require adapting the decoder-layer lookup in `directions.py`.

## Quick start

Run from the repository root. Download the public behavior data:

```bash
python scripts/download_data.py
```

The downloader pins the upstream commit and checks every file against the
original experiment's SHA-256. See [data/README.md](data/README.md) for attribution,
filenames and exact split sizes. No dataset or model weights are bundled.

Run one complete split → extract → train → evaluate pipeline:

```bash
BEHAVIOR=sycophancy bash experiments/run_qwen3_8b.sh
```

Optional environment settings include `MODEL`, `DEVICE`, `METHOD`
(`pr_lora`, `standard`, `frozen`), `TRAIN_SEED` and `DTYPE`:

```bash
BEHAVIOR=power_seeking METHOD=standard TRAIN_SEED=123 DEVICE=cuda:0 \
  bash experiments/run_qwen3_8b.sh
```

Use a separate output directory for every model, method and seed. The training
command refuses to overwrite an existing adapter. For model comparisons, use
the individual commands below with separate paths.

### Individual commands

```bash
python scripts/split_caa.py \
  --input data/raw/sycophancy.jsonl --output-dir data/splits/sycophancy/seed42 \
  --train-seed 42

python scripts/extract_directions.py \
  --model Qwen/Qwen3-8B --input data/splits/sycophancy/seed42/direction.jsonl \
  --output directions/sycophancy.pt

python scripts/train_pr_lora.py \
  --model Qwen/Qwen3-8B --train-data data/splits/sycophancy/seed42/train.jsonl \
  --directions directions/sycophancy.pt --output adapters/sycophancy/seed42

python scripts/evaluate_behavior.py \
  --model Qwen/Qwen3-8B --data data/splits/sycophancy/seed42/eval.jsonl \
  --adapter adapters/sycophancy/seed42 --output results/sycophancy/seed42.json
```

Omit `--adapter` to evaluate the unmodified model. Train the standard control
with `--method standard` (no directions required), or use `--method frozen`
to freeze only the direction-initialized B matrices. Each command supports
`--help`; model commands accept `--device cpu --dtype float32` for small models.

### Protocol defaults

- A fixed seed-42 shuffle reserves direction and evaluation blocks; a separate
  adapter seed samples 50 SFT items from the remaining pool. Split metadata
  records source hashes and row indices.
- Extraction: raw `question + answer`, last-token residual after each decoder
  block, desired minus undesired, unit normalization; maximum length 512.
- Training: `User: {question}\nAssistant: {desired_answer}`, loss on the full
  sequence except padding; rank/alpha 4, Q/K/V/O, 20 steps, batch size 4,
  learning rate `5e-4`, anchor `0.1`, gradient clipping `1.0`, dropout 0,
  `bias="none"`. AdamW weight decay is `.01` (`.1` for Frozen-B).
- Evaluation: raw `question + answer`, mean full-sequence autoregressive loss,
  maximum length 1024. A strict lower loss for the undesirable answer counts
  as one; ties count as zero. This is the paper's scoring implementation,
  not answer-only conditional likelihood.
- All B columns start identical in initialized modules. For O projections
  the coordinates match the residual stream directly; using residual
  coordinates for Q projections is a dimension-compatible initialization prior.

Direction-extraction items are additional behavior-labeled pairs, separate
from the 50 SFT examples. The reported setting is not a total budget of 50
behavior-labeled examples.

## Loading or merging an adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "adapters/sycophancy/seed42")
tokenizer = AutoTokenizer.from_pretrained("adapters/sycophancy/seed42")
model = model.merge_and_unload()  # optional: removes the adapter wrapper
model.eval()
```

The directional prior and anchor are training-time operations; inference uses
the standard PEFT adapter or merged weights without steering hooks.

## Paper experiments and verification

[experiments/paper/README.md](experiments/paper/README.md) maps the original
scripts to the primary comparison, full MMLU, DPO, LoReFT, PiSSA, component
ablations, rank/step sweeps, SVD, cross-model and multi-behavior experiments,
and open-ended generation. [paper_results](paper_results/README.md) contains
the historical numerical artifacts and their provenance.

```bash
pytest -q
```

Tests run locally with tiny randomly initialized Qwen3 and LLaMA models;
they require no model downloads or GPU. They check split parity with the
original protocol, padded direction extraction, gradient updates, anchoring,
Frozen-B behavior, adapter save/reload/merge, and the full command-line pipeline.
Full-scale paper experiments were not rerun as part of this code cleanup.

An optional [GitHub Actions template](ci/github-actions.yml) runs the same
CPU checks. To enable it, copy it to `.github/workflows/tests.yml` and commit
with credentials authorized to manage workflows.

## Citation

```bibtex
@misc{zhang2026prlora,
  title={PR-LoRA: Direction-Regularized Low-Rank Adaptation for Behavioral Alignment in Large Language Models},
  author={Zhang, Yikai and Xu, Rui and Chen, Aili and Zhang, Haoyu and Wu, Libo and Xu, Yinghui and Xiao, Yanghua},
  year={2026},
  url={https://github.com/ykzhang721/PR-LoRA}
}
```

## License

Original code is released under the [MIT License](LICENSE). Third-party data
and model weights retain their own licenses; see [data/README.md](data/README.md).
