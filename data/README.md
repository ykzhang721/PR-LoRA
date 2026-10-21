# Data and splits

The seven behavior files come from [Anthropic's Model-Written Evaluations](https://github.com/anthropics/evals)
at commit `84fcc677e52e1902d696c32cd1a6b663e70d3993`. The five AI-risk files
used in these experiments are in upstream's `advanced-ai-risk/human_generated_evals/`
directory; the two sycophancy files are in `sycophancy/`. The downloader checks
the exact bytes against [checksums.json](checksums.json), derived from the
research inputs and verified against upstream Git blob hashes.

```bash
python scripts/download_data.py
# Or download only one behavior:
python scripts/download_data.py --behaviors corrigibility
```

| Alias | Upstream filename | Total | Direction | SFT | Evaluation |
|---|---|---:|---:|---:|---:|
| sycophancy | sycophancy_on_nlp_survey.jsonl | 9,984 | 500 | 50 | 300 |
| sycophancy_phil | sycophancy_on_philpapers2020.jsonl | 9,867 | 500 | 50 | 300 |
| power_seeking | power-seeking-inclination.jsonl | 998 | 299 | 50 | 299 |
| survival | survival-instinct.jsonl | 953 | 285 | 50 | 285 |
| myopia | myopic-reward.jsonl | 1,000 | 300 | 50 | 300 |
| corrigibility | corrigible-less-HHH.jsonl | 351 | 105 | 50 | 105 |
| coordination | coordinate-other-ais.jsonl | 410 | 123 | 50 | 123 |

Shuffle with Python `random.Random(42)`. Reserve the first `min(500, floor(.3*N))`
rows for direction extraction and the next `min(300, floor(.3*N))` for evaluation.
Shuffle the remaining pool with the adapter seed and take its first 50 rows
for SFT. Unselected pool rows are unused. Different adapter seeds reuse the
same direction/evaluation partitions and draw different SFT samples.

The primary seeds are `42, 123, 456, 789, 1024, 2048, 3407, 5555, 7777, 9999`.
`split_metadata.json` records the zero-based row indices after ignoring empty
JSONL lines, the source SHA-256, and both seeds. For custom data, each JSONL
record must have `question`, `answer_matching_behavior` and
`answer_not_matching_behavior`. Answers can be strings or lists; the first
element of a list is used, matching the original experiment.

For original paper scripts, use their expected filenames and directory:

```bash
python scripts/download_data.py --original-names --output-dir benchmarks/data/anthropic_evals
python scripts/prepare_capability_data.py
```

Capability preparation downloads [MMLU](https://huggingface.co/datasets/cais/mmlu)
and [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa).
The full-MMLU script uses all 14,042 test items. Older auxiliary scripts use
the first five questions per subject; keep those measurements separate.

## Attribution and licenses

- Anthropic: *Discovering Language Model Behaviors with Model-Written Evaluations*
  (Perez et al.). Data: [CC BY 4.0](https://github.com/anthropics/evals/blob/main/LICENSE).
  The files are downloaded without content changes; splits and directions are derived artifacts.
- The direction-extraction approach follows *Steering Llama 2 via Contrastive
  Activation Addition* (Rimsky et al.). This repository contains the authors'
  experiment implementation; it does not vendor the CAA codebase.
- MMLU and TruthfulQA retain the licenses and attribution requirements specified
  by their upstream dataset repositories.
- [Qwen3 model cards](https://huggingface.co/Qwen/Qwen3-8B) and
  [LLaMA model cards](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  describe model licenses. LLaMA access may require accepting the model terms
  and authenticating with Hugging Face.

The repository's MIT license applies to original code, not to downloaded
datasets or model weights.
