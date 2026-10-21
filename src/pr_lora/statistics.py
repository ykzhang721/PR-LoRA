"""Reconstruct the primary table and paired tests from recorded seed results."""

import numpy as np
from scipy import stats

BEHAVIORS = (
    "sycophancy",
    "sycophancy_phil",
    "power_seeking",
    "corrigibility",
    "survival",
    "myopia",
    "coordination",
)
SEEDS = (42, 123, 456, 789, 1024, 2048, 3407, 5555, 7777, 9999)
METHODS = ("baseline", "steering", "frozen_r4", "std_r4", "pr_r4")


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite probabilities")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return adjusted


def summarize_main(results):
    """Require all 10 shared seeds; ddof=0 matches the manuscript tables."""
    rows = {}
    p_values = []
    for behavior in BEHAVIORS:
        rows[behavior] = {}
        for method in METHODS:
            if method in ("baseline", "steering"):
                values = [results[f"{behavior}_{method}"]] * len(SEEDS)
            else:
                values = [results[f"{behavior}_{method}_s{seed}"] for seed in SEEDS]
            rows[behavior][method] = np.array(values, dtype=float)
        p_values.append(stats.ttest_rel(rows[behavior]["pr_r4"], rows[behavior]["std_r4"]).pvalue)
    adjusted = holm_adjust(p_values)
    print(
        "| Behavior | Base | Steering | Frozen-B | Standard LoRA | PR-LoRA | Holm p (PR vs Std) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for behavior, p in zip(BEHAVIORS, adjusted):
        cells = []
        for method in METHODS:
            values = rows[behavior][method]
            cells.append(
                f"{values.mean():.3f}"
                if method in ("baseline", "steering")
                else f"{values.mean():.3f} ± {values.std(ddof=0):.3f}"
            )
        print(f"| {behavior} | " + " | ".join(cells) + f" | {p:.4g} |")
    macro = {method: np.mean([rows[b][method] for b in BEHAVIORS], axis=0) for method in METHODS}
    print("| Macro average | " + " | ".join(f"{macro[m].mean():.3f}" for m in METHODS) + " | — |")
    delta = macro["std_r4"].mean() - macro["pr_r4"].mean()
    p = stats.ttest_rel(macro["pr_r4"], macro["std_r4"]).pvalue
    print(
        f"\nPR vs Std: {delta * 100:.2f} percentage points, "
        f"{delta / macro['std_r4'].mean() * 100:.2f}% relative reduction; paired macro p={p:.6f}"
    )
    return rows, macro
