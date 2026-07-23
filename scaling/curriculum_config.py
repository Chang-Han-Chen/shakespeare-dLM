"""Configuration and exact compute accounting for the AR-to-BD sweep."""

from __future__ import annotations

from config import (
    BATCH_SIZE,
    COMPUTE_BUDGETS,
    MAX_STEPS,
    MIN_STEPS,
    MODEL_SPECS,
    ROOT,
    TARGETED_BUDGETS,
    TOKENS_PER_STEP,
    ModelSpec,
)


RESULTS_DIR = ROOT / "results_p_ar"
FIGURES_DIR = ROOT / "figures_p_ar"
P_AR_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
COMPUTE_ACCOUNTING = "ar_to_dense_dual_stream_attention_v1"


def p_ar_slug(p_ar: float) -> str:
    if p_ar not in P_AR_VALUES:
        raise ValueError(f"Unsupported p_ar={p_ar}")
    return f"p_ar_{p_ar:.1f}"


def split_phase_steps(total_steps: int, p_ar: float) -> tuple[int, int]:
    """Nearest-integer AR allocation with two non-empty phases."""
    if total_steps < 2:
        raise ValueError("A curriculum needs at least two steps")
    if not 0.0 < p_ar < 1.0:
        raise ValueError("p_ar must be strictly between zero and one")
    ar_steps = int(total_steps * p_ar + 0.5)
    ar_steps = min(max(ar_steps, 1), total_steps - 1)
    return ar_steps, total_steps - ar_steps


def average_training_flops_per_clean_token(spec: ModelSpec, p_ar: float) -> float:
    """Continuous-allocation FLOPs used to describe C(N,D,p_ar)."""
    ar = spec.autoregressive_training_flops_per_clean_token
    bd = spec.training_flops_per_clean_token
    return p_ar * ar + (1.0 - p_ar) * bd


def realized_flops(total_steps: int, spec: ModelSpec, p_ar: float) -> int:
    """Exact FLOPs after rounding the phase transition to an optimizer step."""
    ar_steps, bd_steps = split_phase_steps(total_steps, p_ar)
    ar_flops = ar_steps * TOKENS_PER_STEP * spec.autoregressive_training_flops_per_clean_token
    bd_flops = bd_steps * TOKENS_PER_STEP * spec.training_flops_per_clean_token
    return ar_flops + bd_flops


def steps_for(budget: float, spec: ModelSpec, p_ar: float) -> int:
    """Largest whole-step curriculum whose exact realized FLOPs fit C."""
    estimate = int(
        budget
        / (average_training_flops_per_clean_token(spec, p_ar) * TOKENS_PER_STEP)
    )
    steps = max(2, estimate)
    while realized_flops(steps, spec, p_ar) > budget:
        steps -= 1
    while realized_flops(steps + 1, spec, p_ar) <= budget:
        steps += 1
    return steps


def is_feasible(budget: float, spec: ModelSpec, p_ar: float) -> bool:
    if spec.label in TARGETED_BUDGETS and budget not in TARGETED_BUDGETS[spec.label]:
        return False
    steps = steps_for(budget, spec, p_ar)
    return MIN_STEPS <= steps <= MAX_STEPS


def validate_curriculum_config() -> None:
    assert BATCH_SIZE == 128
    for p_ar in P_AR_VALUES:
        for budget in COMPUTE_BUDGETS:
            for spec in MODEL_SPECS:
                steps = steps_for(budget, spec, p_ar)
                assert realized_flops(steps, spec, p_ar) <= budget
                assert realized_flops(steps + 1, spec, p_ar) > budget


validate_curriculum_config()
