"""Exact compute and schedule definitions for the shared-AR curriculum sweep."""

from __future__ import annotations

from config import (
    MAX_STEPS,
    MIN_STEPS,
    TOKENS_PER_STEP,
    ModelSpec,
    is_feasible as baseline_is_feasible,
)


P_AR_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
# AR is cheaper, so the same FLOP budget can modestly exceed the pure-BD
# 25k-step cutoff. The largest requested point is 30,887 steps.
CURRICULUM_MAX_STEPS = 35_000
AR_DECAY_FRACTION = 0.15
AR_WARMUP_REFERENCE_P = 0.1


def split_phase_steps(total_steps: int, p_ar: float) -> tuple[int, int]:
    ar_steps = round(p_ar * total_steps)
    return ar_steps, total_steps - ar_steps


def mixed_steps_for(budget: float, spec: ModelSpec, p_ar: float) -> int:
    if not 0.0 < p_ar < 1.0:
        raise ValueError("p_ar must be strictly between zero and one")
    average = (
        p_ar * spec.autoregressive_training_flops_per_clean_token
        + (1.0 - p_ar) * spec.training_flops_per_clean_token
    )
    total = int(budget / (TOKENS_PER_STEP * average))
    while total > 0:
        ar_steps, bd_steps = split_phase_steps(total, p_ar)
        if realized_flops(ar_steps, bd_steps, spec) <= budget:
            return total
        total -= 1
    return 0


def phase_steps_for(
    budget: float,
    spec: ModelSpec,
    p_ar: float,
) -> tuple[int, int]:
    return split_phase_steps(mixed_steps_for(budget, spec, p_ar), p_ar)


def pure_ar_steps_for(budget: float, spec: ModelSpec) -> int:
    return int(
        budget
        / (TOKENS_PER_STEP * spec.autoregressive_training_flops_per_clean_token)
    )


def ar_decay_steps(ar_steps: int) -> int:
    return max(1, round(AR_DECAY_FRACTION * ar_steps))


def ar_decay_start(ar_steps: int) -> int:
    return ar_steps - ar_decay_steps(ar_steps)


def shared_ar_warmup_steps(budget: float, spec: ModelSpec) -> int:
    shortest_ar, _ = phase_steps_for(budget, spec, AR_WARMUP_REFERENCE_P)
    return max(1, round(0.05 * shortest_ar))


def realized_flops(ar_steps: int, bd_steps: int, spec: ModelSpec) -> int:
    return TOKENS_PER_STEP * (
        ar_steps * spec.autoregressive_training_flops_per_clean_token
        + bd_steps * spec.training_flops_per_clean_token
    )


def is_feasible(budget: float, spec: ModelSpec, p_ar: float) -> bool:
    if not baseline_is_feasible(budget, spec):
        return False
    total = mixed_steps_for(budget, spec, p_ar)
    ar_steps, bd_steps = split_phase_steps(total, p_ar)
    return (
        MIN_STEPS <= total <= CURRICULUM_MAX_STEPS
        and ar_steps > 0
        and bd_steps > 0
    )


def validate_curriculum_config() -> None:
    for left, right in zip(P_AR_VALUES, P_AR_VALUES[1:]):
        assert left < right
    assert MAX_STEPS < CURRICULUM_MAX_STEPS


validate_curriculum_config()
