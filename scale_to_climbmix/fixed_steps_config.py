"""Configuration for the compute-optimal, fixed-total-step curriculum study."""

from __future__ import annotations

from dataclasses import dataclass

from config import MODEL_SPECS, TOKENS_PER_STEP, ModelSpec
from curriculum_config import P_AR_VALUES


# Primary pure-BD allocation law from results/summary.json:
# N_opt(C) = N_OPT_COEFFICIENT * C ** N_OPT_EXPONENT.
# These values are copied here so the experiment targets cannot silently move
# if a later plotting change refits the original IsoFLOP profiles.
N_OPT_COEFFICIENT = 0.014697293557828773
N_OPT_EXPONENT = 0.5138362043386316
ALLOCATION_SOURCE = (
    "pure-BD L2 quadratics through the three lowest-loss points per IsoFLOP "
    "profile; results/summary.json"
)


@dataclass(frozen=True)
class FixedStepTarget:
    spec: ModelSpec
    predicted_compute: float
    total_steps: int

    @property
    def clean_tokens(self) -> int:
        return self.total_steps * TOKENS_PER_STEP

    @property
    def realized_full_bd_compute(self) -> int:
        return self.clean_tokens * self.spec.training_flops_per_clean_token


def compute_optimal_budget(spec: ModelSpec) -> float:
    """Invert the fitted pure-BD N_opt(C) allocation law."""
    return (spec.n_params / N_OPT_COEFFICIENT) ** (1.0 / N_OPT_EXPONENT)


def fixed_step_target(spec: ModelSpec) -> FixedStepTarget:
    predicted_compute = compute_optimal_budget(spec)
    total_steps = int(
        predicted_compute
        / (TOKENS_PER_STEP * spec.training_flops_per_clean_token)
    )
    if total_steps < 150:
        raise ValueError(f"{spec.label} target has only {total_steps} steps")
    return FixedStepTarget(spec, predicted_compute, total_steps)


FIXED_STEP_TARGETS = tuple(fixed_step_target(spec) for spec in MODEL_SPECS)
TARGET_BY_SIZE = {target.spec.label: target for target in FIXED_STEP_TARGETS}


def split_fixed_steps(total_steps: int, p_ar: float) -> tuple[int, int]:
    if p_ar not in P_AR_VALUES:
        raise ValueError(f"Unsupported p_ar={p_ar}")
    ar_steps = round(p_ar * total_steps)
    return ar_steps, total_steps - ar_steps


def validate_fixed_step_config() -> None:
    previous_compute = 0.0
    previous_steps = 0
    for target in FIXED_STEP_TARGETS:
        assert target.predicted_compute > previous_compute
        assert target.total_steps > previous_steps
        assert target.realized_full_bd_compute <= target.predicted_compute
        for p_ar in P_AR_VALUES:
            ar_steps, bd_steps = split_fixed_steps(target.total_steps, p_ar)
            assert ar_steps + bd_steps == target.total_steps
            assert ar_steps > 0 and bd_steps > 0
        previous_compute = target.predicted_compute
        previous_steps = target.total_steps


validate_fixed_step_config()
