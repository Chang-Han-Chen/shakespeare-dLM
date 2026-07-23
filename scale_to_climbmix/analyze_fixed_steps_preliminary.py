"""Preliminary fixed-step analysis using completed baselines and old-grid interpolation."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from config import ROOT
from curriculum_config import P_AR_VALUES
from fixed_steps_config import ALLOCATION_SOURCE, FIXED_STEP_TARGETS
from train import atomic_json_dump


OLD_RESULTS = ROOT / "results_curriculum" / "runs"
RESULTS = ROOT / "results_fixed_steps"
FIGURES = ROOT / "figures_fixed_steps"


def completed_baseline_rows(target):
    rows = []
    for path in (RESULTS / "baseline" / target.spec.label).glob("lr_*/result.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "complete" and math.isfinite(row["val_nelbo"]):
            rows.append(row)
    return sorted(rows, key=lambda row: row["learning_rate"])


def old_curriculum_rows():
    rows = []
    for path in OLD_RESULTS.glob("*/*/p_ar_*/result.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "complete":
            rows.append(row)
    return rows


def interpolate_log_compute(points, target_compute):
    points = sorted(points)
    lower = max((point for point in points if point[0] <= target_compute), default=None)
    upper = min((point for point in points if point[0] >= target_compute), default=None)
    if lower is None or upper is None:
        return None
    if lower[0] == upper[0]:
        return lower[1]
    weight = math.log(target_compute / lower[0]) / math.log(upper[0] / lower[0])
    return lower[1] + weight * (upper[1] - lower[1])


def predictions(target, baseline_loss, curriculum):
    sample = next(row for row in curriculum if row["size"] == target.spec.label)
    ar_over_bd = (
        sample["autoregressive_training_flops_per_clean_token"]
        / sample["block_diffusion_training_flops_per_clean_token"]
    )
    rows = [
        {
            "p_ar": 0.0,
            "target_compute": target.predicted_compute,
            "predicted_val_nelbo": baseline_loss,
            "gain_vs_full_bd": 0.0,
            "support": "measured_new_baseline",
        }
    ]
    for p_ar in P_AR_VALUES:
        matched_compute = target.predicted_compute * (
            (1.0 - p_ar) + p_ar * ar_over_bd
        )
        points = [
            (float(row["budget"]), float(row["val_nelbo"]))
            for row in curriculum
            if row["size"] == target.spec.label
            and math.isclose(float(row["p_ar"]), p_ar)
        ]
        loss = interpolate_log_compute(points, matched_compute)
        rows.append(
            {
                "p_ar": p_ar,
                "target_compute": matched_compute,
                "predicted_val_nelbo": loss,
                "gain_vs_full_bd": (
                    None if loss is None else baseline_loss - loss
                ),
                "support": (
                    "old_grid_log_compute_interpolation"
                    if loss is not None
                    else "unsupported_outside_old_compute_grid"
                ),
            }
        )
    return rows


def main():
    curriculum = old_curriculum_rows()
    baseline_sets = {
        target.spec.label: completed_baseline_rows(target)
        for target in FIXED_STEP_TARGETS
    }
    prediction_sets = {}
    summaries = []
    for target in FIXED_STEP_TARGETS:
        baselines = baseline_sets[target.spec.label]
        if not baselines:
            continue
        best = min(baselines, key=lambda row: row["val_nelbo"])
        predicted = predictions(target, best["val_nelbo"], curriculum)
        prediction_sets[target.spec.label] = predicted
        supported = [
            row
            for row in predicted[1:]
            if row["predicted_val_nelbo"] is not None
        ]
        best_nonzero = (
            min(supported, key=lambda row: row["predicted_val_nelbo"])
            if supported
            else None
        )
        summaries.append(
            {
                "size": target.spec.label,
                "n_params": target.spec.n_params,
                "predicted_compute": target.predicted_compute,
                "total_steps": target.total_steps,
                "clean_tokens": target.clean_tokens,
                "tokens_per_parameter": target.clean_tokens / target.spec.n_params,
                "baseline_lr": best["learning_rate"],
                "baseline_val_nelbo": best["val_nelbo"],
                "lr_selection_provisional": not (
                    RESULTS / "baseline" / target.spec.label / "lr_bracket.json"
                ).exists(),
                "predicted_best_nonzero_p_ar": (
                    None if best_nonzero is None else best_nonzero["p_ar"]
                ),
                "predicted_best_nonzero_gain": (
                    None if best_nonzero is None else best_nonzero["gain_vs_full_bd"]
                ),
            }
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    target_ax, lr_ax, curve_ax, gain_ax = axes.flat
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(FIXED_STEP_TARGETS)))

    n_all = np.array([target.spec.n_params for target in FIXED_STEP_TARGETS])
    tokens = np.array([target.clean_tokens for target in FIXED_STEP_TARGETS])
    ratios = tokens / n_all
    target_ax.plot(n_all, tokens, marker="o", color="#35618d", label="clean tokens")
    target_ax.set_xscale("log")
    target_ax.set_yscale("log")
    target_ax.set_xlabel("counted parameters")
    target_ax.set_ylabel("fixed clean tokens")
    ratio_ax = target_ax.twinx()
    ratio_ax.plot(n_all, ratios, marker="s", ls="--", color="#c46d3b", label="D/N")
    ratio_ax.set_ylabel("tokens per parameter")
    target_ax.set_title("Targets from the pure-BD allocation law")

    for color, target in zip(colors, FIXED_STEP_TARGETS):
        rows = baseline_sets[target.spec.label]
        if not rows:
            continue
        lr_ax.plot(
            [row["learning_rate"] for row in rows],
            [row["val_nelbo"] for row in rows],
            marker="o",
            color=color,
            label=target.spec.label,
        )
    lr_ax.set_xscale("log")
    lr_ax.set_xlabel("peak learning rate")
    lr_ax.set_ylabel("validation NELBO")
    lr_ax.set_title("Measured full-run LR responses")
    lr_ax.legend(ncol=2, fontsize=8)

    supported_summaries = []
    for color, target in zip(colors, FIXED_STEP_TARGETS):
        rows = prediction_sets.get(target.spec.label)
        if not rows or any(row["predicted_val_nelbo"] is None for row in rows[1:]):
            continue
        p = [row["p_ar"] for row in rows]
        gain = [row["gain_vs_full_bd"] for row in rows]
        curve_ax.plot(
            p,
            gain,
            marker="o",
            ls="--",
            color=color,
            label=target.spec.label,
        )
        supported_summaries.append(
            next(row for row in summaries if row["size"] == target.spec.label)
        )
    curve_ax.axhline(0.0, color="black", lw=1)
    curve_ax.set_xlabel(r"$p_{\rm AR}$")
    curve_ax.set_ylabel(r"predicted gain $L(0)-L(p)$")
    curve_ax.set_title("Old-grid prediction at matched total steps")
    curve_ax.legend(title="interpolation only", ncol=2, fontsize=8)

    if supported_summaries:
        n = [row["n_params"] for row in supported_summaries]
        gain = [row["predicted_best_nonzero_gain"] for row in supported_summaries]
        gain_ax.plot(n, gain, marker="o", color="#c46d3b", lw=2)
    gain_ax.axhline(0.0, color="black", lw=1)
    gain_ax.set_xscale("log")
    gain_ax.set_xlabel("counted parameters")
    gain_ax.set_ylabel("predicted best nonzero-p gain")
    gain_ax.set_title("Preliminary expectation—not a new measurement")

    fig.suptitle(
        "Fixed-step study: measured baselines and interpolation-only expectations",
        fontsize=14,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"fixed_steps_preliminary.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    payload = {
        "status": "preliminary",
        "allocation_source": ALLOCATION_SOURCE,
        "warning": (
            "Curriculum values are log-compute interpolations of the old "
            "fixed-FLOP sweep, not measurements from the new fixed-step runs. "
            "Neighboring old budgets may inherit different selected LRs."
        ),
        "summaries": summaries,
        "predictions": prediction_sets,
    }
    atomic_json_dump(payload, RESULTS / "preliminary_analysis.json")
    for row in summaries:
        gain = row["predicted_best_nonzero_gain"]
        print(
            f"{row['size']:>5} D/N={row['tokens_per_parameter']:.1f} "
            f"lr={row['baseline_lr']:.4g} "
            + (
                "matched-step prediction unsupported"
                if gain is None
                else (
                    f"pred_p={row['predicted_best_nonzero_p_ar']:.1f} "
                    f"pred_gain={gain:+.5f}"
                )
            )
        )


if __name__ == "__main__":
    main()
