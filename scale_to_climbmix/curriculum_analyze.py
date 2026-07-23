"""Analyze compute-optimal and fixed-size effects of the shared-AR curriculum."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import fit_l1_quadratic, fit_profiles, load_best_runs, scaling_laws
from config import (
    COMPUTE_BUDGETS,
    MODEL_SPECS,
    ROOT,
    budget_slug,
    is_feasible as baseline_is_feasible,
)
from curriculum_config import P_AR_VALUES, is_feasible


RESULTS = ROOT / "results_curriculum"
FIGURES = ROOT / "figures_curriculum"


def load_rows():
    baseline = load_best_runs()
    for row in baseline:
        row["p_ar"] = 0.0
    curriculum = []
    for path in RESULTS.glob("runs/*/*/p_ar_*/result.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "complete":
            row["result_path"] = str(path)
            curriculum.append(row)
    expected = sum(
        is_feasible(budget, spec, p_ar)
        for budget in COMPUTE_BUDGETS
        for spec in MODEL_SPECS
        for p_ar in P_AR_VALUES
    )
    if len(curriculum) != expected:
        raise RuntimeError(f"Expected {expected} curriculum results, found {len(curriculum)}")
    return baseline, curriculum


def interpolate_mixed_flops(n_params, p_ar):
    x = np.log([spec.n_params for spec in MODEL_SPECS])
    y = np.log(
        [
            p_ar * spec.autoregressive_training_flops_per_clean_token
            + (1 - p_ar) * spec.training_flops_per_clean_token
            for spec in MODEL_SPECS
        ]
    )
    return float(np.exp(np.interp(np.log(n_params), x, y)))


def fit_envelopes(baseline, curriculum):
    rows = baseline + curriculum
    fits = []
    for p_ar in (0.0,) + P_AR_VALUES:
        for budget in COMPUTE_BUDGETS:
            points = sorted(
                [
                    row
                    for row in rows
                    if row["budget"] == budget and row["p_ar"] == p_ar
                ],
                key=lambda row: row["n_params"],
            )
            x = np.log10([row["n_params"] for row in points])
            y = np.array([row["val_nelbo"] for row in points])
            coefficients, support = fit_l1_quadratic(x, y)
            a, b, _ = coefficients
            vertex = -b / (2 * a) if a > 0 else math.nan
            outside = not (a > 0 and x.min() <= vertex <= x.max())
            if outside:
                # Preserve a transparent censored boundary optimum rather than
                # extrapolating an unmeasured quadratic minimum.
                index = int(np.argmin(y))
                vertex = float(x[index])
                fit_kind = "observed_boundary"
            else:
                fit_kind = "interior_l1_quadratic"
            n_opt = 10**vertex
            flops_per_token = interpolate_mixed_flops(n_opt, p_ar)
            fits.append(
                {
                    "budget": budget,
                    "p_ar": p_ar,
                    "fit_kind": fit_kind,
                    "n_points": len(points),
                    "n_opt": n_opt,
                    "effective_n_opt": flops_per_token / 12,
                    "d_opt": budget / flops_per_token,
                    "loss_min": (
                        float(np.polyval(coefficients, vertex))
                        if fit_kind == "interior_l1_quadratic"
                        else float(y[int(np.argmin(y))])
                    ),
                    "coefficients": coefficients.tolist(),
                    "support_indices": list(support),
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                }
            )
    return fits


def observed_envelopes(baseline, curriculum):
    rows = baseline + curriculum
    envelopes = []
    for p_ar in (0.0,) + P_AR_VALUES:
        for budget in COMPUTE_BUDGETS:
            points = [
                row
                for row in rows
                if row["budget"] == budget and row["p_ar"] == p_ar
            ]
            best = min(points, key=lambda row: row["val_nelbo"])
            envelopes.append(
                {
                    "budget": budget,
                    "p_ar": p_ar,
                    "loss_min": best["val_nelbo"],
                    "n_opt": best["n_params"],
                    "size": best["size"],
                }
            )
    return envelopes


def write_csv(path, rows):
    fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"coefficients", "support_indices"}
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_fixed_model_figures(baseline, curriculum, n_opt_law):
    output = FIGURES / "fixed_model_l_vs_p"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = baseline + curriculum
    for spec in MODEL_SPECS:
        budgets = [
            budget
            for budget in COMPUTE_BUDGETS
            if baseline_is_feasible(budget, spec)
        ]
        continuous_c_opt = (
            spec.n_params / n_opt_law["coefficient"]
        ) ** (1 / n_opt_law["exponent"])
        closest_budget = min(
            budgets,
            key=lambda budget: abs(math.log(budget / continuous_c_opt)),
        )
        columns = min(3, len(budgets))
        rows_count = math.ceil(len(budgets) / columns)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(4.2 * columns, 3.4 * rows_count),
            squeeze=False,
        )
        for axis, budget in zip(axes.flat, budgets):
            points = sorted(
                [
                    row
                    for row in all_rows
                    if row["budget"] == budget and row["size"] == spec.label
                ],
                key=lambda row: row["p_ar"],
            )
            p = np.array([row["p_ar"] for row in points])
            loss = np.array([row["val_nelbo"] for row in points])
            axis.plot(p, loss, marker="o", color="#35618d", lw=1.8)
            best = int(np.argmin(loss))
            axis.scatter(p[best], loss[best], marker="*", s=120, color="#d17c37", zorder=3)
            axis.set_title(
                rf"$C={budget:.0e}$",
                color="#c43b3b" if budget == closest_budget else "black",
                fontweight="bold" if budget == closest_budget else "normal",
            )
            axis.set_xlabel(r"$p_{\rm AR}$")
            axis.set_ylabel("validation NELBO")
            axis.set_xticks(np.arange(0, 0.7, 0.1))
            axis.grid(alpha=0.2)
        for axis in axes.flat[len(budgets) :]:
            axis.axis("off")
        fig.suptitle(
            (
                f"Fixed model: {spec.label} ({spec.n_params:,} counted parameters)"
                rf"  ·  predicted $C_{{\rm opt}}\approx{continuous_c_opt:.1e}$"
            ),
            fontsize=13,
        )
        fig.tight_layout()
        stem = spec.label.replace(".", "p")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"fixed_N_{stem}.{extension}", bbox_inches="tight")
        plt.close(fig)


def timing_rows(curriculum):
    rows = []
    for budget in COMPUTE_BUDGETS:
        for spec in MODEL_SPECS:
            if not baseline_is_feasible(budget, spec):
                continue
            pure_path = (
                RESULTS
                / "shared_ar"
                / budget_slug(budget)
                / spec.label
                / "pure_ar_result.json"
            )
            pure = json.loads(pure_path.read_text())
            point_rows = [
                row
                for row in curriculum
                if row["budget"] == budget and row["size"] == spec.label
            ]
            measured_bd = float(np.median([row["bd_seconds_per_step"] for row in point_rows]))
            ar_seconds = pure["seconds_per_step"]
            rows.append(
                {
                    "budget": budget,
                    "size": spec.label,
                    "n_params": spec.n_params,
                    "theoretical_bd_over_ar": (
                        spec.training_flops_per_clean_token
                        / spec.autoregressive_training_flops_per_clean_token
                    ),
                    "measured_bd_over_ar": measured_bd / ar_seconds,
                    "ar_seconds_per_step": ar_seconds,
                    "bd_seconds_per_step": measured_bd,
                }
            )
    return rows


def make_summary_figure(fits, observed, timing):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    loss_ax, gain_ax, n_ax, timing_ax = axes.flat
    for color, budget in zip(colors, COMPUTE_BUDGETS):
        points = sorted(
            [row for row in fits if row["budget"] == budget],
            key=lambda row: row["p_ar"],
        )
        p = np.array([row["p_ar"] for row in points])
        loss = np.array([row["loss_min"] for row in points])
        n_opt = np.array([row["n_opt"] for row in points])
        loss_ax.plot(p, loss, marker="o", color=color, label=rf"$C={budget:.0e}$")
        gain_ax.plot(p, loss[0] - loss, marker="o", color=color)
        n_ax.plot(p, n_opt, marker="o", color=color)
        measured_points = sorted(
            [row for row in observed if row["budget"] == budget],
            key=lambda row: row["p_ar"],
        )
        measured_loss = np.array([row["loss_min"] for row in measured_points])
        loss_ax.scatter(
            p,
            measured_loss,
            marker="x",
            s=42,
            linewidths=1.5,
            color=color,
            zorder=4,
        )
        gain_ax.scatter(
            p,
            measured_loss[0] - measured_loss,
            marker="x",
            s=42,
            linewidths=1.5,
            color=color,
            zorder=4,
        )
        for index, row in enumerate(points):
            if row["fit_kind"] == "observed_boundary":
                loss_ax.scatter(p[index], loss[index], facecolors="none", edgecolors="black", s=75)
    loss_ax.set_xlabel(r"$p_{\rm AR}$")
    loss_ax.set_ylabel("compute-optimal validation NELBO")
    loss_ax.set_title("Curriculum envelope")
    loss_ax.legend(frameon=False, fontsize=8)
    loss_ax.text(
        0.02,
        0.02,
        "circles/lines: L1 quadratic minima\n×: best measured point",
        transform=loss_ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    gain_ax.axhline(0, color="black", lw=0.8)
    gain_ax.set_xlabel(r"$p_{\rm AR}$")
    gain_ax.set_ylabel("NELBO improvement over pure BD")
    gain_ax.set_title("Compute-optimal curriculum gain")

    n_ax.set_yscale("log")
    n_ax.set_xlabel(r"$p_{\rm AR}$")
    n_ax.set_ylabel(r"compute-optimal counted $N$")
    n_ax.set_title("Optimal model size shifts")

    raw_theory = np.array([row["theoretical_bd_over_ar"] for row in timing])
    raw_measured = np.array([row["measured_bd_over_ar"] for row in timing])
    timing_ax.scatter(
        raw_theory,
        raw_measured,
        color="#7a4f83",
        alpha=0.18,
        s=24,
        zorder=1,
    )
    medians = []
    for spec in MODEL_SPECS:
        point_rows = [row for row in timing if row["size"] == spec.label]
        if not point_rows:
            continue
        measured = np.array(
            [row["measured_bd_over_ar"] for row in point_rows]
        )
        medians.append(
            {
                "size": spec.label,
                "theory": point_rows[0]["theoretical_bd_over_ar"],
                "median": float(np.median(measured)),
                "low": float(np.min(measured)),
                "high": float(np.max(measured)),
            }
        )
    theory = np.array([row["theory"] for row in medians])
    measured = np.array([row["median"] for row in medians])
    error = np.array(
        [
            measured - np.array([row["low"] for row in medians]),
            np.array([row["high"] for row in medians]) - measured,
        ]
    )
    timing_ax.errorbar(
        theory,
        measured,
        yerr=error,
        fmt="o",
        color="#7a4f83",
        ecolor="#7a4f83",
        alpha=0.9,
        capsize=3,
        zorder=3,
    )
    for row in medians:
        timing_ax.annotate(
            row["size"],
            (row["theory"], row["median"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    limits = [
        min(raw_theory.min(), raw_measured.min()),
        max(raw_theory.max(), raw_measured.max()),
    ]
    timing_ax.plot(limits, limits, color="black", ls="--", lw=1)
    timing_ax.set_xlabel("theoretical BD / AR step-time ratio")
    timing_ax.set_ylabel("measured BD / AR step-time ratio")
    timing_ax.set_title("Compute model versus wall clock (median + range)")

    fig.suptitle("AR → block-diffusion curriculum on ClimbMix", fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"curriculum_summary.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    baseline, curriculum = load_rows()
    pure_bd_laws = scaling_laws(fit_profiles(baseline))
    fits = fit_envelopes(baseline, curriculum)
    observed = observed_envelopes(baseline, curriculum)
    timing = timing_rows(curriculum)
    pure_ar = [
        json.loads(path.read_text())
        for path in RESULTS.glob("shared_ar/*/*/pure_ar_result.json")
    ]
    write_csv(RESULTS / "curriculum_runs.csv", curriculum)
    write_csv(RESULTS / "optimal_envelopes.csv", fits)
    write_csv(RESULTS / "observed_envelopes.csv", observed)
    write_csv(RESULTS / "phase_timing.csv", timing)
    write_csv(RESULTS / "pure_ar_runs.csv", pure_ar)
    summary = {
        "p_ar_values": [0.0, *P_AR_VALUES],
        "envelopes": fits,
        "observed_envelopes": observed,
        "timing": timing,
        "pure_ar_runs": pure_ar,
        "boundary_optima": sum(
            row["fit_kind"] == "observed_boundary"
            for row in fits
        ),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_fixed_model_figures(baseline, curriculum, pure_bd_laws["n_opt"])
    make_summary_figure(fits, observed, timing)
    for budget in COMPUTE_BUDGETS:
        points = [row for row in fits if row["budget"] == budget]
        best = min(points, key=lambda row: row["loss_min"])
        baseline_loss = next(row["loss_min"] for row in points if row["p_ar"] == 0)
        print(
            f"C={budget:.0e} best_p={best['p_ar']:.1f} "
            f"gain={baseline_loss - best['loss_min']:.5f} "
            f"N*={best['n_opt'] / 1e6:.3f}M"
        )


if __name__ == "__main__":
    main()
