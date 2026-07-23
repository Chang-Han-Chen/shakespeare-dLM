"""Select LRs, fit IsoFLOP parabolas/scaling laws, and make the final figure."""

from __future__ import annotations

import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from config import (
    COMPUTE_BUDGETS,
    FIGURES_DIR,
    FLOP_MULTIPLIER,
    MODEL_SPECS,
    RESULTS_DIR,
    ROOT,
    budget_slug,
    is_feasible,
)


def interpolated_flops_per_clean_token(n_params):
    """Log-linear interpolation of architecture FLOPs at a fitted N vertex."""
    log_n = np.log10([spec.n_params for spec in MODEL_SPECS])
    log_flops = np.log10(
        [spec.training_flops_per_clean_token for spec in MODEL_SPECS]
    )
    target = math.log10(n_params)
    if not log_n.min() <= target <= log_n.max():
        raise ValueError(f"Cannot interpolate FLOPs outside model grid: N={n_params}")
    return float(10 ** np.interp(target, log_n, log_flops))


def load_results():
    rows = []
    for path in sorted((RESULTS_DIR / "runs").glob("*/*/lr_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and math.isfinite(payload["val_nelbo"]):
            payload["result_path"] = str(path.relative_to(RESULTS_DIR.parent))
            rows.append(payload)
    return rows


def select_best(rows):
    grouped = {}
    for row in rows:
        key = (float(row["budget"]), row["size"])
        if key not in grouped or row["val_nelbo"] < grouped[key]["val_nelbo"]:
            grouped[key] = row
    return sorted(grouped.values(), key=lambda row: (row["budget"], row["n_params"]))


def validate_lr_brackets(best_rows):
    for row in best_rows:
        path = (
            RESULTS_DIR
            / "runs"
            / budget_slug(float(row["budget"]))
            / row["size"]
            / "lr_bracket.json"
        )
        if not path.exists():
            raise RuntimeError(f"Missing LR bracket: {path}. Run bracket_lr.py first.")
        bracket = json.loads(path.read_text(encoding="utf-8"))
        if bracket.get("status") != "locally_bracketed":
            raise RuntimeError(f"Invalid LR bracket: {path}")
        if not math.isclose(bracket["selected_lr"], row["learning_rate"]):
            raise RuntimeError(f"Selected LR does not match bracket at {path}")
        if bracket["discrete_curvature"] <= 0:
            raise RuntimeError(f"Non-convex LR neighborhood at {path}")


def fit_l1_quadratic(x, y):
    """Exact least-absolute-deviation quadratic for a full-rank 1-D design.

    A basic optimum of LAD regression with three coefficients can be chosen
    with three zero residuals. Enumerating those support triples is exact here
    and avoids adding a heavyweight optimization dependency for at most eight
    points per IsoFLOP profile.
    """
    candidates = []
    for support in combinations(range(len(x)), 3):
        support_x = x[list(support)]
        design = np.column_stack((support_x**2, support_x, np.ones(3)))
        try:
            coefficients = np.linalg.solve(design, y[list(support)])
        except np.linalg.LinAlgError:
            continue
        residual = y - np.polyval(coefficients, x)
        candidates.append(
            (
                float(np.abs(residual).sum()),
                float(np.square(residual).sum()),
                support,
                coefficients,
            )
        )
    if not candidates:
        raise RuntimeError("Could not construct an L1 quadratic")
    _, _, support, coefficients = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )
    return coefficients, list(support)


def fit_parabolas(best_rows, objective):
    if objective not in {"l1", "l2", "lowest3_l2"}:
        raise ValueError(f"Unknown quadratic objective: {objective}")
    fits = []
    for budget in COMPUTE_BUDGETS:
        points = [row for row in best_rows if float(row["budget"]) == budget]
        if len(points) < 3:
            raise RuntimeError(f"Need at least three points at C={budget:.0e}, found {len(points)}")
        x = np.log10([row["n_params"] for row in points])
        y = np.array([row["val_nelbo"] for row in points])
        if objective == "l1":
            coefficients, support = fit_l1_quadratic(x, y)
            fit_point_indices = list(range(len(points)))
        elif objective == "lowest3_l2":
            fit_point_indices = sorted(np.argsort(y)[:3].tolist())
            coefficients = np.polyfit(
                x[fit_point_indices],
                y[fit_point_indices],
                2,
            )
            support = fit_point_indices
        else:
            coefficients = np.polyfit(x, y, 2)
            support = None
            fit_point_indices = list(range(len(points)))
        curvature, linear, intercept = coefficients
        if curvature <= 0:
            raise RuntimeError(
                f"Non-convex {objective.upper()} IsoFLOP fit at C={budget:.0e}: a={curvature}"
            )
        log_n_opt_raw = -linear / (2 * curvature)
        log_n_opt = float(np.clip(log_n_opt_raw, x.min(), x.max()))
        n_opt = 10**log_n_opt
        loss_min = float(np.polyval(coefficients, log_n_opt))
        prediction = np.polyval(coefficients, x)
        residual = y - prediction
        absolute_error = float(np.abs(residual).sum())
        median_absolute_total = float(np.abs(y - np.median(y)).sum())
        ss_res = float(np.square(y - prediction).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        flops_per_clean_token_opt = interpolated_flops_per_clean_token(n_opt)
        effective_compute_parameters_opt = (
            flops_per_clean_token_opt / FLOP_MULTIPLIER
        )
        d_opt = budget / flops_per_clean_token_opt
        fits.append(
            {
                "budget": budget,
                "fit_objective": objective,
                "coefficients": coefficients.tolist(),
                "support_point_indices": support,
                "fit_point_indices": fit_point_indices,
                "n_points": len(points),
                "n_fit_points": len(fit_point_indices),
                "n_opt": n_opt,
                "d_opt": d_opt,
                "training_flops_per_clean_token_opt": flops_per_clean_token_opt,
                "effective_compute_parameters_opt": effective_compute_parameters_opt,
                "parameter_token_ratio": n_opt / d_opt,
                "tokens_per_parameter": d_opt / n_opt,
                "loss_min": loss_min,
                "r_squared": r_squared,
                "l1_pseudo_r_squared": (
                    1.0 - absolute_error / median_absolute_total
                    if median_absolute_total > 0
                    else 1.0
                ),
                "sum_absolute_error": absolute_error,
                "mean_absolute_error": absolute_error / len(points),
                "residuals": residual.tolist(),
                "vertex_clipped": not bool(x.min() <= log_n_opt_raw <= x.max()),
                "vertex_outside_fit_range": not bool(
                    x[fit_point_indices].min()
                    <= log_n_opt_raw
                    <= x[fit_point_indices].max()
                ),
                "x_min": float(x.min()),
                "x_max": float(x.max()),
                "fit_x_min": float(x[fit_point_indices].min()),
                "fit_x_max": float(x[fit_point_indices].max()),
            }
        )
    return fits


def fit_power_law(x, y):
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    prediction = 10 ** (intercept + slope * np.log10(x))
    ss_res = float(np.square(np.log10(y) - np.log10(prediction)).sum())
    centered = np.log10(y) - np.log10(y).mean()
    r_squared = 1.0 - ss_res / float(np.square(centered).sum())
    return {"coefficient": float(10**intercept), "exponent": float(slope), "r_squared": r_squared}


def fit_loss_law(compute, losses):
    """Fit a descriptive power law over the measured compute range.

    Five compute levels are insufficient to identify a free irreducible-loss
    floor in addition to a coefficient and exponent. Keeping the asymptote
    fixed at zero makes the reported curve identifiable and avoids implying an
    extrapolative floor that moves drastically under robust refitting.
    """
    compute = np.asarray(compute, dtype=float)
    losses = np.asarray(losses, dtype=float)
    law = fit_power_law(compute, losses)
    prediction = law["coefficient"] * compute ** law["exponent"]
    squared_error = float(np.square(losses - prediction).sum())
    centered_error = float(np.square(losses - losses.mean()).sum())
    return {
        "model": "pure_power_over_measured_range",
        "asymptote": 0.0,
        "raw_loss_r_squared": 1.0 - squared_error / centered_error,
        "mean_absolute_error": float(np.abs(losses - prediction).mean()),
        "root_mean_squared_error": math.sqrt(squared_error / len(losses)),
        **law,
    }


def fit_loss_law_with_floor(compute, losses, reference_compute=1e14):
    """Fit L(C) = L_inf + A * (C / reference_compute)^exponent.

    For a fixed exponent, the floor and amplitude are an ordinary two-column
    least-squares problem. A dependency-free golden-section search then finds
    the exponent that minimizes squared residuals in loss space.
    """
    compute = np.asarray(compute, dtype=float)
    losses = np.asarray(losses, dtype=float)
    normalized_compute = compute / reference_compute

    def solve_at_exponent(exponent):
        power = normalized_compute**exponent
        design = np.column_stack((np.ones_like(power), power))
        asymptote, amplitude = np.linalg.lstsq(design, losses, rcond=None)[0]
        prediction = asymptote + amplitude * power
        squared_error = float(np.square(losses - prediction).sum())
        if asymptote < 0 or amplitude <= 0:
            squared_error = math.inf
        return squared_error, float(asymptote), float(amplitude), prediction

    left, right = -5.0, -1e-8
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    inner_left = right - golden_ratio * (right - left)
    inner_right = left + golden_ratio * (right - left)
    left_score = solve_at_exponent(inner_left)[0]
    right_score = solve_at_exponent(inner_right)[0]
    for _ in range(200):
        if left_score < right_score:
            right, inner_right, right_score = inner_right, inner_left, left_score
            inner_left = right - golden_ratio * (right - left)
            left_score = solve_at_exponent(inner_left)[0]
        else:
            left, inner_left, left_score = inner_left, inner_right, right_score
            inner_right = left + golden_ratio * (right - left)
            right_score = solve_at_exponent(inner_right)[0]

    exponent = (left + right) / 2.0
    squared_error, asymptote, amplitude, prediction = solve_at_exponent(exponent)
    centered_error = float(np.square(losses - losses.mean()).sum())
    n_observations = len(losses)
    n_parameters = 3
    mean_absolute_error = float(np.abs(losses - prediction).mean())
    aic = n_observations * math.log(squared_error / n_observations) + 2 * n_parameters
    aicc = aic + (
        2 * n_parameters * (n_parameters + 1)
        / (n_observations - n_parameters - 1)
    )
    return {
        "model": "power_law_with_fitted_asymptote_sensitivity",
        "asymptote": asymptote,
        "coefficient": amplitude / reference_compute**exponent,
        "coefficient_at_reference": amplitude,
        "reference_compute": reference_compute,
        "exponent": exponent,
        "r_squared": 1.0 - squared_error / centered_error,
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": math.sqrt(squared_error / n_observations),
        "aic": aic,
        "aicc": aicc,
        "residual_degrees_of_freedom": n_observations - n_parameters,
    }


def save_best_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "best_runs.csv"
    fields = [
        "budget",
        "size",
        "n_params",
        "steps",
        "learning_rate",
        "val_nelbo",
        "val_masked_ce_t0.5",
        "clean_tokens",
        "realized_flops",
        "compute_accounting",
        "training_flops_per_clean_token",
        "effective_compute_parameters",
        "duration_seconds",
        "result_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_optimal_csv(fits, filename):
    path = RESULTS_DIR / filename
    fields = [
        "budget",
        "fit_objective",
        "n_points",
        "n_fit_points",
        "n_opt",
        "d_opt",
        "training_flops_per_clean_token_opt",
        "effective_compute_parameters_opt",
        "parameter_token_ratio",
        "tokens_per_parameter",
        "loss_min",
        "r_squared",
        "l1_pseudo_r_squared",
        "mean_absolute_error",
        "vertex_clipped",
        "vertex_outside_fit_range",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fits)
    return path


def normalized_power_law_label(name, law, reference=1e14):
    value = law["coefficient"] * reference ** law["exponent"]
    return rf"{name}={value:.2g}(C/$10^{{14}}$)$^{{{law['exponent']:.3f}}}$"


def make_figure(best_rows, parabola_fits, scaling, filename_stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    iso_ax, allocation_ax, ratio_ax, loss_ax = axes.flat
    objective_key = parabola_fits[0]["fit_objective"]
    objective_labels = {
        "l1": "L1",
        "l2": "L2",
        "lowest3_l2": "lowest-three L2",
    }
    objective = objective_labels[objective_key]

    for color, fit in zip(colors, parabola_fits):
        budget = fit["budget"]
        points = [row for row in best_rows if float(row["budget"]) == budget]
        x = np.array([row["n_params"] for row in points])
        y = np.array([row["val_nelbo"] for row in points])
        curve_log_n = np.linspace(fit["x_min"], fit["x_max"], 200)
        curve_y = np.polyval(fit["coefficients"], curve_log_n)
        if objective_key == "lowest3_l2":
            selected = np.array(fit["fit_point_indices"])
            omitted = np.array(
                [index for index in range(len(points)) if index not in fit["fit_point_indices"]]
            )
            if len(omitted):
                iso_ax.scatter(
                    x[omitted],
                    y[omitted],
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.65,
                    s=34,
                    zorder=3,
                )
            iso_ax.scatter(x[selected], y[selected], color=color, s=32, zorder=4)
            iso_ax.plot(
                10**curve_log_n,
                curve_y,
                color=color,
                ls="--",
                alpha=0.35,
                lw=1.4,
            )
            local_log_n = np.linspace(fit["fit_x_min"], fit["fit_x_max"], 160)
            iso_ax.plot(
                10**local_log_n,
                np.polyval(fit["coefficients"], local_log_n),
                color=color,
                lw=2.3,
                label=f"C={budget:.0e}",
            )
        else:
            iso_ax.scatter(x, y, color=color, s=28, zorder=3)
            iso_ax.plot(
                10**curve_log_n,
                curve_y,
                color=color,
                lw=2,
                label=f"C={budget:.0e}",
            )
        iso_ax.scatter(fit["n_opt"], fit["loss_min"], color=color, marker="*", s=130, edgecolor="black", linewidth=0.4, zorder=4)
    iso_ax.set_xscale("log")
    iso_ax.set_xlabel("non-embedding parameters N")
    iso_ax.set_ylabel("validation diffusion NELBO (nats/char)")
    iso_ax.set_title(f"IsoFLOP profiles ({objective} quadratic)")
    iso_ax.legend(frameon=False, fontsize=8, ncol=2)

    compute = np.array([fit["budget"] for fit in parabola_fits])
    n_opt = np.array([fit["n_opt"] for fit in parabola_fits])
    n_effective = np.array(
        [fit["effective_compute_parameters_opt"] for fit in parabola_fits]
    )
    d_opt = np.array([fit["d_opt"] for fit in parabola_fits])
    ratios = np.array([fit["parameter_token_ratio"] for fit in parabola_fits])
    effective_ratios = n_effective / d_opt
    loss_min = np.array([fit["loss_min"] for fit in parabola_fits])
    compute_line = np.geomspace(compute.min(), compute.max(), 200)

    n_law = scaling["n_opt"]
    n_effective_law = scaling["effective_compute_parameters_opt"]
    d_law = scaling["d_opt"]
    allocation_ax.scatter(compute, n_opt, color="#7a8da3", marker="s")
    allocation_ax.plot(
        compute_line,
        n_law["coefficient"] * compute_line ** n_law["exponent"],
        color="#7a8da3",
        ls="--",
        label=normalized_power_law_label(r"$N_\mathrm{opt}$ (counted)", n_law),
    )
    allocation_ax.scatter(compute, n_effective, color="#35618d")
    allocation_ax.plot(
        compute_line,
        n_effective_law["coefficient"]
        * compute_line ** n_effective_law["exponent"],
        color="#35618d",
        label=normalized_power_law_label(r"$N_\mathrm{eff,opt}$", n_effective_law),
    )
    allocation_ax.scatter(compute, d_opt, color="#d17c37")
    allocation_ax.plot(
        compute_line,
        d_law["coefficient"] * compute_line ** d_law["exponent"],
        color="#d17c37",
        label=normalized_power_law_label(r"$D_\mathrm{opt}$", d_law),
    )
    allocation_ax.set_xscale("log")
    allocation_ax.set_yscale("log")
    allocation_ax.set_xlabel("compute C (FLOPs)")
    allocation_ax.set_ylabel("optimal allocation")
    allocation_ax.set_title("Compute-optimal allocations")
    allocation_ax.legend(frameon=False, fontsize=8)

    ratio_law = scaling["parameter_token_ratio"]
    effective_ratio_law = scaling["effective_compute_parameter_token_ratio"]
    ratio_ax.scatter(compute, ratios, color="#5a9b55")
    ratio_ax.plot(
        compute_line,
        ratio_law["coefficient"] * compute_line ** ratio_law["exponent"],
        color="#5a9b55",
        label=normalized_power_law_label(r"$N_\mathrm{opt}/D_\mathrm{opt}$", ratio_law),
    )
    ratio_ax.scatter(compute, effective_ratios, color="#35618d", marker="s")
    ratio_ax.plot(
        compute_line,
        effective_ratio_law["coefficient"]
        * compute_line ** effective_ratio_law["exponent"],
        color="#35618d",
        ls="--",
        label=normalized_power_law_label(
            r"$N_\mathrm{eff,opt}/D_\mathrm{opt}$",
            effective_ratio_law,
        ),
    )
    ratio_ax.set_xscale("log")
    ratio_ax.set_yscale("log")
    ratio_ax.set_xlabel("compute C (FLOPs)")
    ratio_ax.set_ylabel("optimal model / token ratio")
    ratio_ax.set_title("Counted and compute-effective ratios")
    ratio_ax.legend(frameon=False, fontsize=8)

    loss_law = scaling["loss_min"]
    floor_loss_law = scaling["loss_min_with_floor"]
    loss_prediction = (
        loss_law["asymptote"]
        + loss_law["coefficient"] * compute_line ** loss_law["exponent"]
    )
    floor_loss_prediction = (
        floor_loss_law["asymptote"]
        + floor_loss_law["coefficient"]
        * compute_line ** floor_loss_law["exponent"]
    )
    loss_ax.scatter(compute, loss_min, color="#9a4f78")
    pure_line, = loss_ax.plot(compute_line, loss_prediction, color="#9a4f78")
    floor_line, = loss_ax.plot(
        compute_line,
        floor_loss_prediction,
        color="#4d4d4d",
        ls="--",
    )
    loss_ax.set_xscale("log")
    loss_ax.set_xlabel("compute C (FLOPs)")
    loss_ax.set_ylabel("minimum validation NELBO")
    loss_ax.set_title("Minimum-loss scaling with floor sensitivity")
    loss_at_reference = loss_law["coefficient"] * 1e14 ** loss_law["exponent"]
    pure_loss_label = (
        rf"pure: $L_\min={loss_at_reference:.3f}"
        rf"(C/10^{{14}})^{{{loss_law['exponent']:.3f}}}$"
    )
    floor_loss_at_reference = (
        floor_loss_law["coefficient"] * 1e14 ** floor_loss_law["exponent"]
    )
    floor_loss_label = (
        rf"floor sensitivity: $L_\min={floor_loss_law['asymptote']:.3f}"
        rf"+{floor_loss_at_reference:.3f}"
        rf"(C/10^{{14}})^{{{floor_loss_law['exponent']:.3f}}}$"
    )
    loss_ax.legend(
        handles=[pure_line, floor_line],
        labels=[pure_loss_label, floor_loss_label],
        frameon=False,
        fontsize=7.5,
    )

    fig.suptitle(
        f"Block diffusion scaling on TinyShakespeare — {objective} IsoFLOP fits",
        fontsize=15,
    )
    fig.tight_layout()
    png = FIGURES_DIR / f"{filename_stem}.png"
    pdf = FIGURES_DIR / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def scaling_from_fits(fits):
    compute = np.array([fit["budget"] for fit in fits])
    n_opt = np.array([fit["n_opt"] for fit in fits])
    n_effective = np.array(
        [fit["effective_compute_parameters_opt"] for fit in fits]
    )
    d_opt = np.array([fit["d_opt"] for fit in fits])
    ratio = np.array([fit["parameter_token_ratio"] for fit in fits])
    losses = np.array([fit["loss_min"] for fit in fits])
    return {
        "n_opt": fit_power_law(compute, n_opt),
        "effective_compute_parameters_opt": fit_power_law(
            compute,
            n_effective,
        ),
        "d_opt": fit_power_law(compute, d_opt),
        "effective_compute_parameter_token_ratio": fit_power_law(
            compute,
            n_effective / d_opt,
        ),
        "parameter_token_ratio": fit_power_law(compute, ratio),
        "tokens_per_parameter": fit_power_law(compute, 1 / ratio),
        "loss_min": fit_loss_law(compute, losses),
        "loss_min_with_floor": fit_loss_law_with_floor(compute, losses),
    }


def make_fit_comparison(best_rows, l1_fits, l2_fits, l1_scaling, l2_scaling):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
    flat_axes = axes.flat
    for index, (budget, color) in enumerate(zip(COMPUTE_BUDGETS, colors)):
        ax = flat_axes[index]
        points = [row for row in best_rows if float(row["budget"]) == budget]
        x = np.array([row["n_params"] for row in points])
        y = np.array([row["val_nelbo"] for row in points])
        l1_fit = l1_fits[index]
        l2_fit = l2_fits[index]
        curve_log_n = np.linspace(l1_fit["x_min"], l1_fit["x_max"], 300)
        ax.scatter(x, y, color=color, s=30, zorder=3, label="measured")
        ax.plot(
            10**curve_log_n,
            np.polyval(l2_fit["coefficients"], curve_log_n),
            color="#777777",
            ls="--",
            lw=2,
            label=r"$\ell_2$ fit",
        )
        ax.plot(
            10**curve_log_n,
            np.polyval(l1_fit["coefficients"], curve_log_n),
            color=color,
            lw=2.3,
            label=r"$\ell_1$ fit",
        )
        ax.scatter(
            l2_fit["n_opt"],
            l2_fit["loss_min"],
            marker="X",
            s=55,
            color="#777777",
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        ax.scatter(
            l1_fit["n_opt"],
            l1_fit["loss_min"],
            marker="*",
            s=115,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
        largest_residual = int(np.argmax(np.abs(l1_fit["residuals"])))
        ax.scatter(
            x[largest_residual],
            y[largest_residual],
            s=90,
            facecolors="none",
            edgecolors="#c23b3b",
            linewidth=1.3,
            zorder=6,
        )
        ax.set_xscale("log")
        if len(x) <= 4:
            tick_indices = list(range(len(x)))
        else:
            tick_indices = sorted(set([0, 2, 4, len(x) - 1]))
        ax.set_xticks(
            x[tick_indices],
            [points[tick]["size"] for tick in tick_indices],
            rotation=35,
            ha="right",
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"C={budget:.0e}")
        ax.set_xlabel("parameters N")
        ax.set_ylabel("validation NELBO")

    info_ax = flat_axes[-1]
    info_ax.axis("off")
    shifts = [
        100 * (l1_fit["n_opt"] / l2_fit["n_opt"] - 1)
        for l1_fit, l2_fit in zip(l1_fits, l2_fits)
    ]
    info_ax.legend(
        handles=[
            Line2D([], [], color="#777777", ls="--", lw=2, label=r"$\ell_2$ fit"),
            Line2D([], [], color="#333333", lw=2.3, label=r"$\ell_1$ fit"),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#c23b3b",
                ls="",
                label="largest L1 residual",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    info_ax.text(
        0.02,
        0.62,
        "Effect on fitted allocation\n"
        f"N exponent: {l2_scaling['n_opt']['exponent']:.3f} → "
        f"{l1_scaling['n_opt']['exponent']:.3f}\n"
        f"D exponent: {l2_scaling['d_opt']['exponent']:.3f} → "
        f"{l1_scaling['d_opt']['exponent']:.3f}\n"
        f"N/D exponent: {l2_scaling['parameter_token_ratio']['exponent']:.3f} → "
        f"{l1_scaling['parameter_token_ratio']['exponent']:.3f}\n\n"
        "Change in N optimum\n"
        + "\n".join(
            f"{budget:.0e}: {shift:+.1f}%"
            for budget, shift in zip(COMPUTE_BUDGETS, shifts)
        ),
        transform=info_ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.35,
    )
    fig.suptitle(r"IsoFLOP quadratic sensitivity: $\ell_1$ versus $\ell_2$", fontsize=15)
    fig.tight_layout()
    png = FIGURES_DIR / "isoflop_fit_comparison.png"
    pdf = FIGURES_DIR / "isoflop_fit_comparison.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_compute_accounting_comparison(corrected_fits, corrected_scaling):
    legacy_summary_path = ROOT / "results" / "summary.json"
    if not legacy_summary_path.exists():
        return None
    legacy = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    legacy_fits = legacy["parabolas"]
    legacy_scaling = legacy["scaling_laws"]
    compute = np.array([fit["budget"] for fit in corrected_fits])
    compute_line = np.geomspace(compute.min(), compute.max(), 200)

    panels = [
        ("n_opt", "compute-optimal parameters N", r"$N_\mathrm{opt}$"),
        (
            "effective_compute_parameters_opt",
            "compute-effective model allocation",
            r"$N_\mathrm{eff,opt}$",
        ),
        ("d_opt", "compute-optimal clean tokens D", r"$D_\mathrm{opt}$"),
        ("tokens_per_parameter", "optimal tokens per parameter D / N", r"$D_\mathrm{opt}/N_\mathrm{opt}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, symbol) in zip(axes.flat, panels):
        if key == "tokens_per_parameter":
            corrected_values = np.array([fit["tokens_per_parameter"] for fit in corrected_fits])
            legacy_values = np.array([fit["tokens_per_parameter"] for fit in legacy_fits])
        elif key == "effective_compute_parameters_opt":
            corrected_values = np.array(
                [fit["effective_compute_parameters_opt"] for fit in corrected_fits]
            )
            legacy_values = np.array([fit["n_opt"] for fit in legacy_fits])
        else:
            corrected_values = np.array([fit[key] for fit in corrected_fits])
            legacy_values = np.array([fit[key] for fit in legacy_fits])
        corrected_law = corrected_scaling[key]
        legacy_law = (
            legacy_scaling["n_opt"]
            if key == "effective_compute_parameters_opt"
            else legacy_scaling[key]
        )
        ax.scatter(compute, legacy_values, color="#888888", marker="s", s=34)
        ax.plot(
            compute_line,
            legacy_law["coefficient"] * compute_line ** legacy_law["exponent"],
            color="#888888",
            ls="--",
            lw=2,
            label=rf"proxy $12ND$: exponent {legacy_law['exponent']:.3f}",
        )
        ax.scatter(compute, corrected_values, color="#2a788e", s=38)
        ax.plot(
            compute_line,
            corrected_law["coefficient"] * compute_line ** corrected_law["exponent"],
            color="#2a788e",
            lw=2.3,
            label=rf"dense attention: exponent {corrected_law['exponent']:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("compute C (FLOPs)")
        ax.set_ylabel(symbol)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Effect of compute accounting on fitted allocation", fontsize=14)
    fig.tight_layout()
    png = FIGURES_DIR / "compute_accounting_comparison.png"
    pdf = FIGURES_DIR / "compute_accounting_comparison.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


TRAIN_LOG_PATTERN = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)"
)


def load_training_trace(row):
    result_path = RESULTS_DIR.parent / row["result_path"]
    log_path = result_path.parent / "train.log"
    matches = TRAIN_LOG_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        raise RuntimeError(f"No training trace found in {log_path}")
    return [
        {
            "progress": int(step) / int(total),
            "loss": float(loss),
            "lr": float(lr),
        }
        for step, total, loss, lr in matches
    ]


def training_diagnostics(best_rows):
    stable_changes = []
    decay_changes = []
    for row in best_rows:
        trace = load_training_trace(row)
        early = [point["loss"] for point in trace if 0.05 <= point["progress"] < 0.45]
        late = [point["loss"] for point in trace if 0.45 <= point["progress"] < 0.85]
        if not early or not late:
            raise RuntimeError(f"Insufficient stable-phase trace for C={row['budget']}, N={row['size']}")
        early_mean = float(np.mean(early))
        late_mean = float(np.mean(late))
        stable_changes.append(100 * (early_mean - late_mean) / early_mean)
        pre_decay = [point["loss"] for point in trace if point["progress"] < 0.85][-1]
        final = trace[-1]["loss"]
        decay_changes.append(100 * (pre_decay - final) / pre_decay)
    return {
        "selected_runs": len(best_rows),
        "stable_second_half_lower_count": sum(change > 0 for change in stable_changes),
        "median_stable_loss_drop_percent": float(np.median(stable_changes)),
        "min_stable_loss_drop_percent": float(np.min(stable_changes)),
        "decay_lower_count": sum(change > 0 for change in decay_changes),
        "median_decay_loss_drop_percent": float(np.median(decay_changes)),
        "min_decay_loss_drop_percent": float(np.min(decay_changes)),
    }


def make_diagnostic_figure(best_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    budget_colors = dict(zip(COMPUTE_BUDGETS, colors))
    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.05), hspace=0.38, wspace=0.28)
    lr_ax = fig.add_subplot(grid[0, 0])
    bracket_ax = fig.add_subplot(grid[0, 1])
    trace_ax = fig.add_subplot(grid[1, :])

    labels = [spec.label for spec in MODEL_SPECS]
    selected = np.full((len(COMPUTE_BUDGETS), len(labels)), np.nan)
    for row in best_rows:
        budget_index = COMPUTE_BUDGETS.index(float(row["budget"]))
        size_index = labels.index(row["size"])
        selected[budget_index, size_index] = math.log10(row["learning_rate"])
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#eeeeee")
    finite_lrs = selected[np.isfinite(selected)]
    lr_ax.imshow(
        selected,
        aspect="auto",
        cmap=cmap,
        vmin=float(np.min(finite_lrs)),
        vmax=float(np.max(finite_lrs)),
    )
    lr_ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    lr_ax.set_yticks(
        range(len(COMPUTE_BUDGETS)),
        [f"{budget:.0e}" for budget in COMPUTE_BUDGETS],
    )
    lr_ax.set_xlabel("model-size label")
    lr_ax.set_ylabel("compute C (FLOPs)")
    lr_ax.set_title("Selected peak learning rate")
    for budget_index in range(len(COMPUTE_BUDGETS)):
        for size_index in range(len(labels)):
            if np.isfinite(selected[budget_index, size_index]):
                value = 10 ** selected[budget_index, size_index]
                color = "white" if selected[budget_index, size_index] < -2.15 else "black"
                lr_ax.text(
                    size_index,
                    budget_index,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    bracket_values = []
    for row in best_rows:
        bracket_path = (
            RESULTS_DIR
            / "runs"
            / budget_slug(float(row["budget"]))
            / row["size"]
            / "lr_bracket.json"
        )
        bracket = json.loads(bracket_path.read_text(encoding="utf-8"))
        delta = np.array(
            [
                bracket["left_val_nelbo"] - bracket["selected_val_nelbo"],
                0.0,
                bracket["right_val_nelbo"] - bracket["selected_val_nelbo"],
            ]
        )
        bracket_values.append(delta)
        bracket_ax.plot(
            [0, 1, 2],
            delta + 1e-3,
            color=budget_colors[float(row["budget"])],
            alpha=0.3,
            lw=1.2,
        )
    median_delta = np.median(np.stack(bracket_values), axis=0)
    bracket_ax.plot(
        [0, 1, 2],
        median_delta + 1e-3,
        color="black",
        marker="o",
        lw=2.5,
        label="median",
    )
    bracket_ax.set_yscale("log")
    bracket_ax.set_xticks([0, 1, 2], ["selected / 3", "selected", "selected × 3"])
    bracket_ax.set_xlabel("peak LR relative to selected LR")
    bracket_ax.set_ylabel(r"$\Delta$ validation NELBO + $10^{-3}$")
    bracket_ax.set_title(f"Local LR neighborhoods (all {len(best_rows)} points)")
    bracket_ax.legend(frameon=False)

    for row in best_rows:
        trace = [point for point in load_training_trace(row) if point["progress"] >= 0.05]
        baseline = trace[0]["loss"]
        progress = [100 * point["progress"] for point in trace]
        relative_loss = [100 * (point["loss"] / baseline - 1) for point in trace]
        trace_ax.plot(
            progress,
            relative_loss,
            color=budget_colors[float(row["budget"])],
            alpha=0.3,
            lw=1.2,
        )
    for budget in COMPUTE_BUDGETS:
        trace_ax.plot([], [], color=budget_colors[budget], label=f"C={budget:.0e}")
    trace_ax.axvline(5, color="#777777", ls="--", lw=1)
    trace_ax.axvline(85, color="#777777", ls="--", lw=1)
    trace_ax.axhline(0, color="#777777", lw=0.8)
    trace_ax.text(5.8, trace_ax.get_ylim()[1] * 0.9, "warmup ends", color="#666666", fontsize=9)
    trace_ax.text(85.8, trace_ax.get_ylim()[1] * 0.9, "decay starts", color="#666666", fontsize=9)
    trace_ax.set_xlim(4, 101)
    trace_ax.set_xlabel("training progress (%)")
    trace_ax.set_ylabel("relative train-loss change (%)")
    trace_ax.set_title("Selected-run WSD training traces (logged about every 10%)")
    trace_ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower left")

    fig.suptitle("Optimization diagnostics", fontsize=15)
    png = FIGURES_DIR / "optimization_diagnostics.png"
    pdf = FIGURES_DIR / "optimization_diagnostics.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    rows = load_results()
    best_rows = select_best(rows)
    expected_points = sum(
        1
        for budget in COMPUTE_BUDGETS
        for spec in MODEL_SPECS
        if is_feasible(budget, spec)
    )
    if len(best_rows) != expected_points:
        raise RuntimeError(f"Expected {expected_points} completed (C,N) points, found {len(best_rows)}")
    validate_lr_brackets(best_rows)
    l1_fits = fit_parabolas(best_rows, objective="l1")
    l2_fits = fit_parabolas(best_rows, objective="l2")
    lowest3_fits = fit_parabolas(best_rows, objective="lowest3_l2")
    l1_scaling = scaling_from_fits(l1_fits)
    l2_scaling = scaling_from_fits(l2_fits)
    lowest3_scaling = scaling_from_fits(lowest3_fits)
    diagnostics = training_diagnostics(best_rows)
    summary = {
        "fit_objective": "l1",
        "parabolas": l1_fits,
        "scaling_laws": l1_scaling,
        "training_diagnostics": diagnostics,
        "l2_comparison": {
            "parabolas": l2_fits,
            "scaling_laws": l2_scaling,
        },
        "lowest3_l2_comparison": {
            "parabolas": lowest3_fits,
            "scaling_laws": lowest3_scaling,
        },
    }
    l2_summary = {
        "fit_objective": "l2",
        "parabolas": l2_fits,
        "scaling_laws": l2_scaling,
        "training_diagnostics": diagnostics,
    }
    lowest3_summary = {
        "fit_objective": "lowest3_l2",
        "parabolas": lowest3_fits,
        "scaling_laws": lowest3_scaling,
        "training_diagnostics": diagnostics,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    l2_summary_path = RESULTS_DIR / "summary_l2.json"
    l2_summary_path.write_text(
        json.dumps(l2_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lowest3_summary_path = RESULTS_DIR / "summary_lowest3_l2.json"
    lowest3_summary_path.write_text(
        json.dumps(lowest3_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = save_best_csv(best_rows)
    optimal_csv_path = save_optimal_csv(l1_fits, "optimal_allocation.csv")
    l2_optimal_csv_path = save_optimal_csv(l2_fits, "optimal_allocation_l2.csv")
    lowest3_optimal_csv_path = save_optimal_csv(
        lowest3_fits,
        "optimal_allocation_lowest3_l2.csv",
    )
    png, pdf = make_figure(best_rows, l1_fits, l1_scaling, "isoflop_scaling")
    l2_png, l2_pdf = make_figure(
        best_rows,
        l2_fits,
        l2_scaling,
        "isoflop_scaling_l2",
    )
    lowest3_png, lowest3_pdf = make_figure(
        best_rows,
        lowest3_fits,
        lowest3_scaling,
        "isoflop_scaling_lowest3_l2",
    )
    comparison_png, comparison_pdf = make_fit_comparison(
        best_rows,
        l1_fits,
        l2_fits,
        l1_scaling,
        l2_scaling,
    )
    accounting_comparison = make_compute_accounting_comparison(
        l1_fits,
        l1_scaling,
    )
    diagnostics_png, diagnostics_pdf = make_diagnostic_figure(best_rows)
    print(f"saved {summary_path}")
    print(f"saved {l2_summary_path}")
    print(f"saved {lowest3_summary_path}")
    print(f"saved {csv_path}")
    print(f"saved {optimal_csv_path}")
    print(f"saved {l2_optimal_csv_path}")
    print(f"saved {lowest3_optimal_csv_path}")
    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"saved {l2_png}")
    print(f"saved {l2_pdf}")
    print(f"saved {lowest3_png}")
    print(f"saved {lowest3_pdf}")
    print(f"saved {comparison_png}")
    print(f"saved {comparison_pdf}")
    if accounting_comparison is not None:
        print(f"saved {accounting_comparison[0]}")
        print(f"saved {accounting_comparison[1]}")
    print(f"saved {diagnostics_png}")
    print(f"saved {diagnostics_pdf}")


if __name__ == "__main__":
    main()
