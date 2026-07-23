"""Select locally bracketed LRs, fit IsoFLOP laws, and draw final figures."""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from config import (
    COMPUTE_BUDGETS,
    FIGURES_DIR,
    MODEL_SPECS,
    RESULTS_DIR,
    budget_slug,
    is_feasible,
)


def fit_l1_quadratic(x: np.ndarray, y: np.ndarray):
    """Exact LAD solution by enumerating three-point support sets."""
    if len(x) < 3:
        raise ValueError("A quadratic requires at least three points")
    candidates = []
    for support in combinations(range(len(x)), 3):
        design = np.column_stack((x[list(support)] ** 2, x[list(support)], np.ones(3)))
        if abs(np.linalg.det(design)) < 1e-12:
            continue
        coefficients = np.linalg.solve(design, y[list(support)])
        residual = np.abs(y - np.polyval(coefficients, x))
        candidates.append((float(residual.sum()), coefficients, support))
    if not candidates:
        raise ValueError("Could not fit a quadratic")
    _, coefficients, support = min(candidates, key=lambda row: row[0])
    return coefficients, support


def fit_power_law(x, y):
    log_x, log_y = np.log(np.asarray(x)), np.log(np.asarray(y))
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    prediction = intercept + exponent * log_x
    residual = np.square(log_y - prediction).sum()
    centered = np.square(log_y - log_y.mean()).sum()
    return {
        "coefficient": float(np.exp(intercept)),
        "exponent": float(exponent),
        "log_r_squared": float(1 - residual / centered),
    }


def fit_loss_with_floor(compute, loss):
    compute = np.asarray(compute, dtype=float)
    loss = np.asarray(loss, dtype=float)
    reference = 1e15
    best = None
    lower, upper = 0.0, float(loss.min() - 1e-8)
    for _ in range(5):
        for floor in np.linspace(lower, upper, 2001):
            law = fit_power_law(compute / reference, loss - floor)
            prediction = floor + law["coefficient"] * (compute / reference) ** law["exponent"]
            squared_error = float(np.square(loss - prediction).sum())
            if best is None or squared_error < best[0]:
                best = (squared_error, float(floor), law)
        span = (upper - lower) / 2000
        lower = max(0.0, best[1] - 2 * span)
        upper = min(float(loss.min() - 1e-8), best[1] + 2 * span)
    squared_error, floor, law = best
    return {
        "asymptote": floor,
        "coefficient_at_reference": law["coefficient"],
        "reference_compute": reference,
        "exponent": law["exponent"],
        "rmse": math.sqrt(squared_error / len(loss)),
    }


def load_best_runs():
    rows = []
    for budget in COMPUTE_BUDGETS:
        for spec in MODEL_SPECS:
            if not is_feasible(budget, spec):
                continue
            point = RESULTS_DIR / "runs" / budget_slug(budget) / spec.label
            bracket_path = point / "lr_bracket.json"
            if not bracket_path.exists():
                raise FileNotFoundError(f"Missing local LR bracket: {bracket_path}")
            bracket = json.loads(bracket_path.read_text())
            if bracket["discrete_curvature"] <= 0:
                raise ValueError(f"Non-convex LR bracket: {bracket_path}")
            candidates = []
            for path in point.glob("lr_*/result.json"):
                row = json.loads(path.read_text())
                if row.get("status") == "complete":
                    row["result_path"] = str(path)
                    candidates.append(row)
            best = min(candidates, key=lambda row: row["val_nelbo"])
            if not math.isclose(best["learning_rate"], bracket["selected_lr"]):
                raise ValueError(f"Stale LR bracket: {bracket_path}")
            rows.append(best)
    return rows


def interpolate_flops(n_params):
    x = np.log([spec.n_params for spec in MODEL_SPECS])
    y = np.log([spec.training_flops_per_clean_token for spec in MODEL_SPECS])
    return float(np.exp(np.interp(np.log(n_params), x, y)))


def fit_profiles(rows):
    """Fit an L2 quadratic through the three lowest-loss points per profile."""
    fits = []
    for budget in COMPUTE_BUDGETS:
        points = sorted(
            [row for row in rows if row["budget"] == budget],
            key=lambda row: row["n_params"],
        )
        x = np.log10([row["n_params"] for row in points])
        y = np.array([row["val_nelbo"] for row in points])
        support = tuple(sorted(np.argsort(y)[:3].tolist()))
        fit_x = x[list(support)]
        fit_y = y[list(support)]
        coefficients = np.polyfit(fit_x, fit_y, 2)
        a, b, _ = coefficients
        if a <= 0:
            raise ValueError(f"Non-convex IsoFLOP profile at C={budget:.0e}")
        vertex = -b / (2 * a)
        outside = not x.min() <= vertex <= x.max()
        if outside:
            raise ValueError(f"IsoFLOP vertex outside measured range at C={budget:.0e}")
        if not fit_x.min() <= vertex <= fit_x.max():
            raise ValueError(
                f"IsoFLOP vertex outside lowest-3 support at C={budget:.0e}"
            )
        n_opt = 10**vertex
        flops_per_token = interpolate_flops(n_opt)
        d_opt = budget / flops_per_token
        prediction = np.polyval(coefficients, x)
        absolute = np.abs(y - prediction)
        centered = np.abs(y - np.median(y)).sum()
        fits.append(
            {
                "budget": budget,
                "n_points": len(points),
                "n_fit_points": len(support),
                "fit_method": "lowest3_l2_quadratic_in_log10_n",
                "coefficients": coefficients.tolist(),
                "support_indices": list(support),
                "x_min": float(x.min()),
                "x_max": float(x.max()),
                "fit_x_min": float(fit_x.min()),
                "fit_x_max": float(fit_x.max()),
                "n_opt": n_opt,
                "effective_n_opt": flops_per_token / 12,
                "d_opt": d_opt,
                "tokens_per_parameter": d_opt / n_opt,
                "parameter_token_ratio": n_opt / d_opt,
                "loss_min": float(np.polyval(coefficients, vertex)),
                "mean_absolute_error": float(absolute.mean()),
                "l1_pseudo_r_squared": float(1 - absolute.sum() / centered)
                if centered
                else 1.0,
            }
        )
    return fits


def fit_sensitivity_profiles(rows, method):
    if method not in {"full_l1", "full_l2", "lowest3_l2"}:
        raise ValueError(method)
    fits = []
    for budget in COMPUTE_BUDGETS:
        points = sorted(
            [row for row in rows if row["budget"] == budget],
            key=lambda row: row["n_params"],
        )
        fit_points = (
            sorted(
                sorted(points, key=lambda row: row["val_nelbo"])[:3],
                key=lambda row: row["n_params"],
            )
            if method == "lowest3_l2"
            else points
        )
        x = np.log10([row["n_params"] for row in fit_points])
        y = np.array([row["val_nelbo"] for row in fit_points])
        coefficients = (
            fit_l1_quadratic(x, y)[0]
            if method == "full_l1"
            else np.polyfit(x, y, 2)
        )
        a, b, _ = coefficients
        vertex = -b / (2 * a)
        n_opt = 10**vertex
        flops_per_token = interpolate_flops(n_opt)
        fits.append(
            {
                "budget": budget,
                "method": method,
                "n_fit_points": len(fit_points),
                "n_opt": n_opt,
                "effective_n_opt": flops_per_token / 12,
                "d_opt": budget / flops_per_token,
                "loss_min": float(np.polyval(coefficients, vertex)),
                "vertex_outside_fit_range": not float(x.min()) <= vertex <= float(x.max()),
            }
        )
    return fits


def scaling_laws(fits):
    compute = [row["budget"] for row in fits]
    keys = (
        "n_opt",
        "effective_n_opt",
        "d_opt",
        "parameter_token_ratio",
        "loss_min",
    )
    laws = {key: fit_power_law(compute, [row[key] for row in fits]) for key in keys}
    laws["loss_min_with_floor"] = fit_loss_with_floor(
        compute,
        [row["loss_min"] for row in fits],
    )
    return laws


def write_csv(path, rows):
    fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"coefficients", "support_indices", "train_trace"}
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_label(symbol, law, reference=1e15):
    value = law["coefficient"] * reference ** law["exponent"]
    return rf"${symbol}={value:.2g}(C/10^{{15}})^{{{law['exponent']:.3f}}}$"


def clean_model_ticks(axis):
    axis.set_xticks([1.4e5, 3e5, 1e6, 3e6, 8e6])
    axis.set_xticklabels(["0.14M", "0.3M", "1M", "3M", "8M"])
    axis.xaxis.set_minor_formatter(NullFormatter())


def make_figure(rows, fits, laws):
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
    iso, allocation, ratio, loss_ax = axes.flat

    for color, fit in zip(colors, fits):
        points = sorted(
            [row for row in rows if row["budget"] == fit["budget"]],
            key=lambda row: row["n_params"],
        )
        n = np.array([row["n_params"] for row in points])
        loss = np.array([row["val_nelbo"] for row in points])
        support = np.array(fit["support_indices"], dtype=int)
        grid = np.linspace(fit["fit_x_min"], fit["fit_x_max"], 200)
        iso.scatter(n, loss, color=color, s=28, alpha=0.35, zorder=2)
        iso.scatter(
            n[support],
            loss[support],
            color=color,
            edgecolor="black",
            linewidth=0.45,
            s=42,
            zorder=3,
        )
        iso.plot(
            10**grid,
            np.polyval(fit["coefficients"], grid),
            color=color,
            lw=2,
            label=rf"$C={fit['budget']:.0e}$",
        )
        iso.scatter(
            fit["n_opt"],
            fit["loss_min"],
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=130,
            zorder=4,
        )
    iso.set_xscale("log")
    clean_model_ticks(iso)
    iso.set_xlabel("non-embedding parameters $N$")
    iso.set_ylabel("validation diffusion NELBO (nats/token)")
    iso.set_title("IsoFLOP profiles (local L2 quadratic on lowest 3)")
    iso.legend(frameon=False, fontsize=8)

    compute = np.array([row["budget"] for row in fits])
    compute_line = np.geomspace(compute.min(), compute.max(), 200)
    for key, color, marker, symbol in (
        ("n_opt", "#35618d", "o", r"N_{\rm opt}"),
        ("effective_n_opt", "#738da7", "s", r"N_{\rm eff,opt}"),
        ("d_opt", "#d17c37", "o", r"D_{\rm opt}"),
    ):
        values = np.array([row[key] for row in fits])
        law = laws[key]
        allocation.scatter(compute, values, color=color, marker=marker)
        allocation.plot(
            compute_line,
            law["coefficient"] * compute_line ** law["exponent"],
            color=color,
            ls="--" if key == "effective_n_opt" else "-",
            label=normalized_label(symbol, law),
        )
    allocation.set_xscale("log")
    allocation.set_yscale("log")
    allocation.set_xlabel("training compute $C$ (FLOPs)")
    allocation.set_ylabel("parameters or clean tokens")
    allocation.set_title("Compute-optimal allocation")
    allocation.legend(frameon=False, fontsize=8)

    tokens_per_parameter = np.array([row["tokens_per_parameter"] for row in fits])
    ratio.scatter(compute, tokens_per_parameter, color="#5a9b55")
    parameter_ratio_law = laws["parameter_token_ratio"]
    inverse_prediction = 1 / (
        parameter_ratio_law["coefficient"]
        * compute_line ** parameter_ratio_law["exponent"]
    )
    ratio.plot(compute_line, inverse_prediction, color="#5a9b55")
    ratio.set_xscale("log")
    ratio.set_yscale("log")
    ratio.set_xlabel("training compute $C$ (FLOPs)")
    ratio.set_ylabel(r"optimal clean tokens per parameter $D/N$")
    ratio.set_title(
        rf"Optimal data ratio ($D/N\propto C^{{{-parameter_ratio_law['exponent']:.3f}}}$)"
    )

    minima = np.array([row["loss_min"] for row in fits])
    pure = laws["loss_min"]
    floor = laws["loss_min_with_floor"]
    loss_ax.scatter(compute, minima, color="#9a4f78", zorder=3)
    loss_ax.plot(
        compute_line,
        pure["coefficient"] * compute_line ** pure["exponent"],
        color="#9a4f78",
        label=normalized_label(r"L_{\min}", pure),
    )
    loss_ax.plot(
        compute_line,
        floor["asymptote"]
        + floor["coefficient_at_reference"]
        * (compute_line / floor["reference_compute"]) ** floor["exponent"],
        color="#9a4f78",
        ls="--",
        label=(
            rf"$L_\infty={floor['asymptote']:.3f}$ sensitivity fit"
        ),
    )
    loss_ax.set_xscale("log")
    loss_ax.set_xlabel("training compute $C$ (FLOPs)")
    loss_ax.set_ylabel("minimum validation NELBO (nats/token)")
    loss_ax.set_title("Compute-optimal loss")
    loss_ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Block diffusion scaling on ClimbMix", fontsize=14, y=0.99)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"isoflop_scaling.{extension}", bbox_inches="tight")
    plt.close(fig)


def trace_diagnostics(rows):
    diagnostics = []
    for row in rows:
        trace = row["train_trace"]
        total = row["steps"]
        early = [
            point["train_nelbo"]
            for point in trace
            if 0.10 <= point["step"] / total <= 0.40
        ]
        late = [
            point["train_nelbo"]
            for point in trace
            if 0.50 <= point["step"] / total <= 0.80
        ]
        decay = [
            point["train_nelbo"]
            for point in trace
            if point["step"] / total >= 0.90
        ]
        diagnostics.append(
            {
                "budget": row["budget"],
                "size": row["size"],
                "learning_rate": row["learning_rate"],
                "early_stable_mean": float(np.mean(early)),
                "late_stable_mean": float(np.mean(late)),
                "decay_mean": float(np.mean(decay)),
                "stable_relative_drop": float(
                    (np.mean(early) - np.mean(late)) / np.mean(early)
                ),
                "decay_relative_drop": float(
                    (np.mean(late) - np.mean(decay)) / np.mean(late)
                ),
            }
        )
    return diagnostics


def make_optimization_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    lr_axis, trace_axis = axes
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    for color, budget in zip(colors, COMPUTE_BUDGETS):
        points = sorted(
            [row for row in rows if row["budget"] == budget],
            key=lambda row: row["n_params"],
        )
        lr_axis.plot(
            [row["n_params"] for row in points],
            [row["learning_rate"] for row in points],
            marker="o",
            color=color,
            label=rf"$C={budget:.0e}$",
        )
        for row in points:
            progress = np.array(
                [entry["step"] / row["steps"] for entry in row["train_trace"]]
            )
            loss = np.array(
                [entry["train_nelbo"] for entry in row["train_trace"]]
            )
            trace_axis.plot(
                progress,
                loss / loss[0],
                color=color,
                alpha=0.35,
                lw=1,
            )
    lr_axis.set_xscale("log")
    clean_model_ticks(lr_axis)
    lr_axis.set_yscale("log")
    lr_axis.set_xlabel("non-embedding parameters $N$")
    lr_axis.set_ylabel("selected peak learning rate")
    lr_axis.set_title("Full-run locally bracketed LRs")
    lr_axis.grid(alpha=0.2)
    lr_axis.legend(frameon=False, fontsize=8)

    trace_axis.axvline(0.05, color="black", ls=":", lw=1)
    trace_axis.axvline(0.85, color="black", ls=":", lw=1)
    trace_axis.set_xlabel("fraction of optimizer steps")
    trace_axis.set_ylabel("train NELBO / initial NELBO")
    trace_axis.set_title("Selected WSD train traces")
    trace_axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES_DIR / f"optimization_diagnostics.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    rows = load_best_runs()
    fits = fit_profiles(rows)
    laws = scaling_laws(fits)
    sensitivities = {}
    for method in ("full_l1", "full_l2"):
        method_fits = fit_sensitivity_profiles(rows, method)
        sensitivities[method] = {
            "profiles": method_fits,
            "n_opt": fit_power_law(
                [row["budget"] for row in method_fits],
                [row["n_opt"] for row in method_fits],
            ),
            "effective_n_opt": fit_power_law(
                [row["budget"] for row in method_fits],
                [row["effective_n_opt"] for row in method_fits],
            ),
            "d_opt": fit_power_law(
                [row["budget"] for row in method_fits],
                [row["d_opt"] for row in method_fits],
            ),
        }
    diagnostics = trace_diagnostics(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "best_runs.csv", rows)
    write_csv(RESULTS_DIR / "optimal_allocation.csv", fits)
    summary = {
        "profile_fit": "L2 quadratic through the three lowest-loss points in log10(N)",
        "best_runs": len(rows),
        "profiles": fits,
        "scaling_laws": laws,
        "quadratic_sensitivities": sensitivities,
        "optimization_diagnostics": diagnostics,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_figure(rows, fits, laws)
    make_optimization_figure(rows)
    for row in fits:
        print(
            f"C={row['budget']:.0e} N*={row['n_opt'] / 1e6:.3f}M "
            f"D*={row['d_opt'] / 1e6:.1f}M D/N={row['tokens_per_parameter']:.1f} "
            f"L*={row['loss_min']:.4f}"
        )
    print(
        "laws: "
        f"N~C^{laws['n_opt']['exponent']:.3f}, "
        f"N_eff~C^{laws['effective_n_opt']['exponent']:.3f}, "
        f"D~C^{laws['d_opt']['exponent']:.3f}, "
        f"L~C^{laws['loss_min']['exponent']:.4f}"
    )


if __name__ == "__main__":
    main()
