"""Analyze the fixed-LR AR-to-block-diffusion curriculum sweep."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from analyze import fit_l1_quadratic, fit_power_law
from config import (
    COMPUTE_BUDGETS,
    DECAY_FRACTION,
    MODEL_SPECS,
    RESULTS_DIR as BASE_RESULTS_DIR,
    WARMUP_FRACTION,
    budget_slug,
    lr_slug,
)
from curriculum_config import (
    FIGURES_DIR,
    P_AR_VALUES,
    RESULTS_DIR,
    average_training_flops_per_clean_token,
    is_feasible,
    p_ar_slug,
)
from curriculum_run_sweep import baseline_learning_rate


ALL_P_AR_VALUES = (0.0,) + P_AR_VALUES


def result_path(p_ar, budget, spec) -> Path:
    lr = baseline_learning_rate(budget, spec)
    if p_ar == 0.0:
        return (
            BASE_RESULTS_DIR
            / "runs"
            / budget_slug(budget)
            / spec.label
            / f"lr_{lr_slug(lr)}"
            / "result.json"
        )
    return (
        RESULTS_DIR
        / "runs"
        / p_ar_slug(p_ar)
        / budget_slug(budget)
        / spec.label
        / f"lr_{lr_slug(lr)}"
        / "result.json"
    )


def load_required_rows():
    rows = []
    for p_ar in ALL_P_AR_VALUES:
        for budget in COMPUTE_BUDGETS:
            for spec in MODEL_SPECS:
                feasible = (
                    is_feasible(budget, spec, p_ar)
                    if p_ar > 0
                    else (
                        BASE_RESULTS_DIR
                        / "runs"
                        / budget_slug(budget)
                        / spec.label
                        / "lr_bracket.json"
                    ).exists()
                )
                if not feasible:
                    continue
                path = result_path(p_ar, budget, spec)
                if not path.exists():
                    raise RuntimeError(f"Missing required run: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") != "complete" or not math.isfinite(row["val_nelbo"]):
                    raise RuntimeError(f"Incomplete required run: {path}")
                expected_lr = baseline_learning_rate(budget, spec)
                if not math.isclose(float(row["learning_rate"]), expected_lr):
                    raise RuntimeError(f"Wrong inherited LR at {path}")
                row["p_ar"] = p_ar
                row["result_path"] = str(path.relative_to(RESULTS_DIR.parent))
                rows.append(row)
    return sorted(rows, key=lambda row: (row["p_ar"], row["budget"], row["n_params"]))


def interpolated_flops_per_clean_token(n_params, p_ar):
    log_n = np.log10([spec.n_params for spec in MODEL_SPECS])
    log_flops = np.log10(
        [
            average_training_flops_per_clean_token(spec, p_ar)
            for spec in MODEL_SPECS
        ]
    )
    target = math.log10(n_params)
    if not log_n.min() <= target <= log_n.max():
        raise ValueError(f"Cannot interpolate FLOPs outside model grid: N={n_params}")
    return float(10 ** np.interp(target, log_n, log_flops))


def fit_profiles(rows):
    fits = []
    for p_ar in ALL_P_AR_VALUES:
        for budget in COMPUTE_BUDGETS:
            points = [
                row
                for row in rows
                if row["p_ar"] == p_ar and float(row["budget"]) == budget
            ]
            if len(points) < 3:
                raise RuntimeError(
                    f"Need at least three N points at p={p_ar:.1f}, C={budget:.0e}"
                )
            x = np.log10([row["n_params"] for row in points])
            y = np.array([row["val_nelbo"] for row in points])
            coefficients, support = fit_l1_quadratic(x, y)
            curvature, linear, _ = coefficients
            if curvature <= 0:
                raise RuntimeError(
                    f"Non-convex L1 profile at p={p_ar:.1f}, C={budget:.0e}: "
                    f"a={curvature}"
                )
            log_n_raw = -linear / (2 * curvature)
            log_n_opt = float(np.clip(log_n_raw, x.min(), x.max()))
            n_opt = 10**log_n_opt
            loss_min = float(np.polyval(coefficients, log_n_opt))
            prediction = np.polyval(coefficients, x)
            flops_per_token = interpolated_flops_per_clean_token(n_opt, p_ar)
            d_opt = budget / flops_per_token
            fits.append(
                {
                    "p_ar": p_ar,
                    "budget": budget,
                    "coefficients": coefficients.tolist(),
                    "support_point_indices": support,
                    "n_points": len(points),
                    "n_opt": n_opt,
                    "d_opt": d_opt,
                    "effective_compute_parameters_opt": flops_per_token / 12,
                    "training_flops_per_clean_token_opt": flops_per_token,
                    "parameter_token_ratio": n_opt / d_opt,
                    "loss_min": loss_min,
                    "mean_absolute_error": float(np.abs(y - prediction).mean()),
                    "vertex_clipped": not bool(x.min() <= log_n_raw <= x.max()),
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                }
            )
    return fits


def scaling_laws(fits):
    laws = {}
    for p_ar in ALL_P_AR_VALUES:
        group = [fit for fit in fits if fit["p_ar"] == p_ar]
        compute = np.array([fit["budget"] for fit in group])
        n_opt = np.array([fit["n_opt"] for fit in group])
        n_eff = np.array([fit["effective_compute_parameters_opt"] for fit in group])
        d_opt = np.array([fit["d_opt"] for fit in group])
        loss_min = np.array([fit["loss_min"] for fit in group])
        laws[f"{p_ar:.1f}"] = {
            "n_opt": fit_power_law(compute, n_opt),
            "effective_compute_parameters_opt": fit_power_law(compute, n_eff),
            "d_opt": fit_power_law(compute, d_opt),
            "parameter_token_ratio": fit_power_law(compute, n_opt / d_opt),
            "loss_min": fit_power_law(compute, loss_min),
        }
    return laws


def sweep_summary(rows, fits):
    best_by_budget = []
    for budget in COMPUTE_BUDGETS:
        candidates = [fit for fit in fits if fit["budget"] == budget]
        control = next(fit for fit in candidates if fit["p_ar"] == 0)
        best = min(candidates, key=lambda fit: fit["loss_min"])
        best_by_budget.append(
            {
                "budget": budget,
                "best_p_ar": best["p_ar"],
                "best_loss_min": best["loss_min"],
                "pure_bd_loss_min": control["loss_min"],
                "loss_improvement": control["loss_min"] - best["loss_min"],
                "best_n_opt": best["n_opt"],
                "best_d_opt": best["d_opt"],
            }
        )

    control_rows = {
        (float(row["budget"]), row["size"]): row
        for row in rows
        if row["p_ar"] == 0
    }
    curriculum_rows = {}
    for row in rows:
        if row["p_ar"] == 0:
            continue
        key = (float(row["budget"]), row["size"])
        curriculum_rows.setdefault(key, []).append(row)
    pointwise = []
    for key, control in control_rows.items():
        candidates = curriculum_rows.get(key, [])
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["val_nelbo"])
        pointwise.append(
            {
                "budget": key[0],
                "size": key[1],
                "n_params": control["n_params"],
                "best_p_ar": best["p_ar"],
                "pure_bd_val_nelbo": control["val_nelbo"],
                "best_curriculum_val_nelbo": best["val_nelbo"],
                "loss_improvement": control["val_nelbo"] - best["val_nelbo"],
            }
        )

    curriculum_only = [row for row in rows if row["p_ar"] > 0]
    warmup_records = []
    for row in curriculum_only:
        for phase in ("ar", "bd"):
            phase_steps = int(row[f"{phase}_steps"])
            warmup_steps = max(1, round(WARMUP_FRACTION * phase_steps))
            warmup_records.append(
                {
                    "phase": phase,
                    "phase_steps": phase_steps,
                    "warmup_steps": warmup_steps,
                    "fraction": warmup_steps / phase_steps,
                }
            )
    phase_diagnostics = {}
    for phase in ("ar", "bd"):
        stable_changes = []
        decay_changes = []
        for row in curriculum_only:
            trace = [
                point for point in row["train_trace"] if point["phase"] == phase
            ]
            phase_steps = row[f"{phase}_steps"]
            early_stable = [
                point["loss"]
                for point in trace
                if 0.05 <= point["phase_step"] / phase_steps < 0.45
            ]
            late_stable = [
                point["loss"]
                for point in trace
                if 0.45 <= point["phase_step"] / phase_steps < 0.85
            ]
            pre_decay = [
                point["loss"]
                for point in trace
                if 0.75 <= point["phase_step"] / phase_steps < 0.85
            ]
            final_decay = [
                point["loss"]
                for point in trace
                if 0.95 <= point["phase_step"] / phase_steps <= 1.0
            ]
            early_mean = float(np.mean(early_stable))
            late_mean = float(np.mean(late_stable))
            pre_decay_mean = float(np.mean(pre_decay))
            final_decay_mean = float(np.mean(final_decay))
            stable_changes.append(100 * (early_mean - late_mean) / early_mean)
            decay_changes.append(
                100 * (pre_decay_mean - final_decay_mean) / pre_decay_mean
            )
        phase_diagnostics[phase] = {
            "stable_second_half_lower_count": sum(
                change > 0 for change in stable_changes
            ),
            "median_stable_loss_drop_percent": float(np.median(stable_changes)),
            "minimum_stable_loss_drop_percent": float(np.min(stable_changes)),
            "smoothed_decay_lower_count": sum(
                change > 0 for change in decay_changes
            ),
            "median_smoothed_decay_loss_drop_percent": float(
                np.median(decay_changes)
            ),
            "minimum_smoothed_decay_loss_drop_percent": float(
                np.min(decay_changes)
            ),
        }
    diagnostics = {
        "curriculum_runs": len(curriculum_only),
        "pointwise_curriculum_wins": sum(
            point["loss_improvement"] > 0 for point in pointwise
        ),
        "pointwise_control_comparisons": len(pointwise),
        "minimum_ar_phase_steps": min(row["ar_steps"] for row in curriculum_only),
        "minimum_bd_phase_steps": min(row["bd_steps"] for row in curriculum_only),
        "maximum_phase_warmup_fraction": max(
            record["fraction"] for record in warmup_records
        ),
        "maximum_compute_undershoot_fraction": max(
            row["compute_undershoot_fraction"] for row in curriculum_only
        ),
        "optimizer_reset_count": sum(
            bool(row["optimizer_reset_at_transition"]) for row in curriculum_only
        ),
        "phase_training_diagnostics": phase_diagnostics,
    }
    return best_by_budget, pointwise, diagnostics


def save_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def save_figure(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_isoflop_profiles(rows, fits):
    style()
    colors = {
        0.0: "#555555",
        **dict(zip(P_AR_VALUES, plt.cm.plasma(np.linspace(0.12, 0.88, 5)))),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, budget in zip(axes.flat[:5], COMPUTE_BUDGETS):
        for p_ar in ALL_P_AR_VALUES:
            points = [
                row
                for row in rows
                if row["p_ar"] == p_ar and float(row["budget"]) == budget
            ]
            fit = next(
                fit
                for fit in fits
                if fit["p_ar"] == p_ar and fit["budget"] == budget
            )
            x = np.array([row["n_params"] for row in points])
            y = np.array([row["val_nelbo"] for row in points])
            curve_x = np.linspace(fit["x_min"], fit["x_max"], 240)
            ax.scatter(x, y, color=colors[p_ar], s=17, alpha=0.9, zorder=3)
            ax.plot(
                10**curve_x,
                np.polyval(fit["coefficients"], curve_x),
                color=colors[p_ar],
                lw=1.7 if p_ar else 1.4,
                ls="-" if p_ar else "--",
            )
            ax.scatter(
                fit["n_opt"],
                fit["loss_min"],
                color=colors[p_ar],
                marker="*",
                s=70,
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_title(f"C={budget:.0e}")
        ax.set_xlabel("non-embedding parameters N")
        ax.set_ylabel("validation diffusion NELBO")

    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colors[p_ar],
                ls="--" if p_ar == 0 else "-",
                marker="o",
                markersize=4,
                label="pure BD" if p_ar == 0 else f"p_ar={p_ar:.1f}",
            )
            for p_ar in ALL_P_AR_VALUES
        ],
        frameon=False,
        loc="upper left",
        title="AR step/token fraction",
    )
    legend_ax.text(
        0.02,
        0.22,
        "Each point uses the peak LR selected by\n"
        "the corresponding full pure-BD run.\n"
        "Stars are full-profile L1 quadratic minima.",
        transform=legend_ax.transAxes,
        va="bottom",
        linespacing=1.4,
    )
    fig.suptitle(
        "AR warm-start curriculum: IsoFLOP profiles on TinyShakespeare",
        fontsize=15,
    )
    fig.tight_layout()
    return save_figure(fig, "isoflop_profiles")


def make_curriculum_summary(fits, laws, best_by_budget):
    style()
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS)))
    budget_colors = dict(zip(COMPUTE_BUDGETS, colors))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    loss_ax, n_ax, exponent_ax, improvement_ax = axes.flat

    for budget in COMPUTE_BUDGETS:
        group = [fit for fit in fits if fit["budget"] == budget]
        p = [fit["p_ar"] for fit in group]
        loss_ax.plot(
            p,
            [fit["loss_min"] for fit in group],
            marker="o",
            color=budget_colors[budget],
            label=f"C={budget:.0e}",
        )
        n_ax.plot(
            p,
            [fit["n_opt"] for fit in group],
            marker="o",
            color=budget_colors[budget],
        )
    loss_ax.set_xlabel("AR fraction p_ar")
    loss_ax.set_ylabel("fitted minimum validation NELBO")
    loss_ax.set_title("Best attainable loss at each curriculum mix")
    loss_ax.legend(frameon=False, fontsize=8, ncol=2)
    n_ax.set_yscale("log")
    n_ax.set_xlabel("AR fraction p_ar")
    n_ax.set_ylabel("compute-optimal counted parameters N")
    n_ax.set_title("Curriculum changes the optimal model size")

    p = np.array(ALL_P_AR_VALUES)
    n_eff_exponents = np.array(
        [laws[f"{value:.1f}"]["effective_compute_parameters_opt"]["exponent"] for value in p]
    )
    d_exponents = np.array([laws[f"{value:.1f}"]["d_opt"]["exponent"] for value in p])
    exponent_ax.plot(p, n_eff_exponents, marker="o", color="#35618d", label=r"$N_\mathrm{eff,opt}$")
    exponent_ax.plot(p, d_exponents, marker="s", color="#d17c37", label=r"$D_\mathrm{opt}$")
    exponent_ax.axhline(0.5, color="#777777", ls=":", lw=1)
    exponent_ax.set_xlabel("AR fraction p_ar")
    exponent_ax.set_ylabel("compute-scaling exponent")
    exponent_ax.set_title("Effective model/data allocation exponents")
    exponent_ax.legend(frameon=False)

    x = np.arange(len(best_by_budget))
    bars = improvement_ax.bar(
        x,
        [row["loss_improvement"] for row in best_by_budget],
        color=colors,
    )
    improvement_ax.axhline(0, color="#555555", lw=0.8)
    improvement_ax.set_xticks(
        x,
        [f"{row['budget']:.0e}" for row in best_by_budget],
        rotation=25,
    )
    improvement_ax.set_xlabel("compute C (FLOPs)")
    improvement_ax.set_ylabel("NELBO improvement over pure BD")
    improvement_ax.set_title("Best discrete p_ar at each compute budget")
    for bar, row in zip(bars, best_by_budget):
        improvement_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={row['best_p_ar']:.1f}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=8,
        )

    fig.suptitle("AR warm-start curriculum summary", fontsize=15)
    fig.tight_layout()
    return save_figure(fig, "curriculum_summary")


def make_pointwise_heatmaps(pointwise):
    style()
    labels = [spec.label for spec in MODEL_SPECS]
    improvement = np.full((len(COMPUTE_BUDGETS), len(labels)), np.nan)
    best_p = np.full_like(improvement, np.nan)
    for row in pointwise:
        i = COMPUTE_BUDGETS.index(float(row["budget"]))
        j = labels.index(row["size"])
        improvement[i, j] = row["loss_improvement"]
        best_p[i, j] = row["best_p_ar"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    finite = improvement[np.isfinite(improvement)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())))
    image = axes[0].imshow(
        improvement,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
    )
    axes[1].imshow(best_p, aspect="auto", cmap="plasma", vmin=0.1, vmax=0.5)
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(
            range(len(COMPUTE_BUDGETS)),
            [f"{budget:.0e}" for budget in COMPUTE_BUDGETS],
        )
        ax.set_xlabel("model-size label")
        ax.set_ylabel("compute C")
        ax.grid(False)
    axes[0].set_title("Best curriculum improvement over pure BD")
    axes[1].set_title("Best measured p_ar")
    for i in range(len(COMPUTE_BUDGETS)):
        for j in range(len(labels)):
            if np.isfinite(improvement[i, j]):
                axes[0].text(
                    j,
                    i,
                    f"{improvement[i, j]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                axes[1].text(
                    j,
                    i,
                    f"{best_p[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if best_p[i, j] < 0.32 else "black",
                )
    fig.colorbar(image, ax=axes[0], shrink=0.8, label="NELBO decrease")
    fig.suptitle("Pointwise effect of AR warm-start", fontsize=14)
    fig.tight_layout()
    return save_figure(fig, "pointwise_curriculum_effect")


def selected_ar_wd_plot_losses():
    """Return tuned 0.8M losses when the completed ablation is available."""
    path = RESULTS_DIR / "ar_wd_sweep" / "0.8M" / "summary.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    brackets = summary.get("brackets", [])
    return {
        (float(row["budget"]), float(row["p_ar"])): float(
            row["selected_val_nelbo"]
        )
        for row in brackets
        if row.get("size") == "0.8M"
    }


def make_fixed_model_p_curves(rows):
    """Plot measured L(p_ar) at fixed N and C, one model size per figure."""
    style()
    tuned_0p8m_losses = selected_ar_wd_plot_losses()
    output_dir = FIGURES_DIR / "fixed_model_l_vs_p"
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_colors = dict(
        zip(
            COMPUTE_BUDGETS,
            plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS))),
        )
    )
    outputs = []

    for spec in MODEL_SPECS:
        groups = []
        for budget in COMPUTE_BUDGETS:
            points = sorted(
                (
                    row
                    for row in rows
                    if row["size"] == spec.label
                    and float(row["budget"]) == budget
                ),
                key=lambda row: row["p_ar"],
            )
            if points:
                groups.append((budget, points))
        if not groups:
            continue

        ncols = 2 if len(groups) == 4 else min(3, len(groups))
        nrows = math.ceil(len(groups) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.35 * ncols, 3.35 * nrows),
            squeeze=False,
        )
        for ax, (budget, points) in zip(axes.flat, groups):
            p_values = np.array([row["p_ar"] for row in points])
            losses = np.array(
                [
                    tuned_0p8m_losses.get(
                        (budget, float(row["p_ar"])),
                        row["val_nelbo"],
                    )
                    if spec.label == "0.8M" and row["p_ar"] > 0
                    else row["val_nelbo"]
                    for row in points
                ]
            )
            color = budget_colors[budget]
            ax.plot(
                p_values,
                losses,
                marker="o",
                ms=5,
                lw=1.8,
                color=color,
            )
            control = next(
                (row for row in points if row["p_ar"] == 0.0),
                None,
            )
            if control is not None:
                ax.axhline(
                    control["val_nelbo"],
                    color="#777777",
                    ls=":",
                    lw=1,
                    alpha=0.8,
                )
            best_index = int(np.argmin(losses))
            best = points[best_index]
            best_loss = float(losses[best_index])
            ax.scatter(
                [best["p_ar"]],
                [best_loss],
                marker="*",
                s=105,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            annotation = f"best p={best['p_ar']:.1f}"
            if control is not None:
                improvement = control["val_nelbo"] - best_loss
                annotation += f"\ngain={improvement:.3f}"
            ax.text(
                0.97,
                0.96,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
            )
            ax.set_title(f"C={budget:.0e}")
            ax.set_xticks(ALL_P_AR_VALUES)
            ax.set_xlim(-0.025, 0.525)
            ax.set_xlabel(r"AR fraction $p_{\mathrm{AR}}$")
            ax.set_ylabel("validation diffusion NELBO")

        for ax in axes.flat[len(groups):]:
            ax.axis("off")

        fig.suptitle(
            f"Fixed model: {spec.label} "
            f"({spec.n_params:,} counted parameters)",
            fontsize=14,
        )
        fig.text(
            0.5,
            0.008,
            "Each panel holds model size and FLOP budget fixed; "
            "the dotted line is pure BD and the star is the best measured mix.",
            ha="center",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.94))
        stem = f"fixed_N_{spec.label.replace('.', 'p')}"
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        outputs.append((png, pdf))

    return outputs


def main():
    rows = load_required_rows()
    expected_curriculum = sum(
        is_feasible(budget, spec, p_ar)
        for p_ar in P_AR_VALUES
        for budget in COMPUTE_BUDGETS
        for spec in MODEL_SPECS
    )
    if sum(row["p_ar"] > 0 for row in rows) != expected_curriculum:
        raise RuntimeError("Curriculum result count does not match the feasible grid")
    fits = fit_profiles(rows)
    laws = scaling_laws(fits)
    best_by_budget, pointwise, diagnostics = sweep_summary(rows, fits)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "compute_accounting": (
            "C=D[(12-6p)P_layers+(48-36p)L*n_layer*d_model+6*d_model*V]"
        ),
        "learning_rate_policy": "reuse pure-BD full-run locally bracketed optimum",
        "fit_objective": "full-profile L1 quadratic in log10(N)",
        "profiles": fits,
        "scaling_laws_by_p_ar": laws,
        "best_p_ar_by_budget": best_by_budget,
        "diagnostics": diagnostics,
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_csv(
        rows,
        RESULTS_DIR / "fixed_lr_runs.csv",
        [
            "p_ar",
            "budget",
            "size",
            "n_params",
            "steps",
            "ar_steps",
            "bd_steps",
            "learning_rate",
            "val_nelbo",
            "val_masked_ce_t0.5",
            "clean_tokens",
            "realized_flops",
            "effective_compute_parameters",
            "duration_seconds",
            "result_path",
        ],
    )
    save_csv(
        fits,
        RESULTS_DIR / "isoflop_optima.csv",
        [
            "p_ar",
            "budget",
            "n_points",
            "n_opt",
            "d_opt",
            "effective_compute_parameters_opt",
            "parameter_token_ratio",
            "loss_min",
            "mean_absolute_error",
            "vertex_clipped",
        ],
    )
    save_csv(
        pointwise,
        RESULTS_DIR / "pointwise_best_p_ar.csv",
        [
            "budget",
            "size",
            "n_params",
            "best_p_ar",
            "pure_bd_val_nelbo",
            "best_curriculum_val_nelbo",
            "loss_improvement",
        ],
    )
    figures = [
        make_isoflop_profiles(rows, fits),
        make_curriculum_summary(fits, laws, best_by_budget),
        make_pointwise_heatmaps(pointwise),
    ]
    figures.extend(make_fixed_model_p_curves(rows))
    print(f"saved {summary_path}")
    for pair in figures:
        print(f"saved {pair[0]}")
        print(f"saved {pair[1]}")


if __name__ == "__main__":
    main()
