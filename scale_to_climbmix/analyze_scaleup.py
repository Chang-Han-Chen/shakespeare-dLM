"""Analyze the adaptive batch-128 IsoFLOP extension through 1e18 FLOPs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import fit_power_law
from config import MODEL_BY_LABEL, ROOT


HISTORICAL_SUMMARY = ROOT / "results" / "summary.json"
RESULTS = ROOT / "results_scaleup"
RUNS = RESULTS / "runs"
DECISIONS = RESULTS / "lr_decisions.json"
FORECASTS = RESULTS / "forecast_history.json"
FIGURES = ROOT / "figures_scaleup"
ROLLING_WINDOW = 5


def load_json(path: Path):
    return json.loads(path.read_text())


def load_scaleup_rows() -> tuple[list[dict], list[dict]]:
    decisions = load_json(DECISIONS)
    selected = []
    probes = []
    for result_path in sorted(RUNS.glob("*/*/lr_*/result.json")):
        row = load_json(result_path)
        if row.get("status") != "complete":
            continue
        row["result_path"] = str(result_path.relative_to(ROOT))
        slug = f"{row['budget']:.0e}".replace("+", "")
        decision = decisions.get(slug)
        if decision is None:
            continue
        probes.append(row)
        if math.isclose(
            row["learning_rate"],
            decision["selected_lr"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            selected.append(row)
    return selected, probes


def interpolate_flops(n_params: float, rows: list[dict]) -> float:
    """Log-interpolate exact architecture FLOPs/token around a fitted vertex."""
    points = {
        int(row["n_params"]): float(row["training_flops_per_clean_token"])
        for row in rows
    }
    ordered = sorted(points.items())
    x = np.log([item[0] for item in ordered])
    y = np.log([item[1] for item in ordered])
    target = np.log(n_params)
    if target < x[0] or target > x[-1]:
        raise ValueError(
            f"Fitted N={n_params:.0f} is outside architecture support "
            f"[{ordered[0][0]}, {ordered[-1][0]}]"
        )
    return float(np.exp(np.interp(target, x, y)))


def fit_scaleup_profiles(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["budget"]].append(row)
    fits = []
    for budget, points in sorted(grouped.items()):
        points.sort(key=lambda row: row["n_params"])
        if len(points) < 3:
            continue
        losses = np.array([row["val_nelbo"] for row in points])
        minimum = int(np.argmin(losses))
        if minimum == 0 or minimum == len(points) - 1:
            raise ValueError(
                f"Unbracketed profile at C={budget:.0e}: "
                f"minimum is {points[minimum]['size']}"
            )
        support = [minimum - 1, minimum, minimum + 1]
        chosen = [points[index] for index in support]
        x = np.log10([row["n_params"] for row in chosen])
        y = np.array([row["val_nelbo"] for row in chosen])
        coefficients = np.polyfit(x, y, 2)
        a, b, _ = coefficients
        if a <= 0:
            raise ValueError(f"Non-convex profile at C={budget:.0e}")
        vertex = -b / (2 * a)
        if not x.min() <= vertex <= x.max():
            raise ValueError(
                f"Unbracketed profile at C={budget:.0e}: vertex outside lowest three"
            )
        n_opt = 10**vertex
        flops_per_token = interpolate_flops(n_opt, points)
        prediction = np.polyval(coefficients, x)
        fits.append(
            {
                "budget": budget,
                "n_points": len(points),
                "n_fit_points": 3,
                "fit_method": "local_bracketing3_l2_quadratic_in_log10_n",
                "coefficients": coefficients.tolist(),
                "support_sizes": [row["size"] for row in chosen],
                "n_opt": n_opt,
                "effective_n_opt": flops_per_token / 12,
                "d_opt": budget / flops_per_token,
                "tokens_per_parameter": budget / flops_per_token / n_opt,
                "parameter_token_ratio": n_opt * flops_per_token / budget,
                "loss_min": float(np.polyval(coefficients, vertex)),
                "support_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            }
        )
    return fits


def laws_for(profiles: list[dict]) -> dict:
    compute = [row["budget"] for row in profiles]
    laws = {}
    for key in ("n_opt", "effective_n_opt", "d_opt", "loss_min"):
        laws[key] = fit_power_law(compute, [row[key] for row in profiles])
    return laws


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], profiles: list[dict], laws: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    historical_count = len(load_json(HISTORICAL_SUMMARY)["profiles"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    profile_ax, allocation_ax, ratio_ax, loss_ax = axes.ravel()

    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(profiles) - historical_count))
    scale_fits = profiles[historical_count:]
    for color, fit in zip(colors, scale_fits):
        points = sorted(
            [row for row in rows if row["budget"] == fit["budget"]],
            key=lambda row: row["n_params"],
        )
        x = np.log10([row["n_params"] for row in points])
        y = [row["val_nelbo"] for row in points]
        profile_ax.scatter(10**x / 1e6, y, color=color, s=35)
        support = [
            row for row in points if row["size"] in set(fit["support_sizes"])
        ]
        dense_x = np.linspace(
            np.log10(min(row["n_params"] for row in support)),
            np.log10(max(row["n_params"] for row in support)),
            120,
        )
        profile_ax.plot(
            10**dense_x / 1e6,
            np.polyval(fit["coefficients"], dense_x),
            color=color,
            label=f"{fit['budget']:.0e}",
        )
        profile_ax.scatter(
            fit["n_opt"] / 1e6,
            fit["loss_min"],
            marker="*",
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    profile_ax.set_xscale("log")
    profile_ax.set_xlabel("counted parameters N (millions)")
    profile_ax.set_ylabel("validation NELBO")
    profile_ax.set_title("Adaptive batch-128 IsoFLOP profiles")
    profile_ax.legend(frameon=False, fontsize=8)

    compute = np.array([row["budget"] for row in profiles])
    n_opt = np.array([row["n_opt"] for row in profiles])
    d_opt = np.array([row["d_opt"] for row in profiles])
    allocation_ax.loglog(compute, n_opt, "o-", label="N*")
    allocation_ax.loglog(compute, d_opt, "s-", label="D*")
    fit_compute = np.geomspace(compute[-ROLLING_WINDOW], compute[-1], 100)
    for key, marker in (("n_opt", "-"), ("d_opt", "--")):
        law = laws["rolling_highest5"][key]
        allocation_ax.loglog(
            fit_compute,
            law["coefficient"] * fit_compute ** law["exponent"],
            marker,
            color="black",
            alpha=0.65,
        )
    allocation_ax.axvline(
        profiles[historical_count]["budget"],
        color="gray",
        ls=":",
        lw=1,
        label="batch 64→128",
    )
    allocation_ax.set_xlabel("training FLOPs C")
    allocation_ax.set_ylabel("parameters / clean tokens")
    allocation_ax.set_title("Compute-optimal allocation")
    allocation_ax.legend(frameon=False, fontsize=8)

    ratio_ax.loglog(compute, d_opt / n_opt, "o-", color="#9b4f0f")
    ratio_ax.axvline(profiles[historical_count]["budget"], color="gray", ls=":", lw=1)
    ratio_ax.set_xlabel("training FLOPs C")
    ratio_ax.set_ylabel("optimal clean tokens per parameter D*/N*")
    ratio_ax.set_title("Allocation ratio")

    loss_min = np.array([row["loss_min"] for row in profiles])
    loss_ax.semilogx(compute, loss_min, "o-", color="#4b4c9a")
    loss_ax.axvline(profiles[historical_count]["budget"], color="gray", ls=":", lw=1)
    loss_ax.set_xlabel("training FLOPs C")
    loss_ax.set_ylabel("fitted minimum validation NELBO")
    loss_ax.set_title("Compute-optimal loss")

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"isoflop_scaling_to_1e18.{extension}", bbox_inches="tight")
    plt.close(fig)


def make_diagnostics(
    rows: list[dict],
    forecasts: list[dict],
    decisions: dict,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, (forecast_ax, mfu_ax, lr_ax) = plt.subplots(1, 3, figsize=(13.5, 4.1))

    completed_forecasts = [
        row for row in forecasts if row["fitted_n_opt"] is not None
    ]
    budget = np.array([row["budget"] for row in completed_forecasts])
    predicted = np.array([row["forecast_n_opt"] for row in completed_forecasts])
    fitted = np.array([row["fitted_n_opt"] for row in completed_forecasts])
    forecast_ax.loglog(budget, predicted, "o--", label="forecast")
    forecast_ax.loglog(budget, fitted, "s-", label="fitted optimum")
    for row in completed_forecasts:
        forecast_ax.annotate(
            f"{100 * row['n_forecast_relative_error']:+.1f}%",
            (row["budget"], row["fitted_n_opt"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    forecast_ax.set_xlabel("training FLOPs C")
    forecast_ax.set_ylabel("counted parameters N")
    forecast_ax.set_title("Predict-next localization")
    forecast_ax.legend(frameon=False)

    budgets = sorted({row["budget"] for row in rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(budgets)))
    for color, current_budget in zip(colors, budgets):
        points = [row for row in rows if row["budget"] == current_budget]
        mfu_ax.scatter(
            [row["n_params"] / 1e6 for row in points],
            [
                100
                * row["training_flops_per_clean_token"]
                * row["training_tokens_per_second"]
                / 989e12
                for row in points
            ],
            color=color,
            label=f"{current_budget:.0e}",
        )
    mfu_ax.set_xlabel("counted parameters N (millions)")
    mfu_ax.set_ylabel("accounted dense BF16 MFU (%)")
    mfu_ax.set_title("One-GPU H100 efficiency")
    mfu_ax.legend(frameon=False, fontsize=8)

    completed_decisions = [
        (slug, row)
        for slug, row in sorted(
            decisions.items(),
            key=lambda item: float(item[0]),
        )
        if row["accepted_relative_improvement"] is not None
    ]
    labels = [slug for slug, _ in completed_decisions]
    improvements = [
        100 * row["accepted_relative_improvement"]
        for _, row in completed_decisions
    ]
    lr_ax.bar(labels, improvements, color="#8b5a9f")
    lr_ax.axhline(1.0, color="black", ls="--", lw=1, label="acceptance threshold")
    lr_ax.axhline(0.0, color="gray", lw=0.8)
    lr_ax.set_xlabel("training FLOPs C")
    lr_ax.set_ylabel("probe validation improvement (%)")
    lr_ax.set_title("Learning-rate probes")
    lr_ax.legend(frameon=False, fontsize=8)

    for axis in (forecast_ax, mfu_ax, lr_ax):
        axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"scaleup_diagnostics.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def make_profile_detail(
    selected_rows: list[dict],
    probe_rows: list[dict],
    profiles: list[dict],
) -> None:
    scale_profiles = [
        row for row in profiles if row["budget"] >= 3e16
    ]
    fig, axes = plt.subplots(
        1,
        len(scale_profiles),
        figsize=(4.5 * len(scale_profiles), 3.8),
        squeeze=False,
    )
    axes = axes.ravel()
    for axis, fit, color in zip(
        axes,
        scale_profiles,
        plt.cm.viridis(np.linspace(0.08, 0.9, len(scale_profiles))),
    ):
        points = sorted(
            [row for row in selected_rows if row["budget"] == fit["budget"]],
            key=lambda row: row["n_params"],
        )
        x = np.array([row["n_params"] / 1e6 for row in points])
        delta = 1000 * (
            np.array([row["val_nelbo"] for row in points]) - fit["loss_min"]
        )
        axis.plot(x, delta, "o--", color=color, alpha=0.7, label="canonical runs")

        support = [
            row for row in points if row["size"] in set(fit["support_sizes"])
        ]
        dense_log_n = np.linspace(
            np.log10(min(row["n_params"] for row in support)),
            np.log10(max(row["n_params"] for row in support)),
            120,
        )
        axis.plot(
            10**dense_log_n / 1e6,
            1000
            * (
                np.polyval(fit["coefficients"], dense_log_n)
                - fit["loss_min"]
            ),
            color="black",
            lw=1.8,
            label="local bracket fit",
        )
        axis.scatter(
            [row["n_params"] / 1e6 for row in support],
            [
                1000 * (row["val_nelbo"] - fit["loss_min"])
                for row in support
            ],
            s=65,
            facecolors="none",
            edgecolors="black",
            linewidth=1.2,
            label="fit support",
        )

        selected_keys = {
            (row["size"], row["learning_rate"]) for row in points
        }
        rejected = [
            row
            for row in probe_rows
            if row["budget"] == fit["budget"]
            and (row["size"], row["learning_rate"]) not in selected_keys
        ]
        if rejected:
            axis.scatter(
                [row["n_params"] / 1e6 for row in rejected],
                [
                    1000 * (row["val_nelbo"] - fit["loss_min"])
                    for row in rejected
                ],
                marker="x",
                s=55,
                color="#a33d3d",
                label="rejected LR probe",
            )
        axis.axhline(0.0, color="gray", lw=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("counted parameters N (millions)")
        axis.set_title(f"C={fit['budget']:.0e}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("validation NELBO above fitted minimum (×1000)")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for axis in axes[1:]:
        current_handles, current_labels = axis.get_legend_handles_labels()
        by_label.update(zip(current_labels, current_handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=len(by_label),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle("Measured IsoFLOP neighborhoods and local fit support", y=1.13)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"isoflop_profile_details.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    historical = load_json(HISTORICAL_SUMMARY)
    scale_rows, probe_rows = load_scaleup_rows()
    scale_profiles = fit_scaleup_profiles(scale_rows)
    profiles = historical["profiles"] + scale_profiles
    if not scale_profiles:
        raise ValueError("No complete scale-up profiles found")
    windows = {
        "all_budgets": profiles,
        "rolling_highest5": profiles[-ROLLING_WINDOW:],
        "batch128_only": scale_profiles,
    }
    laws = {
        name: laws_for(window)
        for name, window in windows.items()
        if len(window) >= 2
    }
    decisions = load_json(DECISIONS)
    forecasts = load_json(FORECASTS)
    summary = {
        "profile_fit": "L2 quadratic through the measured minimum and its immediate neighbors in log10(N)",
        "forecast_rule": "rolling highest five bracketed optima",
        "batch_regimes": {"historical": 64, "scaleup": 128},
        "profiles": profiles,
        "scaleup_profiles": scale_profiles,
        "scaling_laws": laws,
        "lr_decisions": decisions,
        "forecast_history": forecasts,
        "selected_scaleup_runs": len(scale_rows),
        "completed_probe_runs": len(probe_rows),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(RESULTS / "best_runs.csv", scale_rows)
    write_csv(RESULTS / "optimal_allocation.csv", profiles)
    make_figure(scale_rows, profiles, laws)
    make_diagnostics(probe_rows, forecasts, decisions)
    make_profile_detail(scale_rows, probe_rows, profiles)
    for row in scale_profiles:
        print(
            f"C={row['budget']:.0e} N*={row['n_opt'] / 1e6:.3f}M "
            f"D*={row['d_opt'] / 1e9:.3f}B D/N={row['tokens_per_parameter']:.1f} "
            f"L*={row['loss_min']:.5f}"
        )
    primary = laws["rolling_highest5"]
    print(
        f"rolling-5: N~C^{primary['n_opt']['exponent']:.4f}, "
        f"D~C^{primary['d_opt']['exponent']:.4f}"
    )


if __name__ == "__main__":
    main()
