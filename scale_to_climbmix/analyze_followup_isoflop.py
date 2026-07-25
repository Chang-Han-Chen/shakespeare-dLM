"""Analyze pure-AR or matched-step p_AR=0.4 adaptive IsoFLOP runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import fit_power_law
from config import ROOT


STUDIES = {
    "ar": {
        "results": ROOT / "results_ar",
        "loss": "val_ar_ce",
        "flops_per_token": "flash_causal_training_flops_per_clean_token",
        "title": "Pure autoregressive",
        "loss_label": "validation AR cross-entropy",
    },
    "matched": {
        "results": ROOT / "results_matched_p_ar_0p4",
        "loss": "val_nelbo",
        "flops_per_token": "block_diffusion_training_flops_per_clean_token",
        "title": "Matched-step p_AR=0.4",
        "loss_label": "validation block-diffusion NELBO",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


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


def load_rows(study: dict) -> tuple[list[dict], list[dict]]:
    root = study["results"]
    decisions = load_json(root / "lr_decisions.json")
    selected, all_rows = [], []
    for path in sorted((root / "runs").glob("*/*/**/result.json")):
        row = load_json(path)
        if row.get("status") != "complete":
            continue
        row["result_path"] = str(path.relative_to(ROOT))
        all_rows.append(row)
        key = f"{row['budget']:.0e}/{row['size']}".replace("+", "")
        decision = decisions.get(key)
        if decision is None:
            continue
        if study is STUDIES["ar"]:
            is_selected = math.isclose(
                row["learning_rate"],
                decision["selected_lr"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        else:
            is_selected = math.isclose(
                row["ar_learning_rate"],
                decision["selected_ar_lr"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ) and math.isclose(
                row["bd_learning_rate"],
                decision["selected_bd_lr"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        if is_selected:
            selected.append(row)
    return selected, all_rows


def interpolate(
    n_params: float,
    rows: list[dict],
    key: str,
    *,
    log_values: bool = False,
) -> float:
    points = sorted(
        {int(row["n_params"]): float(row[key]) for row in rows}.items()
    )
    x = np.log([point[0] for point in points])
    y = np.array([point[1] for point in points])
    if log_values:
        y = np.log(y)
    target = math.log(n_params)
    if not x[0] <= target <= x[-1]:
        raise ValueError(f"N={n_params:.0f} is outside measured support")
    value = float(np.interp(target, x, y))
    return math.exp(value) if log_values else value


def fit_profiles(rows: list[dict], study: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["budget"])].append(row)
    profiles = []
    for budget, points in sorted(grouped.items()):
        points.sort(key=lambda row: row["n_params"])
        if len(points) < 3:
            continue
        losses = np.array([row[study["loss"]] for row in points])
        minimum = int(np.argmin(losses))
        if minimum in (0, len(points) - 1):
            raise ValueError(
                f"Unbracketed C={budget:.0e}: measured minimum "
                f"{points[minimum]['size']}"
            )
        support = points[minimum - 1 : minimum + 2]
        x = np.log10([row["n_params"] for row in support])
        y = np.array([row[study["loss"]] for row in support])
        coefficients = np.polyfit(x, y, 2)
        if coefficients[0] <= 0:
            raise ValueError(f"Non-convex local profile at C={budget:.0e}")
        vertex = -coefficients[1] / (2 * coefficients[0])
        if not x[0] <= vertex <= x[-1]:
            raise ValueError(f"Local vertex outside support at C={budget:.0e}")
        n_opt = 10**vertex
        flops_per_token = interpolate(
            n_opt,
            points,
            study["flops_per_token"],
            log_values=True,
        )
        d_opt = budget / flops_per_token
        profile = {
            "budget": budget,
            "n_points": len(points),
            "n_fit_points": 3,
            "fit_method": "local_bracketing3_l2_quadratic_in_log10_n",
            "support_sizes": [row["size"] for row in support],
            "coefficients": coefficients.tolist(),
            "n_opt": n_opt,
            "d_opt": d_opt,
            "tokens_per_parameter": d_opt / n_opt,
            "loss_min": float(np.polyval(coefficients, vertex)),
        }
        if study is STUDIES["matched"]:
            ratio = interpolate(n_opt, points, "realized_to_nominal_compute")
            profile["realized_to_nominal_compute"] = ratio
            profile["realized_compute_opt"] = budget * ratio
        profiles.append(profile)
    return profiles


def fit_laws(profiles: list[dict]) -> dict:
    if len(profiles) < 2:
        return {}

    def laws(rows: list[dict]) -> dict:
        compute = [row["budget"] for row in rows]
        return {
            key: fit_power_law(compute, [row[key] for row in rows])
            for key in ("n_opt", "d_opt", "loss_min")
        }

    output = {"all": laws(profiles)}
    if len(profiles) >= 5:
        output["rolling_highest5"] = laws(profiles[-5:])
    return output


def summarize_forecasts(history: list[dict]) -> dict:
    errors = [
        abs(float(row["n_forecast_relative_error"]))
        for row in history
        if row.get("n_forecast_relative_error") is not None
    ]
    if not errors:
        return {}
    return {
        "n_finalized_forecasts": len(errors),
        "mean_absolute_relative_error": float(np.mean(errors)),
        "median_absolute_relative_error": float(np.median(errors)),
        "fraction_within_25_percent": float(np.mean(np.array(errors) <= 0.25)),
    }


def make_figure(
    rows: list[dict],
    profiles: list[dict],
    laws: dict,
    study: dict,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.1))
    profile_ax, allocation_ax, ratio_ax = axes
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(profiles)))
    for color, profile in zip(colors, profiles):
        points = sorted(
            [row for row in rows if row["budget"] == profile["budget"]],
            key=lambda row: row["n_params"],
        )
        profile_ax.scatter(
            [row["n_params"] / 1e6 for row in points],
            [row[study["loss"]] for row in points],
            color=color,
            s=24,
        )
        support = [
            row for row in points if row["size"] in profile["support_sizes"]
        ]
        dense = np.linspace(
            np.log10(support[0]["n_params"]),
            np.log10(support[-1]["n_params"]),
            100,
        )
        profile_ax.plot(
            10**dense / 1e6,
            np.polyval(profile["coefficients"], dense),
            color=color,
            label=f"{profile['budget']:.0e}",
        )
        profile_ax.scatter(
            profile["n_opt"] / 1e6,
            profile["loss_min"],
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=70,
        )
    profile_ax.set_xscale("log")
    profile_ax.set_xlabel("counted parameters N (millions)")
    profile_ax.set_ylabel(study["loss_label"])
    profile_ax.set_title(f"{study['title']} IsoFLOP profiles")
    profile_ax.legend(frameon=False, fontsize=7, ncol=2)

    compute = np.array([row["budget"] for row in profiles])
    n_opt = np.array([row["n_opt"] for row in profiles])
    d_opt = np.array([row["d_opt"] for row in profiles])
    allocation_ax.loglog(compute, n_opt, "o-", label="N*")
    allocation_ax.loglog(compute, d_opt, "s-", label="D*")
    law_group = laws.get("rolling_highest5", laws.get("all", {}))
    if law_group:
        dense_compute = np.geomspace(compute[-min(5, len(compute))], compute[-1], 100)
        for key, linestyle in (("n_opt", "-"), ("d_opt", "--")):
            law = law_group[key]
            allocation_ax.loglog(
                dense_compute,
                law["coefficient"] * dense_compute ** law["exponent"],
                linestyle,
                color="black",
                alpha=0.65,
            )
    allocation_ax.set_xlabel("nominal training FLOPs C")
    allocation_ax.set_ylabel("parameters / clean tokens")
    allocation_ax.set_title("Compute-optimal allocation")
    allocation_ax.legend(frameon=False)

    ratio_ax.loglog(compute, d_opt / n_opt, "o-", color="#9b4f0f")
    ratio_ax.axhline(20, color="gray", linestyle=":", label="D/N=20 seed")
    ratio_ax.set_xlabel("nominal training FLOPs C")
    ratio_ax.set_ylabel("optimal clean tokens per parameter D*/N*")
    ratio_ax.set_title("Allocation ratio")
    ratio_ax.legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    stem = "pure_ar_isoflop" if study is STUDIES["ar"] else "matched_p_ar_0p4_isoflop"
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    study = STUDIES[args.study]
    results = study["results"]
    figures = (
        ROOT / "figures_ar"
        if args.study == "ar"
        else ROOT / "figures_matched_p_ar_0p4"
    )
    selected, all_rows = load_rows(study)
    profiles = fit_profiles(selected, study)
    laws = fit_laws(profiles)
    forecast_path = results / "forecast_history.json"
    forecast_history = (
        load_json(forecast_path) if forecast_path.exists() else []
    )
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "all_runs.csv", all_rows)
    write_csv(results / "best_runs.csv", selected)
    write_csv(results / "optimal_allocation.csv", profiles)
    summary = {
        "study": args.study,
        "loss_metric": study["loss"],
        "profiles": profiles,
        "laws": laws,
        "forecast_history": forecast_history,
        "forecast_metrics": summarize_forecasts(forecast_history),
        "n_selected_runs": len(selected),
        "n_total_runs": len(all_rows),
    }
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_figure(selected, profiles, laws, study, figures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
