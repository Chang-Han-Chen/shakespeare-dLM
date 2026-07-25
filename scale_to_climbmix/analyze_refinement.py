"""Refit the historical batch-64 IsoFLOP curves with targeted new points."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import fit_power_law, load_best_runs
from config import COMPUTE_BUDGETS, ROOT


RESULTS = ROOT / "results_refinement"
RUNS = RESULTS / "runs"
DECISIONS = RESULTS / "lr_decisions.json"
FIGURES = ROOT / "figures_refinement"


def load_json(path: Path):
    return json.loads(path.read_text())


def load_refinement_rows() -> tuple[list[dict], list[dict]]:
    """Return the threshold-selected new rows and all completed LR probes."""
    decisions = load_json(DECISIONS)
    selected = []
    probes = []
    for result_path in sorted(RUNS.glob("*/*/lr_*/result.json")):
        row = load_json(result_path)
        if row.get("status") != "complete":
            continue
        row["result_path"] = str(result_path.relative_to(ROOT))
        key = f"{row['budget']:.0e}/{row['size']}".replace("+", "")
        decision = decisions.get(key)
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
    """Log-interpolate exact architecture FLOPs/token within one profile."""
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


def fit_local_profiles(rows: list[dict]) -> list[dict]:
    """Fit each minimum using the measured winner and immediate neighbors."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["budget"])].append(row)
    profiles = []
    for budget in COMPUTE_BUDGETS:
        points = sorted(grouped[budget], key=lambda row: row["n_params"])
        losses = np.array([row["val_nelbo"] for row in points])
        minimum = int(np.argmin(losses))
        if minimum == 0 or minimum == len(points) - 1:
            raise ValueError(
                f"Unbracketed historical profile at C={budget:.0e}: "
                f"minimum is {points[minimum]['size']}"
            )
        chosen = points[minimum - 1 : minimum + 2]
        x = np.log10([row["n_params"] for row in chosen])
        y = np.array([row["val_nelbo"] for row in chosen])
        coefficients = np.polyfit(x, y, 2)
        a, b, _ = coefficients
        if a <= 0:
            raise ValueError(f"Non-convex historical profile at C={budget:.0e}")
        vertex = -b / (2 * a)
        if not x.min() <= vertex <= x.max():
            raise ValueError(
                f"Unbracketed historical profile at C={budget:.0e}: "
                "quadratic vertex is outside local support"
            )
        n_opt = 10**vertex
        flops_per_token = interpolate_flops(n_opt, points)
        prediction = np.polyval(coefficients, x)
        profiles.append(
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
    return profiles


def laws_for(profiles: list[dict]) -> dict:
    compute = [row["budget"] for row in profiles]
    return {
        key: fit_power_law(compute, [row[key] for row in profiles])
        for key in ("n_opt", "effective_n_opt", "d_opt", "loss_min")
    }


def write_csv(path: Path, rows: list[dict]) -> None:
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


def make_figure(
    rows: list[dict],
    new_rows: list[dict],
    profiles: list[dict],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    new_paths = {row["result_path"] for row in new_rows}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3))
    for axis, profile in zip(axes.flat, profiles):
        points = sorted(
            [row for row in rows if row["budget"] == profile["budget"]],
            key=lambda row: row["n_params"],
        )
        original = [
            row for row in points if row.get("result_path") not in new_paths
        ]
        targeted = [
            row for row in points if row.get("result_path") in new_paths
        ]
        axis.plot(
            [row["n_params"] / 1e6 for row in points],
            [row["val_nelbo"] for row in points],
            color="#929292",
            lw=1,
        )
        axis.scatter(
            [row["n_params"] / 1e6 for row in original],
            [row["val_nelbo"] for row in original],
            color="#4c78a8",
            s=25,
            label="original",
        )
        axis.scatter(
            [row["n_params"] / 1e6 for row in targeted],
            [row["val_nelbo"] for row in targeted],
            color="#e45756",
            marker="D",
            s=35,
            label="targeted",
        )
        support = [
            row for row in points
            if row["size"] in set(profile["support_sizes"])
        ]
        dense_x = np.linspace(
            np.log10(min(row["n_params"] for row in support)),
            np.log10(max(row["n_params"] for row in support)),
            120,
        )
        axis.plot(
            10**dense_x / 1e6,
            np.polyval(profile["coefficients"], dense_x),
            color="black",
            lw=1.3,
        )
        axis.scatter(
            profile["n_opt"] / 1e6,
            profile["loss_min"],
            marker="*",
            color="#f2cf5b",
            edgecolor="black",
            s=95,
            zorder=4,
            label="refitted optimum",
        )
        axis.set_xscale("log")
        tick_values = np.geomspace(
            min(row["n_params"] for row in points) / 1e6,
            max(row["n_params"] for row in points) / 1e6,
            3,
        )
        axis.set_xticks(tick_values)
        axis.set_xticklabels([f"{value:.2g}" for value in tick_values])
        axis.minorticks_off()
        axis.set_title(f"C={profile['budget']:.0e}")
        axis.set_xlabel("counted parameters N (millions)")
        axis.set_ylabel("validation NELBO")
        axis.grid(alpha=0.2)
    allocation = axes.flat[-1]
    compute = np.array([row["budget"] for row in profiles])
    allocation.loglog(
        compute,
        [row["n_opt"] for row in profiles],
        "o-",
        label="N*",
    )
    allocation.loglog(
        compute,
        [row["d_opt"] for row in profiles],
        "s-",
        label="D*",
    )
    allocation.set_title("Refined historical allocation")
    allocation.set_xlabel("training FLOPs C")
    allocation.set_ylabel("parameters / clean tokens")
    allocation.grid(alpha=0.2)
    allocation.legend(frameon=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"historical_isoflop_refinement.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    original_rows = load_best_runs()
    new_rows, probe_rows = load_refinement_rows()
    if not new_rows:
        raise ValueError("No selected refinement runs found")
    combined = original_rows + new_rows
    profiles = fit_local_profiles(combined)
    laws = laws_for(profiles)
    summary = {
        "profile_fit": (
            "L2 quadratic through the measured minimum and its immediate "
            "neighbors in log10(N)"
        ),
        "batch_size": 64,
        "profiles": profiles,
        "scaling_laws": laws,
        "original_runs": len(original_rows),
        "selected_refinement_runs": len(new_rows),
        "completed_refinement_probes": len(probe_rows),
        "lr_decisions": load_json(DECISIONS),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(RESULTS / "best_runs.csv", combined)
    write_csv(RESULTS / "optimal_allocation.csv", profiles)
    make_figure(combined, new_rows, profiles)
    for row in profiles:
        print(
            f"C={row['budget']:.0e} N*={row['n_opt'] / 1e6:.3f}M "
            f"D*={row['d_opt'] / 1e9:.3f}B "
            f"D/N={row['tokens_per_parameter']:.1f} "
            f"L*={row['loss_min']:.5f}"
        )
    print(
        f"historical: N~C^{laws['n_opt']['exponent']:.4f}, "
        f"D~C^{laws['d_opt']['exponent']:.4f}"
    )


if __name__ == "__main__":
    main()
