"""Analyze whether AR curriculum helps at compute-optimal fixed total steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ROOT
from curriculum_config import P_AR_VALUES
from fixed_steps_config import ALLOCATION_SOURCE, FIXED_STEP_TARGETS
from train import atomic_json_dump


RESULTS = ROOT / "results_fixed_steps"
FIGURES = ROOT / "figures_fixed_steps"


def load_complete(path: Path):
    row = json.loads(path.read_text())
    if row.get("status") != "complete":
        raise RuntimeError(f"Incomplete result: {path}")
    return row


def load_rows(targets):
    rows = []
    for target in targets:
        point = RESULTS / "baseline" / target.spec.label
        bracket = json.loads((point / "lr_bracket.json").read_text())
        if bracket.get("status") not in {
            "locally_bracketed",
            "restricted_two_point",
            "user_truncated_boundary",
            "user_selected_near_tie",
        }:
            raise RuntimeError(f"Invalid LR selection: {point / 'lr_bracket.json'}")
        selected_lr = float(bracket["selected_lr"])
        baseline_path = point / (
            f"lr_{selected_lr:.1e}".replace("+", "")
        ) / "result.json"
        baseline = load_complete(baseline_path)
        rows.append(
            {
                "size": target.spec.label,
                "n_params": target.spec.n_params,
                "predicted_compute": target.predicted_compute,
                "total_steps": target.total_steps,
                "clean_tokens": target.clean_tokens,
                "p_ar": 0.0,
                "ar_steps": 0,
                "bd_steps": target.total_steps,
                "compute_ratio_to_full_bd": 1.0,
                "realized_flops": target.realized_full_bd_compute,
                "learning_rate": selected_lr,
                "val_nelbo": float(baseline["val_nelbo"]),
                "result_path": str(baseline_path),
            }
        )
        for p_ar in P_AR_VALUES:
            slug = f"{p_ar:.1f}".replace(".", "p")
            path = (
                RESULTS
                / "runs"
                / target.spec.label
                / f"p_ar_{slug}"
                / "result.json"
            )
            row = load_complete(path)
            if row.get("comparison_mode") != "fixed_total_steps":
                raise ValueError(f"Wrong comparison mode: {path}")
            if int(row["total_steps"]) != target.total_steps:
                raise ValueError(f"Step mismatch: {path}")
            rows.append(
                {
                    "size": target.spec.label,
                    "n_params": target.spec.n_params,
                    "predicted_compute": target.predicted_compute,
                    "total_steps": target.total_steps,
                    "clean_tokens": target.clean_tokens,
                    "p_ar": p_ar,
                    "ar_steps": int(row["ar_steps"]),
                    "bd_steps": int(row["bd_steps"]),
                    "compute_ratio_to_full_bd": (
                        float(row["realized_flops"])
                        / target.realized_full_bd_compute
                    ),
                    "realized_flops": int(row["realized_flops"]),
                    "learning_rate": float(row["learning_rate"]),
                    "val_nelbo": float(row["val_nelbo"]),
                    "result_path": str(path),
                }
            )
    return rows


def summarize(rows, targets):
    summaries = []
    for target in targets:
        points = sorted(
            [row for row in rows if row["size"] == target.spec.label],
            key=lambda row: row["p_ar"],
        )
        baseline = points[0]
        for row in points:
            row["gain_vs_full_bd"] = baseline["val_nelbo"] - row["val_nelbo"]
        best = min(points, key=lambda row: row["val_nelbo"])
        best_nonzero = min(points[1:], key=lambda row: row["val_nelbo"])
        summaries.append(
            {
                "size": target.spec.label,
                "n_params": target.spec.n_params,
                "predicted_compute": target.predicted_compute,
                "total_steps": target.total_steps,
                "clean_tokens": target.clean_tokens,
                "selected_lr": baseline["learning_rate"],
                "baseline_val_nelbo": baseline["val_nelbo"],
                "optimal_p_ar": best["p_ar"],
                "minimum_val_nelbo": best["val_nelbo"],
                "best_gain": baseline["val_nelbo"] - best["val_nelbo"],
                "best_nonzero_p_ar": best_nonzero["p_ar"],
                "best_nonzero_gain": (
                    baseline["val_nelbo"] - best_nonzero["val_nelbo"]
                ),
                "best_compute_ratio": best["compute_ratio_to_full_bd"],
            }
        )
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows, summaries, targets, output_stem):
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
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(targets)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    loss_ax, gain_ax, p_ax, best_gain_ax = axes.flat

    for color, target in zip(colors, targets):
        points = sorted(
            [row for row in rows if row["size"] == target.spec.label],
            key=lambda row: row["p_ar"],
        )
        p = np.array([row["p_ar"] for row in points])
        loss = np.array([row["val_nelbo"] for row in points])
        gain = loss[0] - loss
        loss_ax.plot(p, loss, marker="o", color=color, label=target.spec.label)
        gain_ax.plot(p, gain, marker="o", color=color)

    n_params = np.array([row["n_params"] for row in summaries])
    optimal_p = np.array([row["optimal_p_ar"] for row in summaries])
    best_nonzero_gain = np.array(
        [row["best_nonzero_gain"] for row in summaries]
    )
    p_ax.plot(n_params, optimal_p, marker="o", color="#35618d", lw=2)
    best_gain_ax.plot(
        n_params,
        best_nonzero_gain,
        marker="o",
        color="#c46d3b",
        lw=2,
    )
    best_gain_ax.axhline(0.0, color="black", lw=1, alpha=0.6)

    loss_ax.set_title("Validation loss at identical total steps")
    loss_ax.set_xlabel(r"$p_{\rm AR}$")
    loss_ax.set_ylabel("validation NELBO")
    loss_ax.legend(title="model", ncol=2, fontsize=8)

    gain_ax.set_title("Curriculum gain over full BD")
    gain_ax.set_xlabel(r"$p_{\rm AR}$")
    gain_ax.set_ylabel(r"$L(p=0)-L(p)$")
    gain_ax.axhline(0.0, color="black", lw=1, alpha=0.6)

    p_ax.set_title("Best curriculum fraction")
    p_ax.set_xscale("log")
    p_ax.set_xlabel("counted parameters")
    p_ax.set_ylabel(r"optimal $p_{\rm AR}$")
    p_ax.set_ylim(-0.03, 0.63)

    best_gain_ax.set_title("Best nonzero curriculum gain")
    best_gain_ax.set_xscale("log")
    best_gain_ax.set_xlabel("counted parameters")
    best_gain_ax.set_ylabel("best nonzero validation-NELBO gain")

    fig.suptitle(
        "AR→BD curriculum at pure-BD compute-optimal token counts\n"
        "same optimizer steps and clean tokens within each model size",
        fontsize=14,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"{output_stem}.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        action="append",
        choices=[target.spec.label for target in FIXED_STEP_TARGETS],
        help="Analyze only these model sizes; repeat for multiple sizes.",
    )
    parser.add_argument(
        "--output-stem",
        default="fixed_steps_curriculum",
        help="Figure stem and prefix for tabular artifacts.",
    )
    args = parser.parse_args()
    selected_sizes = set(args.size) if args.size else None
    targets = [
        target
        for target in FIXED_STEP_TARGETS
        if selected_sizes is None or target.spec.label in selected_sizes
    ]
    rows = load_rows(targets)
    summaries = summarize(rows, targets)
    write_csv(RESULTS / f"{args.output_stem}_all_points.csv", rows)
    write_csv(RESULTS / f"{args.output_stem}_summary.csv", summaries)
    payload = {
        "comparison_mode": "fixed_total_steps",
        "allocation_source": ALLOCATION_SOURCE,
        "hypothesis": (
            "At fixed model size and pure-BD-optimal total steps, an interior "
            "AR curriculum improves validation NELBO and its gain persists "
            "with increasing model size."
        ),
        "points": rows,
        "summaries": summaries,
    }
    atomic_json_dump(payload, RESULTS / f"{args.output_stem}_summary.json")
    make_figure(rows, summaries, targets, args.output_stem)
    for row in summaries:
        print(
            f"{row['size']:>5} steps={row['total_steps']:>6} "
            f"lr={row['selected_lr']:.4g} p*={row['optimal_p_ar']:.1f} "
            f"gain={row['best_gain']:+.5f}"
        )


if __name__ == "__main__":
    main()
