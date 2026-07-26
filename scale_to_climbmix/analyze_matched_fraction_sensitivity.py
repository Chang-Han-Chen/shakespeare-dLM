"""Analyze fixed-size matched-step curriculum-fraction sensitivity."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from config import ROOT


BUDGET = 3e17
SIZE = "19.1M"
RESULTS = ROOT / "results_matched_p_ar_fraction_sensitivity"
FIGURES = ROOT / "figures_matched_p_ar_fraction_sensitivity"


def load(path: Path) -> dict:
    row = json.loads(path.read_text())
    if row["status"] != "complete":
        raise ValueError(f"Incomplete run: {path}")
    return row


def one(pattern: str) -> dict:
    paths = list(ROOT.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one result for {pattern}, found {paths}")
    row = load(paths[0])
    row["_result_path"] = str(paths[0].relative_to(ROOT))
    return row


def main() -> None:
    bd = one("results_scaleup/runs/3e17/19.1M/lr_9.0e-04/result.json")
    curricula = {
        0.1: one(
            "results_matched_p_ar_fraction_sensitivity/runs/"
            "3e17/p0p10/19.1M/*/result.json"
        ),
        0.15: one(
            "results_matched_p_ar_0p15/runs/"
            "3e17/19.1M/*/result.json"
        ),
        0.2: one(
            "results_matched_p_ar_fraction_sensitivity/runs/"
            "3e17/p0p20/19.1M/*/result.json"
        ),
        0.4: one(
            "results_matched_p_ar_0p4/runs/"
            "3e17/19.1M/*/result.json"
        ),
    }
    if float(bd["budget"]) != BUDGET or bd["size"] != SIZE:
        raise ValueError("Unexpected pure-BD comparison point")
    rows = [
        {
            "p_ar": 0.0,
            "n_params": int(bd["n_params"]),
            "total_steps": int(bd["steps"]),
            "clean_tokens": int(bd["clean_tokens"]),
            "ar_learning_rate": None,
            "bd_learning_rate": float(bd["learning_rate"]),
            "val_nelbo": float(bd["val_nelbo"]),
            "realized_flops": int(bd["realized_flops"]),
            "realized_to_nominal_compute": (
                float(bd["realized_flops"]) / BUDGET
            ),
            "wandb_run_id": bd["wandb_run_id"],
            "result_path": bd["_result_path"],
        }
    ]
    for p_ar, row in curricula.items():
        if (
            float(row["budget"]) != BUDGET
            or row["size"] != SIZE
            or not math.isclose(float(row["p_ar"]), p_ar)
        ):
            raise ValueError(f"Unexpected curriculum comparison point: {row}")
        rows.append(
            {
                "p_ar": p_ar,
                "n_params": int(row["n_params"]),
                "total_steps": int(row["total_steps"]),
                "clean_tokens": int(row["clean_tokens"]),
                "ar_learning_rate": float(row["ar_learning_rate"]),
                "bd_learning_rate": float(row["bd_learning_rate"]),
                "val_nelbo": float(row["val_nelbo"]),
                "realized_flops": int(row["realized_flops"]),
                "realized_to_nominal_compute": float(
                    row["realized_to_nominal_compute"]
                ),
                "wandb_run_id": row["wandb_run_id"],
                "result_path": row["_result_path"],
            }
        )
    rows.sort(key=lambda row: row["p_ar"])
    reference = rows[0]
    for row in rows:
        for key in ("n_params", "total_steps", "clean_tokens"):
            if row[key] != reference[key]:
                raise ValueError(f"{key} is not matched: {rows}")
        row["absolute_nelbo_improvement_over_bd"] = (
            reference["val_nelbo"] - row["val_nelbo"]
        )
        row["relative_nelbo_improvement_over_bd"] = (
            reference["val_nelbo"] - row["val_nelbo"]
        ) / reference["val_nelbo"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row["val_nelbo"])
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "comparison": (
                    "fixed_3e17_19.1M_steps_tokens_lr_fraction_sensitivity"
                ),
                "single_seed": True,
                "rows": rows,
                "best_measured": best,
                "nelbo_range_across_curricula": (
                    max(row["val_nelbo"] for row in rows[1:])
                    - min(row["val_nelbo"] for row in rows[1:])
                ),
                "note": (
                    "All points use batch 128, the same model, steps, clean "
                    "tokens, BD LR 9e-4, and seed 1337. Curriculum AR phases "
                    "use LR 2.7e-3. Differences below 1% are treated as "
                    "inconclusive under the study decision rule."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    p_values = [row["p_ar"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    loss_ax, compute_ax = axes
    loss_ax.plot(
        p_values,
        [row["val_nelbo"] for row in rows],
        "o-",
        color="#b35c1e",
    )
    loss_ax.axhline(
        reference["val_nelbo"],
        color="gray",
        linestyle=":",
        linewidth=1,
    )
    loss_ax.set_xlabel("AR step fraction p_AR")
    loss_ax.set_ylabel("validation BD NELBO")
    loss_ax.set_title("Fixed 3e17, 19.1M, steps and tokens")

    compute_ax.plot(
        p_values,
        [100 * row["realized_to_nominal_compute"] for row in rows],
        "s-",
        color="#2b8c6b",
    )
    compute_ax.set_xlabel("AR step fraction p_AR")
    compute_ax.set_ylabel("realized / nominal FLOPs (%)")
    compute_ax.set_title("Cheaper causal phase")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"matched_fraction_sensitivity.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
