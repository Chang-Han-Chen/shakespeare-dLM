"""Compare pure BD, pure AR, and matched-step p_AR=0.4 allocations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ROOT


BD_SUMMARY = ROOT / "results_scaleup" / "summary.json"
AR_SUMMARY = ROOT / "results_ar" / "summary.json"
MATCHED_SUMMARY = ROOT / "results_matched_p_ar_0p4" / "summary.json"
RESULTS = ROOT / "results_followup_comparison"
FIGURES = ROOT / "figures_followup_comparison"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def profiles_by_budget(summary: dict) -> dict[float, dict]:
    return {
        float(row["budget"]): row
        for row in summary["profiles"]
    }


def main() -> None:
    bd_summary = load(BD_SUMMARY)
    bd = profiles_by_budget(bd_summary)
    ar = profiles_by_budget(load(AR_SUMMARY))
    matched = profiles_by_budget(load(MATCHED_SUMMARY))
    bd_scaleup_budgets = {
        float(row["budget"]) for row in bd_summary["scaleup_profiles"]
    }
    budgets = sorted(set(bd) & set(ar) & set(matched))
    if not budgets:
        raise ValueError("No common completed budgets")

    rows = []
    for budget in budgets:
        bd_row, ar_row, matched_row = bd[budget], ar[budget], matched[budget]
        rows.append(
            {
                "budget": budget,
                "bd_batch_size": (
                    bd_summary["batch_regimes"]["scaleup"]
                    if budget in bd_scaleup_budgets
                    else bd_summary["batch_regimes"]["historical"]
                ),
                "ar_batch_size": 128,
                "matched_batch_size": 128,
                "batch_sizes_match": budget in bd_scaleup_budgets,
                "bd_n_opt": bd_row["n_opt"],
                "bd_d_opt": bd_row["d_opt"],
                "bd_d_over_n": bd_row["tokens_per_parameter"],
                "bd_val_nelbo": bd_row["loss_min"],
                "ar_n_opt": ar_row["n_opt"],
                "ar_d_opt": ar_row["d_opt"],
                "ar_d_over_n": ar_row["tokens_per_parameter"],
                "ar_val_ce": ar_row["loss_min"],
                "matched_n_opt": matched_row["n_opt"],
                "matched_d_opt": matched_row["d_opt"],
                "matched_d_over_n": matched_row["tokens_per_parameter"],
                "matched_val_nelbo": matched_row["loss_min"],
                "matched_nelbo_improvement": (
                    bd_row["loss_min"] - matched_row["loss_min"]
                ),
                "matched_relative_nelbo_improvement": (
                    (bd_row["loss_min"] - matched_row["loss_min"])
                    / bd_row["loss_min"]
                ),
                "matched_realized_compute": matched_row[
                    "realized_compute_opt"
                ],
                "matched_realized_to_nominal_compute": matched_row[
                    "realized_to_nominal_compute"
                ],
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (RESULTS / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "comparison": "pure_bd_vs_pure_ar_vs_matched_p_ar_0p4",
                "budgets": budgets,
                "rows": rows,
                "note": (
                    "AR cross-entropy and BD NELBO are distinct objectives; "
                    "only pure-BD and matched final NELBO are subtracted. "
                    "Pure-BD points through 1e16 use the refined historical "
                    "batch-64 study, so batch-matched frontier comparisons "
                    "begin at 3e16."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    compute = np.array(budgets)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    allocation_ax, ratio_ax, gain_ax = axes
    styles = {
        "pure BD": ("o-", "#4b4c9a"),
        "pure AR": ("s-", "#2b8c6b"),
        "matched p_AR=0.4": ("^-", "#b35c1e"),
    }
    for label, key in (
        ("pure BD", "bd_n_opt"),
        ("pure AR", "ar_n_opt"),
        ("matched p_AR=0.4", "matched_n_opt"),
    ):
        style, color = styles[label]
        allocation_ax.loglog(
            compute,
            [row[key] for row in rows],
            style,
            color=color,
            label=label,
        )
    allocation_ax.set_xlabel("nominal training FLOPs C")
    allocation_ax.set_ylabel("compute-optimal parameters N*")
    allocation_ax.set_title("Optimal model size")
    allocation_ax.legend(frameon=False, fontsize=8)

    for label, key in (
        ("pure BD", "bd_d_over_n"),
        ("pure AR", "ar_d_over_n"),
        ("matched p_AR=0.4", "matched_d_over_n"),
    ):
        style, color = styles[label]
        ratio_ax.loglog(
            compute,
            [row[key] for row in rows],
            style,
            color=color,
            label=label,
        )
    ratio_ax.axhline(20, color="gray", linestyle=":", linewidth=1)
    ratio_ax.set_xlabel("nominal training FLOPs C")
    ratio_ax.set_ylabel("optimal D*/N*")
    ratio_ax.set_title("Token allocation")

    gains = np.array(
        [100 * row["matched_relative_nelbo_improvement"] for row in rows]
    )
    batch_matched = np.array([row["batch_sizes_match"] for row in rows])
    gain_ax.semilogx(compute, gains, "-", color="#b35c1e")
    gain_ax.scatter(
        compute[~batch_matched],
        gains[~batch_matched],
        facecolors="none",
        edgecolors="#b35c1e",
        label="BD batch 64 vs matched batch 128",
    )
    gain_ax.scatter(
        compute[batch_matched],
        gains[batch_matched],
        color="#b35c1e",
        label="both batch 128",
    )
    gain_ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    gain_ax.set_xlabel("nominal training FLOPs C")
    gain_ax.set_ylabel("matched NELBO improvement over pure BD (%)")
    gain_ax.set_title("Matched-step frontier effect")
    gain_ax.legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"followup_isoflop_comparison.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
