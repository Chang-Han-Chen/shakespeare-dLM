"""Plot the selected high-scale learning rates by objective and phase."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from config import ROOT


RESULTS = ROOT / "results_followup_comparison"
FIGURES = ROOT / "figures_followup_comparison"
MIN_PARAMS = 3_000_000


def selected_rows(
    path: Path,
    *,
    series: str,
    learning_rate_field: str,
) -> list[dict]:
    with path.open() as handle:
        source = list(csv.DictReader(handle))
    rows = []
    seen = set()
    for row in source:
        if row["status"] != "complete" or int(row["n_params"]) < MIN_PARAMS:
            continue
        key = (float(row["budget"]), int(row["n_params"]))
        if key in seen:
            raise ValueError(f"Duplicate selected run for {series}: {key}")
        seen.add(key)
        rows.append(
            {
                "series": series,
                "budget": float(row["budget"]),
                "size": row["size"],
                "n_params": int(row["n_params"]),
                "learning_rate": float(row[learning_rate_field]),
                "wandb_run_id": row["wandb_run_id"],
            }
        )
    return rows


def main() -> None:
    rows = []
    rows.extend(
        selected_rows(
            ROOT / "results_scaleup" / "best_runs.csv",
            series="pure BD",
            learning_rate_field="learning_rate",
        )
    )
    rows.extend(
        selected_rows(
            ROOT / "results_ar" / "best_runs.csv",
            series="pure AR",
            learning_rate_field="learning_rate",
        )
    )
    rows.extend(
        selected_rows(
            ROOT / "results_matched_p_ar_0p4" / "best_runs.csv",
            series="p_AR=0.4 AR phase",
            learning_rate_field="ar_learning_rate",
        )
    )
    rows.extend(
        selected_rows(
            ROOT / "results_matched_p_ar_0p4" / "best_runs.csv",
            series="p_AR=0.4 BD phase",
            learning_rate_field="bd_learning_rate",
        )
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "learning_rate_comparison.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    styles = {
        "pure BD": ("o", "#4b4c9a"),
        "pure AR": ("s", "#2b8c6b"),
        "p_AR=0.4 AR phase": ("^", "#d18b2c"),
        "p_AR=0.4 BD phase": ("v", "#b35c1e"),
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for series, (marker, color) in styles.items():
        selected = [row for row in rows if row["series"] == series]
        axis.scatter(
            [row["n_params"] for row in selected],
            [row["learning_rate"] for row in selected],
            marker=marker,
            color=color,
            alpha=0.72,
            label=series,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("model parameters N")
    axis.set_ylabel("selected peak learning rate")
    axis.set_title("High-scale objective and curriculum learning rates")
    axis.set_yticks([3e-4, 9e-4, 2.7e-3])
    axis.set_yticklabels(["3e-4", "9e-4", "2.7e-3"])
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"learning_rate_by_model_size.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
