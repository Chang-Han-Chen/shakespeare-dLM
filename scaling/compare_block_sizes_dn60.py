"""Compare block lengths 4 and 32 for the fixed-D/N=60 curriculum sweep."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from analyze_fixed_ratio_wd_sweep import load_rows, summarize
from config import ROOT
from run_fixed_ratio_wd_sweep import P_AR_VALUES, SIZE_TO_LR
from train import atomic_json_dump


TOKEN_PARAMETER_RATIO = 60.0
BLOCK_LENGTHS = (4, 32)
FIGURES_DIR = ROOT / "figures_fixed_ratio"
OUTPUT_CSV = ROOT / "results_fixed_dn60_block_comparison.csv"
OUTPUT_JSON = ROOT / "results_fixed_dn60_block_comparison.json"


def main() -> None:
    rows = []
    for block_len in BLOCK_LENGTHS:
        for row in summarize(load_rows(TOKEN_PARAMETER_RATIO, block_len)):
            rows.append({"block_len": block_len, **row})

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_json_dump(
        {
            "token_parameter_ratio": TOKEN_PARAMETER_RATIO,
            "block_lengths": list(BLOCK_LENGTHS),
            "comparison_mode": (
                "fixed model size, optimizer steps, and clean tokens; "
                "best AR weight decay selected independently at each point"
            ),
            "best_by_p": rows,
        },
        OUTPUT_JSON,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    styles = {
        4: ("#3973a5", "o"),
        32: ("#c6683d", "s"),
    }
    for ax, size in zip(axes, SIZE_TO_LR):
        for block_len in BLOCK_LENGTHS:
            color, marker = styles[block_len]
            points = sorted(
                [
                    row
                    for row in rows
                    if row["size"] == size
                    and row["block_len"] == block_len
                ],
                key=lambda row: row["p_ar"],
            )
            ax.plot(
                [row["p_ar"] for row in points],
                [row["gain_vs_pure_bd"] for row in points],
                marker=marker,
                lw=2,
                color=color,
                label=f"block={block_len}",
            )
        ax.axhline(0.0, color="black", lw=1, alpha=0.65)
        ax.set_title(f"{size}: curriculum gain")
        ax.set_xlabel(r"$p_{\rm AR}$")
        ax.set_xticks(P_AR_VALUES)
        ax.legend()
    axes[0].set_ylabel(r"$L(p=0)-\min_{\rm wd}L(p)$")
    fig.suptitle(
        r"TinyShakespeare at fixed $D/N=60$"
        "\npositive values mean tuned AR curriculum improves over pure BD",
        fontsize=14,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES_DIR
            / f"compare_bl4_bl32_dn60.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
