"""Analyze the fixed-D/N 10M ClimbMix block-size comparison."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt

from config import ROOT
from run_fixed_dn40_10m_blocks import (
    BLOCK_LENGTHS,
    LEARNING_RATE,
    P_AR_VALUES,
    RESULTS_DIR,
    SIZE,
    SPEC,
    TOKEN_PARAMETER_RATIO,
    build_runs,
)
from train import atomic_json_dump


FIGURES_DIR = ROOT / "figures_fixed_dn40_10M"


def load_rows() -> list[dict]:
    rows = []
    for run in build_runs():
        if not run["result"].exists():
            raise RuntimeError(f"Missing result: {run['result']}")
        result = json.loads(run["result"].read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError(f"Incomplete result: {run['result']}")
        rows.append(
            {
                "size": SIZE,
                "n_params": SPEC.n_params,
                "steps": run["steps"],
                "clean_tokens": run["clean_tokens"],
                "realized_d_over_n": run["realized_d_over_n"],
                "block_len": run["block_len"],
                "p_ar": run["p_ar"],
                "learning_rate": LEARNING_RATE,
                "weight_decay": 0.1,
                "val_nelbo": float(result["val_nelbo"]),
                "val_masked_ce_t0.5": float(result["val_masked_ce_t0.5"]),
                "result_path": str(run["result"]),
            }
        )
    for block_len in BLOCK_LENGTHS:
        points = [row for row in rows if row["block_len"] == block_len]
        baseline = next(row for row in points if row["p_ar"] == 0.0)
        for row in points:
            row["gain_vs_pure_bd"] = (
                baseline["val_nelbo"] - row["val_nelbo"]
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict]) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    colors = {4: "#3973a5", 32: "#c6683d"}
    markers = {4: "o", 32: "s"}
    for block_len in BLOCK_LENGTHS:
        points = sorted(
            [row for row in rows if row["block_len"] == block_len],
            key=lambda row: row["p_ar"],
        )
        label = f"block={block_len}"
        axes[0].plot(
            [row["p_ar"] for row in points],
            [row["val_nelbo"] for row in points],
            marker=markers[block_len],
            lw=2,
            color=colors[block_len],
            label=label,
        )
        axes[1].plot(
            [row["p_ar"] for row in points],
            [row["gain_vs_pure_bd"] for row in points],
            marker=markers[block_len],
            lw=2,
            color=colors[block_len],
            label=label,
        )
    axes[0].set_title("Validation diffusion NELBO")
    axes[0].set_xlabel(r"$p_{\rm AR}$")
    axes[0].set_ylabel("validation NELBO")
    axes[0].set_xticks(P_AR_VALUES)
    axes[0].legend()
    axes[1].set_title("Curriculum gain within block size")
    axes[1].set_xlabel(r"$p_{\rm AR}$")
    axes[1].set_ylabel(r"$L(p=0)-L(p)$")
    axes[1].set_xticks(P_AR_VALUES)
    axes[1].axhline(0.0, color="black", lw=1, alpha=0.65)
    axes[1].legend()
    fig.suptitle(
        f"ClimbMix {SIZE} at fixed D/N={TOKEN_PARAMETER_RATIO:g}\n"
        "same optimizer steps and clean tokens within each block size",
        fontsize=14,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES_DIR / f"fixed_dn40_10M_blocks.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_csv(RESULTS_DIR / "all_points.csv", rows)
    summaries = []
    for block_len in BLOCK_LENGTHS:
        points = [row for row in rows if row["block_len"] == block_len]
        best = min(points, key=lambda row: row["val_nelbo"])
        summaries.append(
            {
                "block_len": block_len,
                "optimal_p_ar": best["p_ar"],
                "minimum_val_nelbo": best["val_nelbo"],
                "gain_vs_pure_bd": best["gain_vs_pure_bd"],
            }
        )
    atomic_json_dump(
        {
            "size": SIZE,
            "n_params": SPEC.n_params,
            "token_parameter_ratio_target": TOKEN_PARAMETER_RATIO,
            "rows": rows,
            "summaries": summaries,
        },
        RESULTS_DIR / "summary.json",
    )
    make_figure(rows)
    for row in summaries:
        print(
            f"block={row['block_len']:>2} p*={row['optimal_p_ar']:.1f} "
            f"loss={row['minimum_val_nelbo']:.5f} "
            f"gain={row['gain_vs_pure_bd']:+.5f}"
        )


if __name__ == "__main__":
    main()
