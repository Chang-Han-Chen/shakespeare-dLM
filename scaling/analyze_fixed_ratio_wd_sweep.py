"""Analyze the fixed-D/N TinyShakespeare AR-weight-decay sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import BLOCK_LEN, MODEL_BY_LABEL, ROOT
from run_fixed_ratio_wd_sweep import (
    AR_WEIGHT_DECAYS,
    DEFAULT_TOKEN_PARAMETER_RATIO,
    P_AR_VALUES,
    SIZE_TO_LR,
    build_runs,
    ratio_slug,
    results_dir_for,
)
from train import atomic_json_dump


FIGURES_DIR = ROOT / "figures_fixed_ratio"


def load_rows(
    token_parameter_ratio: float,
    block_len: int = BLOCK_LEN,
) -> list[dict]:
    rows = []
    for run in build_runs(token_parameter_ratio, block_len):
        path = run["result"]
        if not path.exists():
            raise RuntimeError(f"Missing result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError(f"Incomplete result: {path}")
        rows.append(
            {
                "size": run["size"],
                "n_params": run["spec"].n_params,
                "steps": run["steps"],
                "clean_tokens": run["clean_tokens"],
                "realized_d_over_n": run["realized_d_over_n"],
                "block_len": block_len,
                "p_ar": run["p_ar"],
                "ar_weight_decay": run["ar_weight_decay"],
                "bd_weight_decay": 0.1,
                "learning_rate": run["lr"],
                "val_nelbo": float(result["val_nelbo"]),
                "val_masked_ce_t0.5": float(result["val_masked_ce_t0.5"]),
                "result_path": str(path),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for size in SIZE_TO_LR:
        size_rows = [row for row in rows if row["size"] == size]
        baseline = next(row for row in size_rows if row["p_ar"] == 0.0)
        for p_ar in P_AR_VALUES:
            candidates = [row for row in size_rows if row["p_ar"] == p_ar]
            best = min(candidates, key=lambda row: row["val_nelbo"])
            output.append(
                {
                    "size": size,
                    "n_params": best["n_params"],
                    "steps": best["steps"],
                    "clean_tokens": best["clean_tokens"],
                    "realized_d_over_n": best["realized_d_over_n"],
                    "learning_rate": best["learning_rate"],
                    "p_ar": p_ar,
                    "selected_ar_weight_decay": best["ar_weight_decay"],
                    "minimum_val_nelbo": best["val_nelbo"],
                    "gain_vs_pure_bd": (
                        baseline["val_nelbo"] - best["val_nelbo"]
                    ),
                    "selected_result_path": best["result_path"],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    rows: list[dict],
    summaries: list[dict],
    token_parameter_ratio: float,
    block_len: int,
) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(AR_WEIGHT_DECAYS)))

    for ax, size in zip(axes[0], SIZE_TO_LR):
        size_rows = [row for row in rows if row["size"] == size]
        baseline = next(row for row in size_rows if row["p_ar"] == 0.0)
        for color, weight_decay in zip(colors, AR_WEIGHT_DECAYS):
            mixed = sorted(
                [
                    row
                    for row in size_rows
                    if row["ar_weight_decay"] == weight_decay
                ],
                key=lambda row: row["p_ar"],
            )
            x = [0.0] + [row["p_ar"] for row in mixed]
            y = [baseline["val_nelbo"]] + [row["val_nelbo"] for row in mixed]
            ax.plot(
                x,
                y,
                marker="o",
                lw=1.7,
                color=color,
                label=f"wd_AR={weight_decay:g}",
            )
        ax.set_title(
            f"{size}: N={baseline['n_params']:,}, "
            f"steps={baseline['steps']:,}"
        )
        ax.set_xlabel(r"$p_{\rm AR}$")
        ax.set_ylabel("validation diffusion NELBO")
        ax.set_xticks(P_AR_VALUES)
    axes[0, 0].legend(fontsize=8)

    for color, size in zip(("#3973a5", "#c6683d"), SIZE_TO_LR):
        points = sorted(
            [row for row in summaries if row["size"] == size],
            key=lambda row: row["p_ar"],
        )
        axes[1, 0].plot(
            [row["p_ar"] for row in points],
            [row["gain_vs_pure_bd"] for row in points],
            marker="o",
            lw=2,
            color=color,
            label=size,
        )
        nonzero = [row for row in points if row["p_ar"] > 0]
        axes[1, 1].plot(
            [row["p_ar"] for row in nonzero],
            [row["selected_ar_weight_decay"] for row in nonzero],
            marker="o",
            lw=2,
            color=color,
            label=size,
        )

    axes[1, 0].axhline(0.0, color="black", lw=1, alpha=0.65)
    axes[1, 0].set_title("Best WD at each curriculum fraction")
    axes[1, 0].set_xlabel(r"$p_{\rm AR}$")
    axes[1, 0].set_ylabel(r"$L(p=0)-\min_{\rm wd}L(p)$")
    axes[1, 0].set_xticks(P_AR_VALUES)
    axes[1, 0].legend()

    axes[1, 1].set_title("Selected AR-phase weight decay")
    axes[1, 1].set_xlabel(r"$p_{\rm AR}$")
    axes[1, 1].set_ylabel(r"best $\mathrm{wd}_{\rm AR}$")
    axes[1, 1].set_xticks(P_AR_VALUES[1:])
    axes[1, 1].set_yticks(AR_WEIGHT_DECAYS)
    axes[1, 1].legend()

    fig.suptitle(
        "TinyShakespeare "
        + r"AR$\rightarrow$BD at fixed "
        + f"$D/N={token_parameter_ratio:g}$, block={block_len}"
        "\nidentical total steps and clean tokens within each model",
        fontsize=14,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES_DIR
            / (
                f"fixed_dn{ratio_slug(token_parameter_ratio)}"
                f"_bl{block_len}"
                f"_ar_wd.{extension}"
            ),
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_TOKEN_PARAMETER_RATIO,
    )
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    args = parser.parse_args()
    if args.ratio <= 0:
        raise ValueError("ratio must be positive")
    if args.block_len < 1 or 256 % args.block_len:
        raise ValueError("block-len must be a positive divisor of 256")
    results_dir = results_dir_for(args.ratio, args.block_len)
    rows = load_rows(args.ratio, args.block_len)
    summaries = summarize(rows)
    write_csv(results_dir / "all_points.csv", rows)
    write_csv(results_dir / "best_by_p.csv", summaries)
    payload = {
        "token_parameter_ratio_target": args.ratio,
        "block_len": args.block_len,
        "comparison_mode": "fixed total optimizer steps and clean tokens",
        "rows": rows,
        "best_by_p": summaries,
    }
    atomic_json_dump(payload, results_dir / "summary.json")
    make_figure(rows, summaries, args.ratio, args.block_len)
    for row in summaries:
        print(
            f"{row['size']:>4} p={row['p_ar']:.1f} "
            f"wd={row['selected_ar_weight_decay']} "
            f"loss={row['minimum_val_nelbo']:.5f} "
            f"gain={row['gain_vs_pure_bd']:+.5f}"
        )


if __name__ == "__main__":
    main()
