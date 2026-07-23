"""Analyze the per-p_AR AR learning-rate sweep at C=3e14, N=0.5M."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parent
DIAGNOSTIC = ROOT / "results_diagnostics" / "ar_lr_sweep_3e14_0p5M"
P_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
CENTER_AR_LR = 2.7e-3


def load(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for p_ar in P_VALUES:
        center = load(
            ROOT
            / "results_curriculum"
            / "runs"
            / "3e14"
            / "0.5M"
            / f"p_ar_{p_ar:.1f}".replace(".", "p")
            / "result.json"
        )
        rows.append(
            {
                "p_ar": p_ar,
                "ar_learning_rate": CENTER_AR_LR,
                "bd_learning_rate": CENTER_AR_LR,
                "val_nelbo": center["val_nelbo"],
                "source": "original_center",
            }
        )
        for ar_lr, slug in ((9e-4, "9p0e-04"), (8.1e-3, "8p1e-03")):
            result = load(
                DIAGNOSTIC
                / "runs"
                / f"p_ar_{p_ar:.1f}".replace(".", "p")
                / f"ar_lr_{slug}"
                / "result.json"
            )
            rows.append(
                {
                    "p_ar": p_ar,
                    "ar_learning_rate": ar_lr,
                    "bd_learning_rate": CENTER_AR_LR,
                    "val_nelbo": result["val_nelbo"],
                    "source": "diagnostic",
                }
            )
    extension = load(
        DIAGNOSTIC
        / "runs"
        / "p_ar_0p2"
        / "ar_lr_3p0e-04"
        / "result.json"
    )
    rows.append(
        {
            "p_ar": 0.2,
            "ar_learning_rate": 3e-4,
            "bd_learning_rate": CENTER_AR_LR,
            "val_nelbo": extension["val_nelbo"],
            "source": "lower_boundary_extension",
        }
    )

    selected = []
    for p_ar in P_VALUES:
        candidates = sorted(
            [row for row in rows if row["p_ar"] == p_ar],
            key=lambda row: row["ar_learning_rate"],
        )
        best_index = min(
            range(len(candidates)),
            key=lambda index: candidates[index]["val_nelbo"],
        )
        if best_index in (0, len(candidates) - 1):
            raise RuntimeError(f"Unbracketed AR LR at p={p_ar}")
        best = dict(candidates[best_index])
        best["left_ar_lr"] = candidates[best_index - 1]["ar_learning_rate"]
        best["right_ar_lr"] = candidates[best_index + 1]["ar_learning_rate"]
        best["discrete_curvature"] = (
            candidates[best_index - 1]["val_nelbo"]
            + candidates[best_index + 1]["val_nelbo"]
            - 2 * best["val_nelbo"]
        )
        selected.append(best)

    DIAGNOSTIC.mkdir(parents=True, exist_ok=True)
    with (DIAGNOSTIC / "ar_lr_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["p_ar"], row["ar_learning_rate"])))
    with (DIAGNOSTIC / "selected_ar_lr.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    colors = plt.cm.viridis([0.05, 0.22, 0.39, 0.56, 0.73, 0.9])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    lr_axis, p_axis = axes
    for color, p_ar in zip(colors, P_VALUES):
        candidates = sorted(
            [row for row in rows if row["p_ar"] == p_ar],
            key=lambda row: row["ar_learning_rate"],
        )
        lr_axis.plot(
            [row["ar_learning_rate"] for row in candidates],
            [row["val_nelbo"] for row in candidates],
            marker="o",
            color=color,
            label=rf"$p_{{\rm AR}}={p_ar:.1f}$",
        )
        best = next(row for row in selected if row["p_ar"] == p_ar)
        lr_axis.scatter(
            best["ar_learning_rate"],
            best["val_nelbo"],
            marker="*",
            s=100,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
    lr_axis.set_xscale("log")
    lr_axis.set_xticks([3e-4, 9e-4, 2.7e-3, 8.1e-3])
    lr_axis.set_xticklabels(["3e-4", "9e-4", "2.7e-3", "8.1e-3"])
    lr_axis.xaxis.set_minor_formatter(NullFormatter())
    lr_axis.set_xlabel("AR peak learning rate")
    lr_axis.set_ylabel("validation BD NELBO")
    lr_axis.set_title("Full-run AR-LR brackets")
    lr_axis.legend(frameon=False, fontsize=7, ncol=2)
    lr_axis.grid(alpha=0.2)

    original = [
        next(
            row["val_nelbo"]
            for row in rows
            if row["p_ar"] == p and row["ar_learning_rate"] == CENTER_AR_LR
        )
        for p in P_VALUES
    ]
    tuned = [row["val_nelbo"] for row in selected]
    p_axis.plot(P_VALUES, original, marker="o", label="AR LR 0.0027")
    p_axis.plot(P_VALUES, tuned, marker="o", label="per-cell tuned AR LR")
    p_axis.set_xlabel(r"$p_{\rm AR}$")
    p_axis.set_ylabel("validation BD NELBO")
    p_axis.set_title("Effect on the fixed-model curriculum curve")
    p_axis.legend(frameon=False, fontsize=8)
    p_axis.grid(alpha=0.2)

    fig.suptitle(r"AR LR sweep at $C=3\times10^{14}$, $N=0.5$M", fontsize=14)
    fig.tight_layout()
    for extension_name in ("png", "pdf"):
        fig.savefig(
            DIAGNOSTIC / f"ar_lr_sweep.{extension_name}",
            bbox_inches="tight",
        )
    plt.close(fig)

    summary = {
        "budget": 3e14,
        "size": "0.5M",
        "bd_learning_rate": CENTER_AR_LR,
        "selected": selected,
        "best_cell": min(selected, key=lambda row: row["val_nelbo"]),
        "conclusion": (
            "AR LR 0.0027 remains selected for five cells. p_AR=0.2 "
            "selects 0.0009 by less than 0.001 NELBO; per-cell tuning "
            "does not materially change the L(p) curve."
        ),
    }
    (DIAGNOSTIC / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for row in selected:
        print(
            f"p={row['p_ar']:.1f} AR_LR={row['ar_learning_rate']:.1e} "
            f"val={row['val_nelbo']:.5f} curvature={row['discrete_curvature']:.5f}"
        )


if __name__ == "__main__":
    main()
