"""Summarize the fast C=1e15, N=0.5M curriculum diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_diagnostics" / "low_p_1e15_0p5M"


def load(path):
    return json.loads(path.read_text())


def main() -> None:
    baseline = next(
        row
        for row in csv.DictReader((ROOT / "results" / "best_runs.csv").open())
        if float(row["budget"]) == 1e15 and row["size"] == "0.5M"
    )
    baseline_loss = float(baseline["val_nelbo"])
    original_p01 = load(
        ROOT
        / "results_curriculum"
        / "runs"
        / "1e15"
        / "0.5M"
        / "p_ar_0p1"
        / "result.json"
    )
    standard = [
        load(RESULTS / f"p_ar_0p{suffix}.json")
        for suffix in ("03", "06", "08")
    ]
    tuned = load(RESULTS / "p_ar_0p06_bd_lr_2p7e-3.json")
    low_lr = load(RESULTS / "p_ar_0p06_bd_lr_9e-4.json")
    no_decay_006 = load(RESULTS / "p_ar_0p06_ar_no_decay.json")
    no_decay_010 = load(RESULTS / "p_ar_0p10_ar_no_decay.json")
    random_tail = load(RESULTS / "random_init_common_bd_tail_p_ar_0p06.json")
    bd_prefix = load(RESULTS / "compute_matched_bd_prefix_p_ar_0p06.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), dpi=150)
    final_axis, trace_axis = axes
    ratios = [0.0] + [row["ar_tokens_per_parameter"] for row in standard]
    losses = [baseline_loss] + [row["val_nelbo"] for row in standard]
    ratios.append(
        original_p01["ar_steps"] * original_p01["batch_size"]
        * original_p01["sequence_length"] / original_p01["n_params"]
    )
    losses.append(original_p01["val_nelbo"])
    final_axis.plot(ratios, losses, marker="o", color="#35618d", label="BD LR 0.0081")
    final_axis.scatter(
        tuned["ar_tokens_per_parameter"],
        tuned["val_nelbo"],
        marker="D",
        s=55,
        color="#d17c37",
        label="20× AR, BD LR 0.0027",
        zorder=4,
    )
    final_axis.scatter(
        [
            no_decay_006["ar_tokens_per_parameter"],
            no_decay_010["ar_tokens_per_parameter"],
        ],
        [no_decay_006["val_nelbo"], no_decay_010["val_nelbo"]],
        marker="s",
        facecolors="none",
        edgecolors="#9a4f78",
        s=58,
        label="no AR-end decay",
        zorder=4,
    )
    final_axis.axvline(20, color="black", ls=":", lw=1, label="assumed AR optimum")
    final_axis.axhline(baseline_loss, color="#777777", ls="--", lw=1)
    final_axis.set_xlabel(r"AR tokens per counted parameter $D_{\rm AR}/N$")
    final_axis.set_ylabel("final validation BD NELBO")
    final_axis.set_title("Final loss versus AR exposure")
    final_axis.legend(frameon=False, fontsize=8)
    final_axis.grid(alpha=0.2)

    traces = (
        ("AR 20×, BD LR .0081", standard[1], "#35618d"),
        ("AR 20×, BD LR .0027", tuned, "#d17c37"),
        ("AR 20×, BD LR .0009", low_lr, "#9a4f78"),
        ("random init", random_tail, "#555555"),
        ("compute-matched BD prefix", bd_prefix, "#5a9b55"),
    )
    for label, row, color in traces:
        trace_axis.plot(
            [point["bd_step"] for point in row["validation_trace"]],
            [point["val_nelbo"] for point in row["validation_trace"]],
            marker="o",
            color=color,
            label=label,
        )
    trace_axis.scatter(
        [int(baseline["steps"])],
        [baseline_loss],
        marker="*",
        s=120,
        color="black",
        label="uninterrupted pure BD final",
        zorder=5,
    )
    trace_axis.set_xlabel("BD tail steps")
    trace_axis.set_ylabel("validation BD NELBO")
    trace_axis.set_title("Matched 9,723-step BD tail")
    trace_axis.set_ylim(5.05, 9.2)
    trace_axis.legend(frameon=False, fontsize=7)
    trace_axis.grid(alpha=0.2)

    fig.suptitle(r"Diagnosing AR $\rightarrow$ BD at $C=10^{15}$, $N=0.5$M", fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(RESULTS / f"diagnostic_summary.{extension}", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "baseline_pure_bd": baseline_loss,
        "standard_low_p": [
            {
                "p_ar": row["p_ar"],
                "ar_tokens_per_parameter": row["ar_tokens_per_parameter"],
                "val_nelbo": row["val_nelbo"],
            }
            for row in standard
        ],
        "original_p_0.1": original_p01["val_nelbo"],
        "p_0.06_bd_lr_bracket": {
            "0.0009": low_lr["val_nelbo"],
            "0.0027": tuned["val_nelbo"],
            "0.0081": standard[1]["val_nelbo"],
        },
        "ar_end_decay_ablation": {
            "p_0.06_with_decay": standard[1]["val_nelbo"],
            "p_0.06_without_decay": no_decay_006["val_nelbo"],
            "p_0.10_with_decay": original_p01["val_nelbo"],
            "p_0.10_without_decay": no_decay_010["val_nelbo"],
        },
        "matched_tail": {
            "random_init": random_tail["val_nelbo"],
            "compute_matched_bd_prefix": bd_prefix["val_nelbo"],
            "ar_20x_lr_0.0081": standard[1]["val_nelbo"],
            "ar_20x_lr_0.0027": tuned["val_nelbo"],
        },
        "interpretation": (
            "AR improves an identical BD tail, but is less valuable per unit "
            "prefix compute than uninterrupted BD at this data-rich point. "
            "The 20x AR allocation does not close the gap; a 3x lower BD LR "
            "helps modestly and is locally preferred."
        ),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
