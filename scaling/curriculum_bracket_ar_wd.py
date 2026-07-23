"""Adaptively bracket AR-phase weight decay for the 0.8M curriculum runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    COMPUTE_BUDGETS,
    MODEL_BY_LABEL,
    RESULTS_DIR as BASE_RESULTS_DIR,
    ROOT,
    WEIGHT_DECAY,
    budget_slug,
    lr_slug,
)
from curriculum_config import (
    FIGURES_DIR,
    P_AR_VALUES,
    RESULTS_DIR,
    is_feasible,
    p_ar_slug,
    steps_for,
)
from curriculum_run_sweep import baseline_learning_rate, is_complete


SIZE = "0.8M"
SPEC = MODEL_BY_LABEL[SIZE]
BASE_AR_WEIGHT_DECAY = WEIGHT_DECAY
BD_WEIGHT_DECAY = WEIGHT_DECAY
DEFAULT_MAX_AR_WEIGHT_DECAY = 12.8
SWEEP_DIR = RESULTS_DIR / "ar_wd_sweep" / SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-ar-weight-decay",
        type=float,
        default=DEFAULT_MAX_AR_WEIGHT_DECAY,
    )
    return parser.parse_args()


def wd_slug(weight_decay: float) -> str:
    return f"{weight_decay:.1e}".replace("+", "")


def points():
    return [
        (budget, p_ar)
        for budget in COMPUTE_BUDGETS
        for p_ar in P_AR_VALUES
        if is_feasible(budget, SPEC, p_ar)
    ]


def baseline_path(budget: float, p_ar: float) -> Path:
    lr = baseline_learning_rate(budget, SPEC)
    return (
        RESULTS_DIR
        / "runs"
        / p_ar_slug(p_ar)
        / budget_slug(budget)
        / SIZE
        / f"lr_{lr_slug(lr)}"
        / "result.json"
    )


def point_dir(budget: float, p_ar: float) -> Path:
    return SWEEP_DIR / p_ar_slug(p_ar) / budget_slug(budget)


def result_path(budget: float, p_ar: float, ar_weight_decay: float) -> Path:
    return (
        point_dir(budget, p_ar)
        / f"ar_wd_{wd_slug(ar_weight_decay)}"
        / "result.json"
    )


def pure_bd_path(budget: float) -> Path:
    lr = baseline_learning_rate(budget, SPEC)
    return (
        BASE_RESULTS_DIR
        / "runs"
        / budget_slug(budget)
        / SIZE
        / f"lr_{lr_slug(lr)}"
        / "result.json"
    )


def read_complete(path: Path):
    if not is_complete(path):
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if not math.isfinite(float(row["val_nelbo"])):
        return None
    return row


def completed_results(budget: float, p_ar: float):
    baseline = read_complete(baseline_path(budget, p_ar))
    if baseline is None:
        raise RuntimeError(
            f"Missing 0.1 baseline for C={budget:.0e}, p={p_ar:.1f}"
        )
    if not math.isclose(float(baseline["weight_decay"]), BASE_AR_WEIGHT_DECAY):
        raise RuntimeError(f"Unexpected baseline weight decay: {baseline_path(budget, p_ar)}")
    baseline = dict(baseline)
    baseline["ar_weight_decay"] = BASE_AR_WEIGHT_DECAY
    baseline["bd_weight_decay"] = BD_WEIGHT_DECAY
    baseline["result_path"] = str(baseline_path(budget, p_ar))
    rows = [baseline]

    for path in point_dir(budget, p_ar).glob("ar_wd_*/result.json"):
        row = read_complete(path)
        if row is None:
            continue
        if not math.isclose(float(row["bd_weight_decay"]), BD_WEIGHT_DECAY):
            raise RuntimeError(f"BD weight decay changed at {path}")
        if not math.isclose(
            float(row["learning_rate"]),
            baseline_learning_rate(budget, SPEC),
        ):
            raise RuntimeError(f"Learning rate changed at {path}")
        row["result_path"] = str(path)
        rows.append(row)

    unique = {}
    for row in rows:
        unique[float(row["ar_weight_decay"])] = row
    return [unique[key] for key in sorted(unique)]


def bracket_path(budget: float, p_ar: float) -> Path:
    return point_dir(budget, p_ar) / "ar_wd_bracket.json"


def inspect_point(
    budget: float,
    p_ar: float,
    max_ar_weight_decay: float,
):
    rows = completed_results(budget, p_ar)
    best_index = min(
        range(len(rows)),
        key=lambda index: float(rows[index]["val_nelbo"]),
    )
    best = rows[best_index]

    if len(rows) >= 2 and best_index < len(rows) - 1:
        left = rows[best_index - 1] if best_index > 0 else None
        right = rows[best_index + 1]
        status = "locally_bracketed" if left is not None else "lower_boundary"
        payload = {
            "status": status,
            "budget": budget,
            "p_ar": p_ar,
            "size": SIZE,
            "learning_rate": baseline_learning_rate(budget, SPEC),
            "bd_weight_decay": BD_WEIGHT_DECAY,
            "selected_ar_weight_decay": float(best["ar_weight_decay"]),
            "selected_val_nelbo": float(best["val_nelbo"]),
            "left_ar_weight_decay": (
                float(left["ar_weight_decay"]) if left is not None else None
            ),
            "left_val_nelbo": (
                float(left["val_nelbo"]) if left is not None else None
            ),
            "right_ar_weight_decay": float(right["ar_weight_decay"]),
            "right_val_nelbo": float(right["val_nelbo"]),
            "discrete_curvature": (
                float(left["val_nelbo"])
                - 2 * float(best["val_nelbo"])
                + float(right["val_nelbo"])
                if left is not None
                else None
            ),
            "evaluated_ar_weight_decays": [
                float(row["ar_weight_decay"]) for row in rows
            ],
            "evaluated_val_nelbo": [
                float(row["val_nelbo"]) for row in rows
            ],
            "selected_result_path": best["result_path"],
        }
        path = bracket_path(budget, p_ar)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload, None

    candidate = float(rows[-1]["ar_weight_decay"]) * 2
    if candidate > max_ar_weight_decay * (1 + 1e-12):
        payload = {
            "status": "upper_boundary",
            "budget": budget,
            "p_ar": p_ar,
            "size": SIZE,
            "learning_rate": baseline_learning_rate(budget, SPEC),
            "bd_weight_decay": BD_WEIGHT_DECAY,
            "selected_ar_weight_decay": float(best["ar_weight_decay"]),
            "selected_val_nelbo": float(best["val_nelbo"]),
            "evaluated_ar_weight_decays": [
                float(row["ar_weight_decay"]) for row in rows
            ],
            "evaluated_val_nelbo": [
                float(row["val_nelbo"]) for row in rows
            ],
            "selected_result_path": best["result_path"],
        }
        path = bracket_path(budget, p_ar)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload, None
    return None, candidate


def execute_run(
    budget: float,
    p_ar: float,
    ar_weight_decay: float,
    device: str,
):
    path = result_path(budget, p_ar, ar_weight_decay)
    path.parent.mkdir(parents=True, exist_ok=True)
    lr = baseline_learning_rate(budget, SPEC)
    command = [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        "--budget",
        str(budget),
        "--size",
        SIZE,
        "--p-ar",
        str(p_ar),
        "--lr",
        str(lr),
        "--lr-source",
        "pure_bd_full_run_local_optimum",
        "--ar-weight-decay",
        str(ar_weight_decay),
        "--bd-weight-decay",
        str(BD_WEIGHT_DECAY),
        "--output",
        str(path),
        "--device",
        device,
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (path.parent / "train.log").write_text(
        process.stdout
        + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"AR-WD run failed at {path}: {process.stderr[-1200:]}"
        )
    row = json.loads(path.read_text(encoding="utf-8"))
    return budget, p_ar, ar_weight_decay, row


def save_outputs(brackets):
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    brackets = sorted(brackets, key=lambda row: (row["budget"], row["p_ar"]))
    summary = {
        "size": SIZE,
        "n_params": SPEC.n_params,
        "search_policy": (
            "AR-only AdamW weight decay doubles upward from 0.1 until the "
            "measured validation NELBO winner has a higher-WD neighbor"
        ),
        "bd_weight_decay": BD_WEIGHT_DECAY,
        "brackets": brackets,
    }
    summary_path = SWEEP_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = SWEEP_DIR / "brackets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "budget",
            "p_ar",
            "status",
            "learning_rate",
            "selected_ar_weight_decay",
            "selected_val_nelbo",
            "left_ar_weight_decay",
            "left_val_nelbo",
            "right_ar_weight_decay",
            "right_val_nelbo",
            "discrete_curvature",
            "selected_result_path",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(brackets)

    budgets = sorted({float(row["budget"]) for row in brackets})
    colors = dict(
        zip(
            COMPUTE_BUDGETS,
            plt.cm.viridis(np.linspace(0.08, 0.9, len(COMPUTE_BUDGETS))),
        )
    )
    fig, axes = plt.subplots(1, len(budgets), figsize=(13.1, 3.8), squeeze=False)
    for ax, budget in zip(axes.flat, budgets):
        control = read_complete(pure_bd_path(budget))
        if control is None:
            raise RuntimeError(f"Missing pure-BD control: {pure_bd_path(budget)}")
        selected = sorted(
            (
                row
                for row in brackets
                if float(row["budget"]) == budget
            ),
            key=lambda row: row["p_ar"],
        )
        p_values = [0.0] + [float(row["p_ar"]) for row in selected]
        losses = [float(control["val_nelbo"])] + [
            float(row["selected_val_nelbo"]) for row in selected
        ]
        color = colors[budget]
        ax.plot(p_values, losses, marker="o", ms=5, lw=1.8, color=color)
        ax.axhline(
            control["val_nelbo"],
            color="#777777",
            ls=":",
            lw=1,
            alpha=0.8,
        )
        best_index = min(range(len(losses)), key=losses.__getitem__)
        best_p = p_values[best_index]
        best_loss = losses[best_index]
        ax.scatter(
            [best_p],
            [best_loss],
            marker="*",
            s=105,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        gain = float(control["val_nelbo"]) - best_loss
        ax.text(
            0.97,
            0.96,
            f"best p={best_p:.1f}\ngain={gain:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
        ax.set_title(f"C={budget:.0e}")
        ax.set_xticks((0.0,) + P_AR_VALUES)
        ax.set_xlim(-0.025, 0.525)
        ax.set_xlabel(r"AR fraction $p_{\mathrm{AR}}$")
        ax.set_ylabel("validation diffusion NELBO")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Fixed model: {SIZE} ({SPEC.n_params:,} counted parameters)",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.008,
        "Each panel holds model size and FLOP budget fixed; "
        "the dotted line is pure BD and the star is the best measured mix.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    figure_dir = FIGURES_DIR / "fixed_model_l_vs_p"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "fixed_N_0p8M.png"
    pdf = figure_dir / "fixed_N_0p8M.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return summary_path, csv_path, png, pdf


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_ar_weight_decay < BASE_AR_WEIGHT_DECAY * 2:
        raise ValueError("max AR weight decay must permit at least the 0.2 run")
    grid = points()
    print(
        f"points={len(grid)} size={SIZE} "
        f"search=[{BASE_AR_WEIGHT_DECAY:g},...,{args.max_ar_weight_decay:g}] "
        f"workers={args.workers}",
        flush=True,
    )
    for budget, p_ar in grid:
        print(
            f"C={budget:.0e} p={p_ar:.1f} "
            f"steps={steps_for(budget, SPEC, p_ar)} "
            f"lr={baseline_learning_rate(budget, SPEC):.4g}",
            flush=True,
        )
    if args.dry_run:
        return

    started = time.monotonic()
    while True:
        brackets = []
        extensions = []
        for budget, p_ar in grid:
            bracket, candidate = inspect_point(
                budget,
                p_ar,
                args.max_ar_weight_decay,
            )
            if bracket is not None:
                brackets.append(bracket)
            else:
                extensions.append((budget, p_ar, candidate))
        if not extensions:
            outputs = save_outputs(brackets)
            statuses = {}
            for bracket in brackets:
                statuses[bracket["status"]] = statuses.get(bracket["status"], 0) + 1
            print(
                f"complete points={len(brackets)} statuses={statuses} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.3f}",
                flush=True,
            )
            for path in outputs:
                print(f"saved {path}", flush=True)
            return

        print(
            f"running_extensions={len(extensions)} "
            f"wd_values={sorted({candidate for _, _, candidate in extensions})}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    execute_run,
                    budget,
                    p_ar,
                    candidate,
                    args.device,
                ): (budget, p_ar, candidate)
                for budget, p_ar, candidate in extensions
            }
            for future in as_completed(futures):
                budget, p_ar, candidate, row = future.result()
                print(
                    f"done C={budget:.0e} p={p_ar:.1f} "
                    f"ar_wd={candidate:g} val={row['val_nelbo']:.5f} "
                    f"run_s={row['duration_seconds']:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
