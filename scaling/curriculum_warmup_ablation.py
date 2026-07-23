"""Quick paired test of whether AdamW-reset BD training needs re-warmup."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODEL_BY_LABEL, ROOT, budget_slug
from curriculum_config import RESULTS_DIR, p_ar_slug, steps_for
from curriculum_run_sweep import baseline_learning_rate


P_AR = 0.3
WARMUP_FRACTIONS = (0.0, 0.05)
POINTS = (
    (1e13, "0.01M", "tiny/high-LR"),
    (3e13, "0.1M", "short"),
    (1e14, "0.04M", "medium"),
    (3e14, "0.2M", "wider"),
)
OUTPUT_DIR = RESULTS_DIR / "warmup_ablation"


def result_path(budget, size, warmup_fraction):
    return (
        OUTPUT_DIR
        / p_ar_slug(P_AR)
        / budget_slug(budget)
        / size
        / f"bd_warmup_{warmup_fraction:.2f}"
        / "result.json"
    )


def is_complete(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def run_one(budget, size, label, warmup_fraction):
    spec = MODEL_BY_LABEL[size]
    lr = baseline_learning_rate(budget, spec)
    path = result_path(budget, size, warmup_fraction)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        "--budget",
        str(budget),
        "--size",
        size,
        "--p-ar",
        str(P_AR),
        "--lr",
        str(lr),
        "--lr-source",
        "pure_bd_full_run_local_optimum_warmup_ablation",
        "--bd-warmup-fraction",
        str(warmup_fraction),
        "--output",
        str(path),
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (path.parent / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    if process.returncode:
        raise RuntimeError(f"Ablation failed at {path}: {process.stderr[-1000:]}")
    row = json.loads(path.read_text(encoding="utf-8"))
    row["ablation_label"] = label
    return row


def main():
    jobs = [
        (budget, size, label, warmup)
        for budget, size, label in POINTS
        for warmup in WARMUP_FRACTIONS
    ]
    print(f"paired_points={len(POINTS)} runs={len(jobs)} p_ar={P_AR}")
    for budget, size, label, warmup in jobs:
        spec = MODEL_BY_LABEL[size]
        print(
            f"{label:>12} C={budget:.0e} N={size} "
            f"steps={steps_for(budget, spec, P_AR)} "
            f"lr={baseline_learning_rate(budget, spec):.3g} "
            f"bd_warmup={warmup:.0%}"
        )

    rows = []
    started = time.monotonic()
    todo = []
    for job in jobs:
        path = result_path(job[0], job[1], job[3])
        if is_complete(path):
            row = json.loads(path.read_text(encoding="utf-8"))
            row["ablation_label"] = job[2]
            rows.append(row)
        else:
            todo.append(job)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_one, *job): job for job in todo}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"complete {row['ablation_label']} warmup={row['bd_warmup_fraction']:.0%} "
                f"val={row['val_nelbo']:.5f}",
                flush=True,
            )

    comparisons = []
    for budget, size, label in POINTS:
        pair = {
            row["bd_warmup_fraction"]: row
            for row in rows
            if float(row["budget"]) == budget and row["size"] == size
        }
        no_warmup = pair[0.0]
        warmup = pair[0.05]
        comparisons.append(
            {
                "label": label,
                "budget": budget,
                "size": size,
                "steps": warmup["steps"],
                "bd_steps": warmup["bd_steps"],
                "learning_rate": warmup["learning_rate"],
                "val_nelbo_no_warmup": no_warmup["val_nelbo"],
                "val_nelbo_5pct_warmup": warmup["val_nelbo"],
                "delta_no_minus_warmup": (
                    no_warmup["val_nelbo"] - warmup["val_nelbo"]
                ),
            }
        )

    wins = sum(row["delta_no_minus_warmup"] > 0 for row in comparisons)
    mean_delta = sum(row["delta_no_minus_warmup"] for row in comparisons) / len(comparisons)
    summary = {
        "p_ar": P_AR,
        "paired_points": len(comparisons),
        "five_percent_warmup_wins": wins,
        "mean_delta_no_warmup_minus_5pct": mean_delta,
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = OUTPUT_DIR / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"5pct_wins={wins}/{len(comparisons)} mean_delta={mean_delta:+.5f}")
    print(f"saved {summary_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
