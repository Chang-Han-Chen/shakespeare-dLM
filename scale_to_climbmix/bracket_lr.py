"""Extend boundary LR winners by 3x until every optimum is locally convex."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    COMPUTE_BUDGETS,
    LEARNING_RATES,
    MODEL_SPECS,
    RESULTS_DIR,
    ROOT,
    budget_slug,
    is_feasible,
    lr_slug,
    steps_for,
)


MAX_EXTENSIONS_PER_POINT = 4
MIN_LR = min(LEARNING_RATES) / 81
MAX_LR = max(LEARNING_RATES) * 27


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def point_dir(budget, spec):
    return RESULTS_DIR / "runs" / budget_slug(budget) / spec.label


def result_path(budget, spec, lr):
    return point_dir(budget, spec) / f"lr_{lr_slug(lr)}" / "result.json"


def completed_results(budget, spec):
    rows = []
    for path in point_dir(budget, spec).glob("lr_*/result.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "complete" and math.isfinite(row["val_nelbo"]):
            rows.append(row)
    return sorted(rows, key=lambda row: row["learning_rate"])


def run_full_point(budget, spec, lr, device):
    path = result_path(budget, spec, lr)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--budget",
        str(budget),
        "--size",
        spec.label,
        "--lr",
        str(lr),
        "--output",
        str(path),
        "--device",
        device,
    ]
    print(
        f"extend C={budget:.0e} N={spec.label} lr={lr:.6g} "
        f"steps={steps_for(budget, spec)}",
        flush=True,
    )
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (path.parent / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    if process.returncode:
        raise RuntimeError(f"LR extension failed at {path}: {process.stderr[-1200:]}")


def inspect_point(budget, spec):
    rows = completed_results(budget, spec)
    if len(rows) < len(LEARNING_RATES):
        raise RuntimeError(f"Base sweep incomplete at C={budget:.0e}, N={spec.label}")
    best_index = min(range(len(rows)), key=lambda index: rows[index]["val_nelbo"])
    best = rows[best_index]
    extensions = len(rows) - len(LEARNING_RATES)
    if 0 < best_index < len(rows) - 1:
        left, right = rows[best_index - 1], rows[best_index + 1]
        curvature = (
            left["val_nelbo"] - 2 * best["val_nelbo"] + right["val_nelbo"]
        )
        if curvature <= 0:
            raise RuntimeError(
                f"Non-convex LR neighborhood at C={budget:.0e}, N={spec.label}"
            )
        bracket = {
            "status": "locally_bracketed",
            "budget": budget,
            "size": spec.label,
            "selected_lr": best["learning_rate"],
            "selected_val_nelbo": best["val_nelbo"],
            "left_lr": left["learning_rate"],
            "left_val_nelbo": left["val_nelbo"],
            "right_lr": right["learning_rate"],
            "right_val_nelbo": right["val_nelbo"],
            "discrete_curvature": curvature,
            "extensions": extensions,
        }
        path = point_dir(budget, spec) / "lr_bracket.json"
        path.write_text(json.dumps(bracket, indent=2, sort_keys=True) + "\n")
        return bracket, None

    if extensions >= MAX_EXTENSIONS_PER_POINT:
        raise RuntimeError(f"Could not bracket LR at C={budget:.0e}, N={spec.label}")
    candidate = best["learning_rate"] / 3 if best_index == 0 else best["learning_rate"] * 3
    if not MIN_LR <= candidate <= MAX_LR:
        raise RuntimeError(f"LR exceeded safety range at C={budget:.0e}, N={spec.label}")
    return None, candidate


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    points = [
        (budget, spec)
        for budget in COMPUTE_BUDGETS
        for spec in MODEL_SPECS
        if is_feasible(budget, spec)
    ]
    while True:
        brackets, extensions = [], []
        for budget, spec in points:
            bracket, candidate = inspect_point(budget, spec)
            if bracket:
                brackets.append(bracket)
            else:
                extensions.append((budget, spec, candidate))
        if not extensions:
            print(f"locally bracketed {len(brackets)} IsoFLOP datapoints")
            return
        print(f"running {len(extensions)} extensions with {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_full_point, budget, spec, lr, args.device)
                for budget, spec, lr in extensions
            ]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
