"""Tune pure-BD LR at each compute-optimal fixed-step target."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import RESULTS_DIR, ROOT, lr_slug
from fixed_steps_config import ALLOCATION_SOURCE, FIXED_STEP_TARGETS
from train import atomic_json_dump


RESULTS = ROOT / "results_fixed_steps"
MAX_EXTENSIONS = 4
MIN_LR = 1e-5
MAX_LR = 0.1
RESTRICTED_LRS = {
    "4M": (3e-4, 9e-4),
    "8M": (3e-4, 9e-4),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def point_dir(target):
    return RESULTS / "baseline" / target.spec.label


def result_path(target, lr):
    return point_dir(target) / f"lr_{lr_slug(lr)}" / "result.json"


def nearest_original_bracket(target):
    candidates = []
    for path in RESULTS_DIR.glob(f"runs/*/{target.spec.label}/lr_bracket.json"):
        row = json.loads(path.read_text())
        distance = abs(math.log(float(row["budget"]) / target.predicted_compute))
        candidates.append((distance, path, row))
    if not candidates:
        raise FileNotFoundError(f"No original LR bracket for {target.spec.label}")
    _, path, row = min(candidates, key=lambda item: item[0])
    return path, row


def initial_lrs(target):
    if target.spec.label in RESTRICTED_LRS:
        return RESTRICTED_LRS[target.spec.label]
    _, bracket = nearest_original_bracket(target)
    center = float(bracket["selected_lr"])
    return (center / 3.0, center, center * 3.0)


def completed_results(target):
    rows = []
    for path in point_dir(target).glob("lr_*/result.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "complete" and math.isfinite(row["val_nelbo"]):
            rows.append(row)
    return sorted(rows, key=lambda row: row["learning_rate"])


def execute(run, device):
    target, lr = run
    output = result_path(target, lr)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--budget",
        str(target.predicted_compute),
        "--steps",
        str(target.total_steps),
        "--size",
        target.spec.label,
        "--lr",
        str(lr),
        "--output",
        str(output),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output.parent / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return run, process


def run_parallel(runs, args, label):
    todo = [
        run
        for run in runs
        if args.force or not complete(result_path(run[0], run[1]))
    ]
    print(f"{label}: planned={len(runs)} remaining={len(todo)}", flush=True)
    if args.dry_run:
        for target, lr in todo:
            print(
                f"N={target.spec.label} steps={target.total_steps} "
                f"C*={target.predicted_compute:.3e} lr={lr:.6g}"
            )
        return
    failures = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, run, args.device): run
            for run in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run, process = future.result()
            target, lr = run
            if process.returncode:
                failures.append(str(result_path(target, lr)))
                print(
                    f"[{index}/{len(todo)}] FAILED N={target.spec.label} "
                    f"lr={lr:.6g}: {process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            row = json.loads(result_path(target, lr).read_text())
            print(
                f"[{index}/{len(todo)}] N={target.spec.label} lr={lr:.6g} "
                f"val={row['val_nelbo']:.5f} run_s={row['duration_seconds']:.1f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise RuntimeError(f"{len(failures)} fixed-step baseline runs failed: {failures}")


def inspect_target(target):
    rows = completed_results(target)
    base_count = len(initial_lrs(target))
    if len(rows) < base_count:
        return None, None
    if target.spec.label in RESTRICTED_LRS:
        best = min(rows, key=lambda row: row["val_nelbo"])
        other = max(rows, key=lambda row: row["val_nelbo"])
        source_path, source = nearest_original_bracket(target)
        return {
            "status": "restricted_two_point",
            "comparison_mode": "fixed_total_steps",
            "selection_note": (
                "User-requested restricted comparison: only 3e-4 and 9e-4; "
                "local convexity is not claimed"
            ),
            "allocation_source": ALLOCATION_SOURCE,
            "size": target.spec.label,
            "n_params": target.spec.n_params,
            "predicted_compute": target.predicted_compute,
            "realized_full_bd_compute": target.realized_full_bd_compute,
            "total_steps": target.total_steps,
            "clean_tokens": target.clean_tokens,
            "selected_lr": best["learning_rate"],
            "selected_val_nelbo": best["val_nelbo"],
            "other_lr": other["learning_rate"],
            "other_val_nelbo": other["val_nelbo"],
            "source_lr_bracket": str(source_path),
            "source_lr": source["selected_lr"],
            "extensions": 0,
        }, None
    best_index = min(range(len(rows)), key=lambda index: rows[index]["val_nelbo"])
    best = rows[best_index]
    if 0 < best_index < len(rows) - 1:
        left, right = rows[best_index - 1], rows[best_index + 1]
        curvature = (
            left["val_nelbo"] - 2.0 * best["val_nelbo"] + right["val_nelbo"]
        )
        if curvature <= 0:
            raise RuntimeError(f"Non-convex LR neighborhood for {target.spec.label}")
        source_path, source = nearest_original_bracket(target)
        return {
            "status": "locally_bracketed",
            "comparison_mode": "fixed_total_steps",
            "allocation_source": ALLOCATION_SOURCE,
            "size": target.spec.label,
            "n_params": target.spec.n_params,
            "predicted_compute": target.predicted_compute,
            "realized_full_bd_compute": target.realized_full_bd_compute,
            "total_steps": target.total_steps,
            "clean_tokens": target.clean_tokens,
            "selected_lr": best["learning_rate"],
            "selected_val_nelbo": best["val_nelbo"],
            "left_lr": left["learning_rate"],
            "left_val_nelbo": left["val_nelbo"],
            "right_lr": right["learning_rate"],
            "right_val_nelbo": right["val_nelbo"],
            "discrete_curvature": curvature,
            "source_lr_bracket": str(source_path),
            "source_lr": source["selected_lr"],
            "extensions": len(rows) - base_count,
        }, None
    if len(rows) - base_count >= MAX_EXTENSIONS:
        raise RuntimeError(f"Could not bracket fixed-step LR for {target.spec.label}")
    candidate = (
        best["learning_rate"] / 3.0
        if best_index == 0
        else best["learning_rate"] * 3.0
    )
    if not MIN_LR <= candidate <= MAX_LR:
        raise RuntimeError(f"LR extension outside safety range: {candidate}")
    return None, candidate


def write_targets():
    payload = {
        "allocation_source": ALLOCATION_SOURCE,
        "targets": [
            {
                "size": target.spec.label,
                "n_params": target.spec.n_params,
                "predicted_compute": target.predicted_compute,
                "realized_full_bd_compute": target.realized_full_bd_compute,
                "total_steps": target.total_steps,
                "clean_tokens": target.clean_tokens,
            }
            for target in FIXED_STEP_TARGETS
        ],
    }
    atomic_json_dump(payload, RESULTS / "targets.json")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    write_targets()
    base_runs = [
        (target, lr)
        for target in FIXED_STEP_TARGETS
        for lr in initial_lrs(target)
    ]
    run_parallel(base_runs, args, "base LR sweep")
    if args.dry_run:
        return
    while True:
        brackets, extensions = [], []
        for target in FIXED_STEP_TARGETS:
            bracket, candidate = inspect_target(target)
            if bracket is not None:
                brackets.append((target, bracket))
            elif candidate is not None:
                extensions.append((target, candidate))
        for target, bracket in brackets:
            atomic_json_dump(bracket, point_dir(target) / "lr_bracket.json")
        if not extensions:
            print(f"locally bracketed {len(brackets)} fixed-step baselines")
            return
        run_parallel(extensions, args, "LR boundary extensions")


if __name__ == "__main__":
    main()
