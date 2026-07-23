"""Run the complete feasible IsoFLOP by 3-spaced learning-rate grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget", type=float, action="append")
    parser.add_argument("--size", action="append")
    parser.add_argument("--lr", type=float, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def is_complete(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def run_grid(args):
    budgets = tuple(args.budget) if args.budget else COMPUTE_BUDGETS
    selected_sizes = set(args.size) if args.size else None
    learning_rates = tuple(args.lr) if args.lr else LEARNING_RATES
    runs = []
    for budget in budgets:
        for spec in MODEL_SPECS:
            if selected_sizes is not None and spec.label not in selected_sizes:
                continue
            if not is_feasible(budget, spec):
                continue
            for lr in learning_rates:
                run_dir = (
                    RESULTS_DIR
                    / "runs"
                    / budget_slug(budget)
                    / spec.label
                    / f"lr_{lr_slug(lr)}"
                )
                runs.append(
                    {
                        "budget": budget,
                        "spec": spec,
                        "lr": lr,
                        "steps": steps_for(budget, spec),
                        "run_dir": run_dir,
                        "result": run_dir / "result.json",
                    }
                )
    return runs


def execute_run(run, device):
    run["run_dir"].mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--budget",
        str(run["budget"]),
        "--size",
        run["spec"].label,
        "--lr",
        str(run["lr"]),
        "--output",
        str(run["result"]),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (run["run_dir"] / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return run, process


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    runs = run_grid(args)
    print(
        f"planned_runs={len(runs)} "
        f"planned_steps={sum(run['steps'] for run in runs):,}"
    )
    for run in runs:
        print(
            f"{budget_slug(run['budget']):>5} {run['spec'].label:>4} "
            f"steps={run['steps']:>6} lr={run['lr']:.4g}"
        )
    if args.dry_run:
        return

    todo = [run for run in runs if args.force or not is_complete(run["result"])]
    print(
        f"remaining_runs={len(todo)} skipped={len(runs) - len(todo)} "
        f"workers={args.workers}",
        flush=True,
    )
    if not todo:
        return

    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute_run, run, args.device): run
            for run in todo
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            run, process = future.result()
            if process.returncode:
                failures.append(str(run["run_dir"]))
                print(
                    f"[{completed}/{len(todo)}] FAILED {run['run_dir']}: "
                    f"{process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            result = json.loads(run["result"].read_text())
            print(
                f"[{completed}/{len(todo)}] C={run['budget']:.0e} "
                f"N={run['spec'].label} lr={run['lr']:.4g} "
                f"val={result['val_nelbo']:.5f} "
                f"run_s={result['duration_seconds']:.1f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {failures}")
    print(f"all runs complete in {(time.monotonic() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
