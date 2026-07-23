"""Run the complete feasible IsoFLOP x 3-spaced-LR grid on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def run_grid(args) -> list[dict]:
    budgets = tuple(args.budget) if args.budget else COMPUTE_BUDGETS
    sizes = set(args.size) if args.size else None
    learning_rates = tuple(args.lr) if args.lr else LEARNING_RATES
    runs = []
    for budget in budgets:
        for spec in MODEL_SPECS:
            if sizes is not None and spec.label not in sizes:
                continue
            steps = steps_for(budget, spec)
            if not is_feasible(budget, spec):
                continue
            for lr in learning_rates:
                run_dir = RESULTS_DIR / "runs" / budget_slug(budget) / spec.label / f"lr_{lr_slug(lr)}"
                runs.append(
                    {
                        "budget": budget,
                        "spec": spec,
                        "lr": lr,
                        "steps": steps,
                        "run_dir": run_dir,
                        "result": run_dir / "result.json",
                    }
                )
    return runs


def execute_run(run, device):
    result_path = run["result"]
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
        str(result_path),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (run["run_dir"] / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    return run, process


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    runs = run_grid(args)
    total_steps = sum(run["steps"] for run in runs)
    print(f"planned_runs={len(runs)} planned_steps={total_steps:,}")
    for run in runs:
        print(
            f"{budget_slug(run['budget']):>5} {run['spec'].label:>5} "
            f"steps={run['steps']:>6} lr={run['lr']:.4g}"
        )
    if args.dry_run:
        return

    todo = [run for run in runs if args.force or not is_complete(run["result"])]
    skipped = len(runs) - len(todo)
    print(f"remaining_runs={len(todo)} skipped={skipped} workers={args.workers}", flush=True)
    if not todo:
        print("all runs already complete")
        return

    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, run in enumerate(todo, start=1):
            print(
                f"[{index}/{len(todo)}] queue C={run['budget']:.0e} "
                f"N={run['spec'].label} lr={run['lr']:.4g} steps={run['steps']}",
                flush=True,
            )
            futures[executor.submit(execute_run, run, args.device)] = index

        completed = 0
        for future in as_completed(futures):
            completed += 1
            run, process = future.result()
            result_path = run["result"]
            if process.returncode != 0:
                failures.append(str(run["run_dir"]))
                print(
                    f"[{completed}/{len(todo)} complete] FAILED {run['run_dir']} "
                    f"rc={process.returncode}: {process.stderr[-1000:]}",
                    flush=True,
                )
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            elapsed = time.monotonic() - started
            print(
                f"[{completed}/{len(todo)} complete] C={run['budget']:.0e} "
                f"N={run['spec'].label} lr={run['lr']:.4g} "
                f"val={result['val_nelbo']:.5f} run_s={result['duration_seconds']:.1f} "
                f"elapsed_h={elapsed / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {failures}")
    print(f"all runs complete in {(time.monotonic() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
