"""Run the complete feasible IsoFLOP by 3-spaced learning-rate grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BATCH_SIZE,
    COMPUTE_BUDGETS,
    LEARNING_RATES,
    MODEL_SPECS,
    RESULTS_DIR,
    ROOT,
    WANDB_PROJECT,
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
    parser.add_argument(
        "--devices",
        help="Comma-separated devices for independent jobs, e.g. cuda:0,cuda:1",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "cudnn", "efficient", "math"),
        default="auto",
    )
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
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


def execute_run(
    run,
    device,
    compile_model,
    attention_backend,
    wandb_project,
    wandb_entity,
):
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
        "--batch-size",
        str(BATCH_SIZE),
        "--attention-backend",
        attention_backend,
    ]
    if compile_model:
        command.append("--compile")
    if wandb_project:
        command.extend(
            [
                "--wandb-project",
                wandb_project,
                "--wandb-name",
                (
                    f"{budget_slug(run['budget'])}-{run['spec'].label}-"
                    f"lr-{lr_slug(run['lr'])}"
                ),
                "--wandb-group",
                budget_slug(run["budget"]),
            ]
        )
    if wandb_entity:
        command.extend(["--wandb-entity", wandb_entity])
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (run["run_dir"] / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return run, process


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    devices = (
        tuple(part.strip() for part in args.devices.split(",") if part.strip())
        if args.devices
        else (args.device,)
    )
    if not devices:
        raise ValueError("At least one device is required")
    if args.workers > len(devices):
        raise ValueError(
            f"workers={args.workers} would oversubscribe {len(devices)} devices"
        )
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
    executors = [ThreadPoolExecutor(max_workers=1) for _ in range(args.workers)]
    try:
        futures = {
            executors[index % args.workers].submit(
                execute_run,
                run,
                devices[index % args.workers],
                args.compile,
                args.attention_backend,
                args.wandb_project,
                args.wandb_entity,
            ): run
            for index, run in enumerate(todo)
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
    finally:
        for executor in executors:
            executor.shutdown()
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {failures}")
    print(f"all runs complete in {(time.monotonic() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
