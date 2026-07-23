"""Run p_AR branches with identical total steps at each model size."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ROOT
from curriculum_config import P_AR_VALUES
from fixed_steps_config import FIXED_STEP_TARGETS, split_fixed_steps


RESULTS = ROOT / "results_fixed_steps"
SHARED = RESULTS / "shared_ar"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--size",
        action="append",
        help="Restrict to one or more model-size labels",
    )
    return parser.parse_args()


def complete(path):
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def selected_lr(target):
    path = RESULTS / "baseline" / target.spec.label / "lr_bracket.json"
    row = json.loads(path.read_text())
    if row["status"] not in {
        "locally_bracketed",
        "restricted_two_point",
        "user_truncated_boundary",
        "user_selected_near_tie",
    }:
        raise ValueError(f"Invalid LR bracket: {path}")
    return float(row["selected_lr"])


def output_path(target, p_ar):
    slug = f"{p_ar:.1f}".replace(".", "p")
    return RESULTS / "runs" / target.spec.label / f"p_ar_{slug}" / "result.json"


def execute(run, device):
    target, p_ar = run
    output = output_path(target, p_ar)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        "--budget",
        str(target.predicted_compute),
        "--total-steps",
        str(target.total_steps),
        "--size",
        target.spec.label,
        "--p-ar",
        str(p_ar),
        "--trunk-dir",
        str(SHARED / target.spec.label),
        "--bd-lr",
        str(selected_lr(target)),
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


def main() -> None:
    args = parse_args()
    runs = [
        (target, p_ar)
        for target in FIXED_STEP_TARGETS
        for p_ar in P_AR_VALUES
    ]
    if args.size:
        selected = set(args.size)
        known = {target.spec.label for target in FIXED_STEP_TARGETS}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown model sizes: {sorted(unknown)}")
        runs = [run for run in runs if run[0].spec.label in selected]
    # Approximate wall time with the exact AR/BD phase FLOPs, then launch
    # longest-first to avoid leaving a few 8M branches as a serial tail.
    runs.sort(
        key=lambda run: (
            split_fixed_steps(run[0].total_steps, run[1])[0]
            * run[0].spec.autoregressive_training_flops_per_clean_token
            + split_fixed_steps(run[0].total_steps, run[1])[1]
            * run[0].spec.training_flops_per_clean_token
        ),
        reverse=True,
    )
    todo = [run for run in runs if not complete(output_path(*run))]
    print(f"fixed-step curriculum runs={len(runs)} remaining={len(todo)}")
    if args.dry_run:
        for target, p_ar in todo:
            ar_steps, bd_steps = split_fixed_steps(target.total_steps, p_ar)
            print(
                f"N={target.spec.label} p={p_ar:.1f} total={target.total_steps} "
                f"AR={ar_steps} BD={bd_steps} lr={selected_lr(target):.6g}"
            )
        return
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, run, args.device): run
            for run in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run, process = future.result()
            target, p_ar = run
            if process.returncode:
                failures.append(str(output_path(target, p_ar)))
                print(
                    f"[{index}/{len(todo)}] FAILED N={target.spec.label} "
                    f"p={p_ar:.1f}: {process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            row = json.loads(output_path(target, p_ar).read_text())
            print(
                f"[{index}/{len(todo)}] N={target.spec.label} p={p_ar:.1f} "
                f"val={row['val_nelbo']:.5f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} fixed-step curriculum runs failed: {failures}"
        )


if __name__ == "__main__":
    main()
