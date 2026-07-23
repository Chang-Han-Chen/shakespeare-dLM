"""Build shared AR trunks for the compute-optimal fixed-step targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ROOT
from fixed_steps_config import FIXED_STEP_TARGETS


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


def execute(target, device):
    output = SHARED / target.spec.label
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_ar_trunk.py"),
        "--budget",
        str(target.predicted_compute),
        "--total-steps",
        str(target.total_steps),
        "--size",
        target.spec.label,
        "--lr",
        str(selected_lr(target)),
        "--skip-pure-ar-endpoint",
        "--output-dir",
        str(output),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return target, process


def main() -> None:
    args = parse_args()
    # Longest-first keeps the four-worker queue balanced near completion.
    points = sorted(
        FIXED_STEP_TARGETS,
        key=lambda target: target.total_steps,
        reverse=True,
    )
    if args.size:
        selected = set(args.size)
        unknown = selected - {target.spec.label for target in points}
        if unknown:
            raise ValueError(f"Unknown model sizes: {sorted(unknown)}")
        points = [target for target in points if target.spec.label in selected]
    todo = [
        target
        for target in points
        if not complete(SHARED / target.spec.label / "trunk_result.json")
    ]
    print(f"fixed-step AR trunks={len(points)} remaining={len(todo)}")
    if args.dry_run:
        for target in todo:
            print(
                f"N={target.spec.label} steps={target.total_steps} "
                f"lr={selected_lr(target):.6g}"
            )
        return
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, target, args.device): target
            for target in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            target, process = future.result()
            output = SHARED / target.spec.label
            if process.returncode:
                failures.append(str(output))
                print(
                    f"[{index}/{len(todo)}] FAILED N={target.spec.label}: "
                    f"{process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            row = json.loads((output / "trunk_result.json").read_text())
            print(
                f"[{index}/{len(todo)}] N={target.spec.label} "
                f"trunk_steps={row['trunk_steps']} "
                f"run_s={row['trunk_duration_seconds']:.1f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise RuntimeError(f"{len(failures)} fixed-step AR trunks failed: {failures}")


if __name__ == "__main__":
    main()
